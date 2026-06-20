"""
Query-rewriting RAG agent — rewrite the query into a better search key BEFORE
retrieving (rewrite-retrieve-read; Ma et al., 2023).

Switch via env var:
    QR_MODE=vanilla  (default) — retrieve on the raw user query, then generate.
                                 1 LLM call.
    QR_MODE=rewrite            — an LLM first rewrites the query into a concise,
                                 formal search query; retrieve on THAT, then
                                 generate. 2 LLM calls.

Both modes share the same retriever and the same generator. The only difference
is whether the query is rewritten before retrieval — so the comparison isolates
exactly that. Where #14 CRAG fixes bad retrieval *after* the fact (grade +
web-correct), query-rewriting tries to prevent it *before* the fact, by fixing
the retrieval key.

The headline question: rewriting obviously can't hurt a query that already
retrieves well — so *when* does it pay? The dataset splits queries into `clean`
(corpus-aligned vocabulary) and `poor` (vague, layperson phrasing), and the
experiment measures retrieval hit-rate in each.

Trace tree (typed LangWatch spans):

    query_rewriting_rag (workflow root)
    ├─ vanilla:  retrieve (rag) -> generate (llm)
    └─ rewrite:  rewrite (llm) -> retrieve (rag) -> generate (llm)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import langwatch

load_dotenv()

# ---- Configuration ----

QR_MODE = os.getenv("QR_MODE", "vanilla")  # "vanilla" | "rewrite"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
TOP_K = int(os.getenv("TOP_K", "3"))
SERVICE_NAME = "query-rewriting-rag"

os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Corpus ----

def _load_passages(filename: str) -> list[str]:
    text = (Path(__file__).parent / filename).read_text(encoding="utf-8")
    return [line[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]


CORPUS = _load_passages("corpus.md")


# ---- Retriever (keyword overlap; non-LLM) ----

_STOP = set("the a an of to in on and or is are was were be by for with at from as it its what "
            "where who how does do you your i can".split())


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"\w+", s.lower()) if w not in _STOP}


def retrieve(query: str, k: int = TOP_K) -> list[str]:
    q = _tokens(query)
    scored = [(len(q & _tokens(p)), p) for p in CORPUS]
    scored = [(s, p) for s, p in scored if s > 0]
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:k]]


# ---- LLM helpers ----

def _chat(system: str, user: str) -> tuple[str, int, int]:
    c = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE,
    )
    return (c.choices[0].message.content or "").strip(), c.usage.prompt_tokens, c.usage.completion_tokens


_REWRITE_SYSTEM = (
    "Rewrite the user's question into a concise search query for a knowledge base. "
    "Replace vague or colloquial wording with the precise, formal terms a reference "
    "text would use, and keep the key nouns. Output ONLY the rewritten query, no quotes, "
    "no explanation."
)

_GEN_SYSTEM = (
    "Answer the question using ONLY the provided context passages. Be concise — a short "
    "phrase or name. If the context does not contain the answer, reply exactly: I don't know."
)


def _generate(query: str, passages: list[str]) -> tuple[str, int, int]:
    ctx = "\n".join(f"- {p}" for p in passages) if passages else "(no passages)"
    return _chat(_GEN_SYSTEM, f"CONTEXT:\n{ctx}\n\nQUESTION: {query}")


# ---- Result type ----

@dataclass
class QRResult:
    query: str
    mode: str
    answer: str = ""
    search_query: str = ""        # the query actually used for retrieval
    rewritten: bool = False
    llm_calls: int = 0
    retrieved: list[str] = field(default_factory=list)


# ---- The agent ----

@langwatch.trace(name="query_rewriting_rag")
def run(query: str, *, mode: str = QR_MODE) -> QRResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {query}")
    result = QRResult(query=query, mode=mode, search_query=query)

    if mode not in ("vanilla", "rewrite"):
        raise ValueError(f"unknown mode: {mode!r}")

    # --- rewrite (rewrite mode only) ---
    if mode == "rewrite":
        with langwatch.span(name="rewrite", type="llm") as s:
            s.update(input=query)
            rewritten, pt, ct = _chat(_REWRITE_SYSTEM, query)
            s.update(output=rewritten[:200], metrics={"prompt_tokens": pt, "completion_tokens": ct})
        result.llm_calls += 1
        result.search_query = rewritten.strip() or query
        result.rewritten = True

    # --- retrieve (on raw or rewritten query) ---
    with langwatch.span(name="retrieve", type="rag") as s:
        s.update(input=result.search_query)
        passages = retrieve(result.search_query)
        s.update(output="\n".join(passages)[:500])
    result.retrieved = passages

    # --- generate ---
    with langwatch.span(name="generate", type="llm") as s:
        s.update(input=query)
        ans, pt, ct = _generate(query, passages)
        s.update(output=ans[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.answer = ans

    tag = f"rewrite='{result.search_query}'" if result.rewritten else "raw"
    root.update(output=f"[{mode} {tag}] {ans[:160]}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What's the glowing moss people use for achy joints called?"
    mode = os.getenv("QR_MODE", "vanilla")
    print(f"\n=== Query: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    if res.rewritten:
        print(f"Rewritten search query: {res.search_query}")
    print(f"Retrieved ({len(res.retrieved)}):")
    for p in res.retrieved:
        print(f"  - {p[:88]}")
    print(f"\nAnswer: {res.answer}")
    print(f"=== llm calls: {res.llm_calls}")
