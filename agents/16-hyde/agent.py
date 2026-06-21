"""
HyDE agent — Hypothetical Document Embeddings (Gao et al., 2022).

Instead of retrieving with the user's question, HyDE first asks an LLM to write
a *hypothetical answer* — a short passage that would answer the question if it
existed — and retrieves with THAT. The intuition: a corpus is full of answers
(statements), and a question is a poor structural match for an answer. A fake
answer is shaped like the real documents, so it retrieves them better, even if
some invented details are wrong.

Switch via env var:
    HYDE_MODE=vanilla  (default) — retrieve on the raw question, then generate.
                                   1 LLM call.
    HYDE_MODE=hyde               — an LLM writes a hypothetical answer; retrieve
                                   on THAT; then generate from the real passages.
                                   2 LLM calls.

This is the sibling of #15 (query-rewriting): both transform the retrieval key
before retrieving, on the SAME corpus and dataset. #15 keeps the key a *question*
(just better phrased); HyDE turns it into an *answer*. The generation step still
uses only the real retrieved passages — the hypothetical doc is used ONLY as a
search key, never shown to the final generator (so its invented facts can't leak
into the answer).

Trace tree (typed LangWatch spans):

    hyde (workflow root)
    ├─ vanilla:  retrieve (rag) -> generate (llm)
    └─ hyde:     hypothesize (llm) -> retrieve (rag) -> generate (llm)
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

HYDE_MODE = os.getenv("HYDE_MODE", "vanilla")  # "vanilla" | "hyde"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
TOP_K = int(os.getenv("TOP_K", "3"))
SERVICE_NAME = "hyde"

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
            "where who how does do you your i can a an that this these those into once".split())


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"\w+", s.lower()) if w not in _STOP}


def retrieve(key: str, k: int = TOP_K) -> list[str]:
    q = _tokens(key)
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


_HYDE_SYSTEM = (
    "Write a short, plausible passage (1-2 sentences) that directly answers the question, "
    "as if it were an excerpt from a reference book. Be specific and declarative; invent "
    "concrete details if needed. Do NOT hedge or say you are unsure. Output only the passage."
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
class HydeResult:
    query: str
    mode: str
    answer: str = ""
    hypothetical: str = ""        # the fake answer used as the retrieval key
    used_hyde: bool = False
    llm_calls: int = 0
    retrieved: list[str] = field(default_factory=list)


# ---- The agent ----

@langwatch.trace(name="hyde")
def run(query: str, *, mode: str = HYDE_MODE) -> HydeResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {query}")
    result = HydeResult(query=query, mode=mode)

    if mode not in ("vanilla", "hyde"):
        raise ValueError(f"unknown mode: {mode!r}")

    # --- hypothesize (hyde mode only): write a fake answer to search with ---
    search_key = query
    if mode == "hyde":
        with langwatch.span(name="hypothesize", type="llm") as s:
            s.update(input=query)
            hypo, pt, ct = _chat(_HYDE_SYSTEM, query)
            s.update(output=hypo[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
        result.llm_calls += 1
        result.hypothetical = hypo.strip()
        result.used_hyde = True
        # search with the hypothetical answer (fall back to the query if it's empty)
        search_key = result.hypothetical or query

    # --- retrieve (on the raw query OR the hypothetical doc) ---
    with langwatch.span(name="retrieve", type="rag") as s:
        s.update(input=search_key[:300])
        passages = retrieve(search_key)
        s.update(output="\n".join(passages)[:500])
    result.retrieved = passages

    # --- generate (from the REAL passages only; the hypothetical is never shown here) ---
    with langwatch.span(name="generate", type="llm") as s:
        s.update(input=query)
        ans, pt, ct = _generate(query, passages)
        s.update(output=ans[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.answer = ans

    root.update(output=f"[{mode}{' hypo' if result.used_hyde else ''}] {ans[:160]}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What can I give a child for a sore tooth?"
    mode = os.getenv("HYDE_MODE", "vanilla")
    print(f"\n=== Query: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    if res.used_hyde:
        print(f"Hypothetical doc (search key):\n  {res.hypothetical}\n")
    print(f"Retrieved ({len(res.retrieved)}):")
    for p in res.retrieved:
        print(f"  - {p[:88]}")
    print(f"\nAnswer: {res.answer}")
    print(f"=== llm calls: {res.llm_calls}")
