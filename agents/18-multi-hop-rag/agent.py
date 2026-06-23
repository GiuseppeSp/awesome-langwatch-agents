"""
Multi-hop RAG agent — answer questions that require chaining several retrievals,
where the bridge entity for hop N+1 is only learned from hop N.

Switch via env var:
    MH_MODE=single   (default) — retrieve top-k once on the raw question, generate.
                                 1 retrieval, 1 LLM call.
    MH_MODE=multihop           — a self-ask loop: the LLM repeatedly asks for the
                                 next single fact it needs (SEARCH: ...), each is
                                 retrieved and added to the evidence, until it can
                                 ANSWER. Several retrievals, several LLM calls.

The corpus is built so each passage holds exactly ONE fact and the answer-bearing
passage shares no keyword with the original question (the bridge entity isn't in
the question). So single-shot retrieval structurally cannot reach the answer on a
multi-hop question — it can only retrieve the first hop. The experiment measures
when paying for the loop is necessary versus wasted.

Trace tree (typed LangWatch spans):

    multi_hop_rag (workflow root)
    ├─ single:   retrieve (rag) -> generate (llm)
    └─ multihop: hop_k: decide (llm) -> retrieve (rag)  [repeated] -> final answer
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

MH_MODE = os.getenv("MH_MODE", "single")  # "single" | "multihop"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
TOP_K = int(os.getenv("TOP_K", "3"))
MAX_HOPS = int(os.getenv("MAX_HOPS", "4"))

os.environ.setdefault("OTEL_SERVICE_NAME", "multi-hop-rag")
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Corpus ----

def _load_passages(filename: str) -> list[str]:
    text = (Path(__file__).parent / filename).read_text(encoding="utf-8")
    return [line[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]


CORPUS = _load_passages("corpus.md")


# ---- Retriever (keyword overlap with light stemming) ----

_STOP = set("the a an of to in on and or is are was were be by for with at from as it its what "
            "where who how does do you your i can that this these those which place body".split())


def _stem(w: str) -> str:
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def _tokens(s: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"\w+", s.lower()) if w not in _STOP}


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


_GEN_SYSTEM = (
    "Answer the question using ONLY the provided context passages. Be concise — a short "
    "phrase or name. If the context does not contain the answer, reply exactly: I don't know."
)

_HOP_SYSTEM = (
    "You answer questions that may require looking up several facts in sequence, one at a time. "
    "You can search a knowledge base for one fact per step.\n"
    "Given the QUESTION and the FACTS gathered so far, respond with EXACTLY one of:\n"
    "  SEARCH: <a focused query for the single next fact you still need>\n"
    "  ANSWER: <the final answer, a short phrase or name>\n"
    "Search for one missing link at a time (e.g. first which academy a scholar belongs to, then "
    "where that academy is). Only ANSWER once the facts entail it; if they never will, "
    "ANSWER: I don't know."
)


def _generate(query: str, passages: list[str]) -> tuple[str, int, int]:
    ctx = "\n".join(f"- {p}" for p in passages) if passages else "(no passages)"
    return _chat(_GEN_SYSTEM, f"CONTEXT:\n{ctx}\n\nQUESTION: {query}")


# ---- Result type ----

@dataclass
class MHResult:
    query: str
    mode: str
    answer: str = ""
    hops: int = 0
    llm_calls: int = 0
    retrieved: list[str] = field(default_factory=list)   # union of all passages seen
    sub_queries: list[str] = field(default_factory=list)


# ---- The agent ----

@langwatch.trace(name="multi_hop_rag")
def run(query: str, *, mode: str = MH_MODE) -> MHResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {query}")
    result = MHResult(query=query, mode=mode)

    if mode == "single":
        with langwatch.span(name="retrieve", type="rag") as s:
            s.update(input=query)
            passages = retrieve(query)
            s.update(output="\n".join(passages)[:500])
        result.retrieved = passages
        with langwatch.span(name="generate", type="llm") as s:
            s.update(input=query)
            ans, pt, ct = _generate(query, passages)
            s.update(output=ans[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
        result.llm_calls += 1
        result.answer = ans
        root.update(output=f"[single] {ans[:160]}")
        return result

    if mode != "multihop":
        raise ValueError(f"unknown mode: {mode!r}")

    # --- self-ask loop ---
    facts: list[str] = []
    seen: set[str] = set()
    for hop in range(1, MAX_HOPS + 1):
        facts_block = "\n".join(f"- {f}" for f in facts) if facts else "(none yet)"
        with langwatch.span(name=f"hop_{hop}", type="agent") as hs:
            with langwatch.span(name="decide", type="llm") as s:
                user = f"QUESTION: {query}\n\nFACTS SO FAR:\n{facts_block}"
                s.update(input=user[-400:])
                text, pt, ct = _chat(_HOP_SYSTEM, user)
                s.update(output=text[:200], metrics={"prompt_tokens": pt, "completion_tokens": ct})
            result.llm_calls += 1

            m_ans = re.search(r"ANSWER:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
            m_search = re.search(r"SEARCH:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
            if m_ans and (not m_search or m_ans.start() < m_search.start()):
                result.answer = m_ans.group(1).strip().splitlines()[0].strip()
                hs.update(output=f"answer: {result.answer}")
                break
            sub_q = (m_search.group(1) if m_search else query).strip().splitlines()[0].strip()
            result.sub_queries.append(sub_q)
            with langwatch.span(name="retrieve", type="rag") as s:
                s.update(input=sub_q)
                hits = retrieve(sub_q)
                s.update(output="\n".join(hits)[:400])
            result.hops += 1
            for h in hits:
                if h not in seen:
                    seen.add(h); facts.append(h)
            result.retrieved = list(facts)
            hs.update(output=f"search: {sub_q[:60]} -> {len(hits)} passage(s)")
    else:
        # ran out of hops without an explicit ANSWER — make a final attempt
        with langwatch.span(name="generate", type="llm") as s:
            s.update(input=query)
            ans, pt, ct = _generate(query, facts)
            s.update(output=ans[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
        result.llm_calls += 1
        result.answer = ans

    root.update(output=f"[multihop hops={result.hops}] {result.answer[:150]}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "In which place does the scholar Aldric work?"
    mode = os.getenv("MH_MODE", "single")
    print(f"\n=== Query: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    if res.mode == "multihop":
        print(f"Sub-queries ({res.hops}):")
        for i, sq in enumerate(res.sub_queries, 1):
            print(f"  hop {i}: {sq}")
        print("\nFacts gathered:")
        for f in res.retrieved:
            print(f"  - {f}")
    else:
        print(f"Retrieved ({len(res.retrieved)}):")
        for p in res.retrieved:
            print(f"  - {p}")
    print(f"\nAnswer: {res.answer}")
    print(f"=== llm calls: {res.llm_calls}  |  hops: {res.hops}")
