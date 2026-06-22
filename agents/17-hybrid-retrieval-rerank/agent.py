"""
Hybrid-retrieval + rerank agent — fix retrieval at the *ranking* layer instead
of by transforming the query (#15, #16).

Three modes form a staircase, each adding one layer:
    HR_MODE=lexical  (default) — keyword-overlap retrieval -> top-k -> generate.
                                 (This IS the 'vanilla' baseline from #15/#16.)
    HR_MODE=hybrid             — fuse keyword + SEMANTIC (embedding cosine)
                                 rankings via Reciprocal Rank Fusion -> top-k.
    HR_MODE=rerank             — take a wider hybrid candidate pool, have an LLM
                                 rerank it, keep top-k -> generate.

Same corpus + 14 queries as #15/#16. Those agents transformed the QUERY to make
lexical retrieval work on vocabulary-mismatched ("poor") questions; this one
changes the RETRIEVER. The questions worth answering:
  1. Does semantic retrieval fix the vocab-mismatch misses directly, with no
     query rewriting at all?
  2. Does reranking add anything once retrieval is already decent — i.e. does its
     precondition (the gold doc is in the candidate pool but mis-ranked) even hold
     at this corpus size?

Trace tree (typed LangWatch spans):

    hybrid_retrieval_rerank (workflow root)
    ├─ lexical: retrieve (rag) -> generate (llm)
    ├─ hybrid:  retrieve (rag, keyword + embedding fused) -> generate (llm)
    └─ rerank:  retrieve (rag, candidate pool) -> rerank (evaluation) -> generate (llm)
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

HR_MODE = os.getenv("HR_MODE", "lexical")  # "lexical" | "hybrid" | "rerank"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
TOP_K = int(os.getenv("TOP_K", "3"))
POOL = int(os.getenv("POOL", "6"))          # candidate pool size before rerank
RRF_K = 60                                   # standard reciprocal-rank-fusion constant

os.environ.setdefault("OTEL_SERVICE_NAME", "hybrid-retrieval-rerank")
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Corpus ----

def _load_passages(filename: str) -> list[str]:
    text = (Path(__file__).parent / filename).read_text(encoding="utf-8")
    return [line[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]


CORPUS = _load_passages("corpus.md")


# ---- Lexical retrieval (keyword overlap) ----

_STOP = set("the a an of to in on and or is are was were be by for with at from as it its what "
            "where who how does do you your i can that this these those into once".split())


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"\w+", s.lower()) if w not in _STOP}


def _lexical_ranking(query: str) -> list[int]:
    """Return passage indices with >0 keyword overlap, best first."""
    q = _tokens(query)
    scored = [(len(q & _tokens(p)), i) for i, p in enumerate(CORPUS)]
    scored = [(s, i) for s, i in scored if s > 0]
    scored.sort(key=lambda x: -x[0])
    return [i for _, i in scored]


# ---- Semantic retrieval (embeddings, cosine) ----

_PASSAGE_VECS: list[list[float]] | None = None


def _embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _passage_vecs() -> list[list[float]]:
    global _PASSAGE_VECS
    if _PASSAGE_VECS is None:
        _PASSAGE_VECS = _embed(CORPUS)
    return _PASSAGE_VECS


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _semantic_ranking(query: str) -> list[int]:
    qv = _embed([query])[0]
    sims = [( _cosine(qv, pv), i) for i, pv in enumerate(_passage_vecs())]
    sims.sort(key=lambda x: -x[0])
    return [i for _, i in sims]


# ---- Reciprocal Rank Fusion ----

def _rrf(*rankings: list[int]) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])]


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

_RERANK_SYSTEM = (
    "You are a passage reranker. Given a question and numbered candidate passages, return the "
    "numbers of the passages most relevant to answering the question, best first, comma-separated "
    "(e.g. '3, 1, 5'). Only return numbers."
)


def _generate(query: str, passages: list[str]) -> tuple[str, int, int]:
    ctx = "\n".join(f"- {p}" for p in passages) if passages else "(no passages)"
    return _chat(_GEN_SYSTEM, f"CONTEXT:\n{ctx}\n\nQUESTION: {query}")


def _rerank(query: str, candidate_idxs: list[int]) -> tuple[list[int], int, int]:
    listing = "\n".join(f"{n+1}. {CORPUS[idx]}" for n, idx in enumerate(candidate_idxs))
    text, pt, ct = _chat(_RERANK_SYSTEM, f"QUESTION: {query}\n\nCANDIDATES:\n{listing}")
    order = [int(m) for m in re.findall(r"\d+", text)]
    reranked = [candidate_idxs[n - 1] for n in order if 1 <= n <= len(candidate_idxs)]
    # append any candidates the model omitted, preserving original order
    reranked += [i for i in candidate_idxs if i not in reranked]
    return reranked, pt, ct


# ---- Result type ----

@dataclass
class HRResult:
    query: str
    mode: str
    answer: str = ""
    final_idxs: list[int] = field(default_factory=list)   # passages shown to the generator
    pool_idxs: list[int] = field(default_factory=list)     # candidates considered before top-k
    llm_calls: int = 0

    @property
    def retrieved(self) -> list[str]:
        return [CORPUS[i] for i in self.final_idxs]

    @property
    def pool(self) -> list[str]:
        return [CORPUS[i] for i in self.pool_idxs]


# ---- The agent ----

@langwatch.trace(name="hybrid_retrieval_rerank")
def run(query: str, *, mode: str = HR_MODE) -> HRResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {query}")
    result = HRResult(query=query, mode=mode)

    if mode not in ("lexical", "hybrid", "rerank"):
        raise ValueError(f"unknown mode: {mode!r}")

    # --- retrieve (produces a ranked candidate list) ---
    with langwatch.span(name="retrieve", type="rag") as s:
        s.update(input=query)
        if mode == "lexical":
            ranked = _lexical_ranking(query)
        else:
            ranked = _rrf(_lexical_ranking(query), _semantic_ranking(query))
        result.pool_idxs = ranked[: POOL] if mode == "rerank" else ranked
        s.update(output="\n".join(CORPUS[i] for i in result.pool_idxs)[:500])

    # --- rerank (rerank mode only) ---
    if mode == "rerank" and result.pool_idxs:
        with langwatch.span(name="rerank", type="evaluation") as s:
            s.update(input=f"{query} | {len(result.pool_idxs)} candidates")
            reranked, pt, ct = _rerank(query, result.pool_idxs)
            s.update(output="\n".join(CORPUS[i] for i in reranked[:TOP_K])[:400],
                     metrics={"prompt_tokens": pt, "completion_tokens": ct})
        result.llm_calls += 1
        ranked_final = reranked
    else:
        ranked_final = result.pool_idxs

    result.final_idxs = ranked_final[:TOP_K]

    # --- generate ---
    with langwatch.span(name="generate", type="llm") as s:
        s.update(input=query)
        ans, pt, ct = _generate(query, result.retrieved)
        s.update(output=ans[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.answer = ans

    root.update(output=f"[{mode}] {ans[:160]}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What do sailors take so they don't get sick on boats?"
    mode = os.getenv("HR_MODE", "lexical")
    print(f"\n=== Query: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    if mode == "rerank":
        print(f"Candidate pool ({len(res.pool_idxs)}):")
        for i in res.pool_idxs:
            print(f"  - {CORPUS[i][:80]}")
    print(f"Final top-{TOP_K}:")
    for p in res.retrieved:
        print(f"  - {p[:80]}")
    print(f"\nAnswer: {res.answer}")
    print(f"=== llm calls: {res.llm_calls}")
