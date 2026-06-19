"""
Corrective-RAG (CRAG) agent — retrieve, then grade the retrieval and correct it
before generating (Yan et al., 2024).

Switch via env var:
    CRAG_MODE=vanilla  (default) — retrieve top-k from the local corpus, generate.
                                   1 LLM call. No safety net.
    CRAG_MODE=crag               — retrieve local, then a *grader* judges whether
                                   the retrieved passages can answer the question.
                                   CORRECT  -> generate from the local passages.
                                   INCORRECT -> discard them, web_search a fallback
                                   corpus, generate from that instead.
                                   2 LLM calls (grade + generate).

Both modes share the same retriever, the same corpora, and the same generator.
The ONLY thing CRAG adds is the grade-and-correct step — so the comparison
isolates exactly that. The headline question is not "does correcting help?" but
"**how good does the retrieval grader have to be for correction to pay off — and
what happens on the rows where the bad retrieval LOOKS relevant?**"

Trace tree (typed LangWatch spans):

    corrective_rag (workflow root)
    ├─ vanilla:  retrieve (rag) -> generate (llm)
    └─ crag:     retrieve (rag) -> grade (evaluation) ->
                 [if INCORRECT] web_search (rag) -> generate (llm)
                 [if CORRECT]                       generate (llm)
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

CRAG_MODE = os.getenv("CRAG_MODE", "vanilla")  # "vanilla" | "crag"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
TOP_K = int(os.getenv("TOP_K", "3"))
SERVICE_NAME = "corrective-rag"

os.environ.setdefault("OTEL_SERVICE_NAME", SERVICE_NAME)
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Corpora ----

def _load_passages(filename: str) -> list[str]:
    text = (Path(__file__).parent / filename).read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            out.append(line[2:].strip())
    return out


LOCAL_CORPUS = _load_passages("corpus.md")
WEB_CORPUS = _load_passages("web_corpus.md")


# ---- Retriever (keyword overlap; non-LLM) ----

_STOP = set("the a an of to in on and or is are was were be by for with at from as it its".split())


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"\w+", s.lower()) if w not in _STOP}


def retrieve(query: str, corpus: list[str], k: int = TOP_K) -> list[str]:
    q = _tokens(query)
    scored = [(len(q & _tokens(p)), p) for p in corpus]
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
    "Answer the question using ONLY the provided context passages. Be concise — a "
    "short phrase or name. If the context does not contain the answer, reply exactly: "
    "I don't know."
)

_GRADE_SYSTEM = (
    "You are a retrieval grader. Given a question and some retrieved passages, decide "
    "whether the passages actually contain the information needed to answer the question. "
    "Being on the same topic is NOT enough — the specific answer must be present.\n"
    "Reply with exactly one word: CORRECT (the answer is present) or INCORRECT (it is not)."
)


def _generate(query: str, passages: list[str]) -> tuple[str, int, int]:
    ctx = "\n".join(f"- {p}" for p in passages) if passages else "(no passages)"
    return _chat(_GEN_SYSTEM, f"CONTEXT:\n{ctx}\n\nQUESTION: {query}")


# ---- Result type ----

@dataclass
class CragResult:
    query: str
    mode: str
    answer: str = ""
    grade: str = ""            # "CORRECT" | "INCORRECT" | "" (vanilla)
    used_web: bool = False
    llm_calls: int = 0
    retrieved: list[str] = field(default_factory=list)


# ---- The agent ----

@langwatch.trace(name="corrective_rag")
def run(query: str, *, mode: str = CRAG_MODE) -> CragResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {query}")
    result = CragResult(query=query, mode=mode)

    # --- retrieve (both modes) ---
    with langwatch.span(name="retrieve", type="rag") as s:
        s.update(input=query)
        local = retrieve(query, LOCAL_CORPUS)
        s.update(output="\n".join(local)[:500])
    result.retrieved = local

    if mode == "vanilla":
        with langwatch.span(name="generate", type="llm") as s:
            s.update(input=query)
            ans, pt, ct = _generate(query, local)
            s.update(output=ans[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
        result.llm_calls += 1
        result.answer = ans
        root.update(output=f"[vanilla] {ans[:200]}")
        return result

    if mode != "crag":
        raise ValueError(f"unknown mode: {mode!r}")

    # --- grade the retrieval ---
    ctx = "\n".join(f"- {p}" for p in local) if local else "(no passages)"
    with langwatch.span(name="grade", type="evaluation") as s:
        s.update(input=f"Q: {query}\nPASSAGES:\n{ctx}"[:500])
        verdict, pt, ct = _chat(_GRADE_SYSTEM, f"QUESTION: {query}\n\nPASSAGES:\n{ctx}")
        s.update(output=verdict[:100], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.grade = "INCORRECT" if "INCORRECT" in verdict.upper() else "CORRECT"

    # --- correct if needed ---
    passages = local
    if result.grade == "INCORRECT":
        with langwatch.span(name="web_search", type="rag") as s:
            s.update(input=query)
            passages = retrieve(query, WEB_CORPUS)
            s.update(output="\n".join(passages)[:500])
        result.used_web = True
        result.retrieved = passages

    # --- generate ---
    with langwatch.span(name="generate", type="llm") as s:
        s.update(input=query)
        ans, pt, ct = _generate(query, passages)
        s.update(output=ans[:300], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.answer = ans

    root.update(output=f"[crag grade={result.grade}{' +web' if result.used_web else ''}] {ans[:180]}")
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Who is the governor of Ostrov Station?"
    mode = os.getenv("CRAG_MODE", "vanilla")
    print(f"\n=== Query: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    print(f"Retrieved ({'web' if res.used_web else 'local'}):")
    for p in res.retrieved:
        print(f"  - {p[:90]}")
    if res.mode == "crag":
        print(f"\nGrade: {res.grade}{'  -> web_search' if res.used_web else ''}")
    print(f"\nAnswer: {res.answer}")
    print(f"=== llm calls: {res.llm_calls}")
