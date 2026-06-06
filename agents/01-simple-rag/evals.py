"""
Evaluators for the simple-rag agent.

Two programmatic evaluators (cheap, deterministic) and one LLM-as-judge
(more nuanced, costs tokens). Each evaluator returns a bool or numeric
score and a short reason string, so failures are debuggable, not opaque.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

import langwatch

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
_client = OpenAI()


@dataclass
class EvalResult:
    name: str
    passed: bool         # True/False — used for aggregate pass rates
    score: float         # 0..1 (or 1..5 for rubric, normalized)
    reason: str = ""


# ---- Evaluator 1: retrieval@k (programmatic, deterministic) ----

def retrieval_at_k(retrieved: list[dict], expected_source: str, k: int | None = None) -> EvalResult:
    """
    Did any of the top-k retrieved chunks come from the expected_source entry?

    This is a *binary* signal — either the right entry was surfaced or it wasn't.
    Tracks the most common RAG failure mode: the retriever bringing back
    plausible-but-wrong chunks.
    """
    titles = [r["source"] for r in retrieved[: k or len(retrieved)]]
    passed = expected_source in titles
    return EvalResult(
        name="retrieval_at_k",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason=f"Retrieved: {titles}; expected: {expected_source}",
    )


# ---- Evaluator 2: keyword presence (programmatic, lenient) ----

def keyword_match(answer: str, expected_keywords: list[str]) -> EvalResult:
    """
    Does the answer mention every expected keyword (case-insensitive)?

    Lenient proxy for substantive completeness — useful for catching answers
    that retrieve the right context but synthesize a vague or incomplete
    response.
    """
    answer_lc = answer.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in answer_lc]
    missed = [kw for kw in expected_keywords if kw.lower() not in answer_lc]
    score = len(hits) / max(len(expected_keywords), 1)
    passed = len(missed) == 0
    return EvalResult(
        name="keyword_match",
        passed=passed,
        score=score,
        reason=f"Hit {len(hits)}/{len(expected_keywords)}; missing: {missed}" if missed else "All keywords present",
    )


# ---- Evaluator 3: faithfulness (LLM-as-judge, costs tokens) ----

_FAITHFULNESS_RUBRIC = """\
You are evaluating whether an answer is faithful to the retrieved context it
was generated from. Score the answer from 1 to 5:

5 — Fully grounded. Every factual claim in the answer is supported by the context.
4 — Mostly grounded. One minor claim is not directly supported but is plausible.
3 — Partially grounded. About half of the substantive claims come from the context.
2 — Weakly grounded. Most of the answer is invented or pulled from outside the context.
1 — Contradictory. The answer states something the context refutes, OR the answer is fabricated wholesale.

Respond with ONLY a single integer 1-5 followed by a colon and a one-sentence reason.
Example: "5: Every detail in the answer maps to a sentence in the context."
"""


def faithfulness(question: str, retrieved: list[dict], answer: str) -> EvalResult:
    """
    LLM-as-judge: does the answer actually use the retrieved context, or did
    the model invent things?

    Returns a 1-5 score normalized to 0..1. Marked as `passed` when the score
    is 4 or 5.
    """
    context = "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in retrieved)
    user = f"Question: {question}\n\nRetrieved context:\n{context}\n\nAnswer: {answer}"

    with langwatch.span(name="judge_faithfulness", type="evaluation") as s:
        s.update(input=question)
        completion = _client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _FAITHFULNESS_RUBRIC},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        raw = (completion.choices[0].message.content or "").strip()
        s.update(output=raw, metrics={
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
        })

    # Parse "N: reason"
    score_int = 0
    reason = raw
    head, _, tail = raw.partition(":")
    try:
        score_int = int(head.strip())
        reason = tail.strip() or raw
    except ValueError:
        # Judge didn't follow format — treat as a 0 with the raw output as reason.
        reason = f"Judge output unparseable: {raw}"

    score_int = max(1, min(5, score_int)) if score_int else 0
    normalized = (score_int - 1) / 4 if score_int else 0.0
    return EvalResult(
        name="faithfulness",
        passed=score_int >= 4,
        score=normalized,
        reason=f"score={score_int}/5 — {reason}",
    )
