"""
Evaluators for the multi-turn travel chatbot.

Three scorers tuned for the memory-strategy comparison:

    context_recall    — LLM-as-judge. Does the answer reflect the early-turn
                        constraint the user stated? This is the headline eval.
    must_include      — Programmatic. Does the answer mention at least one of
                        the keywords the right answer should mention?
    must_not_include  — Programmatic. Does the answer AVOID keywords that would
                        violate the user's constraint (allergens, etc.)?

The `context_recall` eval is the differentiator: it directly measures whether
memory worked. The other two are cheap programmatic checks that catch obvious
wins and obvious failures.
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
    passed: bool
    score: float
    reason: str = ""


# ---- Evaluator 1: must_include (programmatic, lenient OR) ----

def must_include(answer: str, keywords_any: list[str]) -> EvalResult:
    """
    Does the answer mention AT LEAST ONE of the expected keywords (case-insensitive)?

    `keywords_any` is an OR list — the test passes if any one of them appears.
    For example a pescetarian-traveler test might accept either "fish" or
    "seafood" or "sardine".
    """
    answer_lc = answer.lower()
    hits = [kw for kw in keywords_any if kw.lower() in answer_lc]
    passed = len(hits) > 0
    return EvalResult(
        name="must_include",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason=f"Hits: {hits}" if hits else f"Missed all of: {keywords_any}",
    )


# ---- Evaluator 2: must_not_include (programmatic, strict NONE) ----

def must_not_include(answer: str, forbidden: list[str]) -> EvalResult:
    """
    Does the answer AVOID every forbidden keyword?

    Used for catching memory failures that produce actively wrong recommendations
    — e.g. suggesting steak to a pescetarian, or peanut dishes to someone with a
    peanut allergy.
    """
    if not forbidden:
        return EvalResult(name="must_not_include", passed=True, score=1.0, reason="No forbidden terms specified.")
    answer_lc = answer.lower()
    hits = [kw for kw in forbidden if kw.lower() in answer_lc]
    passed = len(hits) == 0
    return EvalResult(
        name="must_not_include",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="No forbidden terms found." if passed else f"FAIL — contains: {hits}",
    )


# ---- Evaluator 3: context_recall (LLM-as-judge, the headline metric) ----

_CONTEXT_RECALL_RUBRIC = """\
You are evaluating whether an assistant's reply correctly reflects a constraint
the user stated earlier in the conversation.

Reply with ONLY a single integer 1-5 followed by a colon and a one-sentence reason.

5 — The reply explicitly acknowledges the constraint AND adapts its recommendation to honor it.
4 — The reply adapts its recommendation to honor the constraint, even if not explicitly named.
3 — The reply is broadly consistent with the constraint but doesn't directly address it.
2 — The reply ignores the constraint, though doesn't actively contradict it.
1 — The reply contradicts the constraint (e.g. recommends something the user said they couldn't have).

Example: "5: Explicitly mentions vegan options and recommends three plant-based restaurants."
"""


def context_recall(test_question: str, answer: str, expected_constraint: str) -> EvalResult:
    """
    LLM-as-judge: did the assistant's reply actually honor the constraint the
    user stated earlier?

    This is the metric the whole agent exists to measure. When the memory
    strategy works, this score should be high. When the memory strategy drops
    the early constraint, this score should drop sharply.
    """
    user = (
        f"User's earlier constraint: {expected_constraint}\n\n"
        f"User's current question: {test_question}\n\n"
        f"Assistant's reply: {answer}"
    )
    with langwatch.span(name="judge_context_recall", type="evaluation") as s:
        s.update(input=expected_constraint)
        completion = _client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _CONTEXT_RECALL_RUBRIC},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        raw = (completion.choices[0].message.content or "").strip()
        s.update(
            output=raw,
            metrics={
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
            },
        )

    # Parse "N: reason"
    score_int = 0
    reason = raw
    head, _, tail = raw.partition(":")
    try:
        score_int = int(head.strip())
        reason = tail.strip() or raw
    except ValueError:
        reason = f"Judge output unparseable: {raw}"

    score_int = max(1, min(5, score_int)) if score_int else 0
    normalized = (score_int - 1) / 4 if score_int else 0.0
    return EvalResult(
        name="context_recall",
        passed=score_int >= 4,
        score=normalized,
        reason=f"score={score_int}/5 — {reason}",
    )
