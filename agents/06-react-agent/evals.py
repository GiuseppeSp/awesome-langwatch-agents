"""
Evaluators for the ReAct agent.

Three scorers — one programmatic, two LLM-as-judge:

    answer_correctness    — LLM-as-judge. Does the final answer match the
                            labeled expected answer?
    tool_call_efficiency  — Programmatic. Did the agent call the right tools
                            in roughly the right amount (no wasted calls)?
    reasoning_quality     — LLM-as-judge. Are the agent's recorded thoughts
                            actually useful? (Only meaningful in react mode;
                            returns N/A for bare mode where no thoughts exist.)
"""

from __future__ import annotations

import os
from collections import Counter
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


# ---- Evaluator 1: answer_correctness (LLM-as-judge) ----

_CORRECTNESS_RUBRIC = """\
You are checking whether an assistant's answer to a question is correct. You
will get the question, an EXPECTED answer (ground truth, possibly approximate),
and the assistant's actual answer. Score 1-5.

5 — Answer is correct: matches the expected answer's key value (e.g. right number
    within 1, right name, right date).
4 — Answer is correct but with extra detail that's still accurate.
3 — Answer is partially correct (off by a small amount, or correct but missing
    a required component).
2 — Answer is mostly wrong (right topic, wrong specifics — wrong number,
    wrong name).
1 — Answer is fully wrong, off-topic, or empty.

Reply with ONLY a single integer 1-5 followed by a colon and a one-sentence reason.
"""


def answer_correctness(query: str, expected: str, actual: str) -> EvalResult:
    user = (
        f"Question: {query}\n\n"
        f"Expected answer (ground truth):\n{expected}\n\n"
        f"Assistant's actual answer:\n{actual}"
    )
    with langwatch.span(name="judge_correctness", type="evaluation") as s:
        s.update(input=query)
        completion = _client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _CORRECTNESS_RUBRIC},
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
    return _parse_rubric_result("answer_correctness", raw)


# ---- Evaluator 2: tool_call_efficiency (programmatic) ----

def tool_call_efficiency(actual: list[str], expected: list[str]) -> EvalResult:
    """
    Reward calling the right multiset of tools; penalize extras and omissions.

    Score = F1 of the actual-vs-expected tool multisets.
    Passes at >= 0.7. If the agent called no tools at all, score is 0.
    """
    if not actual:
        return EvalResult(
            name="tool_call_efficiency",
            passed=False,
            score=0.0,
            reason="No tools called",
        )

    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    overlap = sum((expected_counts & actual_counts).values())
    n_expected = sum(expected_counts.values())
    n_actual = sum(actual_counts.values())

    precision = overlap / n_actual if n_actual else 0.0
    recall = overlap / n_expected if n_expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return EvalResult(
        name="tool_call_efficiency",
        passed=f1 >= 0.7,
        score=f1,
        reason=(
            f"expected {dict(expected_counts)} → actual {dict(actual_counts)}; "
            f"precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}"
        ),
    )


# ---- Evaluator 3: reasoning_quality (LLM-as-judge, react-mode-only) ----

_REASONING_RUBRIC = """\
You are judging the quality of a chain of explicit reasoning steps an agent
emitted while solving a problem. You will get the question, the agent's recorded
thoughts (one per loop iteration), and the final answer. Score 1-5.

5 — Reasoning is clear and sequential; each thought directly justifies the next
    action; no filler.
4 — Reasoning is mostly clear with one weak or redundant thought.
3 — Reasoning is mixed: some steps are useful, some are filler.
2 — Reasoning is mostly padding; doesn't actually inform the actions.
1 — Reasoning is missing, incoherent, or contradicts the actions taken.

Reply with ONLY a single integer 1-5 followed by a colon and a one-sentence reason.
"""


def reasoning_quality(query: str, thoughts: list[str], final_answer: str) -> EvalResult:
    if not thoughts:
        return EvalResult(
            name="reasoning_quality",
            passed=False,
            score=0.0,
            reason="No thoughts emitted (expected in bare mode; flagged in react mode)",
        )
    thoughts_block = "\n".join(f"- {t}" for t in thoughts)
    user = (
        f"Question: {query}\n\n"
        f"Recorded thoughts (in order):\n{thoughts_block}\n\n"
        f"Final answer: {final_answer}"
    )
    with langwatch.span(name="judge_reasoning", type="evaluation") as s:
        s.update(input=query)
        completion = _client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _REASONING_RUBRIC},
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
    return _parse_rubric_result("reasoning_quality", raw)


# ---- Shared rubric parser ----

def _parse_rubric_result(name: str, raw: str) -> EvalResult:
    head, _, tail = raw.partition(":")
    try:
        score_int = int(head.strip())
    except ValueError:
        return EvalResult(name=name, passed=False, score=0.0, reason=f"Unparseable: {raw}")
    score_int = max(1, min(5, score_int))
    normalized = (score_int - 1) / 4
    return EvalResult(
        name=name,
        passed=score_int >= 4,
        score=normalized,
        reason=f"score={score_int}/5 — {tail.strip() or raw}",
    )
