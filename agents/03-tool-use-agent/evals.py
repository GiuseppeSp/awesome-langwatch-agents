"""
Evaluators for the tool-use agent.

Three scorers, tuned for the tool-description comparison:

    tool_selection      — Programmatic. Did the agent call the right tool
                          (or correctly call NO tool when the query didn't
                          need one)?
    argument_extraction — LLM-as-judge. Were the arguments the agent passed
                          to the tool semantically correct for the query?
    no_tool_correctness — Programmatic. For queries that should NOT trigger a
                          tool, did the agent correctly skip them?

`tool_selection` and `no_tool_correctness` overlap a bit by design — they look
at the same boolean from two angles. Reported separately so the breakdown is
readable: which mode is over-triggering vs which mode is under-triggering.
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


# ---- Evaluator 1: tool_selection (programmatic) ----

def tool_selection(actual_tool: str, expected_tool: str) -> EvalResult:
    """
    Did the agent call the expected tool? Includes the 'expected=none' case
    where the right answer is to skip tools entirely.
    """
    passed = actual_tool == expected_tool
    return EvalResult(
        name="tool_selection",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason=f"actual={actual_tool!r} expected={expected_tool!r}",
    )


# ---- Evaluator 2: no_tool_correctness (programmatic) ----

def no_tool_correctness(actual_tool: str, expected_tool: str) -> EvalResult:
    """
    Narrower lens on the same boolean as tool_selection: only relevant when
    expected_tool is 'none'. Asks: did the agent correctly refrain from
    calling a tool? Helps separate 'over-triggering' from 'wrong tool' errors
    when you read the aggregate scores.
    """
    if expected_tool != "none":
        # Not applicable to this row. Don't count it for or against.
        return EvalResult(name="no_tool_correctness", passed=True, score=1.0, reason="N/A (expected a tool)")
    passed = actual_tool == "none"
    return EvalResult(
        name="no_tool_correctness",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason=f"Tool-free expected; actual={actual_tool!r}",
    )


# ---- Evaluator 3: argument_extraction (LLM-as-judge) ----

_ARG_RUBRIC = """\
You are evaluating whether a tool was called with semantically correct arguments.

Reply with ONLY a single integer 1-5 followed by a colon and a one-sentence reason.

5 — Arguments are exactly right and would produce the correct result.
4 — Arguments are functionally correct (right values, maybe minor unit/format quirk).
3 — Arguments are partially right (one value off, missing a small detail).
2 — Arguments are mostly wrong but show the agent understood the right tool.
1 — Arguments are wrong or nonsensical for the query.

If the expected tool was 'none' and the agent didn't call a tool, reply with "5: N/A — no tool called as expected".
If the expected tool was 'none' and the agent DID call a tool, reply with "1: Over-triggered — should not have called a tool".
"""


def argument_extraction(query: str, expected_tool: str, expected_args_hint: str, tool_calls: list) -> EvalResult:
    """
    LLM-judge: were the arguments the agent passed semantically right for the
    query?

    `tool_calls` is the list of ToolCall objects from the agent. We only score
    the FIRST tool call (the one tool_selection also looks at) to keep the
    rubric clean.
    """
    actual = tool_calls[0] if tool_calls else None
    actual_summary = (
        f"name={actual.name}, args={actual.args}" if actual else "no tool called"
    )
    user = (
        f"Query: {query}\n"
        f"Expected tool: {expected_tool}\n"
        f"Expected-args hint (informal): {expected_args_hint}\n"
        f"Agent's actual tool call: {actual_summary}"
    )
    with langwatch.span(name="judge_argument_extraction", type="evaluation") as s:
        s.update(input=query)
        completion = _client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _ARG_RUBRIC},
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
        name="argument_extraction",
        passed=score_int >= 4,
        score=normalized,
        reason=f"score={score_int}/5 — {reason}",
    )
