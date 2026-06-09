"""
Evaluators for the agentic-router.

Three scorers tuned for the prompt-style comparison:

    route_correctness   — Programmatic. Did the router pick the same category as
                          our label? Exact string match.
    output_format       — Programmatic. Was the model's response a single valid
                          category name? Catches "I think the answer is..."
                          rambling that breaks downstream parsing.
    ambiguity_handling  — LLM-as-judge (1-5). For the 10 deliberately ambiguous
                          queries, is the chosen route a *reasonable* pick even
                          if it doesn't match the label? Surfaces the difference
                          between "label-wrong" and "actually wrong."
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

import langwatch
from agent import CATEGORY_NAMES, CATEGORIES

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
_client = OpenAI()


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


# ---- Evaluator 1: route_correctness (programmatic) ----

def route_correctness(chosen: str, expected: str) -> EvalResult:
    """Exact match on category name."""
    passed = chosen == expected
    return EvalResult(
        name="route_correctness",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason=f"chose={chosen!r} expected={expected!r}",
    )


# ---- Evaluator 2: output_format (programmatic) ----

def output_format(chosen: str) -> EvalResult:
    """
    Did the model output a single valid category name (post-parse)?

    Routers that produce "I think this is creative_task because…" will fail
    this — the chosen field will be "invalid:...". This eval catches when the
    prompt failed to constrain output shape, regardless of whether the
    classification itself was right.
    """
    passed = chosen in CATEGORY_NAMES
    return EvalResult(
        name="output_format",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="valid category" if passed else f"invalid output: {chosen!r}",
    )


# ---- Evaluator 3: ambiguity_handling (LLM-as-judge) ----

_AMBIGUITY_RUBRIC = """\
You are evaluating whether a routing classifier picked a reasonable category
for an ambiguous query. The query genuinely could fit two or more categories —
the "labeled" category in the dataset reflects one defensible choice, but
others may also be defensible.

Reply with ONLY a single integer 1-5 followed by a colon and a one-sentence reason.

5 — The chosen category is fully defensible; a senior PM would agree.
4 — The chosen category is reasonable, even if a different one is also defensible.
3 — The chosen category is one of several plausible options but slightly off.
2 — The chosen category misses the user's primary intent.
1 — The chosen category is clearly wrong.

If the chosen category is "invalid:..." (i.e. the model failed to output a clean
category), reply with "1: Invalid output format — couldn't parse a category".
"""


def ambiguity_handling(query: str, chosen: str, label: str) -> EvalResult:
    """
    LLM-judge: was the route a reasonable pick on an ambiguous query?

    Run only on rows marked `ambiguous=true`. Catches cases where the model
    made a defensible-but-different choice from our label — and would unfairly
    fail route_correctness even though the routing was fine.
    """
    user = (
        f"Query: {query}\n"
        f"Labeled category: {label}\n"
        f"Chosen category: {chosen}\n"
        f"Valid categories: {CATEGORY_NAMES}"
    )
    with langwatch.span(name="judge_ambiguity", type="evaluation") as s:
        s.update(input=query)
        completion = _client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _AMBIGUITY_RUBRIC},
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
        name="ambiguity_handling",
        passed=score_int >= 4,
        score=normalized,
        reason=f"score={score_int}/5 — {reason}",
    )
