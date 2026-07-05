"""
Evaluators for the tool-retrieval agent.

Each query has exactly one correct tool, so everything is programmatic.

    correct_tool     — Did the model pick the gold tool? The core selection metric.
                       Both modes.

    retrieval_recall — retrieved mode only: was the gold tool even IN the shortlist?
                       This is retrieval's ceiling — the model can't pick a tool
                       that retrieval didn't surface. The gap (recall present but
                       correct_tool wrong) is a selection error; recall absent is a
                       retrieval error.

    all_to_retrieved_lift — Paired all_tools -> retrieved per query: +1 if all_tools
                       picked wrong and retrieval fixed it, -1 if the reverse (the
                       correct tool got filtered out), 0 otherwise. The discriminator.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


# ---- Evaluator 1: correct_tool ----

def correct_tool(chosen: str, gold: str) -> EvalResult:
    ok = chosen.strip() == gold.strip()
    return EvalResult("correct_tool", ok, 1.0 if ok else 0.0,
                      f"chose={chosen!r} gold={gold!r}" + ("" if ok else "  <-- WRONG"))


# ---- Evaluator 2: retrieval_recall (retrieved mode) ----

def retrieval_recall(shortlist: list[str], gold: str) -> EvalResult:
    ok = gold in shortlist
    return EvalResult("retrieval_recall", ok, 1.0 if ok else 0.0,
                      "gold tool in shortlist" if ok else "gold tool NOT retrieved")


# ---- Evaluator 3: all_to_retrieved_lift ----

def all_to_retrieved_lift(all_correct: bool, retrieved_correct: bool) -> EvalResult:
    if not all_correct and retrieved_correct:
        v, tag = 1, "HELPED (all_tools wrong, retrieval fixed it)"
    elif all_correct and not retrieved_correct:
        v, tag = -1, "HURT (correct tool filtered out or mis-picked)"
    else:
        v, tag = 0, "no-op"
    return EvalResult("all_to_retrieved_lift", v >= 0, (v + 1) / 2, f"raw_lift={v:+d} - {tag}")
