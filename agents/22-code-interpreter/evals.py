"""
Evaluators for the code-interpreter agent.

Gold answers are exact (computed in Python), so everything is programmatic.

    answer_correct  — Does the answer match the gold value numerically (small
                      tolerance for rounding)? Both modes.

    executes_ok     — code mode: did the program run without error? reason mode:
                      trivially true. The mechanism metric for the code path.

    reason_to_code_lift — Paired reason -> code per question: +1 if reasoning was
                      wrong and code fixed it, -1 if reasoning was right and code
                      broke it (buggy program), 0 otherwise. Normalized to [0,1].
                      The discriminator.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


def _num(s) -> float | None:
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


# ---- Evaluator 1: answer_correct ----

def answer_correct(answer: str, gold: str) -> EvalResult:
    a, g = _num(answer), _num(gold)
    if a is not None and g is not None:
        ok = abs(a - g) < 0.01
    else:
        ok = str(answer).strip().lower() == str(gold).strip().lower()
    return EvalResult("answer_correct", ok, 1.0 if ok else 0.0,
                      f"answer={answer!r} gold={gold!r}" + ("" if ok else "  <-- WRONG"))


# ---- Evaluator 2: executes_ok ----

def executes_ok(mode: str, error: str | None) -> EvalResult:
    ok = mode == "reason" or error is None
    return EvalResult("executes_ok", ok, 1.0 if ok else 0.0,
                      "ran" if ok else f"errored: {error}")


# ---- Evaluator 3: reason_to_code_lift ----

def reason_to_code_lift(reason_correct: bool, code_correct: bool) -> EvalResult:
    if not reason_correct and code_correct:
        v, tag = 1, "HELPED (reasoning wrong, code fixed it)"
    elif reason_correct and not code_correct:
        v, tag = -1, "HURT (reasoning right, code broke it)"
    elif reason_correct and code_correct:
        v, tag = 0, "no-op (both right)"
    else:
        v, tag = 0, "no-op (both wrong)"
    return EvalResult("reason_to_code_lift", v >= 0, (v + 1) / 2, f"raw_lift={v:+d} - {tag}")
