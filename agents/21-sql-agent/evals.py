"""
Evaluators for the SQL agent.

The database is deterministic and gold answers come from running the gold SQL, so
everything is programmatic — no judge noise.

    executes_ok      — Did the model's SQL run at all, or did it error (e.g. on a
                       hallucinated table/column name)? This is the mechanism metric:
                       blind SQL against a non-obvious schema fails HERE.

    answer_correct   — Does the query's result set match the gold result set
                       (order-insensitive, compared as flattened cell strings)?
                       Both modes.

    blind_to_grounded_lift — Paired blind -> grounded per question: +1 if blind was
                       wrong and grounding fixed it, -1 if the reverse, 0 otherwise.
                       Normalized to [0,1]. The discriminator.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


# ---- Evaluator 1: executes_ok ----

def executes_ok(error: str | None) -> EvalResult:
    ok = error is None
    return EvalResult("executes_ok", ok, 1.0 if ok else 0.0,
                      "ran" if ok else f"errored: {error}")


# ---- Evaluator 2: answer_correct ----

def answer_correct(rows: list, gold_rows: list) -> EvalResult:
    # Set comparison: duplicate rows (e.g. a missing DISTINCT) are semantically
    # equivalent, but extra/missing distinct values (wrong shape or logic) are not.
    ok = set(map(str, rows)) == set(map(str, gold_rows))
    return EvalResult("answer_correct", ok, 1.0 if ok else 0.0,
                      f"got={rows} gold={gold_rows}" + ("" if ok else "  <-- WRONG"))


# ---- Evaluator 3: blind_to_grounded_lift ----

def blind_to_grounded_lift(blind_correct: bool, grounded_correct: bool) -> EvalResult:
    if not blind_correct and grounded_correct:
        v, tag = 1, "HELPED (blind wrong, grounding fixed it)"
    elif blind_correct and not grounded_correct:
        v, tag = -1, "HURT (blind right, grounding broke it)"
    elif blind_correct and grounded_correct:
        v, tag = 0, "no-op (both right)"
    else:
        v, tag = 0, "no-op (both wrong)"
    return EvalResult("blind_to_grounded_lift", v >= 0, (v + 1) / 2, f"raw_lift={v:+d} - {tag}")
