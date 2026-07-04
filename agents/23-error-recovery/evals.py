"""
Evaluators for the error-recovery agent.

The mock service is deterministic, so everything is programmatic.

    outcome_correct  — Did the agent reach the RIGHT outcome? For solvable tasks
                       (clean, recoverable) that means it solved; for unsolvable
                       tasks (unrecoverable) that means it correctly did NOT solve
                       (it can't fabricate success — the tool never returns one —
                       so this is always satisfied, and the interesting axis there
                       is efficiency below). Both modes.

    solved           — Did it get a successful result? (Only possible for solvable
                       tasks.) Both modes.

    no_retry_to_retry_lift — Paired no_retry -> retry per task: +1 if retry solved
                       what no_retry couldn't, -1 if retry broke a no_retry success
                       (impossible here), 0 otherwise. Normalized to [0,1].

(run_eval also reports `attempts` — especially on unrecoverable tasks, where a
good retry loop should give up early rather than burn all MAX_ATTEMPTS.)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


# ---- Evaluator 1: solved ----

def solved(res_solved: bool) -> EvalResult:
    return EvalResult("solved", res_solved, 1.0 if res_solved else 0.0,
                      "got a result" if res_solved else "no result")


# ---- Evaluator 2: outcome_correct ----

def outcome_correct(res_solved: bool, solvable: bool) -> EvalResult:
    ok = res_solved == solvable
    return EvalResult("outcome_correct", ok, 1.0 if ok else 0.0,
                      f"solved={res_solved}, solvable={solvable}" + ("" if ok else "  <-- WRONG"))


# ---- Evaluator 3: no_retry_to_retry_lift ----

def no_retry_to_retry_lift(no_retry_solved: bool, retry_solved: bool) -> EvalResult:
    if not no_retry_solved and retry_solved:
        v, tag = 1, "HELPED (retry recovered it)"
    elif no_retry_solved and not retry_solved:
        v, tag = -1, "HURT (retry lost a success)"
    else:
        v, tag = 0, "no-op"
    return EvalResult("no_retry_to_retry_lift", v >= 0, (v + 1) / 2, f"raw_lift={v:+d} - {tag}")
