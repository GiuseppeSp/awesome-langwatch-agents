"""
Evaluators for the human-in-the-loop agent.

Each request has exactly one correct action, so everything is programmatic.

    correct_action  — Did the FINAL executed action match gold? Both modes. In
                      autonomous this is just the agent's own correctness; in hitl the
                      human corrects any escalated action, so it can only go up.

    escalation      — Given (escalated?, raw_correct?), classify the gate's decision:
                        TP  escalated a WRONG action  -> human caught it (the win)
                        FP  escalated a RIGHT action  -> false alarm (pure human burden)
                        FN  proceeded on a WRONG one  -> SILENT ERROR (human never sees it)
                        TN  proceeded on a RIGHT one  -> correct autonomy
                      run_eval aggregates these into escalation precision/recall. Recall
                      is the discriminator: HITL's accuracy gain equals exactly the wrong
                      actions it escalated, so the FNs are its ceiling.

    autonomous_to_hitl_lift — Paired autonomous -> hitl per request: +1 if autonomous
                      acted wrong and hitl ended up correct (escalated + fixed), -1 if
                      hitl is somehow worse (never, with a perfect oracle), 0 otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


# ---- Evaluator 1: correct_action ----

def correct_action(chosen: str, gold: str) -> EvalResult:
    ok = chosen.strip() == gold.strip()
    return EvalResult("correct_action", ok, 1.0 if ok else 0.0,
                      f"chose={chosen!r} gold={gold!r}" + ("" if ok else "  <-- WRONG"))


# ---- Evaluator 2: escalation decision (confusion-matrix cell) ----

def escalation(escalated: bool, raw_correct: bool) -> EvalResult:
    if escalated and not raw_correct:
        cell, good, tag = "TP", True, "escalated a WRONG action (human caught it)"
    elif escalated and raw_correct:
        cell, good, tag = "FP", False, "escalated a RIGHT action (false alarm / burden)"
    elif not escalated and not raw_correct:
        cell, good, tag = "FN", False, "proceeded on a WRONG action (SILENT ERROR)"
    else:
        cell, good, tag = "TN", True, "proceeded on a RIGHT action (correct autonomy)"
    # score = was the gate's decision the right one for this item?
    r = EvalResult("escalation", good, 1.0 if good else 0.0, f"{cell} - {tag}")
    r.cell = cell  # type: ignore[attr-defined]
    return r


# ---- Evaluator 3: autonomous_to_hitl_lift ----

def autonomous_to_hitl_lift(auto_correct: bool, hitl_correct: bool) -> EvalResult:
    if not auto_correct and hitl_correct:
        v, tag = 1, "HELPED (autonomous wrong, human fixed it)"
    elif auto_correct and not hitl_correct:
        v, tag = -1, "HURT (should not happen with a perfect oracle)"
    else:
        v, tag = 0, "no-op"
    return EvalResult("autonomous_to_hitl_lift", v >= 0, (v + 1) / 2, f"raw_lift={v:+d} - {tag}")
