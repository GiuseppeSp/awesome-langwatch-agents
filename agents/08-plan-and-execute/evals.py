"""
Evaluators for the plan-and-execute agent.

Three scorers — all programmatic, because the world is a deterministic knowledge
base and every dataset row has one exact numeric answer. No LLM judges; the
comparison stays noise-free (same choice as agent #7).

    answer_correctness  — Did the final answer match ground truth (within the
                          row's tolerance)? Pass/fail. Applies to both modes.

    adaptation_lift     — Did re-planning change the outcome versus the static
                          plan, and in which direction? Per row, comparing the two
                          modes: +1 if static was wrong but replan was right
                          (replanning rescued a brittle plan), 0 if no change,
                          -1 if static was right but replan was wrong (replanning
                          thrashed — the perverse case). Normalized to [0,1] for
                          averaging. This is the discriminator metric.

    step_efficiency     — How many LLM calls did the run cost? Lower is better when
                          correctness is equal. Surfaces the price of adaptivity:
                          re-planning buys robustness with extra calls. Per mode.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


# ---- Evaluator 1: answer_correctness (programmatic, both modes) ----

def answer_correctness(final_answer, expected: float, tolerance: float) -> EvalResult:
    if final_answer is None:
        return EvalResult("answer_correctness", False, 0.0, "no answer produced")
    ok = abs(float(final_answer) - expected) <= tolerance
    return EvalResult(
        name="answer_correctness",
        passed=ok,
        score=1.0 if ok else 0.0,
        reason=f"got {final_answer}, expected {expected} (±{tolerance})",
    )


# ---- Evaluator 2: adaptation_lift (programmatic, pairs the two modes) ----

def adaptation_lift(static_correct: bool, replan_correct: bool) -> EvalResult:
    """
    Compare static vs replan on the same row.

    -1 replanning broke a row static got right (thrash / perverse)
     0 no change (both right, or both wrong)
    +1 replanning fixed a row static got wrong (rescued brittleness)

    Normalized to [0,1] for averaging: hurt=0.0, neutral=0.5, helped=1.0.
    Pass = lift >= 0 (replanning didn't make things worse).
    """
    if not static_correct and replan_correct:
        lift, label = 1, "HELPED (static wrong, replan right — brittleness rescued)"
    elif static_correct and not replan_correct:
        lift, label = -1, "HURT (static right, replan wrong — replanning thrashed)"
    elif static_correct and replan_correct:
        lift, label = 0, "no-op (both right)"
    else:
        lift, label = 0, "no-op (both wrong)"

    return EvalResult(
        name="adaptation_lift",
        passed=lift >= 0,
        score=(lift + 1) / 2,
        reason=f"raw_lift={lift:+d} — {label}",
    )


# ---- Evaluator 3: step_efficiency (programmatic, both modes) ----

def step_efficiency(llm_calls: int, num_replans: int = 0) -> EvalResult:
    """
    Cost of the run in LLM calls. Not normalized — the score IS the call count, so
    the aggregate mean is 'average LLM calls per run', directly comparable across
    modes. 'Passed' flags a run that stayed within a soft budget (<= 8 calls).
    """
    return EvalResult(
        name="step_efficiency",
        passed=llm_calls <= 8,
        score=float(llm_calls),
        reason=f"{llm_calls} llm calls, {num_replans} replan(s)",
    )
