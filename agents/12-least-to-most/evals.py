"""
Evaluators for the least-to-most agent.

The domain is mechanically checkable (every problem has one integer answer and a
known canonical depth), so all three evaluators are programmatic — no LLM judge,
no judge noise (same choice as #7-#10).

    answer_correctness  — Does the agent's final number equal the gold answer?
                          Pass/fail. Both modes.

    ltm_lift            — Paired cot -> least_to_most comparison per problem:
                          +1 if cot was wrong but ltm got it right, 0 no change,
                          -1 if cot was right but ltm broke it (explicit
                          decomposition introduced an error). Normalized to [0,1].
                          The discriminator — does explicit decomposition add or
                          destroy correctness, and where?

    cost                — LLM calls per run (cot=1, ltm = 1 + #subquestions).
                          The price of explicit decomposition.

The interesting cut is answer_correctness sliced by problem depth (done in
run_eval.py): least-to-most's claim is that it holds as depth grows while a
single chain degrades. That crossover, if it exists, is the whole story.
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

def answer_correctness(predicted: float | None, gold: float) -> EvalResult:
    if predicted is None:
        return EvalResult("answer_correctness", False, 0.0, "no answer parsed")
    ok = abs(predicted - gold) < 1e-6
    return EvalResult(
        "answer_correctness", ok, 1.0 if ok else 0.0,
        f"predicted={predicted:g} gold={gold:g}" + ("" if ok else "  <-- WRONG"),
    )


# ---- Evaluator 2: ltm_lift (programmatic, pairs the two modes) ----

def ltm_lift(cot_correct: bool, ltm_correct: bool) -> EvalResult:
    if not cot_correct and ltm_correct:
        v, tag = 1, "HELPED (cot wrong, decomposition fixed it)"
    elif cot_correct and not ltm_correct:
        v, tag = -1, "HURT (cot right, decomposition broke it)"
    elif cot_correct and ltm_correct:
        v, tag = 0, "no-op (both correct)"
    else:
        v, tag = 0, "no-op (both wrong)"
    return EvalResult("ltm_lift", v >= 0, (v + 1) / 2, f"raw_lift={v:+d} - {tag}")


# ---- Evaluator 3: cost (programmatic, both modes) ----

def cost(llm_calls: int) -> EvalResult:
    return EvalResult("cost", llm_calls <= 1, float(llm_calls), f"{llm_calls} llm call(s)")
