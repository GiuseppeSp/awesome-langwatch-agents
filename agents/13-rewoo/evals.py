"""
Evaluators for the ReWOO agent.

The KB is deterministic and every answer is a single token or number, so all
three evaluators are programmatic — no LLM judge, no judge noise (same as
#7-#10, #12).

    answer_correctness  — Does the agent's final answer match the gold (token or
                          number)? Pass/fail. Both modes.

    llm_call_efficiency  — LLM calls per query. This is the headline ReWOO claim:
                          rewoo = 2 calls regardless of hop count; react scales
                          with the number of tool calls. Lower is better.

    react_to_rewoo_lift  — Paired react -> rewoo per query: +1 if react was wrong
                          but rewoo got it right, -1 if react was right and rewoo
                          broke it (the blind-planning failure), 0 otherwise.
                          Normalized to [0,1]. The discriminator — it isolates
                          where decoupling reasoning from observation costs you.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


def _norm(s: str) -> str:
    return re.sub(r"[,$\s]", "", (s or "").strip().lower())


# ---- Evaluator 1: answer_correctness (programmatic, both modes) ----

def answer_correctness(answer: str, gold: str) -> EvalResult:
    a, g = _norm(answer), _norm(gold)
    if not a:
        return EvalResult("answer_correctness", False, 0.0, "empty answer")
    # numeric compare when both look numeric; else exact-token (gold contained in answer)
    try:
        ok = abs(float(a) - float(g)) < 1e-6
    except ValueError:
        ok = (a == g) or (g in a)
    return EvalResult(
        "answer_correctness", ok, 1.0 if ok else 0.0,
        f"answer={answer!r} gold={gold!r}" + ("" if ok else "  <-- WRONG"),
    )


# ---- Evaluator 2: llm_call_efficiency (programmatic, both modes) ----

def llm_call_efficiency(llm_calls: int) -> EvalResult:
    # rewoo's promise is a flat 2 calls; flag anything above as "scaling cost"
    return EvalResult("llm_call_efficiency", llm_calls <= 2, float(llm_calls), f"{llm_calls} llm call(s)")


# ---- Evaluator 3: react_to_rewoo_lift (programmatic, pairs the two modes) ----

def react_to_rewoo_lift(react_correct: bool, rewoo_correct: bool) -> EvalResult:
    if not react_correct and rewoo_correct:
        v, tag = 1, "HELPED (react wrong, rewoo right)"
    elif react_correct and not rewoo_correct:
        v, tag = -1, "HURT (react right, rewoo broke it — blind planning)"
    elif react_correct and rewoo_correct:
        v, tag = 0, "no-op (both right)"
    else:
        v, tag = 0, "no-op (both wrong)"
    return EvalResult("react_to_rewoo_lift", v >= 0, (v + 1) / 2, f"raw_lift={v:+d} - {tag}")
