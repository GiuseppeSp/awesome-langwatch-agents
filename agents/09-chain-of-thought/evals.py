"""
Evaluators for the chain-of-thought agent.

Three scorers — all programmatic, because every dataset row has one exact
answer (a number, a yes/no, or a name). No LLM judges (same noise-free choice
as agents #7 and #8).

    answer_correctness  — Did the extracted final answer match ground truth?
                          Pass/fail. Applies to all three modes.

    lift                — Generic paired comparison between two modes on the
                          same row: +1 if the baseline was wrong but the
                          treatment was right (the extra compute helped), 0 if
                          no change, -1 if the baseline was right but the
                          treatment was wrong (the extra compute hurt).
                          Normalized to [0,1]. Used twice:
                            direct -> cot   (is reasoning worth it?)
                            cot    -> sc    (is sampling worth it?)

    sample_efficiency   — LLM calls per run. The score IS the call count, so the
                          aggregate mean is 'average calls per row', directly
                          comparable across modes (direct=1, cot=1, sc=N).
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


# ---- grading helpers ----

def _to_number(text: str):
    cleaned = (text or "").lower().replace("$", "").replace(",", "").replace("%", "")
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(m.group()) if m else None


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


# ---- Evaluator 1: answer_correctness (programmatic, all modes) ----

def answer_correctness(final_answer: str, expected: str, answer_type: str,
                       tolerance: float = 0.0) -> EvalResult:
    if answer_type == "numeric":
        got = _to_number(final_answer)
        want = _to_number(expected)
        if got is None or want is None:
            return EvalResult("answer_correctness", False, 0.0,
                              f"could not parse number from {final_answer!r}")
        ok = abs(got - want) <= tolerance
        return EvalResult("answer_correctness", ok, 1.0 if ok else 0.0,
                          f"got {got}, expected {want} (±{tolerance})")

    # text (yes/no, names): substring-tolerant normalized match
    got, want = _norm_text(final_answer), _norm_text(expected)
    ok = bool(want) and want in got
    return EvalResult("answer_correctness", ok, 1.0 if ok else 0.0,
                      f"got {final_answer!r}, expected {expected!r}")


# ---- Evaluator 2: lift (programmatic, pairs two modes) ----

def lift(baseline_correct: bool, treatment_correct: bool, label: str = "") -> EvalResult:
    """
    -1 treatment broke a row the baseline got right
     0 no change (both right, or both wrong)
    +1 treatment fixed a row the baseline got wrong

    Normalized to [0,1]: hurt=0.0, neutral=0.5, helped=1.0. Pass = lift >= 0.
    """
    if not baseline_correct and treatment_correct:
        v, tag = 1, "HELPED"
    elif baseline_correct and not treatment_correct:
        v, tag = -1, "HURT"
    elif baseline_correct and treatment_correct:
        v, tag = 0, "no-op (both right)"
    else:
        v, tag = 0, "no-op (both wrong)"
    name = f"lift[{label}]" if label else "lift"
    return EvalResult(name, v >= 0, (v + 1) / 2, f"raw_lift={v:+d} — {tag}")


# ---- Evaluator 3: sample_efficiency (programmatic, all modes) ----

def sample_efficiency(llm_calls: int) -> EvalResult:
    return EvalResult("sample_efficiency", llm_calls <= 1, float(llm_calls),
                      f"{llm_calls} llm call(s)")
