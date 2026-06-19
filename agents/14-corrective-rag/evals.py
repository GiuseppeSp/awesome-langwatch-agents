"""
Evaluators for the corrective-RAG agent.

Answers are short factual tokens over a fictional corpus, so correctness is
checkable programmatically. The grader's verdict is checkable too: each row is
labelled with whether the answer is actually in the local corpus, which is the
ground truth the grader is trying to recover.

    answer_correctness   — Is the gold answer present in the agent's response?
                           (normalized substring match.) Pass/fail. Both modes.

    vanilla_to_crag_lift — Paired vanilla -> crag per row: +1 if vanilla was wrong
                           and crag fixed it, -1 if vanilla was right and crag broke
                           it (over-correction — the grader routed a good local hit
                           to the web), 0 otherwise. Normalized to [0,1]. The
                           discriminator for whether correction nets out positive.

    grader_accuracy      — CRAG-only. Did the grader's CORRECT/INCORRECT verdict
                           match the truth (answer_in_local)? This is the
                           calibration number — CRAG's whole value rides on it.
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
    return re.sub(r"[^a-z0-9.\- ]", "", (s or "").lower()).strip()


# ---- Evaluator 1: answer_correctness (programmatic, both modes) ----

def answer_correctness(answer: str, gold: str) -> EvalResult:
    a, g = _norm(answer), _norm(gold)
    if not a or a == "i dont know":
        return EvalResult("answer_correctness", False, 0.0, f"no answer ({answer!r})")
    ok = g in a
    return EvalResult(
        "answer_correctness", ok, 1.0 if ok else 0.0,
        f"answer={answer!r} gold={gold!r}" + ("" if ok else "  <-- WRONG"),
    )


# ---- Evaluator 2: vanilla_to_crag_lift (programmatic, pairs the two modes) ----

def vanilla_to_crag_lift(vanilla_correct: bool, crag_correct: bool) -> EvalResult:
    if not vanilla_correct and crag_correct:
        v, tag = 1, "HELPED (vanilla wrong, correction fixed it)"
    elif vanilla_correct and not crag_correct:
        v, tag = -1, "HURT (vanilla right, correction broke it — over-correction)"
    elif vanilla_correct and crag_correct:
        v, tag = 0, "no-op (both right)"
    else:
        v, tag = 0, "no-op (both wrong)"
    return EvalResult("vanilla_to_crag_lift", v >= 0, (v + 1) / 2, f"raw_lift={v:+d} - {tag}")


# ---- Evaluator 3: grader_accuracy (programmatic, crag only) ----

def grader_accuracy(grade: str, answer_in_local: bool) -> EvalResult:
    # The grader should say CORRECT exactly when the answer is in the local corpus.
    predicted_in_local = (grade.upper() == "CORRECT")
    ok = predicted_in_local == answer_in_local
    truth = "in-local" if answer_in_local else "not-in-local"
    return EvalResult(
        "grader_accuracy", ok, 1.0 if ok else 0.0,
        f"grade={grade} truth={truth}" + ("" if ok else "  <-- MISCALIBRATED"),
    )
