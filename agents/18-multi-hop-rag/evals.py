"""
Evaluators for the multi-hop RAG agent.

Short answers over a known fictional corpus → all programmatic, no judge noise
(same as #7-#10, #12-#17).

    answer_correctness   — Is the gold answer present in the response? Pass/fail. Both modes.

    answer_retrieved     — Did the passages the agent actually saw (single: the one
                           retrieval; multihop: the union across hops) contain the
                           gold answer? Isolates retrieval from generation, and shows
                           the core mechanism: single-shot cannot retrieve a multi-hop
                           answer because the answer passage shares no keyword with the
                           original question.

    single_to_multihop_lift — Paired single -> multihop per query: +1 if single was
                           wrong and the loop fixed it, -1 if single was right and the
                           loop broke it, 0 otherwise. Normalized to [0,1]. The
                           discriminator.
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
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


# ---- Evaluator 1: answer_correctness (programmatic, both modes) ----

def answer_correctness(answer: str, gold: str) -> EvalResult:
    a, g = _norm(answer), _norm(gold)
    if not a or a == "i dont know":
        return EvalResult("answer_correctness", False, 0.0, f"no answer ({answer!r})")
    ok = g in a
    return EvalResult("answer_correctness", ok, 1.0 if ok else 0.0,
                      f"answer={answer!r} gold={gold!r}" + ("" if ok else "  <-- WRONG"))


# ---- Evaluator 2: answer_retrieved (programmatic, both modes) ----

def answer_retrieved(retrieved: list[str], gold: str) -> EvalResult:
    g = _norm(gold)
    hit = any(g in _norm(p) for p in retrieved)
    return EvalResult("answer_retrieved", hit, 1.0 if hit else 0.0,
                      "answer passage retrieved" if hit else "answer passage NEVER retrieved")


# ---- Evaluator 3: single_to_multihop_lift (programmatic, pairs the two modes) ----

def single_to_multihop_lift(single_correct: bool, multihop_correct: bool) -> EvalResult:
    if not single_correct and multihop_correct:
        v, tag = 1, "HELPED (single wrong, loop fixed it)"
    elif single_correct and not multihop_correct:
        v, tag = -1, "HURT (single right, loop broke it)"
    elif single_correct and multihop_correct:
        v, tag = 0, "no-op (both right)"
    else:
        v, tag = 0, "no-op (both wrong)"
    return EvalResult("single_to_multihop_lift", v >= 0, (v + 1) / 2, f"raw_lift={v:+d} - {tag}")
