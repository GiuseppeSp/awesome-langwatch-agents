"""
Evaluators for the HyDE agent.

Short answers over a known fictional corpus, so everything is programmatic — no
LLM judge, no judge noise (same as #7-#10, #12-#15). Identical evaluators to #15
so the two can be compared directly.

    answer_correctness  — Is the gold answer present in the response? Pass/fail. Both modes.

    retrieval_hit       — Did the RETRIEVED passages contain the gold answer? HyDE's
                          whole job is to get the right passage into context, so this
                          isolates the retrieval improvement from generation. Both modes.

    hyde_lift           — Paired vanilla -> hyde per query: +1 if vanilla was wrong and
                          HyDE fixed it, -1 if vanilla was right and HyDE broke it (the
                          hypothetical doc retrieved the wrong passage), 0 otherwise.
                          Normalized to [0,1]. The discriminator.
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


# ---- Evaluator 2: retrieval_hit (programmatic, both modes) ----

def retrieval_hit(retrieved: list[str], gold: str) -> EvalResult:
    g = _norm(gold)
    hit = any(g in _norm(p) for p in retrieved)
    return EvalResult("retrieval_hit", hit, 1.0 if hit else 0.0,
                      "gold passage retrieved" if hit else "gold passage MISSED")


# ---- Evaluator 3: hyde_lift (programmatic, pairs the two modes) ----

def hyde_lift(vanilla_correct: bool, hyde_correct: bool) -> EvalResult:
    if not vanilla_correct and hyde_correct:
        v, tag = 1, "HELPED (vanilla wrong, HyDE fixed it)"
    elif vanilla_correct and not hyde_correct:
        v, tag = -1, "HURT (vanilla right, HyDE broke it — hypothetical retrieved wrong doc)"
    elif vanilla_correct and hyde_correct:
        v, tag = 0, "no-op (both right)"
    else:
        v, tag = 0, "no-op (both wrong)"
    return EvalResult("hyde_lift", v >= 0, (v + 1) / 2, f"raw_lift={v:+d} - {tag}")
