"""
Evaluators for the query-rewriting RAG agent.

Answers are short tokens over a known fictional corpus, so everything is
programmatic — no LLM judge, no judge noise (same as #7-#10, #12-#14).

    answer_correctness  — Is the gold answer present in the response? (normalized
                          substring.) Pass/fail. Both modes.

    retrieval_hit       — Did the RETRIEVED passages actually contain the gold
                          answer? This is the mechanism-level metric: rewriting's
                          whole job is to get the right passage into context, so
                          this isolates the retrieval improvement from generation.
                          Both modes.

    rewrite_lift        — Paired vanilla -> rewrite per query: +1 if vanilla was
                          wrong and rewriting fixed it, -1 if vanilla was right and
                          rewriting broke it (drift), 0 otherwise. Normalized to
                          [0,1]. The discriminator.
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


# ---- Evaluator 3: rewrite_lift (programmatic, pairs the two modes) ----

def rewrite_lift(vanilla_correct: bool, rewrite_correct: bool) -> EvalResult:
    if not vanilla_correct and rewrite_correct:
        v, tag = 1, "HELPED (vanilla wrong, rewrite fixed it)"
    elif vanilla_correct and not rewrite_correct:
        v, tag = -1, "HURT (vanilla right, rewrite broke it — drift)"
    elif vanilla_correct and rewrite_correct:
        v, tag = 0, "no-op (both right)"
    else:
        v, tag = 0, "no-op (both wrong)"
    return EvalResult("rewrite_lift", v >= 0, (v + 1) / 2, f"raw_lift={v:+d} - {tag}")
