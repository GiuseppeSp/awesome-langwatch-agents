"""
Evaluators for the hybrid-retrieval + rerank agent.

Short answers over a known fictional corpus, so everything is programmatic — no
LLM judge, no judge noise (same as #7-#10, #12-#16). The answer/retrieval
evaluators match #15/#16 so the three retrieval agents compare directly.

    answer_correctness  — Is the gold answer present in the response? Pass/fail. All modes.

    retrieval_hit       — Did the FINAL top-k passages (what the generator sees)
                          contain the gold answer? The thing the ranking is trying
                          to get right. All modes.

    candidate_recall    — Was the gold answer anywhere in the candidate POOL before
                          the final top-k cut? This is reranking's ceiling: a
                          reranker can only promote a passage that was retrieved.
                          The gap (candidate_recall present, retrieval_hit missing)
                          is exactly the room reranking has to help.
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


# ---- Evaluator 1: answer_correctness (programmatic, all modes) ----

def answer_correctness(answer: str, gold: str) -> EvalResult:
    a, g = _norm(answer), _norm(gold)
    if not a or a == "i dont know":
        return EvalResult("answer_correctness", False, 0.0, f"no answer ({answer!r})")
    ok = g in a
    return EvalResult("answer_correctness", ok, 1.0 if ok else 0.0,
                      f"answer={answer!r} gold={gold!r}" + ("" if ok else "  <-- WRONG"))


# ---- Evaluator 2: retrieval_hit (programmatic, all modes) ----

def retrieval_hit(retrieved: list[str], gold: str) -> EvalResult:
    g = _norm(gold)
    hit = any(g in _norm(p) for p in retrieved)
    return EvalResult("retrieval_hit", hit, 1.0 if hit else 0.0,
                      "gold in final top-k" if hit else "gold MISSED in top-k")


# ---- Evaluator 3: candidate_recall (programmatic; reranking's ceiling) ----

def candidate_recall(pool: list[str], gold: str) -> EvalResult:
    g = _norm(gold)
    hit = any(g in _norm(p) for p in pool)
    return EvalResult("candidate_recall", hit, 1.0 if hit else 0.0,
                      "gold in candidate pool" if hit else "gold MISSED in pool (rerank cannot help)")
