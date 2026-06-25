"""
Evaluators for the graph-RAG agent.

Deterministic graph, short answers → all programmatic, no judge noise (consistent
with the rest of the catalog).

    answer_correctness  — Is the gold answer present in the response? Both modes.

    evidence_complete   — Were ALL the support facts (the entities/edges the answer
                          depends on) present in the retrieved context? This is the
                          mechanism metric: flat top-k physically cannot hold the
                          support for a global question (more edges than k), so it
                          fails this even when it guesses; graph traversal gathers
                          the whole connected subgraph.

    flat_to_graph_lift  — Paired flat -> graph per query: +1 if flat wrong & graph
                          right, -1 if flat right & graph broke it, 0 otherwise.
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


# ---- Evaluator 2: evidence_complete (programmatic, both modes) ----

def evidence_complete(retrieved: list[str], support: list[str]) -> EvalResult:
    ctx = " ".join(_norm(p) for p in retrieved)
    missing = [s for s in support if _norm(s) not in ctx]
    ok = not missing
    return EvalResult("evidence_complete", ok, 1.0 if ok else 0.0,
                      "all support present" if ok else f"missing {len(missing)}/{len(support)}: {missing}")


# ---- Evaluator 3: flat_to_graph_lift (programmatic, pairs the two modes) ----

def flat_to_graph_lift(flat_correct: bool, graph_correct: bool) -> EvalResult:
    if not flat_correct and graph_correct:
        v, tag = 1, "HELPED (flat wrong, graph right)"
    elif flat_correct and not graph_correct:
        v, tag = -1, "HURT (flat right, graph broke it)"
    elif flat_correct and graph_correct:
        v, tag = 0, "no-op (both right)"
    else:
        v, tag = 0, "no-op (both wrong)"
    return EvalResult("flat_to_graph_lift", v >= 0, (v + 1) / 2, f"raw_lift={v:+d} - {tag}")
