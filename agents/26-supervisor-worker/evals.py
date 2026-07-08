"""
Evaluators for the supervisor-worker agent.

Every subtask has an oracle gold, so scoring is programmatic. The critical design
rule (learned the hard way in #20): BOTH modes are scored through the SAME forgiving
normalizer, so a difference can never be an artifact of a stricter parser on one side.

Per subtask, classify the outcome into one of three:
    correct  — answered and matches gold
    wrong    — answered but doesn't match gold
    dropped  — no answer for this task number at all (the bundling failure mode:
               a single pass silently omitting a task as N grows)

    bundle_completeness  — fraction of subtasks answered (not dropped)
    bundle_correctness   — fraction of subtasks correct
    decomposition_ok     — supervisor_worker: did the supervisor emit exactly N subtasks?
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


def _ints(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", s)]


def _clean(s: str) -> str:
    return s.strip().strip('."\'` ').strip()


def check_subtask(sub: dict, answer: str | None) -> str:
    """Return 'correct' | 'wrong' | 'dropped'. Same logic for both modes."""
    if answer is None or not answer.strip():
        return "dropped"
    t, gold, a = sub["type"], sub["gold"], answer

    if t in ("multiply", "add", "maxnum", "wordcount"):
        got = _ints(a)
        return "correct" if got and got[0] == int(gold) else "wrong"
    if t == "sortnums":
        return "correct" if _ints(a) == _ints(gold) else "wrong"
    if t == "upper":
        # the task is to change case, so casing matters
        return "correct" if _clean(a) == gold else "wrong"
    # reverse, nthword, lastchar: case-insensitive exact
    return "correct" if _clean(a).casefold() == gold.casefold() else "wrong"


def score_bundle(subtasks: list[dict], answers: dict[str, str]) -> dict:
    """Classify every subtask; return counts + the two rates."""
    cells = {"correct": 0, "wrong": 0, "dropped": 0}
    per = []
    for i, sub in enumerate(subtasks, 1):
        outcome = check_subtask(sub, answers.get(str(i)))
        cells[outcome] += 1
        per.append((sub, answers.get(str(i)), outcome))
    n = len(subtasks)
    return {
        "n": n, "cells": cells, "per": per,
        "completeness": (n - cells["dropped"]) / n if n else 0.0,
        "correctness": cells["correct"] / n if n else 0.0,
    }


def bundle_correctness(subtasks: list[dict], answers: dict[str, str]) -> EvalResult:
    r = score_bundle(subtasks, answers)
    ok = r["correctness"] == 1.0
    c = r["cells"]
    return EvalResult("bundle_correctness", ok, r["correctness"],
                      f"{c['correct']}/{r['n']} correct ({c['wrong']} wrong, {c['dropped']} dropped)")


def decomposition_ok(n_decomposed: int, n_expected: int) -> EvalResult:
    ok = n_decomposed == n_expected
    return EvalResult("decomposition_ok", ok, 1.0 if ok else 0.0,
                      f"supervisor produced {n_decomposed}/{n_expected} subtasks"
                      + ("" if ok else "  <-- MIS-DECOMPOSED"))
