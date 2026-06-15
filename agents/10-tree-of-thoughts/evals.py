"""
Evaluators for the tree-of-thoughts agent.

Three scorers — all programmatic, because Game of 24 is exactly checkable and
the brute-force oracle gives ground truth for every partial state.

    solved              — Does the final expression use each given number exactly
                          once and evaluate to 24? Pass/fail. Both modes. The
                          outcome metric.

    evaluator_accuracy  — For every partial state the LLM evaluator judged, did
                          its keep/prune decision match the brute-force oracle?
                          Fraction correct. ToT-only. This is the discriminator:
                          ToT's whole advantage is pruning, and pruning is only
                          as good as the evaluator. (Analogous to #7's
                          self_critique_accuracy.)

    search_efficiency   — LLM calls per puzzle. The score IS the count, so the
                          aggregate mean is 'average calls per puzzle' — cot is 1,
                          tot is many. The cost axis.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import _eval_expr, numbers_in  # reuse the agent's arithmetic

TOL = 1e-6


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    reason: str = ""


# ---- Evaluator 1: solved (programmatic, both modes) ----

def solved(final_expression: str, numbers: list) -> EvalResult:
    expr = (final_expression or "").strip()
    if not expr:
        return EvalResult("solved", False, 0.0, "no expression produced")
    try:
        value = _eval_expr(expr)
    except Exception as e:  # noqa: BLE001
        return EvalResult("solved", False, 0.0, f"unparseable: {e}")

    used = sorted(numbers_in(expr))
    want = sorted(float(n) for n in numbers)
    if used != want:
        return EvalResult("solved", False, 0.0,
                          f"used {used} but must use exactly {want}")
    ok = abs(value - 24) < 1e-4
    return EvalResult("solved", ok, 1.0 if ok else 0.0,
                      f"expr = {value} ({'==' if ok else '!='} 24)")


# ---- Evaluator 2: evaluator_accuracy (programmatic, tot-only) ----

def evaluator_accuracy(judgements: list) -> EvalResult:
    """judgements: list of agent.EvalJudgement (needs .kept and .truly_solvable)."""
    if not judgements:
        return EvalResult("evaluator_accuracy", False, float("nan"),
                          "no states evaluated (cot mode or empty search)")
    correct = sum(1 for j in judgements if j.kept == j.truly_solvable)
    # break out the two error types — they mean different things
    pruned_winners = sum(1 for j in judgements if not j.kept and j.truly_solvable)
    kept_deadends = sum(1 for j in judgements if j.kept and not j.truly_solvable)
    score = correct / len(judgements)
    return EvalResult(
        "evaluator_accuracy", score >= 0.7, score,
        f"{correct}/{len(judgements)} judgements matched oracle; "
        f"pruned-reachable={pruned_winners} (fatal), kept-deadend={kept_deadends} (wasteful)",
    )


# ---- Evaluator 3: search_efficiency (programmatic, both modes) ----

def search_efficiency(llm_calls: int) -> EvalResult:
    return EvalResult("search_efficiency", llm_calls <= 1, float(llm_calls),
                      f"{llm_calls} llm call(s)")
