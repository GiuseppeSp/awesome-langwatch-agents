"""
Driver: run the dataset through the ReAct agent in both modes and print a
comparison table.

Usage:
    python run_eval.py                       # bare vs react, all rows
    python run_eval.py --mode react          # one mode only
    python run_eval.py --limit 3             # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import EvalResult, answer_correctness, reasoning_quality, tool_call_efficiency


@dataclass
class Row:
    query: str
    expected_answer: str
    expected_tools: list[str]
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tools = [t.strip() for t in r["expected_tools"].split(";") if t.strip()]
            rows.append(
                Row(
                    query=r["query"],
                    expected_answer=r["expected_answer"],
                    expected_tools=tools,
                    notes=r["notes"],
                )
            )
    return rows


def run_one(row: Row, mode: str) -> dict:
    result = agent.run(row.query, mode=mode)
    r_correct = answer_correctness(row.query, row.expected_answer, result.answer)
    r_efficiency = tool_call_efficiency(result.tool_calls, row.expected_tools)
    if mode == "react":
        r_reasoning = reasoning_quality(row.query, result.thoughts, result.answer)
    else:
        r_reasoning = EvalResult(
            name="reasoning_quality",
            passed=False,
            score=float("nan"),
            reason="N/A in bare mode (no thoughts emitted by design)",
        )
    return {
        "query": row.query,
        "mode": mode,
        "answer": result.answer,
        "iterations": result.iterations,
        "tool_calls": result.tool_calls,
        "thoughts": result.thoughts,
        "hit_limit": result.hit_iteration_limit,
        "evals": {
            "answer_correctness": r_correct,
            "tool_call_efficiency": r_efficiency,
            "reasoning_quality": r_reasoning,
        },
    }


def aggregate(results: list[dict]) -> dict:
    per_mode: dict[str, dict[str, list[EvalResult]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for name, ev in r["evals"].items():
            per_mode[r["mode"]][name].append(ev)
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for mode, by_eval in per_mode.items():
        out[mode] = {}
        for name, evs in by_eval.items():
            valid = [e for e in evs if not _is_nan(e.score)]
            if not valid:
                out[mode][name] = (float("nan"), float("nan"))
                continue
            mean_score = sum(e.score for e in valid) / len(valid)
            pass_rate = sum(1 for e in valid if e.passed) / len(valid)
            out[mode][name] = (mean_score, pass_rate)
    return out


def _is_nan(x: float) -> bool:
    return x != x  # NaN != NaN


def _fmt_cell(value: tuple[float, float]) -> str:
    score, rate = value
    if _is_nan(score):
        return "n/a"
    return f"{score:.2f} ({rate * 100:.0f}%)"


def _print_summary(summary: dict) -> None:
    eval_names = ["answer_correctness", "tool_call_efficiency", "reasoning_quality"]
    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>10} | " + " | ".join(f"{n:>22}" for n in eval_names))
    print("-" * (12 + sum(24 for _ in eval_names)))
    for mode, by_eval in summary.items():
        cells = [_fmt_cell(by_eval.get(n, (float("nan"), float("nan")))) for n in eval_names]
        print(f"{mode:>10} | " + " | ".join(f"{c:>22}" for c in cells))
    print("\n(score is mean; % is pass rate)\n")


def _print_per_row(results: list[dict]) -> None:
    print("=== Per-row detail ===\n")
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["query"]][r["mode"]] = r
    for q, by_mode in by_q.items():
        print(f"  {q}")
        for mode, r in by_mode.items():
            correct = r["evals"]["answer_correctness"]
            efficiency = r["evals"]["tool_call_efficiency"]
            reasoning = r["evals"]["reasoning_quality"]
            reasoning_str = (
                f"reason={reasoning.score:.2f}" if not _is_nan(reasoning.score) else "reason=n/a"
            )
            limit_marker = "  [hit limit]" if r["hit_limit"] else ""
            print(
                f"    [{mode:>5}] iters={r['iterations']} "
                f"tools={r['tool_calls']} "
                f"correct={correct.score:.2f} "
                f"efficiency={efficiency.score:.2f} "
                f"{reasoning_str}{limit_marker}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["bare", "react", "both"], default="both",
        help="Which reasoning mode to evaluate. Default: both.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only first N rows.")
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = ["bare", "react"] if args.mode == "both" else [args.mode]

    print(
        f"Loaded {len(rows)} dataset rows. Running {len(modes)} mode(s) = "
        f"{len(rows) * len(modes)} agent calls."
    )

    results: list[dict] = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  {row.query[:60]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(aggregate(results))
    _print_per_row(results)


if __name__ == "__main__":
    main()
