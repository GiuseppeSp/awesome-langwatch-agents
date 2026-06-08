"""
Driver: run the dataset through the agent with each tool-description mode and
print a comparison table.

Usage:
    python run_eval.py                          # vague vs precise, all rows
    python run_eval.py --mode precise           # one mode only
    python run_eval.py --limit 5                # first N rows (smoke test)
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import EvalResult, argument_extraction, no_tool_correctness, tool_selection


@dataclass
class Row:
    query: str
    expected_tool: str
    expected_args_hint: str
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    with open(path, encoding="utf-8") as f:
        return [
            Row(
                query=r["query"],
                expected_tool=r["expected_tool"],
                expected_args_hint=r["expected_args_hint"],
                notes=r["notes"],
            )
            for r in csv.DictReader(f)
        ]


def run_one(row: Row, mode: str) -> dict:
    result = agent.answer(row.query, mode=mode)
    r_select = tool_selection(result.first_tool, row.expected_tool)
    r_no_tool = no_tool_correctness(result.first_tool, row.expected_tool)
    r_args = argument_extraction(
        row.query, row.expected_tool, row.expected_args_hint, result.tool_calls
    )
    return {
        "query": row.query,
        "expected_tool": row.expected_tool,
        "mode": mode,
        "actual_tool": result.first_tool,
        "answer": result.answer,
        "evals": {
            "tool_selection": r_select,
            "no_tool_correctness": r_no_tool,
            "argument_extraction": r_args,
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
            mean_score = sum(e.score for e in evs) / len(evs)
            pass_rate = sum(1 for e in evs if e.passed) / len(evs)
            out[mode][name] = (mean_score, pass_rate)
    return out


def _print_summary(summary: dict) -> None:
    eval_names = ["tool_selection", "argument_extraction", "no_tool_correctness"]
    print("\n=== Tool-description comparison ===\n")
    print(f"{'mode':>10} | " + " | ".join(f"{n:>22}" for n in eval_names))
    print("-" * (12 + sum(24 for _ in eval_names)))
    for mode, by_eval in summary.items():
        cells = []
        for n in eval_names:
            score, rate = by_eval.get(n, (0.0, 0.0))
            cells.append(f"{score:.2f} ({rate * 100:.0f}%)")
        print(f"{mode:>10} | " + " | ".join(f"{c:>22}" for c in cells))
    print("\n(score is mean across dataset; % is pass rate)\n")


def _print_per_row(results: list[dict]) -> None:
    print("=== Per-row detail ===\n")
    by_query: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_query[r["query"]][r["mode"]] = r
    for query, by_mode in by_query.items():
        expected = next(iter(by_mode.values()))["expected_tool"]
        print(f'  "{query}"  [expected: {expected}]')
        for mode, r in by_mode.items():
            sel = r["evals"]["tool_selection"]
            args = r["evals"]["argument_extraction"]
            print(
                f"    [{mode:>7}] picked={r['actual_tool']:<20} "
                f"select={'✓' if sel.passed else '✗'}  "
                f"args={args.score:.2f}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["vague", "precise", "both"], default="both",
        help="Which description mode to evaluate. Default: both (the comparison).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only first N rows.")
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = ["vague", "precise"] if args.mode == "both" else [args.mode]

    print(f"Loaded {len(rows)} dataset rows. Running {len(modes)} mode(s) = {len(rows) * len(modes)} agent calls.\n")

    results: list[dict] = []
    for mode in modes:
        print(f"--- Running mode={mode} ---")
        for row in rows:
            print(f"  {row.query[:60]}...", flush=True)
            results.append(run_one(row, mode))
        print()

    _print_summary(aggregate(results))
    _print_per_row(results)


if __name__ == "__main__":
    main()
