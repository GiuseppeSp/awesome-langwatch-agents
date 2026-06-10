"""
Driver: run the dataset through the pipeline with each mode and print a
comparison table.

Usage:
    python run_eval.py                       # basic vs fact_checked, all rows
    python run_eval.py --mode fact_checked   # one mode only
    python run_eval.py --limit 5             # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import EvalResult, answer_quality, fact_coverage, hallucination_judge


@dataclass
class Row:
    query: str
    expected_facts: list[str]
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            facts = [x.strip() for x in r["expected_facts"].split(";") if x.strip()]
            rows.append(Row(query=r["query"], expected_facts=facts, notes=r["notes"]))
    return rows


def run_one(row: Row, mode: str) -> dict:
    result = agent.run(row.query, mode=mode)
    r_facts = fact_coverage(result.answer, row.expected_facts)
    r_halluc = hallucination_judge(row.query, row.expected_facts, result.answer)
    r_quality = answer_quality(row.query, result.answer)
    return {
        "query": row.query,
        "mode": mode,
        "answer": result.answer,
        "revised": result.revised,
        "evals": {
            "fact_coverage": r_facts,
            "hallucination_judge": r_halluc,
            "answer_quality": r_quality,
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
    eval_names = ["fact_coverage", "hallucination_judge", "answer_quality"]
    print("\n=== Pipeline mode comparison ===\n")
    print(f"{'mode':>14} | " + " | ".join(f"{n:>22}" for n in eval_names))
    print("-" * (16 + sum(24 for _ in eval_names)))
    for mode, by_eval in summary.items():
        cells = []
        for n in eval_names:
            score, rate = by_eval.get(n, (0.0, 0.0))
            cells.append(f"{score:.2f} ({rate * 100:.0f}%)")
        print(f"{mode:>14} | " + " | ".join(f"{c:>22}" for c in cells))
    print("\n(score is mean; % is pass rate)\n")


def _print_per_row(results: list[dict]) -> None:
    print("=== Per-row detail ===\n")
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["query"]][r["mode"]] = r
    for q, by_mode in by_q.items():
        print(f"  {q}")
        for mode, r in by_mode.items():
            facts = r["evals"]["fact_coverage"]
            halluc = r["evals"]["hallucination_judge"]
            quality = r["evals"]["answer_quality"]
            revised_marker = "  [revised]" if r["revised"] else ""
            print(
                f"    [{mode:>12}] facts={facts.score:.2f} halluc={halluc.score:.2f} "
                f"quality={quality.score:.2f}{revised_marker}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["basic", "fact_checked", "both"], default="both",
        help="Which pipeline mode to evaluate. Default: both.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only first N rows.")
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = ["basic", "fact_checked"] if args.mode == "both" else [args.mode]

    print(
        f"Loaded {len(rows)} dataset rows. Running {len(modes)} mode(s) = "
        f"{len(rows) * len(modes)} pipeline calls."
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
