"""
Driver: run the dataset through the router with each prompt style and print a
comparison table.

Splits the per-row breakdown into CLEAR rows and AMBIGUOUS rows so you can see
where each style wins independently — the whole point is that ambiguous queries
are where the tuned prompt has room to actually help.

Usage:
    python run_eval.py                       # bare vs tuned, all rows
    python run_eval.py --style tuned         # just one style
    python run_eval.py --limit 5             # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import EvalResult, ambiguity_handling, output_format, route_correctness


@dataclass
class Row:
    query: str
    expected_category: str
    ambiguous: bool
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                Row(
                    query=r["query"],
                    expected_category=r["expected_category"],
                    ambiguous=r["ambiguous"].strip().lower() == "true",
                    notes=r["notes"],
                )
            )
    return rows


def run_one(row: Row, style: str) -> dict:
    result = agent.route(row.query, style=style)
    r_correct = route_correctness(result.chosen, row.expected_category)
    r_format = output_format(result.chosen)
    # Only run the LLM-judge on ambiguous rows — saves tokens and is the only
    # place that signal is meaningful.
    if row.ambiguous:
        r_ambig = ambiguity_handling(row.query, result.chosen, row.expected_category)
    else:
        r_ambig = EvalResult(
            name="ambiguity_handling",
            passed=True,
            score=1.0,
            reason="N/A (clear row, judge skipped)",
        )
    return {
        "query": row.query,
        "expected": row.expected_category,
        "ambiguous": row.ambiguous,
        "style": style,
        "chosen": result.chosen,
        "evals": {
            "route_correctness": r_correct,
            "output_format": r_format,
            "ambiguity_handling": r_ambig,
        },
    }


def aggregate(results: list[dict]) -> dict:
    """Split aggregates into all / clear / ambiguous so the story is readable."""
    out: dict[str, dict] = {}
    for subset_name, predicate in [
        ("all", lambda r: True),
        ("clear", lambda r: not r["ambiguous"]),
        ("ambiguous", lambda r: r["ambiguous"]),
    ]:
        per_style: dict[str, dict[str, list[EvalResult]]] = defaultdict(lambda: defaultdict(list))
        for r in results:
            if not predicate(r):
                continue
            for name, ev in r["evals"].items():
                per_style[r["style"]][name].append(ev)
        out[subset_name] = {
            style: {
                name: (sum(e.score for e in evs) / len(evs), sum(1 for e in evs if e.passed) / len(evs))
                for name, evs in by_eval.items()
            }
            for style, by_eval in per_style.items()
        }
    return out


def _print_table(label: str, subset: dict) -> None:
    eval_names = ["route_correctness", "output_format", "ambiguity_handling"]
    print(f"\n--- {label} ---")
    print(f"{'style':>10} | " + " | ".join(f"{n:>22}" for n in eval_names))
    print("-" * (12 + sum(24 for _ in eval_names)))
    for style, by_eval in subset.items():
        cells = []
        for n in eval_names:
            score, rate = by_eval.get(n, (0.0, 0.0))
            cells.append(f"{score:.2f} ({rate * 100:.0f}%)")
        print(f"{style:>10} | " + " | ".join(f"{c:>22}" for c in cells))


def _print_per_row(results: list[dict]) -> None:
    by_query: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_query[r["query"]][r["style"]] = r

    # Sort: ambiguous rows first (more interesting).
    sorted_queries = sorted(
        by_query.keys(),
        key=lambda q: (not next(iter(by_query[q].values()))["ambiguous"], q),
    )

    print("\n=== Per-row detail (ambiguous rows first) ===\n")
    for query in sorted_queries:
        by_style = by_query[query]
        first = next(iter(by_style.values()))
        tag = "AMBIG" if first["ambiguous"] else "clear"
        print(f'  [{tag}] "{query}"  [labeled: {first["expected"]}]')
        for style, r in by_style.items():
            corr = r["evals"]["route_correctness"]
            ambig = r["evals"]["ambiguity_handling"]
            ambig_marker = f"  judge={ambig.score:.2f}" if r["ambiguous"] else ""
            print(
                f"    [{style:>5}] picked={r['chosen']:<22} "
                f"correct={'✓' if corr.passed else '✗'}{ambig_marker}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style", choices=["bare", "tuned", "both"], default="both",
        help="Which prompt style to evaluate. Default: both.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only first N rows.")
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    styles = ["bare", "tuned"] if args.style == "both" else [args.style]

    print(
        f"Loaded {len(rows)} dataset rows "
        f"({sum(1 for r in rows if r.ambiguous)} ambiguous, "
        f"{sum(1 for r in rows if not r.ambiguous)} clear). "
        f"Running {len(styles)} style(s) = {len(rows) * len(styles)} agent calls."
    )

    results: list[dict] = []
    for style in styles:
        print(f"\n--- Running style={style} ---")
        for row in rows:
            print(f"  {row.query[:60]}...", flush=True)
            results.append(run_one(row, style))

    summary = aggregate(results)
    print("\n=== Prompt-style comparison ===")
    _print_table("ALL rows", summary["all"])
    _print_table("CLEAR rows only (15)", summary["clear"])
    _print_table("AMBIGUOUS rows only (10)", summary["ambiguous"])
    print("\n(score is mean; % is pass rate)")

    _print_per_row(results)


if __name__ == "__main__":
    main()
