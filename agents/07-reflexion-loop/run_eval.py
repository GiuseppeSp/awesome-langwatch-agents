"""
Driver: run the dataset through the reflexion agent in both modes and print
a comparison table.

Usage:
    python run_eval.py                       # single_pass vs reflexion, all rows
    python run_eval.py --mode reflexion      # one mode only
    python run_eval.py --limit 3             # smoke test
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import EvalResult, constraint_satisfaction, retry_lift, self_critique_accuracy


@dataclass
class Row:
    prompt: str
    constraint_type: str
    primary: str
    secondary: str
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                Row(
                    prompt=r["prompt"],
                    constraint_type=r["constraint_type"],
                    primary=r["primary"],
                    secondary=r.get("secondary", "") or "",
                    notes=r.get("notes", "") or "",
                )
            )
    return rows


def run_one(row: Row, mode: str) -> dict:
    result = agent.run(row.prompt, mode=mode)
    r_constraint = constraint_satisfaction(
        result.final_output, row.constraint_type, row.primary, row.secondary
    )
    if mode == "reflexion":
        r_lift = retry_lift(
            result.first_attempt_output,
            result.final_output,
            row.constraint_type,
            row.primary,
            row.secondary,
        )
        r_critique = self_critique_accuracy(
            result.attempts, row.constraint_type, row.primary, row.secondary
        )
    else:
        nan = EvalResult(
            name="retry_lift", passed=False, score=float("nan"),
            reason="N/A in single_pass (no retry)",
        )
        nan2 = EvalResult(
            name="self_critique_accuracy", passed=False, score=float("nan"),
            reason="N/A in single_pass (no critique step)",
        )
        r_lift, r_critique = nan, nan2

    return {
        "prompt": row.prompt,
        "mode": mode,
        "first_output": result.first_attempt_output,
        "final_output": result.final_output,
        "num_attempts": result.num_attempts,
        "evals": {
            "constraint_satisfaction": r_constraint,
            "retry_lift": r_lift,
            "self_critique_accuracy": r_critique,
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
            valid = [e for e in evs if not math.isnan(e.score)]
            if not valid:
                out[mode][name] = (float("nan"), float("nan"))
                continue
            mean_score = sum(e.score for e in valid) / len(valid)
            pass_rate = sum(1 for e in valid if e.passed) / len(valid)
            out[mode][name] = (mean_score, pass_rate)
    return out


def _fmt_cell(value: tuple[float, float]) -> str:
    score, rate = value
    if math.isnan(score):
        return "n/a"
    return f"{score:.2f} ({rate * 100:.0f}%)"


def _print_summary(summary: dict) -> None:
    eval_names = ["constraint_satisfaction", "retry_lift", "self_critique_accuracy"]
    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>12} | " + " | ".join(f"{n:>24}" for n in eval_names))
    print("-" * (14 + sum(26 for _ in eval_names)))
    for mode, by_eval in summary.items():
        cells = [_fmt_cell(by_eval.get(n, (float("nan"), float("nan")))) for n in eval_names]
        print(f"{mode:>12} | " + " | ".join(f"{c:>24}" for c in cells))
    print("\n(score is mean; % is pass rate)\n")


def _print_per_row(results: list[dict]) -> None:
    print("=== Per-row detail ===\n")
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["prompt"]][r["mode"]] = r
    for prompt, by_mode in by_q.items():
        print(f"  {prompt}")
        for mode, r in by_mode.items():
            constraint = r["evals"]["constraint_satisfaction"]
            lift = r["evals"]["retry_lift"]
            critique = r["evals"]["self_critique_accuracy"]
            lift_str = f"lift={lift.score:.2f}" if not math.isnan(lift.score) else "lift=n/a"
            crit_str = (
                f"critique={critique.score:.2f}"
                if not math.isnan(critique.score)
                else "critique=n/a"
            )
            print(
                f"    [{mode:>12}] attempts={r['num_attempts']} "
                f"constraint={constraint.score:.0f} {lift_str} {crit_str}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["single_pass", "reflexion", "both"], default="both",
        help="Which reflexion mode to evaluate. Default: both.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only first N rows.")
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = ["single_pass", "reflexion"] if args.mode == "both" else [args.mode]

    print(
        f"Loaded {len(rows)} dataset rows. Running {len(modes)} mode(s) = "
        f"{len(rows) * len(modes)} agent calls."
    )

    results: list[dict] = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  {row.prompt[:60]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(aggregate(results))
    _print_per_row(results)


if __name__ == "__main__":
    main()
