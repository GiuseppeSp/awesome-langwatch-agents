"""
Driver: run the tasks through both modes and print the mode comparison, solve
rate by category (clean / recoverable / unrecoverable), the paired lift, and the
efficiency of the retry loop on unrecoverable tasks (did it give up early?).

Usage:
    python run_eval.py                 # no_retry vs retry, all rows
    python run_eval.py --mode retry
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import solved, outcome_correct, no_retry_to_retry_lift

MODES = ["no_retry", "retry"]
CATS = ["clean", "recoverable", "unrecoverable"]


@dataclass
class Row:
    task: str
    category: str
    solvable: bool


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(task=r["task"], category=r["category"],
                            solvable=r["solvable"].strip().lower() == "yes"))
    return rows


def run_one(row: Row, mode: str) -> dict:
    res = agent.run(row.task, mode=mode)
    return {
        "task": row.task, "category": row.category, "solvable": row.solvable, "mode": mode,
        "solved": res.solved, "attempts": res.attempts, "gave_up": res.gave_up,
        "solved_eval": solved(res.solved),
        "outcome": outcome_correct(res.solved, row.solvable),
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>9} | {'solved':>14} | {'outcome correct':>16} | {'mean attempts':>14}")
    print("-" * 62)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        s = sum(1 for r in rs if r["solved"])
        o = sum(1 for r in rs if r["outcome"].passed)
        a = sum(r["attempts"] for r in rs) / n
        print(f"{mode:>9} | {f'{s}/{n} ({s/n*100:.0f}%)':>14} | {f'{o}/{n} ({o/n*100:.0f}%)':>16} | {a:>14.1f}")


def _print_by_category(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Solved by category ===\n")
    print(f"{'mode':>9} | " + " | ".join(f"{c:>13}" for c in CATS))
    print("-" * (11 + 16 * len(CATS)))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        cells = []
        for cat in CATS:
            crs = [r for r in rs if r["category"] == cat]
            s = sum(1 for r in crs if r["solved"])
            cells.append(f"{s}/{len(crs)}")
        print(f"{mode:>9} | " + " | ".join(f"{c:>13}" for c in cells))


def _print_lift_and_efficiency(results: list[dict]) -> None:
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["task"]][r["mode"]] = r
    if all("no_retry" in m and "retry" in m for m in by_q.values()):
        helped = hurt = neutral = 0
        for q, m in by_q.items():
            L = no_retry_to_retry_lift(m["no_retry"]["solved"], m["retry"]["solved"])
            if L.score == 1.0:
                helped += 1
            elif L.score == 0.0:
                hurt += 1
            else:
                neutral += 1
        total = helped + hurt + neutral
        mean = (helped + neutral * 0.5) / total if total else 0.0
        print(f"\n=== Paired lift: no_retry -> retry ===\n")
        print(f"  no_retry_to_retry_lift mean = {mean:.2f}  (helped {helped}, hurt {hurt}, neutral {neutral})")

    # efficiency: attempts spent on unrecoverable tasks in retry mode
    unrec = [r for r in results if r["mode"] == "retry" and r["category"] == "unrecoverable"]
    if unrec:
        print(f"\n=== Retry efficiency on unrecoverable tasks (MAX_ATTEMPTS={agent.MAX_ATTEMPTS}) ===\n")
        for r in unrec:
            print(f"  {r['attempts']} attempt(s){' + gave_up' if r['gave_up'] else ' (no give-up)'}: {r['task'][:50]}")
        gave_up = sum(1 for r in unrec if r["gave_up"])
        print(f"\n  gave up early on {gave_up}/{len(unrec)} unrecoverable tasks; "
              f"mean attempts {sum(r['attempts'] for r in unrec)/len(unrec):.1f} (min 1, max {agent.MAX_ATTEMPTS})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = MODES if args.mode == "all" else [args.mode]

    print(f"Loaded {len(rows)} tasks. Running {len(modes)} mode(s).")
    results: list[dict] = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  [{row.category:>13}] {row.task[:44]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_by_category(results)
    if args.mode == "all":
        _print_lift_and_efficiency(results)


if __name__ == "__main__":
    main()
