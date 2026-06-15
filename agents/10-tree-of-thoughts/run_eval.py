"""
Driver: run the Game of 24 dataset through the agent in both modes and print
the comparison, the solve rate by difficulty, and — for ToT — how accurate the
state evaluator's keep/prune judgements were against the brute-force oracle.

Usage:
    python run_eval.py                  # cot vs tot, all puzzles
    python run_eval.py --mode tot       # one mode
    python run_eval.py --limit 3        # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import solved, evaluator_accuracy, search_efficiency

MODES = ["cot", "tot"]


@dataclass
class Row:
    numbers: list
    difficulty: str
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(
                numbers=[float(x) for x in r["numbers"].split()],
                difficulty=r["difficulty"], notes=r.get("notes", "") or "",
            ))
    return rows


def run_one(row: Row, mode: str) -> dict:
    result = agent.run(row.numbers, mode=mode)
    return {
        "numbers": row.numbers, "difficulty": row.difficulty, "mode": mode,
        "expr": result.final_expression, "llm_calls": result.llm_calls,
        "solved": solved(result.final_expression, row.numbers),
        "eval_acc": evaluator_accuracy(result.judgements),
        "efficiency": search_efficiency(result.llm_calls),
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)

    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>6} | {'solved':>14} | {'mean llm_calls':>15} | {'evaluator_accuracy':>20}")
    print("-" * 66)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        c = sum(1 for r in rs if r["solved"].passed)
        mean_calls = sum(r["llm_calls"] for r in rs) / n
        accs = [r["eval_acc"].score for r in rs if r["eval_acc"].score == r["eval_acc"].score]
        acc = f"{sum(accs)/len(accs)*100:.0f}%" if accs else "n/a"
        print(f"{mode:>6} | {f'{c}/{n} ({c/n*100:.0f}%)':>14} | {mean_calls:>15.1f} | {acc:>20}")

    print("\n=== Solve rate by difficulty ===\n")
    diffs = ["easy", "medium", "hard"]
    print(f"{'mode':>6} | " + " | ".join(f"{d:>8}" for d in diffs))
    print("-" * (8 + 11 * len(diffs)))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        by_d: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_d[r["difficulty"]].append(r)
        cells = []
        for d in diffs:
            drs = by_d.get(d, [])
            c = sum(1 for r in drs if r["solved"].passed)
            cells.append(f"{c}/{len(drs)}" if drs else "-")
        print(f"{mode:>6} | " + " | ".join(f"{c:>8}" for c in cells))


def _print_eval_breakdown(results: list[dict]) -> None:
    tot = [r for r in results if r["mode"] == "tot"]
    if not tot:
        return
    pruned_winners = kept_deadends = 0
    for r in tot:
        # re-derive from the stored reasons isn't ideal; recompute from agent run is costly.
        # Instead parse the counts the evaluator embedded in its reason string.
        reason = r["eval_acc"].reason
        import re
        pw = re.search(r"pruned-reachable=(\d+)", reason)
        kd = re.search(r"kept-deadend=(\d+)", reason)
        if pw:
            pruned_winners += int(pw.group(1))
        if kd:
            kept_deadends += int(kd.group(1))
    print("\n=== ToT evaluator error breakdown (across all puzzles) ===\n")
    print(f"  pruned-reachable states (FATAL — threw away a path to 24): {pruned_winners}")
    print(f"  kept-deadend states   (WASTEFUL — spent budget on dead ends): {kept_deadends}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = MODES if args.mode == "all" else [args.mode]

    print(f"Loaded {len(rows)} puzzles. Running {len(modes)} mode(s). "
          f"(tot makes many calls per puzzle — this can take a few minutes.)")

    results: list[dict] = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            nums = " ".join(str(int(x)) for x in row.numbers)
            print(f"  [{row.difficulty:>6}] {nums}", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    if "tot" in modes:
        _print_eval_breakdown(results)


if __name__ == "__main__":
    main()
