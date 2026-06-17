"""
Driver: run the dataset through the agent in both modes and print the mode
comparison, accuracy sliced by problem depth (where least-to-most's claim
lives), the paired cot -> least_to_most lift, and a decomposition-granularity
diagnostic.

Usage:
    python run_eval.py                    # cot vs least_to_most, all rows
    python run_eval.py --mode least_to_most
    python run_eval.py --limit 4          # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import answer_correctness, ltm_lift, cost

MODES = ["cot", "least_to_most"]


@dataclass
class Row:
    problem: str
    depth: int
    answer: float
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(
                problem=r["problem"], depth=int(r["depth"]),
                answer=float(r["answer"]), notes=r.get("notes", "") or "",
            ))
    return rows


def run_one(row: Row, mode: str) -> dict:
    result = agent.run(row.problem, mode=mode)
    return {
        "problem": row.problem, "depth": row.depth, "gold": row.answer, "mode": mode,
        "answer": result.answer, "n_subq": len(result.subquestions),
        "llm_calls": result.llm_calls,
        "correct": answer_correctness(result.answer, row.answer),
        "cost": cost(result.llm_calls),
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)

    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>15} | {'correct':>16} | {'mean llm_calls':>15}")
    print("-" * 54)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        c = sum(1 for r in rs if r["correct"].passed)
        mean_calls = sum(r["llm_calls"] for r in rs) / n
        print(f"{mode:>15} | {f'{c}/{n} ({c/n*100:.0f}%)':>16} | {mean_calls:>15.1f}")


def _print_by_depth(results: list[dict]) -> None:
    depths = sorted({r["depth"] for r in results})
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)

    print("\n=== Accuracy by problem depth (the crossover, if any) ===\n")
    print(f"{'mode':>15} | " + " | ".join(f"depth {d:>2}" for d in depths))
    print("-" * (17 + 11 * len(depths)))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        cells = []
        for d in depths:
            drs = [r for r in rs if r["depth"] == d]
            c = sum(1 for r in drs if r["correct"].passed)
            cells.append(f"{c}/{len(drs)}" if drs else "-")
        print(f"{mode:>15} | " + " | ".join(f"{c:>8}" for c in cells))


def _print_lift(results: list[dict]) -> None:
    by_problem: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_problem[r["problem"]][r["mode"]] = r
    if not all("cot" in m and "least_to_most" in m for m in by_problem.values()):
        return

    print("\n=== Paired lift: cot -> least_to_most ===\n")
    helped = hurt = neutral = 0
    for problem, m in by_problem.items():
        d, l = m["cot"], m["least_to_most"]
        L = ltm_lift(d["correct"].passed, l["correct"].passed)
        if L.score == 1.0:
            helped += 1; tag = "  HELPED"
        elif L.score == 0.0:
            hurt += 1; tag = "    HURT"
        else:
            neutral += 1; tag = ""
        if tag:
            print(f"  [depth {d['depth']}] cot={'OK ' if d['correct'].passed else 'BAD'} "
                  f"ltm={'OK ' if l['correct'].passed else 'BAD'} "
                  f"(cot={d['answer']}, ltm={l['answer']}, gold={d['gold']:g}){tag}")
            print(f"             {problem[:72]}")
    total = helped + hurt + neutral
    mean = (helped + neutral * 0.5) / total if total else 0.0
    print(f"\n  ltm_lift mean = {mean:.2f}  (helped {helped}, hurt {hurt}, neutral {neutral})")

    # decomposition granularity: how many subquestions did ltm produce vs the
    # canonical depth? (a proxy for whether it decomposed at the right grain)
    ltm = [r for r in results if r["mode"] == "least_to_most"]
    match = sum(1 for r in ltm if r["n_subq"] == r["depth"])
    over = sum(1 for r in ltm if r["n_subq"] > r["depth"])
    under = sum(1 for r in ltm if r["n_subq"] < r["depth"])
    mean_subq = sum(r["n_subq"] for r in ltm) / len(ltm) if ltm else 0
    print(f"\n  decomposition granularity: matched canonical depth {match}/{len(ltm)} "
          f"(finer {over}, coarser {under}); mean subquestions = {mean_subq:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = MODES if args.mode == "all" else [args.mode]

    print(f"Loaded {len(rows)} problems. Running {len(modes)} mode(s).")
    results: list[dict] = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  [depth {row.depth}] {row.problem[:50]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_by_depth(results)
    if args.mode == "all":
        _print_lift(results)


if __name__ == "__main__":
    main()
