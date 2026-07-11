"""
Driver: run each question through single, union, and moa, and print the three-way
recall / precision / F1, the paired union->moa delta (does the aggregator add over a
naive merge, and where — precision or recall?), a by-size breakdown (does ensembling
help more on the large sets a single pass under-recalls?), and cost.

Usage:
    python run_eval.py                 # single + union + moa, all rows
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import score_set

MODES = ["single", "union", "moa"]


@dataclass
class Row:
    question: str
    gold: list
    size: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    with open(path, encoding="utf-8") as f:
        return [Row(r["question"], json.loads(r["gold_json"]), r["size"]) for r in csv.DictReader(f)]


def run_one(row: Row, mode: str) -> dict:
    res = agent.run(row.question, mode=mode)
    sc = score_set(res.items, row.gold)
    return {"row": row, "mode": mode, "items": res.items, "score": sc, "llm_calls": res.llm_calls}


def _mean(results, mode, key):
    rs = [r["score"][key] for r in results if r["mode"] == mode]
    return sum(rs) / len(rs) if rs else 0.0


def _print_summary(results):
    n = len({r["row"].question for r in results})
    print("\n=== Three-way: recall / precision / F1 (mean over questions) ===\n")
    print(f"{'mode':>8} | {'recall':>7} | {'precision':>9} | {'F1':>6} | {'mean calls':>10}")
    print("-" * 54)
    for mode in MODES:
        calls = sum(r["llm_calls"] for r in results if r["mode"] == mode) / n
        print(f"{mode:>8} | {_mean(results,mode,'recall'):>7.2f} | {_mean(results,mode,'precision'):>9.2f} "
              f"| {_mean(results,mode,'f1'):>6.2f} | {calls:>10.1f}")


def _print_delta(results):
    by_q = defaultdict(dict)
    for r in results:
        by_q[r["row"].question][r["mode"]] = r
    print("\n=== union -> moa: what does the aggregator add over a naive merge? ===\n")
    dr = dp = 0.0
    rec_up = prec_up = 0
    for q, m in by_q.items():
        if "union" not in m or "moa" not in m:
            continue
        u, a = m["union"]["score"], m["moa"]["score"]
        dr += a["recall"] - u["recall"]
        dp += a["precision"] - u["precision"]
        if a["recall"] > u["recall"] + 1e-9:
            rec_up += 1
        if a["precision"] > u["precision"] + 1e-9:
            prec_up += 1
    k = len(by_q)
    print(f"  mean recall change    union->moa: {dr/k:+.3f}   (moa recall > union on {rec_up}/{k} q)")
    print(f"  mean precision change union->moa: {dp/k:+.3f}   (moa precision > union on {prec_up}/{k} q)")


def _print_single_vs_union(results):
    by_q = defaultdict(dict)
    for r in results:
        by_q[r["row"].question][r["mode"]] = r
    print("\n=== single -> union: does ensembling add coverage (recall)? at what precision cost? ===\n")
    dr = dp = 0.0
    for q, m in by_q.items():
        s, u = m["single"]["score"], m["union"]["score"]
        dr += u["recall"] - s["recall"]
        dp += u["precision"] - s["precision"]
    k = len(by_q)
    print(f"  mean recall change    single->union: {dr/k:+.3f}")
    print(f"  mean precision change single->union: {dp/k:+.3f}")


def _print_by_size(results):
    print("\n=== By set size: recall (single -> union -> moa) ===\n")
    for size in ("small", "medium", "large"):
        rs = [r for r in results if r["row"].size == size]
        if not rs:
            continue
        cells = " -> ".join(f"{m} {_mean(rs, m, 'recall'):.2f}" for m in MODES)
        nq = len({r["row"].question for r in rs})
        print(f"  {size:>6} ({nq} q): {cells}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loaded {len(rows)} questions. Running {len(MODES)} modes.")
    results = []
    for mode in MODES:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  [{row.size:>6}] {row.question[:50]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_single_vs_union(results)
    _print_delta(results)
    _print_by_size(results)


if __name__ == "__main__":
    main()
