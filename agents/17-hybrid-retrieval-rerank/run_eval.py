"""
Driver: run the dataset through all three modes (lexical / hybrid / rerank) and
print the staircase — accuracy + retrieval-hit per mode, the same split by
category (clean vs poor), and the candidate-recall vs retrieval-hit gap that
shows whether reranking even had room to help.

Usage:
    python run_eval.py                 # all three modes, all rows
    python run_eval.py --mode hybrid
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import answer_correctness, retrieval_hit, candidate_recall

MODES = ["lexical", "hybrid", "rerank"]
CATS = ["clean", "poor"]


@dataclass
class Row:
    query: str
    category: str
    gold: str
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(query=r["query"], category=r["category"],
                            gold=r["gold_answer"], notes=r.get("notes", "") or ""))
    return rows


def run_one(row: Row, mode: str) -> dict:
    res = agent.run(row.query, mode=mode)
    return {
        "query": row.query, "category": row.category, "gold": row.gold, "mode": mode,
        "answer": res.answer, "llm_calls": res.llm_calls,
        "correct": answer_correctness(res.answer, row.gold),
        "hit": retrieval_hit(res.retrieved, row.gold),
        "recall": candidate_recall(res.pool, row.gold),
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Mode staircase ===\n")
    print(f"{'mode':>8} | {'correct':>14} | {'retrieval hit':>14} | {'mean llm_calls':>15}")
    print("-" * 60)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        c = sum(1 for r in rs if r["correct"].passed)
        h = sum(1 for r in rs if r["hit"].passed)
        mc = sum(r["llm_calls"] for r in rs) / n
        print(f"{mode:>8} | {f'{c}/{n} ({c/n*100:.0f}%)':>14} | {f'{h}/{n} ({h/n*100:.0f}%)':>14} | {mc:>15.1f}")


def _print_by_category(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Accuracy by category ===\n")
    print(f"{'mode':>8} | " + " | ".join(f"{c:>12}" for c in CATS))
    print("-" * (10 + 15 * len(CATS)))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        cells = []
        for cat in CATS:
            crs = [r for r in rs if r["category"] == cat]
            c = sum(1 for r in crs if r["correct"].passed)
            cells.append(f"{c}/{len(crs)}")
        print(f"{mode:>8} | " + " | ".join(f"{c:>12}" for c in cells))


def _print_rerank_ceiling(results: list[dict]) -> None:
    rer = [r for r in results if r["mode"] == "rerank"]
    if not rer:
        return
    n = len(rer)
    recall = sum(1 for r in rer if r["recall"].passed)
    hit = sum(1 for r in rer if r["hit"].passed)
    print("\n=== Reranking's room to help (rerank mode) ===\n")
    print(f"  candidate_recall (gold in pool):     {recall}/{n}")
    print(f"  retrieval_hit   (gold in final top-k): {hit}/{n}")
    gap = [r for r in rer if r["recall"].passed and not r["hit"].passed]
    print(f"  rows where gold was in the pool but NOT in final top-k (rerank's job): {len(gap)}")
    for r in gap:
        print(f"    [{r['category']:>5}] {r['query'][:54]}")
    missed_pool = [r for r in rer if not r["recall"].passed]
    if missed_pool:
        print(f"  rows where gold was NOT even in the pool (rerank cannot help): {len(missed_pool)}")
        for r in missed_pool:
            print(f"    [{r['category']:>5}] {r['query'][:54]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = MODES if args.mode == "all" else [args.mode]

    print(f"Loaded {len(rows)} queries. Running {len(modes)} mode(s).")
    results: list[dict] = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  [{row.category:>5}] {row.query[:48]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_by_category(results)
    _print_rerank_ceiling(results)


if __name__ == "__main__":
    main()
