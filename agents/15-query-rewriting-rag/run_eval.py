"""
Driver: run the dataset through the agent in both modes and print the mode
comparison, accuracy + retrieval hit-rate split by category (clean vs poor —
where rewriting earns its keep), and the paired vanilla -> rewrite lift.

Usage:
    python run_eval.py                 # vanilla vs rewrite, all rows
    python run_eval.py --mode rewrite
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import answer_correctness, retrieval_hit, rewrite_lift

MODES = ["vanilla", "rewrite"]
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
        "answer": res.answer, "search_query": res.search_query, "llm_calls": res.llm_calls,
        "correct": answer_correctness(res.answer, row.gold),
        "hit": retrieval_hit(res.retrieved, row.gold),
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Mode comparison ===\n")
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
    print("\n=== Accuracy / retrieval-hit by category ===\n")
    print(f"{'mode':>8} | " + " | ".join(f"{c+' acc':>10} {c+' hit':>10}" for c in CATS))
    print("-" * (10 + 23 * len(CATS)))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        cells = []
        for cat in CATS:
            crs = [r for r in rs if r["category"] == cat]
            c = sum(1 for r in crs if r["correct"].passed)
            h = sum(1 for r in crs if r["hit"].passed)
            cells.append(f"{f'{c}/{len(crs)}':>10} {f'{h}/{len(crs)}':>10}")
        print(f"{mode:>8} | " + " | ".join(cells))


def _print_lift(results: list[dict]) -> None:
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["query"]][r["mode"]] = r
    if not all("vanilla" in m and "rewrite" in m for m in by_q.values()):
        return
    print("\n=== Paired lift: vanilla -> rewrite ===\n")
    helped = hurt = neutral = 0
    for q, m in by_q.items():
        d, w = m["vanilla"], m["rewrite"]
        L = rewrite_lift(d["correct"].passed, w["correct"].passed)
        if L.score == 1.0:
            helped += 1; tag = "  HELPED"
        elif L.score == 0.0:
            hurt += 1; tag = "    HURT"
        else:
            neutral += 1; tag = ""
        if tag:
            print(f"  [{d['category']:>5}] vanilla={'OK ' if d['correct'].passed else 'BAD'} "
                  f"rewrite={'OK ' if w['correct'].passed else 'BAD'} (gold={d['gold']!r}){tag}")
            print(f"           q: {q[:60]}")
            print(f"           rewritten -> {w['search_query'][:60]!r}")
    total = helped + hurt + neutral
    mean = (helped + neutral * 0.5) / total if total else 0.0
    print(f"\n  rewrite_lift mean = {mean:.2f}  (helped {helped}, hurt {hurt}, neutral {neutral})")


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
            print(f"  [{row.category:>5}] {row.query[:50]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_by_category(results)
    if args.mode == "all":
        _print_lift(results)


if __name__ == "__main__":
    main()
