"""
Driver: run the dataset through both modes and print the mode comparison,
accuracy + evidence-completeness split by category (local / multihop / global —
where graph structure matters), and the paired flat -> graph lift.

Usage:
    python run_eval.py                 # flat vs graph, all rows
    python run_eval.py --mode graph
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import answer_correctness, evidence_complete, flat_to_graph_lift

MODES = ["flat", "graph"]
CATS = ["local", "multihop", "global"]


@dataclass
class Row:
    query: str
    category: str
    gold: str
    support: list[str]
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(query=r["query"], category=r["category"], gold=r["gold_answer"],
                            support=[s for s in r["support"].split("|") if s], notes=r.get("notes", "") or ""))
    return rows


def run_one(row: Row, mode: str) -> dict:
    res = agent.run(row.query, mode=mode)
    return {
        "query": row.query, "category": row.category, "gold": row.gold, "mode": mode,
        "answer": res.answer, "n_facts": len(res.retrieved),
        "correct": answer_correctness(res.answer, row.gold),
        "evidence": evidence_complete(res.retrieved, row.support),
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>6} | {'correct':>14} | {'evidence complete':>18} | {'mean facts':>11}")
    print("-" * 60)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        c = sum(1 for r in rs if r["correct"].passed)
        e = sum(1 for r in rs if r["evidence"].passed)
        mf = sum(r["n_facts"] for r in rs) / n
        print(f"{mode:>6} | {f'{c}/{n} ({c/n*100:.0f}%)':>14} | {f'{e}/{n} ({e/n*100:.0f}%)':>18} | {mf:>11.1f}")


def _print_by_category(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Accuracy by category ===\n")
    print(f"{'mode':>6} | " + " | ".join(f"{c:>10}" for c in CATS))
    print("-" * (8 + 13 * len(CATS)))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        cells = []
        for cat in CATS:
            crs = [r for r in rs if r["category"] == cat]
            c = sum(1 for r in crs if r["correct"].passed)
            cells.append(f"{c}/{len(crs)}")
        print(f"{mode:>6} | " + " | ".join(f"{c:>10}" for c in cells))


def _print_lift(results: list[dict]) -> None:
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["query"]][r["mode"]] = r
    if not all("flat" in m and "graph" in m for m in by_q.values()):
        return
    print("\n=== Paired lift: flat -> graph ===\n")
    helped = hurt = neutral = 0
    for q, m in by_q.items():
        d, w = m["flat"], m["graph"]
        L = flat_to_graph_lift(d["correct"].passed, w["correct"].passed)
        if L.score == 1.0:
            helped += 1; tag = "  HELPED"
        elif L.score == 0.0:
            hurt += 1; tag = "    HURT"
        else:
            neutral += 1; tag = ""
        if tag:
            print(f"  [{d['category']:>8}] flat={'OK ' if d['correct'].passed else 'BAD'} "
                  f"graph={'OK ' if w['correct'].passed else 'BAD'} "
                  f"(flat={d['answer']!r}, graph={w['answer']!r}, gold={d['gold']!r}){tag}")
            print(f"             {q[:62]}")
    total = helped + hurt + neutral
    mean = (helped + neutral * 0.5) / total if total else 0.0
    print(f"\n  flat_to_graph_lift mean = {mean:.2f}  (helped {helped}, hurt {hurt}, neutral {neutral})")


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
            print(f"  [{row.category:>8}] {row.query[:46]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_by_category(results)
    if args.mode == "all":
        _print_lift(results)


if __name__ == "__main__":
    main()
