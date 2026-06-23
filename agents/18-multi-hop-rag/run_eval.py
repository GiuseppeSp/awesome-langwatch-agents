"""
Driver: run the dataset through both modes and print the mode comparison,
accuracy + answer-retrieval split by category (single vs multi — where single-shot
structurally fails), and the paired single -> multihop lift.

Usage:
    python run_eval.py                 # single vs multihop, all rows
    python run_eval.py --mode multihop
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import answer_correctness, answer_retrieved, single_to_multihop_lift

MODES = ["single", "multihop"]
CATS = ["single", "multi"]


@dataclass
class Row:
    query: str
    category: str
    hops: int
    gold: str
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(query=r["query"], category=r["category"], hops=int(r["hops"]),
                            gold=r["gold_answer"], notes=r.get("notes", "") or ""))
    return rows


def run_one(row: Row, mode: str) -> dict:
    res = agent.run(row.query, mode=mode)
    return {
        "query": row.query, "category": row.category, "hops": row.hops, "gold": row.gold,
        "mode": mode, "answer": res.answer, "llm_calls": res.llm_calls, "agent_hops": res.hops,
        "correct": answer_correctness(res.answer, row.gold),
        "retrieved": answer_retrieved(res.retrieved, row.gold),
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>9} | {'correct':>14} | {'answer retrieved':>16} | {'mean llm_calls':>15}")
    print("-" * 64)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        c = sum(1 for r in rs if r["correct"].passed)
        h = sum(1 for r in rs if r["retrieved"].passed)
        mc = sum(r["llm_calls"] for r in rs) / n
        print(f"{mode:>9} | {f'{c}/{n} ({c/n*100:.0f}%)':>14} | {f'{h}/{n} ({h/n*100:.0f}%)':>16} | {mc:>15.1f}")


def _print_by_category(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Accuracy by category ===\n")
    print(f"{'mode':>9} | {'single (1-hop)':>16} | {'multi (2-3 hop)':>16}")
    print("-" * 48)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        cells = []
        for cat in CATS:
            crs = [r for r in rs if r["category"] == cat]
            c = sum(1 for r in crs if r["correct"].passed)
            cells.append(f"{c}/{len(crs)}")
        print(f"{mode:>9} | {cells[0]:>16} | {cells[1]:>16}")


def _print_lift(results: list[dict]) -> None:
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["query"]][r["mode"]] = r
    if not all("single" in m and "multihop" in m for m in by_q.values()):
        return
    print("\n=== Paired lift: single -> multihop ===\n")
    helped = hurt = neutral = 0
    for q, m in by_q.items():
        d, w = m["single"], m["multihop"]
        L = single_to_multihop_lift(d["correct"].passed, w["correct"].passed)
        if L.score == 1.0:
            helped += 1; tag = "  HELPED"
        elif L.score == 0.0:
            hurt += 1; tag = "    HURT"
        else:
            neutral += 1; tag = ""
        if tag:
            print(f"  [{d['category']:>6} {d['hops']}h] single={'OK ' if d['correct'].passed else 'BAD'} "
                  f"multihop={'OK ' if w['correct'].passed else 'BAD'} (gold={d['gold']!r}, hops={w['agent_hops']}){tag}")
            print(f"             {q[:64]}")
    total = helped + hurt + neutral
    mean = (helped + neutral * 0.5) / total if total else 0.0
    print(f"\n  single_to_multihop_lift mean = {mean:.2f}  (helped {helped}, hurt {hurt}, neutral {neutral})")
    # cost
    tot_s = sum(r["llm_calls"] for r in results if r["mode"] == "single")
    tot_m = sum(r["llm_calls"] for r in results if r["mode"] == "multihop")
    print(f"  total LLM calls — single={tot_s}, multihop={tot_m} ({tot_m/tot_s:.1f}x)")


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
            print(f"  [{row.category:>6} {row.hops}h] {row.query[:46]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_by_category(results)
    if args.mode == "all":
        _print_lift(results)


if __name__ == "__main__":
    main()
