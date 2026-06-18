"""
Driver: run the dataset through the agent in both modes and print the mode
comparison, accuracy + LLM-call cost split by category (fixed vs branch — where
ReWOO's blind planning shows), and the paired react -> rewoo lift.

Usage:
    python run_eval.py                 # react vs rewoo, all rows
    python run_eval.py --mode rewoo
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import answer_correctness, llm_call_efficiency, react_to_rewoo_lift

MODES = ["react", "rewoo"]


@dataclass
class Row:
    query: str
    category: str
    expected: str
    canonical_tool_calls: int
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(
                query=r["query"], category=r["category"], expected=r["expected_answer"],
                canonical_tool_calls=int(r["canonical_tool_calls"]), notes=r.get("notes", "") or "",
            ))
    return rows


def run_one(row: Row, mode: str) -> dict:
    res = agent.run(row.query, mode=mode)
    return {
        "query": row.query, "category": row.category, "gold": row.expected,
        "mode": mode, "answer": res.answer, "llm_calls": res.llm_calls,
        "tool_calls": res.tool_calls, "canonical_tool_calls": row.canonical_tool_calls,
        "correct": answer_correctness(res.answer, row.expected),
        "efficiency": llm_call_efficiency(res.llm_calls),
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)

    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>8} | {'correct':>14} | {'mean llm_calls':>15} | {'mean tool_calls':>16}")
    print("-" * 64)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        c = sum(1 for r in rs if r["correct"].passed)
        mc = sum(r["llm_calls"] for r in rs) / n
        mt = sum(r["tool_calls"] for r in rs) / n
        print(f"{mode:>8} | {f'{c}/{n} ({c/n*100:.0f}%)':>14} | {mc:>15.1f} | {mt:>16.1f}")


def _print_by_category(results: list[dict]) -> None:
    cats = ["fixed", "branch"]
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)

    print("\n=== Accuracy + mean LLM calls by category ===\n")
    print(f"{'mode':>8} | " + " | ".join(f"{c+' acc':>12} {c+' calls':>12}" for c in cats))
    print("-" * (10 + 27 * len(cats)))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        cells = []
        for cat in cats:
            crs = [r for r in rs if r["category"] == cat]
            if crs:
                c = sum(1 for r in crs if r["correct"].passed)
                mc = sum(r["llm_calls"] for r in crs) / len(crs)
                cells.append(f"{c}/{len(crs):<10} {mc:>11.1f}")
            else:
                cells.append(f"{'-':>12} {'-':>12}")
        print(f"{mode:>8} | " + " | ".join(cells))


def _print_lift(results: list[dict]) -> None:
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["query"]][r["mode"]] = r
    if not all("react" in m and "rewoo" in m for m in by_q.values()):
        return

    print("\n=== Paired lift: react -> rewoo ===\n")
    helped = hurt = neutral = 0
    for q, m in by_q.items():
        d, w = m["react"], m["rewoo"]
        L = react_to_rewoo_lift(d["correct"].passed, w["correct"].passed)
        if L.score == 1.0:
            helped += 1; tag = "  HELPED"
        elif L.score == 0.0:
            hurt += 1; tag = "    HURT"
        else:
            neutral += 1; tag = ""
        if tag:
            print(f"  [{d['category']:>6}] react={'OK ' if d['correct'].passed else 'BAD'} "
                  f"rewoo={'OK ' if w['correct'].passed else 'BAD'} "
                  f"(react={d['answer']!r}, rewoo={w['answer']!r}, gold={d['gold']!r}){tag}")
            print(f"           {q[:70]}")
    total = helped + hurt + neutral
    mean = (helped + neutral * 0.5) / total if total else 0.0
    print(f"\n  react_to_rewoo_lift mean = {mean:.2f}  (helped {helped}, hurt {hurt}, neutral {neutral})")

    # efficiency headline: total LLM calls each mode spent
    tot_react = sum(r["llm_calls"] for r in results if r["mode"] == "react")
    tot_rewoo = sum(r["llm_calls"] for r in results if r["mode"] == "rewoo")
    print(f"\n  total LLM calls — react={tot_react}, rewoo={tot_rewoo} "
          f"({tot_react / tot_rewoo:.1f}x more for react)")


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
            print(f"  [{row.category:>6}] {row.query[:50]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_by_category(results)
    if args.mode == "all":
        _print_lift(results)


if __name__ == "__main__":
    main()
