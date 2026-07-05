"""
Driver: run the queries through both modes and print the mode comparison,
selection accuracy, the retrieval recall ceiling, the paired lift, and the
prompt-token cost difference (the other reason to pre-filter tools).

Usage:
    python run_eval.py                 # all_tools vs retrieved, all rows
    python run_eval.py --mode retrieved
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import correct_tool, retrieval_recall, all_to_retrieved_lift

MODES = ["all_tools", "retrieved"]


@dataclass
class Row:
    query: str
    correct_tool: str
    cluster: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(query=r["query"], correct_tool=r["correct_tool"], cluster=r["cluster"]))
    return rows


def run_one(row: Row, mode: str) -> dict:
    res = agent.run(row.query, mode=mode)
    d = {
        "query": row.query, "gold": row.correct_tool, "cluster": row.cluster, "mode": mode,
        "chosen": res.chosen_tool, "prompt_tokens": res.prompt_tokens,
        "correct": correct_tool(res.chosen_tool, row.correct_tool),
    }
    if mode == "retrieved":
        d["recall"] = retrieval_recall(res.shortlist, row.correct_tool)
    return d


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>10} | {'correct tool':>15} | {'mean prompt tokens':>19}")
    print("-" * 52)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        c = sum(1 for r in rs if r["correct"].passed)
        pt = sum(r["prompt_tokens"] for r in rs) / n
        print(f"{mode:>10} | {f'{c}/{n} ({c/n*100:.0f}%)':>15} | {pt:>19.0f}")


def _print_recall(results: list[dict]) -> None:
    rr = [r for r in results if r["mode"] == "retrieved" and "recall" in r]
    if not rr:
        return
    n = len(rr)
    recall = sum(1 for r in rr if r["recall"].passed)
    correct = sum(1 for r in rr if r["correct"].passed)
    print("\n=== Retrieval ceiling (retrieved mode) ===\n")
    print(f"  retrieval_recall (gold tool in shortlist): {recall}/{n}")
    print(f"  correct_tool     (gold tool then picked):  {correct}/{n}")
    miss_recall = [r for r in rr if not r["recall"].passed]
    if miss_recall:
        print(f"  gold tool NOT retrieved (retrieval error): {len(miss_recall)}")
        for r in miss_recall:
            print(f"    [{r['cluster']:>8}] {r['query'][:44]} (gold={r['gold']})")
    miss_select = [r for r in rr if r["recall"].passed and not r["correct"].passed]
    if miss_select:
        print(f"  gold retrieved but mis-picked (selection error): {len(miss_select)}")
        for r in miss_select:
            print(f"    [{r['cluster']:>8}] {r['query'][:40]} chose={r['chosen']} (gold={r['gold']})")


def _print_lift(results: list[dict]) -> None:
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["query"]][r["mode"]] = r
    if not all("all_tools" in m and "retrieved" in m for m in by_q.values()):
        return
    print("\n=== Paired lift: all_tools -> retrieved ===\n")
    helped = hurt = neutral = 0
    for q, m in by_q.items():
        a, r = m["all_tools"], m["retrieved"]
        L = all_to_retrieved_lift(a["correct"].passed, r["correct"].passed)
        if L.score == 1.0:
            helped += 1; tag = "  HELPED"
        elif L.score == 0.0:
            hurt += 1; tag = "    HURT"
        else:
            neutral += 1; tag = ""
        if tag:
            print(f"  [{a['cluster']:>8}] all={'OK ' if a['correct'].passed else 'BAD'}({a['chosen']}) "
                  f"ret={'OK ' if r['correct'].passed else 'BAD'}({r['chosen']}) gold={a['gold']}{tag}")
            print(f"             {q[:60]}")
    total = helped + hurt + neutral
    mean = (helped + neutral * 0.5) / total if total else 0.0
    print(f"\n  all_to_retrieved_lift mean = {mean:.2f}  (helped {helped}, hurt {hurt}, neutral {neutral})")
    ta = sum(r["prompt_tokens"] for r in results if r["mode"] == "all_tools")
    tr = sum(r["prompt_tokens"] for r in results if r["mode"] == "retrieved")
    print(f"  prompt tokens — all_tools={ta}, retrieved={tr} ({ta/tr:.1f}x more for all_tools)")


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
            print(f"  [{row.cluster:>8}] {row.query[:44]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_recall(results)
    if args.mode == "all":
        _print_lift(results)


if __name__ == "__main__":
    main()
