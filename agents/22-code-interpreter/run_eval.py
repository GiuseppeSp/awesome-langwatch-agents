"""
Driver: run the questions through both modes and print the mode comparison,
accuracy by category (compute / trivial), and the paired reason -> code lift.

Usage:
    python run_eval.py                 # reason vs code, all rows
    python run_eval.py --mode code
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import answer_correct, executes_ok, reason_to_code_lift

MODES = ["reason", "code"]
CATS = ["compute", "trivial"]


@dataclass
class Row:
    question: str
    category: str
    gold: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(question=r["question"], category=r["category"], gold=r["gold"]))
    return rows


def run_one(row: Row, mode: str) -> dict:
    res = agent.run(row.question, mode=mode)
    return {
        "question": row.question, "category": row.category, "gold": row.gold, "mode": mode,
        "answer": res.answer, "error": res.error,
        "correct": answer_correct(res.answer, row.gold),
        "exec": executes_ok(mode, res.error),
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>7} | {'answer correct':>15} | {'executes ok':>13}")
    print("-" * 42)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        c = sum(1 for r in rs if r["correct"].passed)
        e = sum(1 for r in rs if r["exec"].passed)
        print(f"{mode:>7} | {f'{c}/{n} ({c/n*100:.0f}%)':>15} | {f'{e}/{n} ({e/n*100:.0f}%)':>13}")


def _print_by_category(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Answer-correct by category ===\n")
    print(f"{'mode':>7} | " + " | ".join(f"{c:>10}" for c in CATS))
    print("-" * (9 + 13 * len(CATS)))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        cells = []
        for cat in CATS:
            crs = [r for r in rs if r["category"] == cat]
            c = sum(1 for r in crs if r["correct"].passed)
            cells.append(f"{c}/{len(crs)}")
        print(f"{mode:>7} | " + " | ".join(f"{c:>10}" for c in cells))


def _print_lift(results: list[dict]) -> None:
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["question"]][r["mode"]] = r
    if not all("reason" in m and "code" in m for m in by_q.values()):
        return
    print("\n=== Paired lift: reason -> code ===\n")
    helped = hurt = neutral = 0
    for q, m in by_q.items():
        d, c = m["reason"], m["code"]
        L = reason_to_code_lift(d["correct"].passed, c["correct"].passed)
        if L.score == 1.0:
            helped += 1; tag = "  HELPED"
        elif L.score == 0.0:
            hurt += 1; tag = "    HURT"
        else:
            neutral += 1; tag = ""
        if tag:
            print(f"  [{d['category']:>7}] reason={'OK ' if d['correct'].passed else 'BAD'}({d['answer']}) "
                  f"code={'OK ' if c['correct'].passed else 'BAD'}({c['answer']}) gold={d['gold']}{tag}")
            print(f"             {q[:62]}")
    total = helped + hurt + neutral
    mean = (helped + neutral * 0.5) / total if total else 0.0
    print(f"\n  reason_to_code_lift mean = {mean:.2f}  (helped {helped}, hurt {hurt}, neutral {neutral})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = MODES if args.mode == "all" else [args.mode]

    print(f"Loaded {len(rows)} questions. Running {len(modes)} mode(s).")
    results: list[dict] = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  [{row.category:>7}] {row.question[:46]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_by_category(results)
    if args.mode == "all":
        _print_lift(results)


if __name__ == "__main__":
    main()
