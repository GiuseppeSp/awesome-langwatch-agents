"""
Driver: run the questions through both modes and print the mode comparison,
accuracy + execution success by tier (easy / join / aggregate), and the paired
blind -> grounded lift.

Usage:
    python run_eval.py                 # blind vs grounded, all rows
    python run_eval.py --mode grounded
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import executes_ok, answer_correct, blind_to_grounded_lift

MODES = ["blind", "grounded"]
TIERS = ["easy", "join", "aggregate"]


@dataclass
class Row:
    question: str
    tier: str
    gold_sql: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(question=r["question"], tier=r["tier"], gold_sql=r["gold_sql"]))
    return rows


def run_one(row: Row, mode: str) -> dict:
    gold_rows, gold_err = agent.run_sql(row.gold_sql)
    assert gold_err is None, f"gold SQL failed for {row.question!r}: {gold_err}"
    res = agent.run(row.question, mode=mode)
    return {
        "question": row.question, "tier": row.tier, "mode": mode,
        "sql": res.sql, "rows": res.rows, "error": res.error,
        "exec": executes_ok(res.error),
        "correct": answer_correct(res.rows, gold_rows),
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>9} | {'executes ok':>14} | {'answer correct':>15}")
    print("-" * 46)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        e = sum(1 for r in rs if r["exec"].passed)
        c = sum(1 for r in rs if r["correct"].passed)
        print(f"{mode:>9} | {f'{e}/{n} ({e/n*100:.0f}%)':>14} | {f'{c}/{n} ({c/n*100:.0f}%)':>15}")


def _print_by_tier(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Answer-correct by tier ===\n")
    print(f"{'mode':>9} | " + " | ".join(f"{t:>10}" for t in TIERS))
    print("-" * (11 + 13 * len(TIERS)))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        cells = []
        for tier in TIERS:
            trs = [r for r in rs if r["tier"] == tier]
            c = sum(1 for r in trs if r["correct"].passed)
            cells.append(f"{c}/{len(trs)}")
        print(f"{mode:>9} | " + " | ".join(f"{c:>10}" for c in cells))


def _print_lift(results: list[dict]) -> None:
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["question"]][r["mode"]] = r
    if not all("blind" in m and "grounded" in m for m in by_q.values()):
        return
    print("\n=== Paired lift: blind -> grounded ===\n")
    helped = hurt = neutral = 0
    for q, m in by_q.items():
        b, g = m["blind"], m["grounded"]
        L = blind_to_grounded_lift(b["correct"].passed, g["correct"].passed)
        if L.score == 1.0:
            helped += 1; tag = "  HELPED"
        elif L.score == 0.0:
            hurt += 1; tag = "    HURT"
        else:
            neutral += 1; tag = ""
        if tag:
            b_why = "ok" if b["exec"].passed else f"err:{(b['error'] or '')[:30]}"
            print(f"  [{b['tier']:>9}] blind={'OK ' if b['correct'].passed else 'BAD'}({b_why}) "
                  f"grounded={'OK ' if g['correct'].passed else 'BAD'}{tag}")
            print(f"             {q[:64]}")
    total = helped + hurt + neutral
    mean = (helped + neutral * 0.5) / total if total else 0.0
    print(f"\n  blind_to_grounded_lift mean = {mean:.2f}  (helped {helped}, hurt {hurt}, neutral {neutral})")
    # how many blind failures were execution errors vs wrong answers?
    blind = [r for r in results if r["mode"] == "blind"]
    exec_fail = sum(1 for r in blind if not r["exec"].passed)
    print(f"  blind execution failures (hallucinated schema): {exec_fail}/{len(blind)}")


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
            print(f"  [{row.tier:>9}] {row.question[:46]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_by_tier(results)
    if args.mode == "all":
        _print_lift(results)


if __name__ == "__main__":
    main()
