"""
Driver: run the bios through both modes and print the mode comparison, accuracy
by category (clean / tricky / incomplete), and the fabrication tally on ABSENT
fields (the schema's slot-filling failure mode).

Usage:
    python run_eval.py                 # freeform vs schema, all rows
    python run_eval.py --mode schema
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import parse_success, record_accuracy, field_accuracy, fabrication_count, FIELDS

MODES = ["freeform", "schema"]
CATS = ["clean", "tricky", "incomplete"]


@dataclass
class Row:
    bio: str
    category: str
    gold: dict


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(bio=r["bio"], category=r["category"], gold={
                "name": r["gold_name"], "role": r["gold_role"],
                "city": r["gold_city"], "years_experience": r["gold_years"],
            }))
    return rows


def run_one(row: Row, mode: str) -> dict:
    res = agent.run(row.bio, mode=mode)
    return {
        "bio": row.bio, "category": row.category, "gold": row.gold, "mode": mode,
        "record": res.record,
        "parse": parse_success(res.parse_ok),
        "record_acc": record_accuracy(res.record, row.gold),
        "field_acc": field_accuracy(res.record, row.gold),
        "fabricated": fabrication_count(res.record, row.gold),
        "absent_fields": sum(1 for k in FIELDS if row.gold[k] == "ABSENT"),
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>9} | {'parse ok':>13} | {'record correct':>15} | {'field acc':>10}")
    print("-" * 56)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        p = sum(1 for r in rs if r["parse"].passed)
        rc = sum(1 for r in rs if r["record_acc"].passed)
        fa = sum(r["field_acc"].score for r in rs) / n
        print(f"{mode:>9} | {f'{p}/{n} ({p/n*100:.0f}%)':>13} | {f'{rc}/{n} ({rc/n*100:.0f}%)':>15} | {fa*100:>9.0f}%")


def _print_by_category(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Record-correct by category ===\n")
    print(f"{'mode':>9} | " + " | ".join(f"{c:>11}" for c in CATS))
    print("-" * (11 + 14 * len(CATS)))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        cells = []
        for cat in CATS:
            crs = [r for r in rs if r["category"] == cat]
            c = sum(1 for r in crs if r["record_acc"].passed)
            cells.append(f"{c}/{len(crs)}")
        print(f"{mode:>9} | " + " | ".join(f"{c:>11}" for c in cells))


def _print_fabrication(results: list[dict]) -> None:
    print("\n=== Fabrication on ABSENT fields (incomplete rows) ===\n")
    total_absent = sum(r["absent_fields"] for r in results if r["mode"] == MODES[0])
    for mode in MODES:
        rs = [r for r in results if r["mode"] == mode]
        fab = sum(r["fabricated"] for r in rs)
        print(f"  {mode:>9}: fabricated {fab}/{total_absent} absent fields")
        for r in rs:
            if r["fabricated"]:
                made = {k: r["record"].get(k) for k in FIELDS if r["gold"][k] == "ABSENT" and r["record"].get(k) is not None}
                print(f"      [{r['category']}] {r['bio'][:46]} -> fabricated {made}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = MODES if args.mode == "all" else [args.mode]

    print(f"Loaded {len(rows)} bios. Running {len(modes)} mode(s).")
    results: list[dict] = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  [{row.category:>10}] {row.bio[:44]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_by_category(results)
    _print_fabrication(results)


if __name__ == "__main__":
    main()
