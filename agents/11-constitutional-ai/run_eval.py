"""
Driver: run the dataset through the agent in both modes and print the
comparison, appropriateness by category (where over-refusal shows up), and the
paired direct -> constitutional revision lift.

Usage:
    python run_eval.py                    # direct vs constitutional, all rows
    python run_eval.py --mode constitutional
    python run_eval.py --limit 4          # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import appropriate_response, revision_lift, cost

MODES = ["direct", "constitutional"]


@dataclass
class Row:
    prompt: str
    category: str
    should_refuse: bool
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(
                prompt=r["prompt"], category=r["category"],
                should_refuse=r["should_refuse"].strip().lower() == "yes",
                notes=r.get("notes", "") or "",
            ))
    return rows


def run_one(row: Row, mode: str) -> dict:
    result = agent.run(row.prompt, mode=mode)
    appropriate = appropriate_response(row.prompt, result.final_response, row.should_refuse)
    return {
        "prompt": row.prompt, "category": row.category, "should_refuse": row.should_refuse,
        "mode": mode, "final_response": result.final_response,
        "revised": result.revised, "llm_calls": result.llm_calls,
        "appropriate": appropriate, "cost": cost(result.llm_calls),
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)

    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>15} | {'appropriate':>16} | {'mean llm_calls':>15}")
    print("-" * 54)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        c = sum(1 for r in rs if r["appropriate"].passed)
        mean_calls = sum(r["llm_calls"] for r in rs) / n
        print(f"{mode:>15} | {f'{c}/{n} ({c/n*100:.0f}%)':>16} | {mean_calls:>15.1f}")

    print("\n=== Appropriateness by category ===\n")
    cats = ["harmful", "benign_touchy", "neutral"]
    print(f"{'mode':>15} | " + " | ".join(f"{c:>14}" for c in cats))
    print("-" * (17 + 17 * len(cats)))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_cat[r["category"]].append(r)
        cells = []
        for cat in cats:
            crs = by_cat.get(cat, [])
            c = sum(1 for r in crs if r["appropriate"].passed)
            cells.append(f"{c}/{len(crs)}" if crs else "-")
        print(f"{mode:>15} | " + " | ".join(f"{c:>14}" for c in cells))


def _print_lift(results: list[dict]) -> None:
    by_prompt: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_prompt[r["prompt"]][r["mode"]] = r
    if not all("direct" in m and "constitutional" in m for m in by_prompt.values()):
        return

    print("\n=== Paired lift: direct -> constitutional ===\n")
    helped = hurt = neutral = 0
    for prompt, m in by_prompt.items():
        d, c = m["direct"], m["constitutional"]
        L = revision_lift(d["appropriate"].passed, c["appropriate"].passed)
        if L.score == 1.0:
            helped += 1; tag = "  HELPED"
        elif L.score == 0.0:
            hurt += 1; tag = "    HURT"
        else:
            neutral += 1; tag = ""
        if tag:
            print(f"  [{d['category']:>13}] direct={'OK ' if d['appropriate'].passed else 'BAD'} "
                  f"const={'OK ' if c['appropriate'].passed else 'BAD'}"
                  f"{' revised' if c['revised'] else ''}{tag}")
            print(f"               {prompt[:74]}")
    total = helped + hurt + neutral
    mean = (helped + neutral * 0.5) / total if total else 0.0
    print(f"\n  revision_lift mean = {mean:.2f}  (helped {helped}, hurt {hurt}, neutral {neutral})")

    # How often did the critique fire a revision, and on what?
    revised = [r for r in results if r["mode"] == "constitutional" and r["revised"]]
    by_cat: dict[str, int] = defaultdict(int)
    for r in revised:
        by_cat[r["category"]] += 1
    print(f"\n  revisions fired on {len(revised)} rows: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = MODES if args.mode == "all" else [args.mode]

    print(f"Loaded {len(rows)} prompts. Running {len(modes)} mode(s).")
    results: list[dict] = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  [{row.category:>13}] {row.prompt[:50]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    if args.mode == "all":
        _print_lift(results)


if __name__ == "__main__":
    main()
