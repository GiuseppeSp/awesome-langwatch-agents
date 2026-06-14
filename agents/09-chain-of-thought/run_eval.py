"""
Driver: run the dataset through the chain-of-thought agent in all three modes
and print the comparison plus the two paired lifts the experiment turns on:
direct -> cot (is reasoning worth it?) and cot -> self_consistency (is sampling
worth it?).

Usage:
    python run_eval.py                       # all three modes, all rows
    python run_eval.py --mode cot            # one mode only
    python run_eval.py --limit 3             # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import answer_correctness, lift, sample_efficiency

MODES = ["direct", "cot", "self_consistency"]


@dataclass
class Row:
    prompt: str
    band: str
    answer: str
    answer_type: str
    tolerance: float
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(
                prompt=r["prompt"], band=r["band"], answer=r["answer"],
                answer_type=r["answer_type"], tolerance=float(r["tolerance"] or 0),
                notes=r.get("notes", "") or "",
            ))
    return rows


def run_one(row: Row, mode: str) -> dict:
    result = agent.run(row.prompt, mode=mode)
    correctness = answer_correctness(result.final_answer, row.answer,
                                     row.answer_type, row.tolerance)
    return {
        "prompt": row.prompt, "band": row.band, "mode": mode,
        "final_answer": result.final_answer, "llm_calls": result.llm_calls,
        "vote_agreement": result.vote_agreement if mode == "self_consistency" else None,
        "correctness": correctness,
        "efficiency": sample_efficiency(result.llm_calls),
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)

    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>17} | {'answer_correctness':>20} | {'mean llm_calls':>15}")
    print("-" * 60)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        c = sum(1 for r in rs if r["correctness"].passed)
        mean_calls = sum(r["llm_calls"] for r in rs) / n
        print(f"{mode:>17} | {f'{c}/{n} ({c/n*100:.0f}%)':>20} | {mean_calls:>15.1f}")

    print("\n=== Correctness by band ===\n")
    bands = ["easy", "multistep", "logic", "trick"]
    print(f"{'mode':>17} | " + " | ".join(f"{b:>10}" for b in bands))
    print("-" * (19 + 13 * len(bands)))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        by_band: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_band[r["band"]].append(r)
        cells = []
        for b in bands:
            brs = by_band.get(b, [])
            c = sum(1 for r in brs if r["correctness"].passed)
            cells.append(f"{c}/{len(brs)}" if brs else "-")
        print(f"{mode:>17} | " + " | ".join(f"{c:>10}" for c in cells))

    # Self-consistency only makes a difference where the samples DISAGREE.
    sc = by_mode.get("self_consistency", [])
    agreements = [r["vote_agreement"] for r in sc if r["vote_agreement"] is not None]
    if agreements:
        split = sum(1 for a in agreements if a < 1.0)
        print("\n=== Self-consistency vote agreement ===\n")
        print(f"  mean agreement across {len(agreements)} rows = "
              f"{sum(agreements) / len(agreements):.0%}")
        print(f"  rows where the 5 samples were NOT unanimous (vote had a job) = "
              f"{split}/{len(agreements)}")


def _print_lift(results: list[dict], baseline: str, treatment: str) -> None:
    by_prompt: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_prompt[r["prompt"]][r["mode"]] = r
    if not all(baseline in m and treatment in m for m in by_prompt.values()):
        return

    print(f"\n=== Paired lift: {baseline} -> {treatment} ===\n")
    helped = hurt = neutral = 0
    for prompt, modes in by_prompt.items():
        b, t = modes[baseline], modes[treatment]
        L = lift(b["correctness"].passed, t["correctness"].passed)
        if L.score == 1.0:
            helped += 1; tag = "  HELPED"
        elif L.score == 0.0:
            hurt += 1; tag = "    HURT"
        else:
            neutral += 1; tag = ""
        if tag:
            agree = t.get("vote_agreement")
            extra = f"  [sc agreement {agree:.0%}]" if agree is not None else ""
            print(f"  [{b['band']:>9}] {baseline}={'OK ' if b['correctness'].passed else 'BAD'} "
                  f"{treatment}={'OK ' if t['correctness'].passed else 'BAD'}{tag}{extra}")
            print(f"             {prompt[:78]}")
    total = helped + hurt + neutral
    mean = (helped * 1.0 + neutral * 0.5) / total if total else 0.0
    print(f"\n  lift mean = {mean:.2f}  (helped {helped}, hurt {hurt}, neutral {neutral})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = MODES if args.mode == "all" else [args.mode]

    total_calls = sum(1 if m != "self_consistency" else agent.SC_SAMPLES for m in modes)
    print(f"Loaded {len(rows)} rows. Running {len(modes)} mode(s) "
          f"≈ {len(rows) * total_calls} llm calls.")

    results: list[dict] = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  [{row.band:>9}] {row.prompt[:50]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    if args.mode == "all":
        _print_lift(results, "direct", "cot")
        _print_lift(results, "cot", "self_consistency")


if __name__ == "__main__":
    main()
