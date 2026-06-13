"""
Driver: run the dataset through the plan-and-execute agent in both modes and
print a comparison plus the static-vs-replan pairing that the whole experiment
turns on.

Usage:
    python run_eval.py                  # static vs replan, all rows
    python run_eval.py --mode replan    # one mode only
    python run_eval.py --limit 3        # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import answer_correctness, adaptation_lift, step_efficiency


@dataclass
class Row:
    prompt: str
    kind: str
    answer: float
    tolerance: float
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                Row(
                    prompt=r["prompt"],
                    kind=r["kind"],
                    answer=float(r["answer"]),
                    tolerance=float(r["tolerance"]),
                    notes=r.get("notes", "") or "",
                )
            )
    return rows


def run_one(row: Row, mode: str) -> dict:
    result = agent.run(row.prompt, mode=mode)
    correctness = answer_correctness(result.final_answer, row.answer, row.tolerance)
    efficiency = step_efficiency(result.llm_calls, result.num_replans)
    return {
        "prompt": row.prompt,
        "kind": row.kind,
        "mode": mode,
        "final_answer": result.final_answer,
        "num_steps": result.num_steps,
        "num_replans": result.num_replans,
        "llm_calls": result.llm_calls,
        "correctness": correctness,
        "efficiency": efficiency,
    }


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)

    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>10} | {'answer_correctness':>22} | {'mean llm_calls':>15} | {'mean replans':>12}")
    print("-" * 70)
    for mode, rs in by_mode.items():
        n = len(rs)
        correct = sum(1 for r in rs if r["correctness"].passed)
        mean_calls = sum(r["llm_calls"] for r in rs) / n
        mean_replans = sum(r["num_replans"] for r in rs) / n
        print(f"{mode:>10} | {f'{correct}/{n} ({correct/n*100:.0f}%)':>22} | "
              f"{mean_calls:>15.1f} | {mean_replans:>12.2f}")

    # Correctness split by row kind — where does each mode win or lose?
    print("\n=== Correctness by row kind ===\n")
    print(f"{'mode':>10} | {'kind':>8} | {'correct':>10}")
    print("-" * 36)
    for mode, rs in by_mode.items():
        by_kind: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_kind[r["kind"]].append(r)
        for kind, krs in sorted(by_kind.items()):
            c = sum(1 for r in krs if r["correctness"].passed)
            print(f"{mode:>10} | {kind:>8} | {f'{c}/{len(krs)}':>10}")


def _print_pairing(results: list[dict]) -> None:
    """The heart of the experiment: per-row static-vs-replan diff."""
    by_prompt: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_prompt[r["prompt"]][r["mode"]] = r

    have_both = all("static" in m and "replan" in m for m in by_prompt.values())
    if not have_both:
        return

    print("\n=== Static vs replan pairing (adaptation_lift) ===\n")
    lifts = []
    for prompt, modes in by_prompt.items():
        s, rp = modes["static"], modes["replan"]
        lift = adaptation_lift(s["correctness"].passed, rp["correctness"].passed)
        lifts.append(lift.score)
        tag = {1.0: "  HELPED", 0.0: "    HURT", 0.5: "    --"}[lift.score]
        print(f"  [{s['kind']:>6}] static={'OK ' if s['correctness'].passed else 'BAD'} "
              f"replan={'OK ' if rp['correctness'].passed else 'BAD'} "
              f"(static {s['llm_calls']} calls, replan {rp['llm_calls']} calls){tag}")
        print(f"           {prompt[:80]}")
    mean_lift = sum(lifts) / len(lifts)
    helped = sum(1 for l in lifts if l == 1.0)
    hurt = sum(1 for l in lifts if l == 0.0)
    print(f"\n  adaptation_lift mean = {mean_lift:.2f}  "
          f"(helped {helped}, hurt {hurt}, neutral {len(lifts) - helped - hurt})")
    print("  >0.5 means replanning was net-positive; <0.5 means it did net harm.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "replan", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Only first N rows.")
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = ["static", "replan"] if args.mode == "both" else [args.mode]

    print(f"Loaded {len(rows)} rows. Running {len(modes)} mode(s) = "
          f"{len(rows) * len(modes)} agent calls.")

    results: list[dict] = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  [{row.kind:>6}] {row.prompt[:55]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    if len(modes) == 2:
        _print_pairing(results)


if __name__ == "__main__":
    main()
