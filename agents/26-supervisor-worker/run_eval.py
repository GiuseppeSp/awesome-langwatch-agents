"""
Driver: run every bundle through both modes and print the mode comparison, the
CROSSOVER table (correctness by bundle size — the headline), the drop/wrong
breakdown, the call-cost difference, and supervisor decomposition fidelity.

Usage:
    python run_eval.py                 # single vs supervisor_worker, all rows
    python run_eval.py --mode single
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import decomposition_ok, score_bundle

MODES = ["single", "supervisor_worker"]


@dataclass
class Row:
    bundle_size: int
    subtasks: list


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(int(r["bundle_size"]), json.loads(r["subtasks_json"])))
    return rows


def run_one(row: Row, mode: str) -> dict:
    instrs = [s["instr"] for s in row.subtasks]
    res = agent.run(instrs, mode=mode)
    sc = score_bundle(row.subtasks, res.answers)
    return {
        "bundle_size": row.bundle_size, "mode": mode, "subtasks": row.subtasks,
        "score": sc, "llm_calls": res.llm_calls, "n_decomposed": res.n_decomposed,
    }


def _agg(results, mode):
    rs = [r for r in results if r["mode"] == mode]
    tot = sum(r["score"]["n"] for r in rs)
    cor = sum(r["score"]["cells"]["correct"] for r in rs)
    wrong = sum(r["score"]["cells"]["wrong"] for r in rs)
    drop = sum(r["score"]["cells"]["dropped"] for r in rs)
    calls = sum(r["llm_calls"] for r in rs)
    return tot, cor, wrong, drop, calls, len(rs)


def _print_summary(results):
    print("\n=== Mode comparison (all subtasks) ===\n")
    print(f"{'mode':>18} | {'subtasks correct':>17} | {'wrong':>6} | {'dropped':>8} | {'mean llm calls':>15}")
    print("-" * 78)
    for mode in MODES:
        tot, cor, wrong, drop, calls, nrows = _agg(results, mode)
        if not tot:
            continue
        print(f"{mode:>18} | {f'{cor}/{tot} ({cor/tot*100:.0f}%)':>17} | {wrong:>6} | "
              f"{drop:>8} | {calls/nrows:>15.1f}")


def _print_crossover(results):
    print("\n=== Crossover: subtask correctness by bundle size ===\n")
    sizes = sorted({r["bundle_size"] for r in results})
    print(f"{'bundle size':>12} | " + " | ".join(f"{m:>18}" for m in MODES))
    print("-" * 56)
    for k in sizes:
        cells = []
        for mode in MODES:
            rs = [r for r in results if r["mode"] == mode and r["bundle_size"] == k]
            tot = sum(r["score"]["n"] for r in rs)
            cor = sum(r["score"]["cells"]["correct"] for r in rs)
            cells.append(f"{cor}/{tot} ({cor/tot*100:.0f}%)" if tot else "-")
        print(f"{k:>12} | " + " | ".join(f"{c:>18}" for c in cells))


def _print_failures(results):
    print("\n=== Where single dropped or fumbled (per subtask) ===\n")
    for mode in MODES:
        rs = [r for r in results if r["mode"] == mode]
        bad = []
        for r in rs:
            for sub, ans, outcome in r["score"]["per"]:
                if outcome != "correct":
                    bad.append((r["bundle_size"], sub["type"], outcome, sub["gold"], ans))
        if not bad:
            print(f"  {mode}: clean — every subtask correct")
            continue
        print(f"  {mode}: {len(bad)} non-correct subtasks")
        for k, typ, outcome, gold, ans in bad:
            shown = "(dropped)" if outcome == "dropped" else repr((ans or "")[:24])
            print(f"    [size {k}] {typ:<9} {outcome:<8} gold={gold!r} got={shown}")


def _print_decomposition(results):
    rs = [r for r in results if r["mode"] == "supervisor_worker"]
    if not rs:
        return
    ok = sum(1 for r in rs if decomposition_ok(r["n_decomposed"], r["score"]["n"]).passed)
    print("\n=== Supervisor decomposition fidelity ===\n")
    print(f"  exact-N decompositions: {ok}/{len(rs)}")
    for r in rs:
        d = decomposition_ok(r["n_decomposed"], r["score"]["n"])
        if not d.passed:
            print(f"    [size {r['score']['n']}] {d.reason}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = MODES if args.mode == "all" else [args.mode]

    print(f"Loaded {len(rows)} bundles. Running {len(modes)} mode(s).")
    results = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  [size {row.bundle_size}] bundle of {row.bundle_size} tasks...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    if args.mode == "all":
        _print_crossover(results)
    _print_failures(results)
    _print_decomposition(results)


if __name__ == "__main__":
    main()
