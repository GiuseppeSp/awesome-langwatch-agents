"""
Driver: run every request through both modes and print the mode comparison, the
escalation confusion matrix (precision / recall — recall is the ceiling), the human
burden, the count of dangerous silent errors (unescalated wrong RISKY actions), and
the paired autonomous -> hitl lift.

Usage:
    python run_eval.py                 # autonomous vs hitl, all rows
    python run_eval.py --mode hitl
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from actions import action_risk
from evals import autonomous_to_hitl_lift, correct_action, escalation

MODES = ["autonomous", "hitl"]


@dataclass
class Row:
    request: str
    gold_action: str
    risk: str
    cluster: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(r["request"], r["gold_action"], r["risk"], r["cluster"]))
    return rows


def run_one(row: Row, mode: str) -> dict:
    res = agent.run(row.request, mode=mode, gold=row.gold_action)
    raw_correct = res.proposed_action.strip() == row.gold_action.strip()
    d = {
        "request": row.request, "gold": row.gold_action, "risk": row.risk, "cluster": row.cluster,
        "mode": mode, "proposed": res.proposed_action, "final": res.final_action,
        "confidence": res.confidence, "escalated": res.escalated, "raw_correct": raw_correct,
        "correct": correct_action(res.final_action, row.gold_action),
    }
    if mode == "hitl":
        d["escalation"] = escalation(res.escalated, raw_correct)
    return d


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>11} | {'correct action':>15} | {'% escalated':>11}")
    print("-" * 45)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        c = sum(1 for r in rs if r["correct"].passed)
        esc = sum(1 for r in rs if r["escalated"]) / n * 100 if mode == "hitl" else 0.0
        esc_str = f"{esc:.0f}%" if mode == "hitl" else "—"
        print(f"{mode:>11} | {f'{c}/{n} ({c/n*100:.0f}%)':>15} | {esc_str:>11}")


def _print_escalation(results: list[dict]) -> None:
    hh = [r for r in results if r["mode"] == "hitl" and "escalation" in r]
    if not hh:
        return
    cells = defaultdict(int)
    for r in hh:
        cells[r["escalation"].cell] += 1
    tp, fp, fn, tn = cells["TP"], cells["FP"], cells["FN"], cells["TN"]
    n = len(hh)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    burden = (tp + fp) / n

    print("\n=== Escalation calibration (hitl mode) ===\n")
    print("                        agent WRONG   agent RIGHT")
    print(f"  escalated to human      TP={tp:<7}    FP={fp}")
    print(f"  proceeded alone         FN={fn:<7}    TN={tn}")
    print()
    print(f"  escalation_precision (escalated & wrong / all escalated): {precision:.2f}")
    print(f"  escalation_recall    (escalated & wrong / all wrong):     {recall:.2f}  <-- HITL's ceiling")
    print(f"  human_burden         (% of actions sent to a human):      {burden*100:.0f}%")

    silent = [r for r in hh if r["escalation"].cell == "FN"]
    if silent:
        dangerous = [r for r in silent if action_risk(r["proposed"]) == "risky"]
        print(f"\n  SILENT ERRORS (wrong + NOT escalated): {len(silent)}"
              f"  — of which irreversible/RISKY: {len(dangerous)}")
        for r in silent:
            mark = "  ** RISKY **" if action_risk(r["proposed"]) == "risky" else ""
            print(f"    [{r['cluster']:>8}] conf={r['confidence']:.2f} "
                  f"did={r['proposed']} (gold={r['gold']}){mark}")
            print(f"             {r['request'][:64]}")
    false_alarms = [r for r in hh if r["escalation"].cell == "FP"]
    if false_alarms:
        print(f"\n  FALSE ALARMS (right, but escalated anyway): {len(false_alarms)}")
        for r in false_alarms:
            print(f"    [{r['cluster']:>8}] conf={r['confidence']:.2f} {r['request'][:60]}")


def _print_lift(results: list[dict]) -> None:
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["request"]][r["mode"]] = r
    if not all("autonomous" in m and "hitl" in m for m in by_q.values()):
        return
    print("\n=== Paired lift: autonomous -> hitl ===\n")
    helped = hurt = neutral = 0
    for q, m in by_q.items():
        a, h = m["autonomous"], m["hitl"]
        L = autonomous_to_hitl_lift(a["correct"].passed, h["correct"].passed)
        if L.score == 1.0 and not (a["correct"].passed and h["correct"].passed):
            helped += 1; tag = "  HELPED"
        elif L.score == 0.0:
            hurt += 1; tag = "    HURT"
        else:
            neutral += 1; tag = ""
        if tag:
            print(f"  [{a['cluster']:>8}] auto={'OK ' if a['correct'].passed else 'BAD'}({a['proposed']}) "
                  f"hitl={'OK ' if h['correct'].passed else 'BAD'}({h['final']}) gold={a['gold']}{tag}")
            print(f"             {q[:60]}")
    total = helped + hurt + neutral
    mean = (helped + neutral * 0.5) / total if total else 0.0
    print(f"\n  autonomous_to_hitl_lift mean = {mean:.2f}  (helped {helped}, hurt {hurt}, neutral {neutral})")
    print("  (HITL's accuracy gain = exactly the wrong actions it escalated; the silent")
    print("   errors above are the ones it did NOT, and its ceiling.)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = MODES if args.mode == "all" else [args.mode]

    print(f"Loaded {len(rows)} requests. Running {len(modes)} mode(s).")
    results: list[dict] = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  [{row.cluster:>8}] {row.request[:48]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_escalation(results)
    if args.mode == "all":
        _print_lift(results)


if __name__ == "__main__":
    main()
