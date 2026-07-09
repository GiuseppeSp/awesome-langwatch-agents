"""
Driver: run each question through single and debate, DERIVE self-consistency from the
debate run's round 1 (so SC and debate share the same samples — a paired comparison),
and print the three-way accuracy, the paired debate delta (conformity vs correction),
the split by whether round-1 agents already agreed, the agent-level flips, and cost.

Usage:
    python run_eval.py                 # single + debate (SC derived), all rows
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import agent_flips, correct, debate_delta, initial_agreement


@dataclass
class Row:
    question: str
    gold: str
    category: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    with open(path, encoding="utf-8") as f:
        return [Row(r["question"], r["gold"], r["category"]) for r in csv.DictReader(f)]


def run_one(row: Row) -> dict:
    single = agent.run(row.question, mode="single")
    debate = agent.run(row.question, mode="debate")
    return {
        "row": row,
        "single": single.single_answer,
        "sc": debate.sc_answer,
        "debate": debate.debate_answer,
        "rounds": debate.rounds,
        "single_calls": single.llm_calls,
        "debate_calls": debate.llm_calls,
        "agreed": initial_agreement(debate.rounds),
    }


def _acc(results, key):
    c = sum(1 for r in results if correct(r[key], r["row"].gold).passed)
    return c, len(results)


def _print_summary(results):
    n = len(results)
    sc_calls = agent.N_AGENTS  # round 1 of the debate run
    print("\n=== Three-way accuracy ===\n")
    print(f"{'mode':>18} | {'correct':>13} | {'mean llm calls':>15}")
    print("-" * 52)
    for label, key, calls in [
        ("single", "single", sum(r["single_calls"] for r in results) / n),
        ("self_consistency", "sc", float(sc_calls)),
        ("debate", "debate", sum(r["debate_calls"] for r in results) / n),
    ]:
        c, tot = _acc(results, key)
        print(f"{label:>18} | {f'{c}/{tot} ({c/tot*100:.0f}%)':>13} | {calls:>15.1f}")


def _print_delta(results):
    print("\n=== Paired SC -> debate (what the arguing changed) ===\n")
    harm = corr = same = 0
    for r in results:
        d = debate_delta(r["sc"], r["debate"], r["row"].gold)
        if d.reason.startswith("CONFORMITY"):
            harm += 1
            print(f"  HARM  [{r['row'].category:>4}] SC={r['sc']} -> debate={r['debate']} "
                  f"(gold={r['row'].gold})  {r['row'].question[:46]}")
        elif d.reason.startswith("CORRECTION"):
            corr += 1
            print(f"  FIX   [{r['row'].category:>4}] SC={r['sc']} -> debate={r['debate']} "
                  f"(gold={r['row'].gold})  {r['row'].question[:46]}")
        else:
            same += 1
    print(f"\n  corrections (wrong->right): {corr}   conformity harm (right->wrong): {harm}   "
          f"unchanged: {same}")
    print(f"  net debate effect vs self-consistency: {corr - harm:+d} questions")


def _print_agreement(results):
    print("\n=== Does debate only matter where round-1 agents disagreed? ===\n")
    for agreed in (True, False):
        rs = [r for r in results if r["agreed"] == agreed]
        if not rs:
            continue
        sc_c, _ = _acc(rs, "sc")
        db_c, _ = _acc(rs, "debate")
        tag = "already AGREED at round 1" if agreed else "DISAGREED at round 1"
        print(f"  {tag} ({len(rs)} q): self_consistency {sc_c}/{len(rs)} -> debate {db_c}/{len(rs)}")


def _print_flips(results):
    r2w = w2r = 0
    for r in results:
        f = agent_flips(r["rounds"], r["row"].gold)
        r2w += f["right_to_wrong"]
        w2r += f["wrong_to_right"]
    print("\n=== Agent-level flips across rounds (individual agents) ===\n")
    print(f"  wrong -> right: {w2r}    right -> wrong (conformity): {r2w}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loaded {len(rows)} questions. Running single + debate (SC derived from debate round 1).")
    results = []
    for row in rows:
        print(f"  [{row.category:>4}] {row.question[:50]}...", flush=True)
        results.append(run_one(row))

    _print_summary(results)
    _print_delta(results)
    _print_agreement(results)
    _print_flips(results)


if __name__ == "__main__":
    main()
