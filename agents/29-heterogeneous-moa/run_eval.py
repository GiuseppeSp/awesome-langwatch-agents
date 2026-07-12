"""
Driver: run each question through single (gpt-4o-mini AND gpt-4o), homogeneous MoA
(3x gpt-4o-mini), and heterogeneous MoA (mini + gpt-3.5-turbo + gpt-4o), all with the
same aggregator (gpt-4o-mini). Derive the union of each MoA's drafts and print:

  - recall / precision / F1 for every config
  - INDEPENDENCE: of the gold items gpt-4o-mini's draft missed, how many did the OTHER
    drafters recover — same-model (homo) vs different-model (hetero)? The mechanism test.
  - SYNERGY: does heterogeneous MoA beat the BEST individual model? (else the "win" is
    just gpt-4o carrying it, not diversity.)
  - cost.

Usage:
    python run_eval.py
    python run_eval.py --limit 4
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass

import agent
from evals import canon, dedup, score_set

MINI, GPT4O, GPT35 = "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"


@dataclass
class Row:
    question: str
    gold: list
    size: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    with open(path, encoding="utf-8") as f:
        return [Row(r["question"], json.loads(r["gold_json"]), r["size"]) for r in csv.DictReader(f)]


def run_one(row: Row) -> dict:
    single_mini = agent.run(row.question, mode="single", models=[MINI], label=MINI)
    single_4o = agent.run(row.question, mode="single", models=[GPT4O], label=GPT4O)
    homo = agent.run(row.question, mode="moa", models=[MINI, MINI, MINI], label="homogeneous")
    hetero = agent.run(row.question, mode="moa", models=[MINI, GPT35, GPT4O], label="heterogeneous")

    def union(res):
        return dedup([it for d in res.draft_items for it in d])

    return {
        "row": row,
        "single_mini": single_mini.items, "single_4o": single_4o.items,
        "homo_union": union(homo), "homo_moa": homo.items,
        "hetero_union": union(hetero), "hetero_moa": hetero.items,
        # first draft is always gpt-4o-mini in both; the "others" differ (mini,mini vs 3.5,4o)
        "mini_draft": homo.draft_items[0],
        "homo_others": homo.draft_items[1:], "hetero_others": hetero.draft_items[1:],
        "calls": single_mini.llm_calls + single_4o.llm_calls + homo.llm_calls + hetero.llm_calls,
    }


CONFIGS = [
    ("single gpt-4o-mini", "single_mini", 1),
    ("single gpt-4o", "single_4o", 1),
    ("homo union", "homo_union", 3),
    ("homo moa", "homo_moa", 4),
    ("hetero union", "hetero_union", 3),
    ("hetero moa", "hetero_moa", 4),
]


def _mean(results, key, metric):
    xs = [score_set(r[key], r["row"].gold)[metric] for r in results]
    return sum(xs) / len(xs) if xs else 0.0


def _print_summary(results):
    print("\n=== Configs: recall / precision / F1 (mean over questions) ===\n")
    print(f"{'config':>20} | {'recall':>7} | {'precision':>9} | {'F1':>6} | {'calls':>5}")
    print("-" * 60)
    for name, key, calls in CONFIGS:
        print(f"{name:>20} | {_mean(results,key,'recall'):>7.2f} | {_mean(results,key,'precision'):>9.2f} "
              f"| {_mean(results,key,'f1'):>6.2f} | {calls:>5}")


def _print_independence(results):
    """Of the gold items gpt-4o-mini's draft missed, what fraction did the OTHER
    drafters recover — same-model (homo) vs different-model (hetero)?"""
    homo_rec = hetero_rec = 0
    total_missed = 0
    for r in results:
        gold = {canon(g) for g in r["row"].gold}
        mini_got = {canon(x) for x in r["mini_draft"]}
        missed = gold - mini_got
        if not missed:
            continue
        total_missed += len(missed)
        homo_others = {canon(x) for d in r["homo_others"] for x in d}
        hetero_others = {canon(x) for d in r["hetero_others"] for x in d}
        homo_rec += len(missed & homo_others)
        hetero_rec += len(missed & hetero_others)
    print("\n=== INDEPENDENCE: recovering gpt-4o-mini's misses ===\n")
    print(f"  gold items gpt-4o-mini's draft missed (total across questions): {total_missed}")
    if total_missed:
        print(f"  recovered by the OTHER drafts...")
        print(f"    homogeneous (2 more gpt-4o-mini):        {homo_rec}/{total_missed} ({homo_rec/total_missed*100:.0f}%)")
        print(f"    heterogeneous (gpt-3.5-turbo + gpt-4o):  {hetero_rec}/{total_missed} ({hetero_rec/total_missed*100:.0f}%)")


def _print_synergy(results):
    print("\n=== SYNERGY: does heterogeneous MoA beat the best individual model? ===\n")
    best_ind_f1 = max(_mean(results, "single_mini", "f1"), _mean(results, "single_4o", "f1"))
    hetero_f1 = _mean(results, "hetero_moa", "f1")
    print(f"  best individual F1 (max of mini / gpt-4o): {best_ind_f1:.3f}")
    print(f"  heterogeneous MoA F1:                      {hetero_f1:.3f}")
    print(f"  delta (hetero_moa - best individual):      {hetero_f1 - best_ind_f1:+.3f}")
    wins = sum(1 for r in results
               if score_set(r["hetero_moa"], r["row"].gold)["f1"]
               > max(score_set(r["single_mini"], r["row"].gold)["f1"],
                     score_set(r["single_4o"], r["row"].gold)["f1"]) + 1e-9)
    print(f"  questions where hetero_moa > best individual: {wins}/{len(results)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loaded {len(rows)} questions. single(mini,4o) + homo + hetero per question.")
    results = []
    for row in rows:
        print(f"  [{row.size:>6}] {row.question[:48]}...", flush=True)
        results.append(run_one(row))

    _print_summary(results)
    _print_independence(results)
    _print_synergy(results)


if __name__ == "__main__":
    main()
