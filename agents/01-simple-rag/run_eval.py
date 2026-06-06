"""
Driver: run the golden dataset through the agent and report aggregate scores.

By default it runs once with the current CHUNK_SIZE env var. Pass --sweep to
run all three chunk sizes (128 / 256 / 512) back-to-back and print a table
so the tuning experiment is just one command.

Usage:
    python run_eval.py                      # one run with current config
    python run_eval.py --sweep              # the chunk-size tuning experiment
    python run_eval.py --limit 5            # only first 5 dataset rows (smoke test)
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass

import agent  # local module
from evals import EvalResult, faithfulness, keyword_match, retrieval_at_k


@dataclass
class Row:
    question: str
    expected_source: str
    expected_keywords: list[str]
    difficulty: str
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            keywords = [k.strip() for k in r["expected_keywords"].split(",") if k.strip()]
            rows.append(
                Row(
                    question=r["question"],
                    expected_source=r["expected_source"],
                    expected_keywords=keywords,
                    difficulty=r["difficulty"],
                    notes=r["notes"],
                )
            )
    return rows


def run_one_config(chunk_size: int, dataset: list[Row], corpus: list[agent.Chunk]) -> dict:
    """Run the full dataset against one chunk_size. Returns aggregate stats."""
    per_eval: dict[str, list[EvalResult]] = defaultdict(list)
    per_row: list[dict] = []

    for row in dataset:
        result = agent.query(row.question, corpus, chunk_size=chunk_size)
        r1 = retrieval_at_k(result.retrieved, row.expected_source)
        r2 = keyword_match(result.answer, row.expected_keywords)
        r3 = faithfulness(row.question, result.retrieved, result.answer)
        for ev in (r1, r2, r3):
            per_eval[ev.name].append(ev)
        per_row.append(
            {
                "question": row.question,
                "retrieval_at_k": r1.passed,
                "keyword_match": round(r2.score, 2),
                "faithfulness": round(r3.score, 2),
            }
        )

    return {
        "chunk_size": chunk_size,
        "pass_rate": {name: sum(e.passed for e in evs) / len(evs) for name, evs in per_eval.items()},
        "avg_score": {name: sum(e.score for e in evs) / len(evs) for name, evs in per_eval.items()},
        "rows": per_row,
    }


def _print_summary(results: list[dict]) -> None:
    eval_names = ["retrieval_at_k", "keyword_match", "faithfulness"]
    print("\n=== Tuning experiment summary ===\n")
    print(f"{'chunk_size':>10} | " + " | ".join(f"{n:>18}" for n in eval_names))
    print("-" * (12 + sum(20 for _ in eval_names)))
    for r in results:
        cells = [f"{r['avg_score'][n]:.2f} ({r['pass_rate'][n] * 100:.0f}%)" for n in eval_names]
        print(f"{r['chunk_size']:>10} | " + " | ".join(f"{c:>18}" for c in cells))
    print("\n(score is mean across the dataset; % is pass rate)\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run chunk_size 128/256/512 back to back.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N dataset rows.")
    args = parser.parse_args()

    corpus = agent.load_corpus()
    dataset = load_dataset()
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"Loaded {len(corpus)} corpus entries and {len(dataset)} dataset rows.")

    if args.sweep:
        sizes = [128, 256, 512]
    else:
        sizes = [int(os.getenv("CHUNK_SIZE", "256"))]

    results = []
    for size in sizes:
        print(f"\n--- Running chunk_size={size} ---")
        results.append(run_one_config(size, dataset, corpus))

    _print_summary(results)


if __name__ == "__main__":
    main()
