"""
Driver: run each conversation through the agent with each memory strategy and
print a comparison table.

Usage:
    python run_eval.py                       # window vs summary, all conversations
    python run_eval.py --strategy window     # just one strategy
    python run_eval.py --conversation tokyo-vegan   # one specific conversation
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import agent  # local module
from evals import EvalResult, context_recall, must_include, must_not_include


def load_conversations(path: str = "conversations.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_one(convo: dict, strategy: str) -> dict:
    """Run one conversation through the agent with a strategy. Score it."""
    answer = agent.chat(
        convo["setup"],
        convo["test_question"],
        strategy=strategy,
        thread_id=f"{convo['id']}-{strategy}",
    )
    r_include = must_include(answer, convo.get("must_include_any", []))
    r_forbid = must_not_include(answer, convo.get("must_not_include", []))
    r_recall = context_recall(convo["test_question"], answer, convo["expected_constraint"])
    return {
        "convo_id": convo["id"],
        "strategy": strategy,
        "answer": answer,
        "evals": {
            "context_recall": r_recall,
            "must_include": r_include,
            "must_not_include": r_forbid,
        },
    }


def aggregate(results: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate scores per strategy across the dataset."""
    per_strategy: dict[str, dict[str, list[EvalResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        for name, ev in r["evals"].items():
            per_strategy[r["strategy"]][name].append(ev)

    summary: dict[str, dict[str, float]] = {}
    for strategy, by_eval in per_strategy.items():
        summary[strategy] = {}
        for eval_name, evs in by_eval.items():
            avg_score = sum(e.score for e in evs) / len(evs)
            pass_rate = sum(1 for e in evs if e.passed) / len(evs)
            summary[strategy][eval_name] = (avg_score, pass_rate)
    return summary


def _print_summary(summary: dict[str, dict[str, float]]) -> None:
    eval_names = ["context_recall", "must_include", "must_not_include"]
    print("\n=== Memory strategy comparison ===\n")
    print(f"{'strategy':>10} | " + " | ".join(f"{n:>18}" for n in eval_names))
    print("-" * (12 + sum(20 for _ in eval_names)))
    for strategy, by_eval in summary.items():
        cells = []
        for n in eval_names:
            score, rate = by_eval.get(n, (0.0, 0.0))
            cells.append(f"{score:.2f} ({rate * 100:.0f}%)")
        print(f"{strategy:>10} | " + " | ".join(f"{c:>18}" for c in cells))
    print("\n(score is mean across the dataset; % is pass rate)\n")


def _print_per_row(results: list[dict]) -> None:
    print("=== Per-conversation detail ===\n")
    by_convo: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_convo[r["convo_id"]][r["strategy"]] = r

    for convo_id, strats in by_convo.items():
        print(f"  {convo_id}:")
        for strategy, r in strats.items():
            recall = r["evals"]["context_recall"]
            inc = r["evals"]["must_include"]
            forbid = r["evals"]["must_not_include"]
            print(
                f"    [{strategy:>7}] recall={recall.score:.2f} "
                f"({'✓' if recall.passed else '✗'})  "
                f"include={'✓' if inc.passed else '✗'}  "
                f"avoid={'✓' if forbid.passed else '✗'}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["window", "summary", "both"],
        default="both",
        help="Which strategy to evaluate. Default: both (the comparison).",
    )
    parser.add_argument(
        "--conversation",
        default=None,
        help="Run only one conversation by id (for debugging).",
    )
    args = parser.parse_args()

    conversations = load_conversations()
    if args.conversation:
        conversations = [c for c in conversations if c["id"] == args.conversation]
        if not conversations:
            raise SystemExit(f"No conversation with id={args.conversation!r}")

    strategies = ["window", "summary"] if args.strategy == "both" else [args.strategy]

    print(
        f"Loaded {len(conversations)} conversation(s). "
        f"Running {len(strategies)} strategy(ies) = "
        f"{len(conversations) * len(strategies)} agent calls.\n"
    )

    results: list[dict] = []
    for strategy in strategies:
        print(f"--- Running strategy={strategy} ---")
        for convo in conversations:
            print(f"  {convo['id']}...", flush=True)
            results.append(run_one(convo, strategy))
        print()

    summary = aggregate(results)
    _print_summary(summary)
    _print_per_row(results)


if __name__ == "__main__":
    main()
