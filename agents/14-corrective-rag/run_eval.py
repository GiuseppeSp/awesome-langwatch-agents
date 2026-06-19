"""
Driver: run the dataset through the agent in both modes and print the mode
comparison, accuracy by category (in_corpus / out_obvious / out_tricky — where
grader calibration shows), the paired vanilla -> crag lift, and the grader's
accuracy split by category.

Usage:
    python run_eval.py                 # vanilla vs crag, all rows
    python run_eval.py --mode crag
    python run_eval.py --limit 4       # smoke test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass

import agent  # local
from evals import answer_correctness, vanilla_to_crag_lift, grader_accuracy

MODES = ["vanilla", "crag"]
CATS = ["in_corpus", "out_obvious", "out_tricky"]


@dataclass
class Row:
    query: str
    category: str
    gold: str
    answer_in_local: bool
    notes: str


def load_dataset(path: str = "dataset.csv") -> list[Row]:
    rows: list[Row] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(Row(
                query=r["query"], category=r["category"], gold=r["gold_answer"],
                answer_in_local=r["answer_in_local"].strip().lower() == "yes",
                notes=r.get("notes", "") or "",
            ))
    return rows


def run_one(row: Row, mode: str) -> dict:
    res = agent.run(row.query, mode=mode)
    d = {
        "query": row.query, "category": row.category, "gold": row.gold,
        "answer_in_local": row.answer_in_local, "mode": mode,
        "answer": res.answer, "grade": res.grade, "used_web": res.used_web,
        "llm_calls": res.llm_calls,
        "correct": answer_correctness(res.answer, row.gold),
    }
    if mode == "crag":
        d["grader"] = grader_accuracy(res.grade, row.answer_in_local)
    return d


def _print_summary(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Mode comparison ===\n")
    print(f"{'mode':>8} | {'correct':>14} | {'mean llm_calls':>15}")
    print("-" * 44)
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        n = len(rs)
        c = sum(1 for r in rs if r["correct"].passed)
        mc = sum(r["llm_calls"] for r in rs) / n
        print(f"{mode:>8} | {f'{c}/{n} ({c/n*100:.0f}%)':>14} | {mc:>15.1f}")


def _print_by_category(results: list[dict]) -> None:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    print("\n=== Accuracy by category ===\n")
    print(f"{'mode':>8} | " + " | ".join(f"{c:>12}" for c in CATS))
    print("-" * (10 + 15 * len(CATS)))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        cells = []
        for cat in CATS:
            crs = [r for r in rs if r["category"] == cat]
            c = sum(1 for r in crs if r["correct"].passed)
            cells.append(f"{c}/{len(crs)}" if crs else "-")
        print(f"{mode:>8} | " + " | ".join(f"{c:>12}" for c in cells))


def _print_grader(results: list[dict]) -> None:
    crag = [r for r in results if r["mode"] == "crag" and "grader" in r]
    if not crag:
        return
    print("\n=== Grader accuracy by category (CRAG's calibration) ===\n")
    print(f"{'':>8} | " + " | ".join(f"{c:>12}" for c in CATS) + " | overall")
    print("-" * (10 + 15 * len(CATS) + 10))
    cells = []
    for cat in CATS:
        crs = [r for r in crag if r["category"] == cat]
        c = sum(1 for r in crs if r["grader"].passed)
        cells.append(f"{c}/{len(crs)}" if crs else "-")
    overall = sum(1 for r in crag if r["grader"].passed)
    print(f"{'grader':>8} | " + " | ".join(f"{c:>12}" for c in cells) + f" | {overall}/{len(crag)}")
    # show every miscalibration
    bad = [r for r in crag if not r["grader"].passed]
    if bad:
        print("\n  miscalibrated rows:")
        for r in bad:
            print(f"    [{r['category']:>11}] grade={r['grade']:>9} (truth={'in-local' if r['answer_in_local'] else 'not-in-local'})  {r['query'][:50]}")


def _print_lift(results: list[dict]) -> None:
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["query"]][r["mode"]] = r
    if not all("vanilla" in m and "crag" in m for m in by_q.values()):
        return
    print("\n=== Paired lift: vanilla -> crag ===\n")
    helped = hurt = neutral = 0
    for q, m in by_q.items():
        d, c = m["vanilla"], m["crag"]
        L = vanilla_to_crag_lift(d["correct"].passed, c["correct"].passed)
        if L.score == 1.0:
            helped += 1; tag = "  HELPED"
        elif L.score == 0.0:
            hurt += 1; tag = "    HURT"
        else:
            neutral += 1; tag = ""
        if tag:
            print(f"  [{d['category']:>11}] vanilla={'OK ' if d['correct'].passed else 'BAD'} "
                  f"crag={'OK ' if c['correct'].passed else 'BAD'} "
                  f"(grade={c['grade']}{'+web' if c['used_web'] else ''}, gold={d['gold']!r}){tag}")
            print(f"               {q[:66]}")
    total = helped + hurt + neutral
    mean = (helped + neutral * 0.5) / total if total else 0.0
    print(f"\n  vanilla_to_crag_lift mean = {mean:.2f}  (helped {helped}, hurt {hurt}, neutral {neutral})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    modes = MODES if args.mode == "all" else [args.mode]

    print(f"Loaded {len(rows)} queries. Running {len(modes)} mode(s).")
    results: list[dict] = []
    for mode in modes:
        print(f"\n--- Running mode={mode} ---")
        for row in rows:
            print(f"  [{row.category:>11}] {row.query[:48]}...", flush=True)
            results.append(run_one(row, mode))

    _print_summary(results)
    _print_by_category(results)
    _print_grader(results)
    if args.mode == "all":
        _print_lift(results)


if __name__ == "__main__":
    main()
