"""
Deterministic dataset generator for the supervisor-worker agent.

Every subtask is an ATOMIC operation with an oracle-computed gold answer that a
capable model gets right ~100% of the time IN ISOLATION (word counts, small
arithmetic, string reversals, pick-the-nth-word, sorts). That is the whole point:
if a bundled request of N such tasks is answered wrong, the failure is attributable
to BUNDLING (attention split across N simultaneous instructions), not to any one
task being hard.

Composite requests bundle N DISTINCT op types (heterogeneous, so the model must
switch modes N times in one pass). Bundle sizes 1/3/5/8 sweep for the crossover
where single-agent instruction-following frays and per-worker isolation starts to
pay — the same "push the stress variable until it breaks" method as #12 (depth) and
#24 (registry size).

Run:  python build_dataset.py   ->  writes dataset.csv (bundle_size, subtasks_json)

Golds are computed here, never by a model, so the dataset is self-verifying.
"""

from __future__ import annotations

import csv
import json
import random

SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "a journey of a thousand miles begins with a single step",
    "she sells sea shells by the sea shore",
    "all that glitters is not gold in the end",
    "to be or not to be that is the question",
    "the early bird catches the worm every morning",
    "practice makes perfect if you keep at it",
    "better late than never but never late is better",
]
WORDS = ["hello", "banana", "system", "orange", "planet", "guitar",
         "window", "pencil", "rocket", "silver", "forest", "dragon"]


def op_wordcount(rng):
    s = rng.choice(SENTENCES)
    return {"type": "wordcount",
            "instr": f'Count the number of words in this sentence: "{s}"',
            "gold": str(len(s.split()))}


def op_multiply(rng):
    a, b = rng.randint(11, 99), rng.randint(11, 99)
    return {"type": "multiply", "instr": f"Multiply {a} by {b}.", "gold": str(a * b)}


def op_add(rng):
    a, b = rng.randint(100, 999), rng.randint(100, 999)
    return {"type": "add", "instr": f"Add {a} and {b}.", "gold": str(a + b)}


def op_reverse(rng):
    s = rng.choice(WORDS)
    return {"type": "reverse", "instr": f'Reverse the string "{s}".', "gold": s[::-1]}


def op_upper(rng):
    s = rng.choice(WORDS)
    return {"type": "upper", "instr": f'Write "{s}" in all capital letters.', "gold": s.upper()}


def op_nthword(rng):
    s = rng.choice(SENTENCES)
    w = s.split()
    n = rng.randint(2, len(w))
    return {"type": "nthword",
            "instr": f'What is word number {n} in this sentence: "{s}"?',
            "gold": w[n - 1]}


def op_sortnums(rng):
    nums = rng.sample(range(10, 99), 4)
    return {"type": "sortnums",
            "instr": f"Sort these numbers in ascending order: {', '.join(map(str, nums))}.",
            "gold": ", ".join(map(str, sorted(nums)))}


def op_maxnum(rng):
    nums = rng.sample(range(10, 99), 5)
    return {"type": "maxnum",
            "instr": f"What is the largest of these numbers: {', '.join(map(str, nums))}?",
            "gold": str(max(nums))}


def op_lastchar(rng):
    s = rng.choice(WORDS)
    return {"type": "lastchar", "instr": f'What is the last letter of the word "{s}"?', "gold": s[-1]}


OPS = [op_wordcount, op_multiply, op_add, op_reverse, op_upper,
       op_nthword, op_sortnums, op_maxnum, op_lastchar]

BUNDLE_SIZES = [1, 3, 6, 9, 12]
ROWS_PER_SIZE = 4
SEED = 7


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for k in BUNDLE_SIZES:
        for _ in range(ROWS_PER_SIZE):
            # distinct op types for variety; past the 9 available, allow repeats
            # (same type, fresh random params -> still a distinct task)
            if k <= len(OPS):
                ops = rng.sample(OPS, k)
            else:
                ops = rng.sample(OPS, len(OPS)) + rng.choices(OPS, k=k - len(OPS))
            subtasks = [op(rng) for op in ops]
            rows.append({"bundle_size": k, "subtasks_json": json.dumps(subtasks, ensure_ascii=False)})

    with open("dataset.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bundle_size", "subtasks_json"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote dataset.csv — {len(rows)} rows, bundle sizes {BUNDLE_SIZES}")


if __name__ == "__main__":
    main()
