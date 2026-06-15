# 10 · Tree-of-Thoughts

A tree-of-thoughts agent on **Game of 24**: given four numbers, combine them with + − × ÷ (each used once) to make 24 — e.g. `4 9 10 13 → (13 − 9) × (10 − 4) = 24`. This is the headline benchmark from the ToT paper, and the canonical place a single linear chain fails: one wrong early combination dooms the whole attempt with no way back.

The variable under test is **linear reasoning vs deliberate search**:

- **cot** — one chain-of-thought pass → a final expression. 1 LLM call.
- **tot** — beam search. At each depth the next states are enumerated mechanically (combining two numbers is just arithmetic), and the LLM is the **value function**: it scores each partial state `sure / likely / impossible`, the search keeps the top-`BEAM`(=5) and prunes the rest, repeating until one number remains.

The PM question isn't "is search better?" — it's "**when you turn reasoning into search, where does the win actually come from, and what does it cost?**" For Game of 24, move *generation* is trivial arithmetic; all the leverage is in the **evaluator** that decides which branches to keep. So the experiment isolates exactly that: ToT's performance is bounded by the quality of its state evaluator, nothing else.

And because Game of 24 is a tiny closed world, a brute-force oracle can say, for *every* state the LLM evaluator judged, whether 24 was actually still reachable — turning "is the pruning grounded?" into a measured number.

## The pipeline

```
cot:  solve (one chain) → expression

tot:  depth 1: enumerate all 3-number states → evaluate each → keep top 5
      depth 2: enumerate all 2-number states from the beam → evaluate → keep top 5
      depth 3: enumerate 1-number states → is any == 24?  (checked in code, no LLM)
```

Each `evaluate_dN` is a typed LangWatch `evaluation` span recording every state's verdict, so the trace shows what the value function kept and what it threw away. See [`agent.py`](agent.py) — ~330 lines (it carries the Game-of-24 mechanics, a brute-force solvability oracle, and the search), raw OpenAI + LangWatch, no agent framework.

> **Implementation note — full-precision arithmetic.** Intermediate values are carried as exact floats, never rounded. Rounding to a few decimals silently destroys non-terminating fractions like 8/3, which is exactly what the hardest puzzle (`3 3 8 8 = 8/(3 − 8/3)`) depends on — round it and the oracle wrongly calls it unsolvable. This bit during development; the fix is to round only for display, never for computation.

## The dataset

12 Game-of-24 puzzles ([`dataset.csv`](dataset.csv)), **all verified solvable by the oracle** (so a perfect searcher scores 12/12), spread across difficulty by solution density and whether a fraction is required:

| Difficulty | Examples | Count |
|---|---|---|
| `easy` | `1 2 3 4`, `6 6 6 6` — hundreds of solution paths | 4 |
| `medium` | `2 3 4 5`, `5 5 5 5` — fewer paths, maybe one clean fraction | 4 |
| `hard` | `4 9 10 13` (the paper's classic), `1 3 4 6`, `3 3 8 8` — few paths, fractions needed | 4 |

## The evaluators

Three scorers ([`evals.py`](evals.py)) — **all programmatic** (same noise-free choice as #7–#9):

| Evaluator | Type | Measures |
|---|---|---|
| `solved` | Programmatic | Does the final expression use each given number exactly once and evaluate to 24? Pass/fail. Both modes. |
| `evaluator_accuracy` | Programmatic | For every state the LLM value function judged, did its keep/prune decision match the brute-force oracle? Plus the breakdown into the two error types. **ToT-only — the discriminator.** |
| `search_efficiency` | Programmatic | LLM calls per puzzle (cot=1, tot=many). The cost axis. |

## The tuning experiment

> **cot vs tot on 12 solvable Game-of-24 puzzles**

### Three hypotheses to discriminate

| | Outcome | What it would teach |
|---|---|---|
| **H1** (textbook) | tot ≫ cot — search cracks puzzles the single chain can't | Deliberate search with backtracking is worth the cost. The ToT-paper result. |
| **H2** (null) | tot ≈ cot | Either the model is good enough that a chain suffices, or search adds no reachable structure. |
| **H3** (evaluator-bound) | tot > cot but far below the 12/12 ceiling; `evaluator_accuracy` is mediocre and fatal prunes appear | The search is only as good as its value function. A weak evaluator throws away winning branches and the search can't recover. |

### Results

**Aggregate across 12 puzzles**

| mode | solved | mean llm_calls | evaluator_accuracy |
|---|---|---|---|
| `cot` | 0.25 (3/12) | 1.0 | — |
| `tot` | **0.42 (5/12)** | **77.1** | **64%** |

**Solve rate by difficulty**

| mode | easy | medium | hard |
|---|---|---|---|
| `cot` | 2/4 | 1/4 | **0/4** |
| `tot` | 3/4 | 1/4 | **1/4** |

**ToT evaluator error breakdown (all puzzles)**

| Error type | Count | Consequence |
|---|---|---|
| pruned-reachable (said *impossible* about a state that could reach 24) | **52** | **Fatal** — the search threw away a path to the answer |
| kept-deadend (said *likely/sure* about a true dead end) | **293** | Wasteful — spent the evaluation budget on doomed branches |

### What's actually happening — H1 in direction, H3 in magnitude

**ToT does beat CoT.** +17 percentage points (25% → 42%), and it wins in every band — including one `hard` puzzle (`4 9 10 13`, the paper's classic) that CoT got 0/4 on. The direction of the textbook result reproduces: search beats a single chain on a problem that needs backtracking.

**But it is nowhere near the ToT paper's 4% → 74%,** and it costs **77× the LLM calls** (77.1 vs 1.0) to get there. The reason is in the discriminator: **the value function is only 64% accurate**, and its errors are exactly what cap the search.

- All 12 puzzles are solvable, so the ceiling is 12/12. ToT solved 5. The gap is not a search-depth problem — it's that the evaluator **fatally pruned 52 states that could actually reach 24.** When the value function says "impossible" about a live branch, beam search drops it and there is no recovery. Those 52 fatal prunes are unsolved puzzles.
- In the other direction it was far too generous: **293 dead-end states were kept** as "likely/sure," so most of the 77 calls per puzzle were spent scoring branches that could never reach 24. The cost is high *and* mostly wasted.

So the model is simultaneously **too lenient** (293 dead-ends kept — wasting budget) and **occasionally fatally strict** (52 reachable states pruned — losing answers). A 64%-accurate judge produces a search that is both expensive and leaky.

### The lesson

> **Tree-of-thoughts turns reasoning into search — and a search is only ever as good as the value function that prunes it. ToT's headline win is really a bet on your evaluator's calibration, paid for at a large multiple of the tokens.**

On Game of 24, search genuinely helps (+17pp over a single chain). But the entire distance between "helps a bit" (5/12) and "solves it" (12/12) is evaluator quality: every fatal prune is an answer the search will never find again, no matter how wide the beam. Spending more on search (more calls, bigger beam) cannot fix a value function that throws away the goal — it just prunes more confidently in the wrong place.

This is the search-tier instance of the catalog's recurring finding. The bolted-on control mechanism only works under a precondition that must be *measured*, not assumed:

| Agent | Mechanism | Precondition for it to help |
|---|---|---|
| #7 reflexion | self-critique + retry | the critic must be **calibrated** |
| #8 plan-and-execute | replan on divergence | the monitor must be **calibrated** |
| #9 chain-of-thought | self-consistency vote | the samples must actually **disagree** |
| **#10 tree-of-thoughts** | **search + pruning** | **the value function must be calibrated** |

For an AI PM in 2026 evaluating "let's add tree search to the agent," the operational takeaway is concrete: the number that predicts whether ToT will pay off is not the search width or depth — it's `evaluator_accuracy`, and specifically the **fatal-prune rate**. Measure that on a labeled set before you pay 77× per query for search.

## Quick start

```bash
cd agents/10-tree-of-thoughts
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY and LANGWATCH_API_KEY (Project API Key, type Service)

# One puzzle, each mode (the classic hard one)
python agent.py 4 9 10 13
TOT_MODE=tot python agent.py 4 9 10 13

# Full comparison (tot makes ~77 calls/puzzle — this takes several minutes)
python run_eval.py
python run_eval.py --limit 3     # smoke test
```

If you hit the macOS Xcode-Python SSL issue (`Errno 2` on `ssl.py`):

```bash
pip install certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

## What you'll learn

How a tree-of-thoughts search looks in LangWatch when each depth's evaluation is its own span — and how to read, from the verdicts, whether the value function is throwing away winning branches. The PM-relevant takeaway is that ToT's much-cited advantage is conditional and measurable: it beats a single chain when the problem needs search, but the size of the win — and whether the large compute cost is justified — is set almost entirely by the calibration of the state evaluator, which `evaluator_accuracy` and the fatal-prune count make visible.

## Status

✅ Complete. ToT beats CoT on Game of 24 (42% vs 25%, +17pp, winning in every band including a `hard` puzzle CoT scored 0/4 on), reproducing the direction of the textbook result — but at **77× the LLM calls** and far below the 12/12 ceiling, because the value function is only **64% accurate**: it fatally pruned **52** reachable states (lost answers) while wastefully keeping **293** dead ends. The search-tier entry in the calibration thread (#7/#8/#9/#10): a bolted-on mechanism — here, search with pruning — is only as good as the judge that drives it.
