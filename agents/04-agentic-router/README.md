# 04 · Agentic Router

An LLM-as-router agent that classifies incoming queries into one of five categories. This is the generic, reusable version of the routing pattern that production AI products live and die by — the moment an LLM has to decide *which downstream pipeline* should answer a query, you're in router territory.

The experiment is one variable: **bare vs tuned system prompt**. Same model (gpt-4o-mini), same 25-query dataset. The bare prompt is the minimum-viable classifier ("classify into one of these categories"). The tuned prompt adds three things the production-AI prompt-engineering literature swears by — but does it actually move the needle on real-world routing?

## The five categories

| Name | Routes to (in a real product) |
|---|---|
| `code_question` | Programming-help pipeline (search code docs, run examples) |
| `factual_question` | RAG over knowledge base |
| `creative_task` | Long-form generation model with creative-writing prompt |
| `math_calculation` | Calculator tool / formal solver |
| `general_chat` | Default chat path (the fallback) |

The categories are deliberately distinct *and* deliberately overlapping. Code and math overlap on "calculate compound interest in Excel." Creative and general_chat overlap on "suggest a name for my dog." That overlap is where prompt quality earns its keep.

Definitions live in [`categories.json`](categories.json) — swap them out and the agent works for any routing domain.

## The dataset

25 hand-labeled queries ([`dataset.csv`](dataset.csv)) split intentionally:

- **15 clear rows** — single-category queries where the right answer is obvious to a human ("What's 17 times 84?" → `math_calculation`)
- **10 ambiguous rows** — deliberately overloaded queries that fit multiple categories ("Write a story that uses the number pi exactly three times" → math + creative)

The clear rows show whether each prompt style does the basics right. The ambiguous rows are where the experiment actually matters.

## The two prompt styles

| Style | What's in the prompt |
|---|---|
| `bare` | Just the category names + 1-line descriptions. Minimum-viable classifier. |
| `tuned` | Bare + (1) 3 example queries per category, (2) explicit disambiguation rules for the common edge cases ("generation deliverable beats factual lookup"), (3) a fallback rule ("when in doubt, use `general_chat`") |

This is the textbook "improve your system prompt" recipe. The eval framework tells you whether the textbook recipe actually pays off on your data.

## The pipeline

```
classify  →  parse output
```

Two steps — one LangWatch span for the LLM call, plus a thin parser that's tolerant of "I think the answer is..." rambling (it grabs the last valid category name). The trace tree is intentionally simple: the interesting work is all in the prompt, not the orchestration.

See [`agent.py`](agent.py) — ~150 lines.

## The evaluators

Three scorers ([`evals.py`](evals.py)):

| Evaluator | Type | Measures |
|---|---|---|
| `route_correctness` | Programmatic (exact match) | Did the router pick the labeled category? Strict, binary. |
| `output_format` | Programmatic | Was the response a single valid category name (post-parse)? Catches rambling that breaks downstream parsing. |
| `ambiguity_handling` | LLM-as-judge (rubric 1-5) | Run only on ambiguous rows — was the chosen category *defensible* even if it didn't match the label? Surfaces "label-wrong" vs "actually wrong" so a great router doesn't get unfairly punished. |

Only ambiguity_handling costs LLM calls. The other two are pure code.

## The tuning experiment

> **Bare vs tuned system prompt on 25 queries (15 clear + 10 ambiguous)**

The driver script ([`run_eval.py`](run_eval.py)) splits the comparison three ways — **all rows**, **clear rows only**, and **ambiguous rows only** — so you can see exactly where each style wins.

### Results

> _TBD after first run._
>
> Expected story: clear rows are at ceiling for both styles (gpt-4o-mini gets the easy ones). The action is on ambiguous rows. Tuned should lift `route_correctness` 20-30 points there (the disambiguation rules give the model explicit guidance) AND maintain `output_format` near 100% (the bare prompt sometimes rambles).

## Quick start

```bash
cd agents/04-agentic-router
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY and LANGWATCH_API_KEY (Project API Key, type Service)

# Smoke test on a clear and an ambiguous case
python agent.py "What's 17 times 84?"
PROMPT_STYLE=tuned python agent.py "Write a story that uses pi exactly three times."

# Full comparison
python run_eval.py
```

If you hit the macOS Xcode-Python SSL issue (`Errno 2: No such file or directory` on `ssl.py`):

```bash
pip install certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

## What you'll learn

How to build an LLM-as-router with full LangWatch tracing on every classification, and — more importantly — how to measure whether the standard prompt-engineering tricks (few-shot examples, edge-case rules, fallback paths) actually move the needle on your specific routing problem. Routers are everywhere in production AI; "I tuned the prompt and it felt better" isn't an answer that survives a code review.

## Status

🚧 Code complete. Baseline numbers + LangWatch trace screenshot landing after the first run.
