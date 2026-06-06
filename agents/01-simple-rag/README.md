# 01 · Simple RAG

The most basic RAG pattern: chunk a small corpus, embed the chunks, retrieve the top-k by cosine similarity, and synthesize an answer. Wired end-to-end with LangWatch so every step is visible in the trace tree.

## The corpus

A short mini-encyclopedia of ten AI agent patterns ([`corpus.md`](corpus.md)) — RAG, ReAct, Chain-of-Thought, Tree of Thoughts, Function Calling, Agentic Router, Multi-Agent Orchestration, Reflexion, Constitutional AI, and Plan-and-Execute. Each entry is one paragraph (~150-200 words). Distinctive enough that retrieval matters, with enough overlapping terminology to make the task realistic.

## The golden dataset

15 hand-labeled questions ([`dataset.csv`](dataset.csv)) spanning three difficulty bands:
- **Easy** — direct definition lookups ("What does ReAct stand for?")
- **Medium** — comparative ("How is Reflexion different from a single Chain-of-Thought attempt?")
- **Hard** — indirect, use-case framed ("What pattern would you use to solve a logic puzzle that needs trying multiple approaches?")

Each row carries the expected source entry, expected keywords in the answer, and a difficulty tag.

## The pipeline

```
chunk corpus  →  embed chunks  →  embed query  →  retrieve top-k  →  synthesize
```

Every step is a LangWatch span. The trace's root carries the user's question as input and the synthesized answer as output; children carry per-step inputs, outputs, model, and token usage. The result is a complete audit trail of one query — exactly the kind of artifact you'd want in production to debug an "I don't trust this answer" report.

See [`agent.py`](agent.py) — ~150 lines, no frameworks, raw OpenAI + LangWatch.

## The evaluators

Three scorers, two cheap and one LLM-judged ([`evals.py`](evals.py)):

| Evaluator | Type | Measures |
|---|---|---|
| `retrieval_at_k` | Programmatic | Did the top-k retrieved chunks include the *expected source* entry? Binary. |
| `keyword_match` | Programmatic | Does the answer contain every expected keyword? Lenient proxy for completeness. |
| `faithfulness` | LLM-as-judge (gpt-4o-mini, rubric 1-5) | Is every factual claim in the answer supported by the retrieved context? Penalizes hallucinations. |

## The tuning experiment

> **Chunk size 128 vs 256 vs 512 — measured impact on retrieval and faithfulness**
>
> _Baseline (chunk size 256): TBD after first run._
> _After sweep: TBD._

Run all three sizes with one command — the driver script ([`run_eval.py`](run_eval.py)) supports a `--sweep` flag that runs the whole dataset against each chunk size in sequence and prints a comparison table.

## Quick start

```bash
cd agents/01-simple-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY and LANGWATCH_API_KEY

# Smoke test — one query
python agent.py "What does ReAct stand for?"

# Full eval against the golden dataset
python run_eval.py

# The tuning experiment
python run_eval.py --sweep
```

Every query produces a trace in your LangWatch dashboard with the full step-by-step breakdown.

## What you'll learn

How to wire LangWatch around the simplest possible RAG so the next time you add complexity (re-ranking, hybrid search, citation generation), the trace tree shows you exactly which step changed and why. And how a chunk-size sweep — the most basic RAG tuning experiment there is — gives you the language to talk about retrieval quality with numbers instead of vibes.

## Status

🚧 Code complete. Baseline numbers + LangWatch screenshot landing after the first run.
