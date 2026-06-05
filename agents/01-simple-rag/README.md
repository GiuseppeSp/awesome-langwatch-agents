# 01 · Simple RAG

The most basic RAG pattern: chunk a small corpus, embed the chunks, retrieve the top-k for a query, and synthesize an answer. Instrumented end-to-end with LangWatch so every step is visible: which chunks were retrieved, how long each LLM call took, what tokens were spent.

## What you'll learn

How to wire LangWatch around the simplest possible RAG so the next time you add complexity (re-ranking, hybrid search, citation generation), the trace tree shows you exactly which step changed.

## The tuning experiment (placeholder)

> **Chunk size 256 vs 512 vs 1024 — measured impact on faithfulness and recall**
>
> _Baseline (chunk size 512): faithfulness 0.78, recall@3 0.65._
> _After tuning: TBD._

## Status

🚧 In progress. Coming next session.

## Files (planned)

- `agent.py` — main loop: load corpus → chunk → embed → retrieve → synthesize
- `dataset.csv` — ~20 question/answer pairs hand-labeled against the corpus
- `evals.py` — retrieval-correctness + answer-faithfulness evaluators
- `requirements.txt` — `openai`, `langwatch`, `chromadb` (or similar)
- `.env.example` — required env vars
