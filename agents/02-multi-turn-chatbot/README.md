# 02 · Multi-turn Chatbot

A travel-planning chatbot wired around the question every conversational AI faces eventually: **how do you remember what the user told you 10 turns ago without paying for the full transcript on every call?**

Two memory strategies are implemented as a single env var (`MEMORY_STRATEGY`):

| Strategy | What it does | When it wins |
|---|---|---|
| `window` | Keep only the last N turns verbatim. Drop everything older. | Short conversations, or when only recent context matters. Cheap and predictable. |
| `summary` | Keep the last N turns verbatim AND prepend an LLM-generated summary of everything older. | Long conversations where the user shared constraints early ("I'm vegan", "tight budget", "back injury") that the answer still needs to honor. |

The whole agent is one chat function with strategy as a parameter, so the comparison is apples-to-apples — same conversations, same questions, same evaluator, only the memory pipeline changes.

## The dataset

Six labeled travel-planning conversations ([`conversations.json`](conversations.json)). Each one follows the same shape:

1. **Setup turns** establish a constraint early (turn 1) — vegan, allergic, tight budget, back injury, traveling with a toddler, photographing winter Iceland, pescetarian
2. **Filler turns** carry the conversation forward for ~10 more turns with normal travel-planning chatter
3. **One test question** at the end that forces the agent to recall the early constraint

A memory strategy that successfully preserves the early constraint will produce an answer that adapts to it. A strategy that loses it will produce a generic answer that contradicts the user's needs.

## The pipeline

```
build_memory  →  synthesize
```

Two steps, both LangWatch spans. The `build_memory` step is the interesting one — its behavior changes based on the strategy. When `MEMORY_STRATEGY=summary` and the history exceeds the window, it spawns a child `summarize_history` LLM span so you can see exactly what summary was produced for each query.

Every call is wrapped in a `@langwatch.trace`, and `thread_id` is set per conversation so all turns of one conversation group together in the LangWatch **Thread** tab. That's the right way to view a multi-turn agent — one row per conversation, not one row per turn.

See [`agent.py`](agent.py) — ~150 lines, raw OpenAI + LangWatch.

## The evaluators

Three scorers ([`evals.py`](evals.py)):

| Evaluator | Type | Measures |
|---|---|---|
| `context_recall` | LLM-as-judge (rubric 1-5) | Did the assistant's reply actually honor the constraint the user stated earlier? **This is the headline metric** — when memory fails, this drops sharply. |
| `must_include` | Programmatic (OR) | Does the answer mention at least one of the keywords a correct answer should mention? Cheap completeness proxy. |
| `must_not_include` | Programmatic (NONE) | Does the answer AVOID keywords that would actively contradict the constraint? Catches the worst failure mode — recommending steak to a pescetarian. |

## The tuning experiment

> **Sliding window vs LLM-summarized memory on 6 multi-turn conversations**

Run both strategies with one command — `python run_eval.py` runs all 6 conversations with each strategy and prints a comparison table.

### Results

> _TBD after first run._
>
> Expected story: window scores collapse on `context_recall` and `must_not_include` because the constraint-stating turn falls off the window. Summary preserves the constraint and stays high.

## Quick start

```bash
cd agents/02-multi-turn-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY and LANGWATCH_API_KEY

# Smoke test — one conversation, one strategy
python agent.py tokyo-vegan window
python agent.py tokyo-vegan summary

# The full comparison
python run_eval.py
```

If you run into the macOS Xcode-Python SSL issue (Errno 2: No such file or directory on `ssl.py`), fix it with:

```bash
pip install certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

## What you'll learn

How to instrument a multi-turn agent in LangWatch so each conversation collapses into one Thread (not 12 disconnected traces), how to A/B two memory strategies on the same data, and how to write an evaluator (`context_recall`) that directly measures the thing your agent is actually being tested on — not a proxy.

## Status

🚧 Code complete. Baseline numbers + Thread-view screenshot landing after the first run.
