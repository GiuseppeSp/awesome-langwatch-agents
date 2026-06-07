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

| strategy | context_recall | must_include | must_not_include |
|---|---|---|---|
| `window` | 0.50 (50%) | 0.50 (50%) | 0.67 (67%) |
| `summary` | **1.00 (100%)** | **1.00 (100%)** | 0.50 (50%) |

*(Score is the mean across the dataset; % is the pass rate. `context_recall` is an LLM-as-judge 1-5 rubric normalized to 0..1; the others are programmatic.)*

**Summary wins decisively on the metric that actually matters.** `context_recall` doubles from 50% to 100% — meaning the strategy that compresses early turns into a summary preserved the user's stated constraint on every single conversation, while the strategy that drops older turns lost it on half.

### The headline demo: `tokyo-vegan`

The user states "I'm vegan and severely allergic to peanuts" in turn 1, then has 9 turns of unrelated chat (transit, cherry blossoms, tipping), then asks for restaurant recommendations.

- **`window`** recommended seven restaurants including Sushi Dai, Ichiran Ramen, kushikatsu (deep-fried meat skewers), and — the kill shot — *Matsuzakagyu Yakiniku, "high-quality Matsusaka beef."* Beef to a vegan. The constraint was completely lost.
- **`summary`** opened with *"Here are some vegan-friendly restaurants in Tokyo that cater to your dietary needs and are safe for your peanut allergy"* and recommended seven plant-based spots — T's Tantan, Ain Soph. Journey, Nagi Shokudo, vegan sushi at Sushi Ken. The closing sentence: *"Make sure to communicate your peanut allergy clearly when ordering, as some dishes may have hidden ingredients."*

Same agent code. Same conversation. Same question. Different memory strategy. One produces a recommendation that could actively harm the user; the other produces a recommendation that explicitly addresses both constraints.

### The unexpected finding: a subtle eval-design lesson

Look at `must_not_include` in the table: summary "loses" 50% vs window's 67%. That's counterintuitive — how can the strategy that *correctly remembered the constraint* fail the avoidance check more often than the one that forgot it?

The eval rule is too literal. For `tokyo-vegan`, the forbidden list includes the word "peanut." When `window` forgot the constraint, it recommended sushi and beef — neither of which contains the string "peanut," so it accidentally passes. When `summary` correctly engaged with the constraint, it said things like *"be sure to mention your peanut allergy when ordering"* — and the literal keyword `peanut` triggered the rule, even though it appeared in the context of *warning about* peanuts.

**This is real and worth keeping in the writeup.** Programmatic keyword-avoidance checks penalize strategies that engage with constraints by name. The fix would be either:

- An LLM-as-judge replacement for `must_not_include` that reasons about whether the keyword's *role* in the answer is positive or negative
- More precise forbidden patterns ("eat peanuts" vs the bare word "peanut")

The broader lesson: **LLM-as-judge (here, `context_recall`) was the most reliable signal in this experiment**. The two programmatic evals agreed with it on the headline result but each had a blind spot. That's the case for spending the extra tokens on a rubric judge for the metric that actually drives decisions.

### Per-conversation breakdown

| Conversation | window recall | summary recall | Notes |
|---|---|---|---|
| `tokyo-vegan` | ❌ 0.00 | ✅ 1.00 | The kill demo. Window recommended beef. |
| `paris-budget-anniversary` | ❌ 0.25 | ✅ 1.00 | Window forgot the 1500 EUR budget. |
| `bali-back-injury` | ✅ 0.75 | ✅ 1.00 | Window narrowly held on — back injury was mentioned across multiple turns. |
| `iceland-photography-winter` | ✅ 1.00 | ✅ 1.00 | Both pass — winter context is reinforced in recent turns. |
| `costa-rica-toddler` | ❌ 0.00 | ✅ 1.00 | Window forgot the toddler. |
| `portugal-pescetarian` | ✅ 1.00 | ✅ 1.00 | Both pass — pescetarian came up in turn 1 but Lisbon is famously seafood-heavy so the model defaults to fish anyway. |

The pattern is clear: `window` survives when the constraint either (a) recurs in recent turns or (b) aligns with the default tendencies of the model's training data. It fails the moment the constraint is single-stated, far back, and specific.

That's a useful finding by itself — it tells you when you can get away with the cheap strategy and when you need to pay for summarization.

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

✅ Complete. Code, dataset, evaluators, and the memory-strategy comparison are all shipped. Real run produced a clean 50→100 point lift on `context_recall` for the summary strategy, plus an unexpected finding about the limits of programmatic keyword-avoidance checks.
