# 06 · ReAct Agent

A multi-turn agent that interleaves reasoning, tool calls, and observations until it converges on an answer. This is the classic ReAct pattern (Yao et al., 2022) — "Thought → Action → Observation" — applied to multi-hop factual questions that genuinely require chaining 2-3 tool calls.

The experiment is one variable: **bare vs react reasoning mode**. Same model (gpt-4o-mini), same three tools, same loop, same 15 multi-hop questions. The only difference is the system prompt — `bare` lets the model call tools freely with no externalized reasoning; `react` forces the model to emit a `Thought:` line before every tool call.

The hypothesis worth testing: does explicit reasoning still earn its keep in 2026? Or have base models gotten good enough at internal chain-of-thought that the ReAct scaffolding adds nothing but tokens?

## The three tools

| Tool | What it does | LangWatch span type |
|---|---|---|
| `search(query)` | Lenient keyword-overlap lookup over a local 30-fact knowledge base ([`kb.json`](kb.json)) | `tool` |
| `calculator(expression)` | Evaluates an arithmetic expression (digits + `+ - * / ( )` only) | `tool` |
| `current_date()` | Returns today's date as `YYYY-MM-DD` — forces the agent to NOT hardcode the year | `tool` |

> **Design call: why a local KB, not Tavily?** A live web API introduces retrieval variance that confounds the bare-vs-react comparison — when bare fails, was it the reasoning or the flaky search? The local KB makes every retrieval deterministic, so any score gap between modes is attributable to the reasoning prompt alone. This is the same eval-design discipline agent #5's writeup argues for.

## The loop

```
┌──────────────────────────────────┐
│  user query + system prompt      │
└────────────────┬─────────────────┘
                 ▼
        ┌────────────────┐
        │ reason (LLM)   │◄────┐
        └────────┬───────┘     │
                 ▼             │ observe
        ┌────────────────┐     │
        │ tool call(s)   │─────┘
        └────────────────┘
                 │  (no tool call = final answer)
                 ▼
            final answer
```

Up to 8 iterations. Each iteration is a `langwatch.span(type="agent", name="iteration_N")` under the `react_agent` workflow root, with the LLM call (`type="llm"`) and each tool call (`type="tool"`) as nested children. The trace tree literally mirrors the loop — that's the whole pitch.

See [`agent.py`](agent.py) — ~200 lines, raw OpenAI function calling + LangWatch, no agent framework.

## The dataset

15 multi-hop questions ([`dataset.csv`](dataset.csv)) where the answer can only be assembled by chaining at least two tool calls. Examples:

| Question | Expected tool sequence | Why it forces multi-hop |
|---|---|---|
| How many years apart were the Eiffel Tower's completion and the Statue of Liberty's dedication? | `search` → `search` → `calculator` | Two date lookups → one subtraction |
| How old would Albert Einstein be today? | `search` → `current_date` → `calculator` | The current year MUST come from the tool, not the model's training data |
| How many seconds would light take to travel Earth's equatorial circumference? | `search` → `search` → `calculator` | Two physical constants → one division |

Every row's `expected_tools` column lists the minimum tool sequence — used by `tool_call_efficiency` to score whether the agent took a clean path or wandered.

## The evaluators

Three scorers ([`evals.py`](evals.py)):

| Evaluator | Type | Measures |
|---|---|---|
| `answer_correctness` | LLM-as-judge (1-5) | Does the final answer match the expected answer's key value (right number / right name / right date)? |
| `tool_call_efficiency` | Programmatic (F1 of tool multisets) | Did the agent call roughly the right tools roughly the right number of times — i.e., no thrash, no missing steps? Passes at ≥0.7. |
| `reasoning_quality` | LLM-as-judge (1-5) | Are the recorded `Thought:` lines actually useful — each one justifying the next action — or is the agent just generating reasoning theater? Returns `N/A` in bare mode (no thoughts emitted by design). |

Two of three are LLM calls; `tool_call_efficiency` is pure code.

## The tuning experiment

> **bare vs react reasoning on 15 multi-hop questions that require 2-3 chained tool calls**

The driver ([`run_eval.py`](run_eval.py)) runs every row twice — once bare, once react — and prints a side-by-side comparison plus a per-row detail view with iteration count, tool sequence, and `[hit limit]` markers for runs that exhausted MAX_ITERATIONS.

### Results

**Aggregate across 15 multi-hop rows**

| mode | answer_correctness | tool_call_efficiency | reasoning_quality |
|---|---|---|---|
| `bare` | **1.00 (100%)** | **0.97 (100%)** | n/a |
| `react` | **1.00 (100%)** | 0.87 (80%) | 0.70 (67%) |

Both modes got every question right. But on `tool_call_efficiency`, **react is 10 percentage points worse than bare** — the opposite of what the textbook ReAct paper predicts.

### Why react is worse: it bypasses tools MORE

Three rows where react skipped verification steps that bare ran:

| Question | bare tools | react tools | What happened |
|---|---|---|---|
| How old would Einstein be today, in 2026? | `search, calculator` | `calculator` only | Model hardcoded 1879 + 2026 from training data |
| Years between Sputnik and Apollo 11? | `search, search, calculator` | `calculator` only | Model just computed 1969 - 1957 directly |
| Years between WWI end and WWII end in Europe? | `search, search, calculator` | `calculator` only | Same — bypassed both searches |

In all three cases, react mode emitted a `Thought:` line that explicitly justified skipping the search: *"I know that Sputnik launched in 1957 and Apollo 11 landed in 1969, so I'll just calculate the difference."* That's the externalized reasoning **giving the model rhetorical cover to act on training-data knowledge instead of verifying via the tool**. Bare mode — no scratchpad — followed the tool protocol more faithfully on the exact same questions.

The Einstein row is the cleanest demonstration: it was specifically designed to require `current_date` so the agent couldn't hardcode the year. Neither mode called `current_date`, but react skipped `search` as well — going from one tool short to two tools short.

### The eval-design finding hiding inside this

Look at `reasoning_quality` per row in react mode. Three rows scored a perfect **1.00** — and all three are rows where the model **bypassed search**. Here's a representative judged thought:

> *"I already know the year Sputnik launched (1957) and the year Apollo 11 landed (1969), so I will calculate the difference directly."*

The LLM judge rated this as MORE coherent reasoning than the procedural narration on rows where the agent did follow the protocol (*"I need to look up X, then look up Y, then subtract them"*). And it's right in a narrow sense — bypass reasoning IS more decisive. But:

> **`reasoning_quality` rewards the exact behavior that hurts `tool_call_efficiency`.**

If a team tuned prompts using `reasoning_quality` alone (which sounds like a smart, defensible metric), they would push the agent toward more confident tool-bypass over time. The three orthogonal evaluators in this experiment are what made the divergence visible. With only one of them, the optimization signal would point the wrong way.

### The lesson

> **Asking a model to externalize its reasoning also gives it confidence to commit to that reasoning — including reasoning that bypasses your defined process.**

The textbook ReAct paper (Yao et al., 2022) reported large lifts because 2022-era base models needed the scaffolding to reason at all. By 2026, gpt-4o-mini reasons internally just fine — and externalizing the chain doesn't add new reasoning, it just commits the model to whichever conclusion the chain reaches first. When that conclusion is "I already know this," tool calls get skipped.

For PMs designing AI products in 2026, three concrete takeaways:

1. **Don't import 2022-2023 prompt-engineering wisdom without re-measuring on a current base model.** The deltas have collapsed on many tasks.
2. **Tool-bypass is a real production failure mode**, and explicit reasoning prompts can make it MORE likely, not less. If you need the model to always call a specific tool, the system prompt is not where you enforce it — the loop/guard is.
3. **Reasoning-quality metrics can be misaligned with process-quality metrics**, and you cannot tell from a single LLM-judge run. Build orthogonal evaluators or you'll optimize the wrong direction.

## Quick start

```bash
cd agents/06-react-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY and LANGWATCH_API_KEY (Project API Key, type Service)

# Smoke test on one query, bare mode (no scratchpad)
python agent.py "How many years apart were the Eiffel Tower's completion and the Statue of Liberty's dedication?"

# Same query, react mode (visible Thought: lines before each action)
REASONING_MODE=react python agent.py "How many years apart were the Eiffel Tower's completion and the Statue of Liberty's dedication?"

# Full comparison
python run_eval.py                  # both modes, all 15 rows
python run_eval.py --limit 3        # smoke test
python run_eval.py --mode react
```

If you hit the macOS Xcode-Python SSL issue (`Errno 2: No such file or directory` on `ssl.py`):

```bash
pip install certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

## What you'll learn

How a ReAct loop looks in LangWatch when each iteration is its own typed span — and the deeper PM-relevant question: whether explicit scratchpad reasoning is still worth the prompt complexity (and token cost) when modern base models already reason internally before tool calls. The 2022 paper got 30+ point lifts on hard reasoning; how much of that survives on a 2026 base model is what the eval framework is here to answer.

## Status

✅ Complete. Headline is a null on answer correctness (100% in both modes) but react is **10 points worse** on tool efficiency — the model bypasses verification steps on 3/15 rows when its reasoning is externalized, and the `reasoning_quality` judge happens to reward those bypasses as the most coherent reasoning of the whole run. The PM lesson is the inverse of the textbook ReAct claim: in 2026, forcing externalized reasoning doesn't make models more careful — it makes them more committed to whatever conclusion they reach, including conclusions that bypass your defined process. First catalog finding where the *direction-of-effect* is the opposite of the textbook prediction.
