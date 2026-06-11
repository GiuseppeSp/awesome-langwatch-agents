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

🚧 _Pending — baseline run not yet executed. Update once the run is complete._

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

🚧 Code shipped — baseline run pending. The honest framing will land after the local run: either the textbook lift survives (`react` clearly beats `bare`), or it doesn't (modern base models have closed the gap, in which case explicit scratchpads are pure overhead on this kind of task).
