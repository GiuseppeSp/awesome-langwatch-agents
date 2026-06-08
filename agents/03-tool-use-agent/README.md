# 03 · Tool-use Agent

A function-calling agent with four tools — `calculator`, `date_parser`, `unit_converter`, `currency_converter` — wired around the production lesson that's hardest to internalize: **the quality of your tool descriptions matters more than the model you're using.**

The experiment is one variable: vague vs precise tool descriptions. Two sentences of "use this when… do NOT use this for…" framing can shift tool-selection accuracy by 20-40 percentage points. Same model, same dataset, same code — only the JSON schemas change.

## The tools

Four self-contained Python functions ([`tools.py`](tools.py)). No external APIs — clone, run, reproduce on any machine. Each has two schema variants exposed to the model:

| Tool | Vague description | Precise description (excerpt) |
|---|---|---|
| `calculator` | *"Does math."* | *"Evaluate an arithmetic expression like '2 * (3 + 4)'… Do NOT use this for unit conversions or currency conversions — those have their own dedicated tools."* |
| `date_parser` | *"Handles dates."* | *"Convert a natural-language date description ('next Tuesday', 'tomorrow') into ISO format (YYYY-MM-DD)…"* |
| `unit_converter` | *"Converts things."* | *"Convert a physical measurement from one unit to another… Do NOT use this for currency — currency has its own tool. Supported units: m/km/mi/ft/in/cm for length…"* |
| `currency_converter` | *"Money stuff."* | *"Convert an amount of money from one currency to another (e.g. EUR to USD)… Supported currencies: USD, EUR, GBP, JPY…"* |

The contrast is deliberate. Vague descriptions are exactly the kind a developer writes when they're focused on getting the integration working rather than getting the agent to make good choices. The precise versions add three things the model actually uses: a concrete example, a "use this when" frame, and an explicit "do NOT use this for…" boundary.

## The dataset

20 hand-labeled queries ([`dataset.csv`](dataset.csv)) split across two intents:

- **15 should-trigger-a-tool** queries — at least one row per tool, plus borderline cases (e.g., *"how many minutes are in 2.5 hours?"* — could be `calculator` or `unit_converter`; the precise schema makes it unambiguous)
- **5 should-NOT-trigger-a-tool** queries — general-knowledge questions like *"What's the capital of France?"* that exist purely to catch over-triggering

Each row carries the expected tool, an informal hint of what the args should look like (for the LLM-judge), and a one-line note explaining why it's in the set.

## The pipeline

```
model_turn  →  [tool call(s) if requested]  →  model_turn  →  …  →  final answer
```

Standard OpenAI tool-use loop, capped at 5 turns. Every model turn is one LangWatch `llm` span; every executed tool call is a `tool` span with the args and result captured inline. The trace tree for a query that needed two tools will show: root → model_turn → tool → model_turn → final answer.

See [`agent.py`](agent.py) — ~150 lines, no frameworks.

## The evaluators

Three scorers ([`evals.py`](evals.py)):

| Evaluator | Type | Measures |
|---|---|---|
| `tool_selection` | Programmatic | Did the agent call the expected tool — or correctly skip tools when the query didn't need one? Binary. |
| `argument_extraction` | LLM-as-judge (rubric 1-5) | Were the arguments the agent passed to the tool semantically right for the query? Tolerant of phrasing differences. |
| `no_tool_correctness` | Programmatic | Narrower lens: only for the 5 should-NOT-trigger rows. Surfaces over-triggering separately from wrong-tool errors. |

## The tuning experiment

> **Vague vs precise tool descriptions on the same 20-query dataset**

Run both modes with one command — `python run_eval.py` runs the dataset with each schema set and prints a comparison table.

### Results

> _TBD after first run._
>
> Expected story: vague mode confuses calculator with unit_converter on time questions ("how many minutes in 2.5 hours"), uses currency_converter for non-monetary unit questions, and over-triggers on general-knowledge queries. Precise mode resolves most of this — same model, just better descriptions.

## Quick start

```bash
cd agents/03-tool-use-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY and LANGWATCH_API_KEY (Project API Key, type Service)

# Smoke test — one query, one mode
python agent.py "How many minutes are in 2.5 hours?"
TOOL_DESCRIPTIONS=precise python agent.py "How many minutes are in 2.5 hours?"

# The full comparison
python run_eval.py
```

If you hit the macOS Xcode-Python SSL issue (`Errno 2: No such file or directory` on `ssl.py`):

```bash
pip install certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

## What you'll learn

How to wire OpenAI function calling with LangWatch tracing so every tool call shows up as a typed span (args + result inline), and how to A/B test tool descriptions on a labeled dataset to actually measure the impact instead of guessing. This is the kind of measurement a production team needs to make the case for spending 30 minutes rewriting a tool description instead of swapping out the model.

## Status

🚧 Code complete. Baseline numbers + LangWatch trace screenshot landing after the first run.
