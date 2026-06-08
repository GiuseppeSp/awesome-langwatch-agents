# Awesome LangWatch Agents

A curated, runnable catalog of **AI agents you can actually trust** — because every one of them ships with full observability, a golden dataset, and pass/fail evaluators wired into [LangWatch](https://langwatch.ai).

Most "awesome AI agents" lists show you code. This one shows you **what good looks like, measured**: the trace tree, the per-step token + cost breakdown, the evaluator scores before and after every tuning decision.

## Why this exists

Building an AI agent is easy. Knowing whether your agent is actually working — and proving it stays working after every prompt change — is hard. That gap is what kills production AI features. This repo is a working answer: a series of small, focused agents, each one measured end-to-end, each one paired with the kind of tuning experiment that turns "vibes" into "I shipped a 23% improvement."

## Agent catalog

| # | Agent | Pattern | Status |
|---|---|---|---|
| 1 | [simple-rag](agents/01-simple-rag/) | Retrieve-then-generate over a small corpus | ✅ Shipped — chunk-size sweep ran, finding is a null result (and the README explains why that's the interesting answer) |
| 2 | [multi-turn-chatbot](agents/02-multi-turn-chatbot/) | Conversation memory (window vs summary) | ✅ Shipped — summary doubled context recall (50→100%) and surfaced a subtle eval-design pitfall |
| 3 | [tool-use-agent](agents/03-tool-use-agent/) | Function calling with multiple tools | ✅ Shipped — another null aggregate, with a sharp finding about *when* tool-description tuning actually matters |
| 4 | [agentic-router](agents/04-agentic-router/) | LLM-as-router across knowledge sources | ⏳ Planned |
| 5 | [multi-agent-pipeline](agents/05-multi-agent-pipeline/) | Orchestrator + specialized workers | ⏳ Planned |

Each folder contains:
- `README.md` — pattern walkthrough, what it does, what the tuning experiment showed
- `agent.py` — self-contained runnable agent
- `dataset.csv` — hand-labeled golden dataset (~20-50 rows)
- `evals.py` — programmatic + LLM-as-judge evaluators
- A LangWatch trace screenshot showing the parent/child span tree

## How to read this repo

Pick any agent. The `README.md` will tell you:

1. What pattern it implements
2. The before/after metrics from one tuning experiment
3. A link to a public LangWatch dashboard (or screenshots, if the run is private)
4. How to clone and run it yourself in under 5 minutes

## Quick start (for any agent)

```bash
git clone https://github.com/GiuseppeSp/awesome-langwatch-agents.git
cd awesome-langwatch-agents/agents/01-simple-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your OPENAI_API_KEY and LANGWATCH_API_KEY
python agent.py
```

Every agent follows this same shape. No framework lock-in, no magic — just `openai` + `langwatch` + a clear example of what good instrumentation looks like.

## Background

These agents grew out of practical experience instrumenting a 6-stage production RAG pipeline with LangWatch and running a real eval loop against a hand-labeled golden dataset. The lessons from that build — how to actually set up traces, what evaluators catch, where instrumentation pays off — are packaged here as small, focused examples anyone can copy.

## Author

Giuseppe Spano · Technical Product Manager · [LinkedIn](https://www.linkedin.com/in/giuseppe-spano/) · [GitHub](https://github.com/GiuseppeSp)
