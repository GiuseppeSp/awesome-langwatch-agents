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
| 4 | [agentic-router](agents/04-agentic-router/) | LLM-as-router with 5-way classification | ✅ Shipped — tuned prompt +30 points on ambiguous rows; all wins from one specific named failure mode |
| 5 | [multi-agent-pipeline](agents/05-multi-agent-pipeline/) | Planner → writer → optional fact-checker (with revision) | ✅ Shipped — fact-checker fired 0/15 revisions; real failures lived upstream in the planner, invisible to the verifier's scope |
| 6 | [react-agent](agents/06-react-agent/) | ReAct loop (Thought → Action → Observation) over a multi-hop tool dataset | ✅ Shipped — tie on correctness but react is 10pts WORSE on tool efficiency: externalized reasoning made the model more confident skipping verification tools |
| 7 | [reflexion-loop](agents/07-reflexion-loop/) | Self-critique + retry on constraint-satisfaction tasks | ✅ Shipped — aggregate looks like reflexion +7pp wins but orthogonal evals reveal it's temperature noise + a perverse case; critic wrong 60% of the time; completes the three-agent verification arc |
| 8 | [plan-and-execute](agents/08-plan-and-execute/) | Plan the whole task upfront, then execute (static vs replan) | ✅ Shipped — replanning lifts correctness +14pp on brittle branch rows, but the paired metric shows it's H1 and H3 at once: it fixed 4 rows and broke 2, at ~2x the cost — an adaptation trigger is only as good as its calibration |
| 9 | [chain-of-thought](agents/09-chain-of-thought/) | direct vs CoT vs self-consistency over a mixed-reasoning set | ✅ Shipped — CoT +11pp but all of it in multi-step arithmetic (the CRT "trick" band was already saturated in direct); self-consistency was a pure 5× tax, 0/18 rows changed because the samples were 100% unanimous — extra compute only pays where the model is genuinely uncertain |
| 10 | [tree-of-thoughts](agents/10-tree-of-thoughts/) | Beam search over partial solutions vs a single chain (Game of 24) | ✅ Shipped — ToT beats CoT (42% vs 25%, +17pp) but at 77× the calls and far below the 12/12 ceiling: the value function is only 64% accurate, fatally pruning 52 reachable states — search is only as good as its evaluator |
| 11 | [constitutional-ai](agents/11-constitutional-ai/) | Answer, then critique-and-revise against a written constitution (direct vs constitutional) | ✅ Shipped — both modes 18/18 appropriate: gpt-4o-mini already satisfies the constitution, so the loop is +130% cost for 0pp change (revision_lift=0.50); the over-refusal never fired, and the critic's only activity was 6/6 false positives on already-refused prompts — a self-critique loop only pays when there's measured headroom |
| 12 | [least-to-most](agents/12-least-to-most/) | Explicit decompose-then-solve-sequentially vs a single chain (dependency-chain arithmetic, depth 2→14) | ✅ Shipped — both modes 26/26 correct, even on the depth-14 stress chains: a single chain already decomposes implicitly, so explicit decomposition is a ~7.6× cost no-op (ltm_lift=0.50) that also over-decomposes — the crossover never comes because single-pass reasoning never drops a sub-result; explicit decomposition only pays past the depth where implicit decomposition breaks |
| 13 | [rewoo](agents/13-rewoo/) | Plan all tool calls blind → execute → solve (2 LLM calls) vs interleaved ReAct (multi-hop tool use over a fictional KB) | ✅ Shipped — ReWOO matches ReAct on fixed-structure multi-hop (12/12) at half the LLM calls (2.0 vs 3.9; 2× fewer overall), and even absorbs threshold branches by over-gathering both outcomes for the solver to pick (4/4) — but fails comparison branches where an observed value decides which entity to look up next (0/2 → UNKNOWN); the flat 2-call cost is free only when the plan structure is observation-independent |

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
