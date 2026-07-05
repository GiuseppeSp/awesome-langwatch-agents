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
| 14 | [corrective-rag](agents/14-corrective-rag/) | Grade retrieval, web-correct if weak (CRAG) vs plain RAG (fictional corpus) | ✅ Shipped — CRAG lifts accuracy 40% → 100% (+60pp, 9 helped / 0 hurt) for one extra LLM call, because its retrieval grader was 15/15 calibrated: it caught all 9 out-of-corpus rows (incl. 4 look-alike traps) and over-flagged none of the 6 in-corpus rows. The thread's positive control — the first mechanism whose precondition (a calibrated grader) demonstrably holds, which is exactly why it pays off |
| 15 | [query-rewriting-rag](agents/15-query-rewriting-rag/) | Rewrite the query into a better search key before retrieving vs plain RAG (fictional corpus) | ✅ Shipped — rewriting lifts accuracy 79% → 93% purely through retrieval (answer_correctness == retrieval_hit), with zero drift on clean queries (7/7 both); it fixed 2 of the 3 genuinely-missed poor queries but drifted on the third ("sore tooth" → "pediatric dental pain", not "toothache") — an open-loop, corpus-blind fix that only helps when the rewrite guesses the corpus's vocabulary. The before-retrieval complement to #14's after-retrieval CRAG |
| 16 | [hyde](agents/16-hyde/) | Retrieve with a hypothetical LLM-written answer instead of the query vs plain RAG (same corpus/queries as #15) | ✅ Shipped — HyDE scores 79% → 93%, *identical* to #15 query-rewriting (same +2 rows, same miss), but the traces show why it's riskier: on a fictional (out-of-distribution) corpus its hypothetical answers are real-world hallucinations (meclizine, acetaminophen) that only help when they incidentally echo the query's concept. HyDE's famous edge needs an in-distribution corpus — absent here it collapses to "query-rewriting with extra hallucinated text" |
| 17 | [hybrid-retrieval-rerank](agents/17-hybrid-retrieval-rerank/) | lexical vs hybrid (keyword+semantic) vs +LLM rerank, same corpus/queries as #15/#16 | ✅ Shipped — staircase 79% → 93% → 100%: semantic retrieval lifts retrieval_hit to 100%, fixing every vocab-mismatch miss that #15 needed a query-rewrite and #16 a hypothetical doc for (one embedding, no drift); reranking recovers the last row (the sore-tooth query both #15/#16 failed) by disambiguating a "sore throat" distractor the embedding ranked as a neighbor. The classic rerank "promote a buried doc" job never fired (candidate_recall == retrieval_hit) — at 8 passages nothing is buried, so rerank earned its keep on precision, not recall |
| 18 | [multi-hop-rag](agents/18-multi-hop-rag/) | single-shot RAG vs an iterative self-ask retrieve-reason loop on chained questions | ✅ Shipped — 38% → 92%: single-shot scores 0/8 on multi-hop questions because the answer passage shares no keyword with the question and is *never retrieved* (answer_retrieved == answer_correctness) — a structural impossibility, not a tuning gap; the loop recovers 7/8 by discovering each bridge entity and re-querying (+54pp, 0 hurt) at 3× the LLM calls. Clean no-op on single-hop rows (5/5 both); its one miss is a depth-3 chain that re-runs correctly — iterative retrieval is required where chaining is, wasted where it isn't, and frays with depth |
| 19 | [graph-rag](agents/19-graph-rag/) | flat keyword RAG vs traversing a knowledge graph (subgraph / whole-graph gather) | ✅ Shipped — 36% → 91%: flat RAG is *structurally* locked out of multi-hop (0/3) and global/aggregate (0/4) questions — top-k can't hold the support, so evidence_complete is just 36%; graph traversal gathers the connected subgraph (evidence_complete 100%), fixing all multi-hop and 3/4 global with 0 regressions. Its one miss is a 4-level aggregate with complete evidence but a wrong count — traversal solves retrieval, deep aggregation is then a reasoning problem. No-op on local lookups (4/4 both, 4× the context) |
| 20 | [structured-extraction](agents/20-structured-extraction/) | free-form extraction vs a typed JSON-mode schema (extract fields from bios) | ✅ Shipped — schema's accuracy edge is modest (90% vs 80% record-correct; the one gap is free-form *summing* distractor numbers, 14 vs 9). The real win is reliability: free-form parse success swung 0% → 80% purely on parser forgiveness (the model wrote "Full Name", a naïve parser dropped everything), while JSON mode is guaranteed parseable by construction. Neither mode fabricated on absent fields (0/4) given a "null if not stated" rule — structured output's value is consumable structure, not intelligence, and fabrication is a prompt issue not an inherent one |
| 21 | [sql-agent](agents/21-sql-agent/) | text-to-SQL with vs without the schema in the prompt, executed against SQLite | ✅ Shipped — opens Tier 4 (tools). blind (no schema) scores **0/8, failing every query at execution** — it hallucinates plausible table/column names (`adventurers`, `name`, `reward`) and the SQL won't run; grounded runs 8/8 and answers 7/8. Schema grounding is a prerequisite, not an optimization (the failure without it is total, at execution). The one residual miss ran fine but returned an extra column — a query-shape error, showing that once identifiers are grounded, query semantics is the separate remaining problem |
| 22 | [code-interpreter](agents/22-code-interpreter/) | reason it out vs write-and-run Python, on computational questions | ✅ Shipped — code scores 100% vs reason's 58% (lift 0.71, 0 hurt, 0 buggy programs), but reason wasn't hopeless (4/9 compute — it did powers, dates, compound interest, medians in-head). Code's wins were a specific profile: exact 5-digit multiplication (7222029041 vs 7230130061), modulo, letter counting (a structural tokenization blind spot — the model doesn't see characters), precise stdev. The tell for reaching for an interpreter is the *kind* of computation (exact / character-level), not the presence of a number; overhead on trivial arithmetic |
| 23 | [error-recovery](agents/23-error-recovery/) | give up on tool error vs retry-and-repair (feed the error back), over classifiable failures | ✅ Shipped — retry lifts solved 22% → 67% and outcome-correct 56% → 100%: it recovers all 4 recoverable failures (0/4 → 4/4) by copying the fix out of the error message, with 0 regressions — *and* it's disciplined, giving up early on all 3 unrecoverable tasks (1 attempt, not the max of 4), keeping mean attempts at 1.4. The value rides on errors being classifiable/actionable (the fix reachable from the message, unrecoverable failures announcing themselves) — not on a cleverer retry policy |
| 24 | [tool-retrieval](agents/24-tool-retrieval/) | put all 101 tools in the prompt vs semantically retrieve a top-k shortlist first | ✅ Shipped — the "too many tools confuses the model" folklore didn't reproduce: `all_tools` and `retrieved` **both score 20/20** on a 101-tool registry with near-duplicates (pushed 30→100 hunting for degradation — it isn't there for this model). Retrieval matched that accuracy at **10× fewer prompt tokens** (140 vs 1,415/query) with perfect recall (20/20). So tool retrieval is a cost/latency win, not accuracy insurance — and its only risk is dropping the gold tool from the shortlist, making `retrieval_recall` (not selection) the metric that governs whether the savings are free |

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
