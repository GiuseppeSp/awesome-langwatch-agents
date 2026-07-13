# The Precondition Thread

### What 29 instrumented agents taught me about when agentic patterns actually help

I set out to build a catalog of the famous agent patterns — RAG, ReAct, reflexion, chain-of-thought, tree-of-thoughts, corrective-RAG, multi-hop, tool use, debate, mixture-of-agents — each one small, runnable, and wired into [LangWatch](https://langwatch.ai) with a golden dataset and pass/fail evaluators. The idea was a reference: *here is what each pattern looks like, measured.*

Somewhere around the seventh agent, a through-line appeared that I hadn't planned, and it turned out to be the more interesting result than any single pattern. Every one of these mechanisms is sold as a general upgrade — "add a critic and it self-corrects," "let the agents debate and they converge," "retrieve the tools and the model stops getting confused." Almost none of them are general upgrades. **Each one helps only when a specific, measurable precondition holds — and for a capable model, that precondition often doesn't hold, so the mechanism is pure cost.**

That sentence is the whole thread. This essay is the argument for it, walked through the 29 agents, and then the part I actually care about: the *discipline* it implies, and why observability plus evals is what makes that discipline possible.

---

## The method, in one paragraph

Every agent is a single controlled experiment. One variable changes between two (sometimes three) modes — critic on/off, schema vs free-form, all-tools vs retrieved, one agent vs many. A hand-labeled dataset of ~20 rows runs through both modes. Programmatic or LLM-judge evaluators score every row, and one number — the *discriminator* — says whether the mechanism paid. Every step is a typed LangWatch span, so the trace tree shows not just the score but the *mechanism*: where the retrieval happened, which tool was picked, whether the human was ever consulted, whether the three agents actually disagreed. The framing for each is deliberately adversarial to the hype: *here is the folklore; here is what the measurement says.*

The point of that rigor isn't to debunk. It's that a measured "no" with a named reason is worth more than an assumed "yes." A team that knows *why* a pattern won't help them has learned something; a team that adopts it on reputation has just added cost and a new failure surface.

---

## The thread, tier by tier

### Foundations: the method finds its shape (#1–#6)

The first agents are where the method got sharp, mostly by producing **nulls that teach**. A chunk-size sweep on simple RAG (#1) moved nothing — and the interesting finding was *why* the eval framework, not the chunk size, was the deliverable. Tool-description tuning (#3) was a null in aggregate with a sharp caveat: it only mattered on the borderline cases. The three-agent pipeline (#5) shipped a fact-checker that fired **0 revisions in 15 rows** — because the real failures lived *upstream* of what the verifier could see. That last one seeded the whole thread: a mechanism can be perfectly implemented and still do nothing, because its precondition (an error inside its scope) never occurred.

Agents #5, #6, and #7 turned out to be a verification arc — three different ways checking-your-own-work fails. The pipeline's checker had the wrong *scope*. ReAct (#6) tied on correctness but scored **10 points worse on tool efficiency**, because externalizing its reasoning made the model *more* confident about skipping verification tools — a *commitment* failure. And reflexion (#7) looked like a +7pp win until orthogonal evals revealed it was temperature noise plus a perverse case, with the critic **wrong 60% of the time** — a *calibration* failure. Self-correction is only as good as the self-critic, and nobody measures the critic.

### Reasoning and compute: pay only where the model is unsure (#7–#13)

The reasoning tier is where the precondition became explicit, and it has a single shape: **extra compute or structure pays only where the model has genuine headroom to be wrong.**

- **Chain-of-thought (#9)** lifted accuracy +11pp — but *all of it* was in multi-step arithmetic; the CRT "trick" band was already saturated without it. And self-consistency, sampled five times, changed **0 of 18 rows** — because the samples were 100% unanimous. There was no uncertainty to resolve, so voting on it was a 5× tax for nothing.
- **Tree-of-thoughts (#10)** beat CoT on Game of 24 (+17pp) but at **77× the calls** and far below the reachable ceiling, because its value function was only **64% accurate** — it fatally pruned 52 reachable states. Search is only as good as its evaluator.
- **Constitutional AI (#11)** and **least-to-most (#12)** are the purest nulls: both scored *identically* to the baseline (18/18 and 26/26) because the base model already satisfied the constitution and already decomposed implicitly — even out to depth-14 stress chains. A self-critique loop, or an explicit decomposition, only pays past the point where the base behavior actually breaks. It didn't break.
- **Plan-and-execute (#8)** and **ReWOO (#13)** sharpened it further: replanning helped +14pp on brittle branches but *broke* 2 rows it should have left alone (the adaptation trigger was miscalibrated), and ReWOO's flat 2-call cost was free only when the plan structure was observation-independent — it failed the moment an observed value had to decide the next step.

The recurring lesson: measure the base model's failure rate *first*. The mechanism's ceiling is exactly the headroom you can prove exists.

### Retrieval: the mechanism is fine; the precondition is the whole story (#14–#20)

The retrieval tier is where the thread got its **positive control** — proof the method finds wins, not just nulls.

**Corrective-RAG (#14)** lifted accuracy **40% → 100%** for one extra LLM call. Why did it pay when so much else didn't? Because its precondition — a *calibrated retrieval grader* — demonstrably held: the grader was 15/15, catching all 9 out-of-corpus rows (including 4 look-alike traps) and over-flagging none of the 6 good ones. That's the shape of the entire thread stated positively: *the mechanism pays exactly to the degree its precondition holds, and here it holds, so it pays.*

The rest of the tier is variations on *which* precondition:

- **Query-rewriting (#15)** and **HyDE (#16)** both lifted 79% → 93% — identically — but their precondition is corpus vocabulary. Rewriting is open-loop (it never sees the corpus), so it drifted "sore tooth" → "pediatric dental pain," away from the corpus word "toothache." HyDE's hypothetical answers were real-world hallucinations that only helped when they happened to echo the query's concept. Neither can verify it guessed right.
- **Hybrid + rerank (#17)** resolved both by upgrading the *retriever*: semantic embeddings lifted retrieval-hit to 100% in one step, fixing every vocab-mismatch the query-transform tricks were working around. On a fixed vocabulary problem, a better retriever beats a cleverer query.
- **Multi-hop (#18)** and **graph-RAG (#19)** are the tier's *structural necessity* cases: single-shot RAG scored **0/8** and **0/7** not because it was tuned badly but because the answer passage was *never retrievable* — a structural impossibility. The mechanism here isn't an optimization; it's a prerequisite for a class of question, and a pure no-op (and 3–4× cost) on the questions that don't need it.
- **Structured extraction (#20)** reframed "does it help" entirely: the schema's accuracy edge was modest (90% vs 80%), but free-form *parse success* swung **0% → 80%** purely on parser forgiveness. The value of structured output is guaranteed-consumable structure, not smarter extraction — and a naïve parser will manufacture a fake difference if you let it. (That lesson became a rule for the whole catalog: score both modes through the *same* forgiving parser, or you're measuring your parser.)

### Tools: grounding is a prerequisite, not an optimization (#21–#25)

The tools tier's precondition is starker, because the failure mode is *total*.

- **SQL agent (#21)** without the schema in the prompt scored **0/8 — every query failed at execution**, hallucinating plausible table names that don't exist. Grounding isn't a knob that improves quality; without it the thing doesn't run. With it, 7/8.
- **Code interpreter (#22)** beat in-head reasoning 100% vs 58%, but the wins were a *specific profile*: exact 5-digit multiplication, modulo, letter-counting (a tokenization blind spot), precise stdev. The tell for reaching for a tool is the *kind* of computation, not the presence of a number.
- **Error-recovery (#23)** lifted solved-rate 22% → 67%, but the value rode entirely on errors being *classifiable* — the fix reachable from the error message, unrecoverable failures announcing themselves. It recovered all 4 recoverable failures by copying the fix out of the error, and gave up early on all 3 unrecoverable ones. A vague error defeats retry in both directions.
- **Tool-retrieval (#24)** is a clean folklore-rejection: the "too many tools confuses the model" belief *didn't reproduce* — all-tools and retrieved both scored **20/20** on a 101-tool registry with near-duplicates. Retrieval matched that at **10× fewer prompt tokens**. So it's a cost win, not accuracy insurance — and its one risk is dropping the right tool, making `retrieval_recall` (not selection accuracy) the number that governs whether the savings are free.
- **Human-in-the-loop (#25)** is the tier's sharpest caution: "a human makes it safe" *didn't hold*. Escalation recall was **0.00–0.25** — the three actions that actually hurt (permanently deleting files) came at **0.90–1.00 confidence** and were *never* escalated. HITL's safety ceiling is escalation recall, and self-reported confidence is a low-recall, unstable trigger that is blind, by construction, to the confident errors most worth catching. Gate irreversible actions on *risk*, not the model's mood.

### Multi-agent: mostly cost for a capable model — four times, four reasons (#26–#29)

The multi-agent tier produced four nulls in a row, and the reason I'm proud of it rather than embarrassed is that **each null has a distinct, measured cause.** Together they're a stronger statement than any single win: for a single capable model, orchestrating copies of it is mostly cost, and here is exactly why, each time.

| # | Pattern | Why multi-agent didn't help |
|---|---|---|
| 26 | supervisor-worker | **No load bottleneck to relieve.** Splitting a bundle of N subtasks across workers tied single-pass (108/124 both) with **0 dropped tasks out to 12 simultaneous ones**, at 7.3× the calls. Decomposition relieves attention overload; there wasn't any. |
| 27 | debate | **Accuracy is ensembling, not arguing.** Debate tied self-consistency (20/20 both) at 3× the cost; agents already agreed on 18/20, so the arguing had no surface. Deriving SC from debate's own round 1 isolated the interaction to *zero* net, plus a conformity risk voting doesn't carry. |
| 28 | mixture-of-agents | **Correlated errors (same model).** On a synthesis task built to favor it, unioning 3 same-model drafts recovered <10% of the first draft's misses — because same-model drafts miss the *same* items. Ensembling needs error independence; N copies of one model don't have it. |
| 29 | heterogeneous-moa | **Correlated errors (same vendor).** Testing #28's own fix — *use different models* — three different OpenAI models tied a single gpt-4o call (F1 0.90, winning 0/19). Different models recovered the *same 9%* of the misses that more copies did. Shared training lineage → shared blind spots. |

The tier taught me that "multi-agent" is not a quality lever; it's a lever for *specific bottlenecks* — genuine load, genuine disagreement, genuine error-independence — and if you can't measure the bottleneck, you're paying N× for correlated votes. #28 concluded the fix was model diversity; #29 measured that and found *same-vendor* diversity doesn't supply independence either. The honest open question — genuinely different knowledge sources (other vendors, retrieval, tools) — is the one thing I couldn't test on an OpenAI-only key, and naming that boundary is part of the result.

---

## The patterns behind the patterns

Four habits show up across all 29, and they're the actual transferable skill:

**1. Find the discriminator metric.** For every mechanism there is one number that governs whether it pays, and it is usually *not* the headline accuracy. It's `retrieval_hit` (not answer accuracy) for query transforms; the *grader's* accuracy for CRAG; the *value function's* accuracy for tree-of-thoughts; `escalation_recall` (not task success) for human-in-the-loop; `retrieval_recall` for tool-retrieval; *error independence* for mixture-of-agents. Naming that metric is 80% of the analysis. Once you know what governs the win, you know what to measure before you ship.

**2. A null with a reason beats an assumed win.** Roughly a third of these agents are nulls. Every one names *why*, and that "why" is the deliverable — self-consistency changed nothing because samples were unanimous; constitutional AI because the base model already complied; supervisor-worker because there was no load to split. "It didn't help, and here's the measured reason" is a shippable insight. "It probably helps" is not.

**3. When there's no headroom, stress the variable until it breaks — or prove it doesn't.** Several experiments started with the capable model acing everything (no signal). The move each time was to escalate difficulty deliberately — pad the tool registry 30→100, push decomposition depth to 14, swap memorized riddles for hard combinatorics, grow the recall sets to 50 items — and either find the crossover or honestly report that it never came in a reasonable range. A flat line at 100% is only interesting once you've tried to bend it.

**4. Isolate the mechanism with a middle mode.** The sharpest experiments add a third mode that strips the mechanism to its ensemble. Debate vs *self-consistency-derived-from-its-own-round-1* isolated arguing from voting. Mixture-of-agents vs *union-without-aggregator* isolated synthesis from merging. That controlled subtraction is what let me say "the gain is X, not Y" instead of "it's better."

---

## Why this needed observability, specifically

None of this works from the eval score alone. The score tells you *that* a mode won; the trace tells you *why*, and the "why" is where the precondition lives.

The LangWatch trace tree is where the mechanism becomes visible and falsifiable. You *see* the `retrieve` span hand the model a clean shortlist, and the token metric on the `select` span showing 10× fewer tokens. You *see* the `human_review` span that is simply **absent** on the confidently-wrong deletion — the precondition failing, rendered as a missing node. You *see* debate's three round-spans converge on the answer round 1 already had. You *see* the heterogeneous fan-out — three different models drafting — land on the exact same set a single call produced. The span tree is the difference between "the number went up" and "here is the causal path, and here is the step where it broke."

And the eval layer is what turns folklore into a number you can defend in a review. "I think debate is better" becomes "debate tied self-consistency at 3× the cost, net +0 corrections, and here's the paired metric." That shift — from vibes to a regression you can re-run after every prompt change — is the entire reason the discipline exists, and it's exactly the surface LangWatch is built for.

---

## The takeaway

If there's one thing to carry out of 29 experiments, it's this: **the skill in agent engineering is not choosing patterns; it's identifying and measuring each pattern's precondition before adopting it.** Nearly every agentic technique is real and works — *in the regime it was designed for*. Cargo-culting it into a regime where its precondition doesn't hold buys you cost, latency, and a new failure surface, and the only way to know which regime you're in is to measure.

For a practitioner in 2026, that turns into a short checklist per mechanism:

- **What is this mechanism's precondition?** (A load bottleneck? Model uncertainty? A calibrated sub-model? Error independence? Classifiable failures?)
- **What single metric would tell me it holds?** (Not the headline score — the discriminator.)
- **Does it hold on my task?** Run 20 rows, both modes, and look. If it doesn't, use the simpler baseline and save the N× bill.

That checklist is unglamorous, and it's most of the value. The 29 agents in this repo are 29 worked examples of running it — each one a small, honest answer to "does this actually help, and how would we know?"

---

*Every claim above is backed by a runnable agent, a golden dataset, and LangWatch traces in [the catalog](../README.md). The findings are what the measurements said, including — especially — where they said "no."*
