# Why evals matter (the short version)

Most AI features ship by vibe. A PM tries five queries, the answers look fine, the feature goes live. Three weeks later someone tweaks the system prompt to fix one bug — and silently breaks a different class of queries that no one is actively testing. The team finds out from a user complaint.

Evals are the cheapest insurance against that pattern.

A good eval setup has three things:

1. **A golden dataset** — 20-50 hand-labeled examples covering the patterns you care about. The labels are the ground truth.
2. **A scorer** — anything that produces a pass/fail or numeric quality signal for each row. Can be programmatic (exact match, regex), can be an LLM-as-judge with a rubric.
3. **A regression workflow** — run the dataset before any change, run it after, compare. Two numbers. Did it go up, down, or sideways?

Once you have those three things, the conversation about prompt changes shifts from "I think this is better" to "this raised the answer-quality score from 3.4 to 3.8 with no regression in routing accuracy." That's the moment AI development starts to look like normal software development.

The agents in this repo are designed to be the smallest possible demonstration of each step.
