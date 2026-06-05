# shared/

Cross-agent utilities — kept small and dependency-free.

Planned contents:

- `langwatch_setup.py` — single `setup_tracing(service_name)` function that calls `setupObservability` + returns a tracer, with `forceFlush()` wired correctly for both serverless and long-running contexts
- `openai_client.py` — thin wrapper around the OpenAI SDK that's pre-instrumented, so every chat completion automatically lands as a typed LLM span
- `eval_helpers.py` — tiny helpers for the patterns that repeat across agents (string-equality evaluator, LLM-as-judge rubric runner)

Nothing here is a framework. Each utility is ~50 lines and copy-pasteable into your own project.
