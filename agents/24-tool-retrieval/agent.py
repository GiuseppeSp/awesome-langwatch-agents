"""
Tool-retrieval agent — with a large tool registry, does the model pick the right
tool better when it sees ALL tools, or a RETRIEVED shortlist?

Switch via env var:
    TR_MODE=all_tools   (default) — put all 30 tool descriptions in the prompt,
                                    let the model pick one and call it.
    TR_MODE=retrieved             — first retrieve the top-k tools relevant to the
                                    query (keyword overlap over descriptions), put
                                    ONLY those in the prompt, then pick and call.

Both make one selection LLM call and one (mock) tool call; the only difference is
how many tools the model has to choose among. The registry has near-duplicate
clusters (currency/units/timezone converters, current/forecast/air weather, ...),
so choosing from all 30 means disambiguating close neighbors. The experiment
measures whether pre-filtering the tool set improves selection — and its risk:
retrieval can drop the correct tool from the shortlist entirely.

Trace tree (typed LangWatch spans):

    tool_retrieval (workflow root)
    ├─ all_tools:  select (llm, 30 tools) -> call (tool)
    └─ retrieved:  retrieve (rag, top-k) -> select (llm, k tools) -> call (tool)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

import langwatch
from registry import TOOLS, call_tool

load_dotenv()

# ---- Configuration ----

TR_MODE = os.getenv("TR_MODE", "all_tools")  # "all_tools" | "retrieved"
MODEL = os.getenv("MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
TOP_K = int(os.getenv("TOP_K", "5"))

os.environ.setdefault("OTEL_SERVICE_NAME", "tool-retrieval")
langwatch.setup(api_key=os.environ["LANGWATCH_API_KEY"])

client = OpenAI()


# ---- Tool retrieval (semantic: embed each tool's name+description, cosine to the query) ----
#
# Real tool-retrieval systems embed the tool descriptions — keyword overlap fails on
# vocabulary mismatch ("will it rain" vs "weather forecast"), which is #15's lesson,
# not this agent's. Embeddings isolate the actual question: does pre-filtering help?

_TOOL_VECS: dict[str, list[float]] | None = None


def _embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _tool_vecs() -> dict[str, list[float]]:
    global _TOOL_VECS
    if _TOOL_VECS is None:
        names = list(TOOLS)
        vecs = _embed([f"{n}: {TOOLS[n]}" for n in names])
        _TOOL_VECS = dict(zip(names, vecs))
    return _TOOL_VECS


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def retrieve_tools(query: str, k: int = TOP_K) -> list[str]:
    qv = _embed([query])[0]
    scored = [(_cosine(qv, v), n) for n, v in _tool_vecs().items()]
    scored.sort(key=lambda x: -x[0])
    return [n for _, n in scored[:k]]


# ---- LLM selection ----

_SYSTEM = (
    "You select the single best tool to answer the user's request, then call it.\n"
    "Respond with ONE JSON object: {\"tool\": \"<exact tool name>\"}.\n"
    "Choose the tool whose description most precisely matches the request."
)


def _tool_menu(names: list[str]) -> str:
    return "\n".join(f"- {n}: {TOOLS[n]}" for n in names)


def _chat(system: str, user: str) -> tuple[str, int, int]:
    c = client.chat.completions.create(
        model=MODEL, temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (c.choices[0].message.content or "").strip(), c.usage.prompt_tokens, c.usage.completion_tokens


# ---- Result type ----

@dataclass
class TRResult:
    query: str
    mode: str
    chosen_tool: str = ""
    result: str = ""
    shortlist: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    llm_calls: int = 0

    @property
    def in_shortlist(self) -> bool:
        return True  # placeholder; run_eval checks correct-tool recall


# ---- The agent ----

@langwatch.trace(name="tool_retrieval")
def run(query: str, *, mode: str = TR_MODE) -> TRResult:
    root = langwatch.get_current_trace().root_span
    root.update(input=f"[mode={mode}] {query}")
    result = TRResult(query=query, mode=mode)

    if mode not in ("all_tools", "retrieved"):
        raise ValueError(f"unknown mode: {mode!r}")

    if mode == "retrieved":
        with langwatch.span(name="retrieve", type="rag") as s:
            s.update(input=query)
            names = retrieve_tools(query)
            s.update(output=", ".join(names))
        result.shortlist = names
    else:
        result.shortlist = list(TOOLS)

    with langwatch.span(name="select", type="llm") as s:
        user = f"REQUEST: {query}\n\nAVAILABLE TOOLS:\n{_tool_menu(result.shortlist)}"
        s.update(input=f"{query} | {len(result.shortlist)} tools")
        raw, pt, ct = _chat(_SYSTEM, user)
        s.update(output=raw[:200], metrics={"prompt_tokens": pt, "completion_tokens": ct})
    result.llm_calls += 1
    result.prompt_tokens = pt
    try:
        result.chosen_tool = json.loads(raw).get("tool", "").strip()
    except json.JSONDecodeError:
        result.chosen_tool = raw.strip().strip('"')

    with langwatch.span(name="call", type="tool") as s:
        s.update(input=result.chosen_tool)
        result.result = call_tool(result.chosen_tool)
        s.update(output=result.result)

    root.update(output=f"[{mode} tool={result.chosen_tool}] {result.result}"[:180])
    return result


# ---- CLI ----

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What's the weather like in Oslo right now?"
    mode = os.getenv("TR_MODE", "all_tools")
    print(f"\n=== Query: {q}\n=== Mode: {mode}\n")
    res = run(q, mode=mode)
    if mode == "retrieved":
        print(f"Retrieved shortlist ({len(res.shortlist)}): {res.shortlist}")
    print(f"Chosen tool: {res.chosen_tool}")
    print(f"Result: {res.result}")
    print(f"Prompt tokens: {res.prompt_tokens}")
