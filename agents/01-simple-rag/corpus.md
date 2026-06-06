# A Small Encyclopedia of AI Agent Patterns

A short, opinionated reference covering ten foundational patterns used to build modern AI agents. Each entry is self-contained and assumes only basic familiarity with language models.

## Retrieval-Augmented Generation

Retrieval-Augmented Generation, or RAG, is a pattern where an external knowledge source is consulted before the language model produces an answer. The corpus is split into chunks, each chunk is embedded into a vector space, and at query time the most semantically similar chunks are retrieved and concatenated into the prompt as context. The model is then instructed to answer using only the retrieved material. RAG is the dominant pattern for grounding language models in proprietary or up-to-date knowledge without retraining the model itself. Its quality depends heavily on chunk size, retrieval algorithm, and the prompt that frames the retrieved context. RAG is often paired with a relevance gate that decides whether the retrieved chunks are actually useful or whether the model should fall back to its own training knowledge.

## ReAct

ReAct stands for Reasoning and Acting. It is a prompting technique that interleaves chain-of-thought reasoning with tool calls in a single loop. At each step the model writes a short thought about what it should do next, then either calls a tool (such as a search engine or calculator) or produces a final answer. The output of each tool call is appended to the prompt, and the loop continues until the model decides it has enough information. ReAct popularized the idea that agents need both deliberation and action, and most modern tool-using agents are descendants of this pattern. Its main weakness is that the loop can spiral into long, expensive trajectories when the model fails to commit to a final answer.

## Chain-of-Thought

Chain-of-Thought prompting, abbreviated CoT, asks the model to write out intermediate reasoning steps before producing its final answer. The technique was popularized by the observation that simply appending "let's think step by step" to a math or logic prompt dramatically improves accuracy. Chain-of-Thought is a single linear trajectory: the model reasons forward through one path. It is cheap, requires no orchestration, and is built into the default behavior of many modern reasoning models. Its limitations are that the reasoning is non-recoverable once committed and the model cannot easily explore alternative branches when its initial path turns out to be wrong.

## Tree of Thoughts

Tree of Thoughts, or ToT, generalizes Chain-of-Thought by exploring multiple reasoning branches and selecting between them. At each decision point the model generates several candidate next steps, evaluates them (sometimes using the model itself as the evaluator), and continues down the most promising branch. ToT is particularly effective on problems where the right approach is not obvious from the start, such as puzzles and combinatorial search. It is far more expensive than Chain-of-Thought because it generates many more tokens, and orchestrating the search loop requires real engineering, but it can solve problems that linear reasoning cannot.

## Function Calling

Function calling, also called tool use, is the pattern in which a language model produces structured calls to external functions instead of free-form text. The model is given a list of available tools, each with a name, a description, and a JSON schema for its arguments, and it learns to emit calls in the same JSON shape when it determines a tool is needed. Modern function calling is built directly into the API of OpenAI, Anthropic, and Google, so the model itself is trained to produce well-formed tool calls. The pattern's reliability depends heavily on the quality of tool descriptions: a clearer description of when to use a tool produces dramatically fewer wrong calls.

## Agentic Router

An agentic router uses a language model to decide which downstream pipeline should handle a query. A typical router looks at an incoming question and routes it to one of several specialized handlers: a knowledge-base RAG path for questions about saved content, a general-knowledge path for questions the model can answer directly, or a web-search path for questions that need fresh information. The router is itself a small LLM prompt, evaluated against a labeled dataset of question-to-path mappings. Routing quality is the single largest determinant of end-user experience in production AI systems because every wrong route produces a misshaped answer.

## Multi-Agent Orchestration

Multi-agent orchestration is the pattern in which several specialized agents collaborate to complete a task. An orchestrator agent breaks the work into subtasks and delegates each to a worker agent that has been prompted for a specific role, such as a researcher, a writer, or a fact-checker. The worker outputs are passed back to the orchestrator, which may combine them, request revisions, or terminate the loop. The pattern's appeal is modularity: each worker can be improved in isolation. Its weakness is that failures cascade silently across agents and debugging requires careful instrumentation, since the orchestrator may simply accept and concatenate broken intermediate results.

## Reflexion

Reflexion is a self-improvement pattern in which an agent critiques its own output and tries again. After producing an initial answer the agent is prompted to evaluate that answer against a set of criteria, identify specific weaknesses, and generate a revised attempt. The loop continues for a fixed number of iterations or until the self-critique judges the output acceptable. Reflexion is particularly effective on writing and coding tasks where the model has the capability to do better but does not always do so on the first attempt. Its cost is high because each iteration consumes additional tokens, and the critic loop must be carefully prompted to avoid converging on stylistic preferences over substantive improvements.

## Constitutional AI

Constitutional AI is a training and prompting technique developed by Anthropic in which the model is given a written "constitution" — a set of principles describing how it should behave — and is trained to critique and revise its own outputs against those principles. At training time the model generates an initial response, then evaluates that response against the constitution, then produces a revised response that better adheres to the principles. The technique reduces the need for direct human feedback on every example by substituting principle-based self-critique. Constitutional AI is one of the main alternatives to Reinforcement Learning from Human Feedback for shaping model behavior at scale.

## Plan-and-Execute

Plan-and-Execute is a pattern that separates an agent's reasoning into two distinct phases. A planner first reads the user's request and produces a complete, ordered plan of steps to execute, expressed as a sequence of subtasks. An executor then carries out each step in turn, calling tools as needed and tracking intermediate results, without re-deliberating about the high-level strategy. The pattern reduces token consumption compared to looping patterns like ReAct because the planner only deliberates once, and it produces traces that are easier to audit because the plan is visible upfront. Its weakness is brittleness: when the world deviates from the plan, the executor cannot adapt without replanning, and naive implementations can blunder through plans that no longer make sense.
