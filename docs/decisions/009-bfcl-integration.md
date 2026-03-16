# ADR-009: BFCL Integration

**Status:** Implemented (az/bfcl_eval branch, Feb 2026)

## Context

The eval harness and ablation study measure forge's guardrails on our own scenarios. We needed an external, industry-standard benchmark to validate forge's value-add on neutral ground. Berkeley Function Calling Leaderboard (BFCL) v4 is the de facto standard for evaluating LLM function calling — UC Berkeley's Gorilla project, ICML 2025.

## Decision

Run BFCL through forge's prod path — every test case goes through a real `Workflow` + `WorkflowRunner`. No bypass, no thin wrapper. The `SchemaAdapter` builds a dynamic Workflow per test case, and the full forge pipeline handles it.

**Key design choice: bypass the BFCL framework.** BFCL's `BaseHandler` loop is `@final` and manages turns itself. We use the same test data and scoring criteria but run through forge's orchestration (guardrails active throughout). Results are comparable but not identical to leaderboard runs.

### Architecture

- **`schema_adapter.py`** — Translates BFCL's OpenAI-style function JSON schemas into forge `ToolDef` objects on the fly, per test case. Builds a `Workflow` with those tools.
- **`runner.py`** — Single-turn and multi-turn BFCL runner. Single-turn: BFCL function is the terminal tool. Multi-turn: BFCL functions as regular tools, synthetic `done` as terminal, multiple `WorkflowRunner.run()` calls with message history carrying forward via `initial_messages`.
- **`scorer.py`** — Pass/fail scoring against ground truth. Single-turn: AST comparison of tool calls. Multi-turn: end-state comparison on backend instances (attribute-level deep equality).
- **`executors.py`** — BFCL backend execution wrappers. Stateful backend instances where tool calls mutate state.
- **`backend_wiring.py`** — BFCL → forge backend setup (client, server, budget).
- **`batch_runner.py`** — Batch runner across all configs, JSONL output with automatic resume.
- **`bfcl_report.py`** — ASCII table report from BFCL JSONL.

### Categories

11 categories (~2,183 entries): 7 single-turn (`simple_python`, `simple_java`, `simple_javascript`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance`) and 4 multi-turn (`base`, `miss_func`, `miss_param`, `long_context`).

### Multi-turn design

Each turn is a separate `WorkflowRunner.run()` call with conversation history via `initial_messages`. Forge guardrails (nudges, retries, compaction) are active on every turn. After all turns complete, scoring compares final backend state against BFCL ground truth — end state, not per-turn sequence matching.

## Alternatives Considered

- **Extend BFCL's `BaseHandler`** — Would have given leaderboard-compatible results but the `@final` loop prevents forge guardrails from participating. Defeats the purpose.
- **Raw baseline via `OpenAICompletionsHandler`** at Ollama — Considered as Phase 1 but deferred. The forge-through path is what matters for measuring forge's value.

## Consequences

- 11 BFCL categories runnable against any forge backend (Ollama, llama-server, Anthropic)
- `bfcl_results.jsonl` with automatic resume — same pattern as `eval_results.jsonl`
- Parallel FC category works naturally since parallel tool calling was implemented (ADR-005)
- Multi-turn uses the same `initial_messages` pattern documented in the README for consumer multi-turn flows

## References

- Implementation: `tests/eval/bfcl/` (7 modules)
- BFCL repo: https://github.com/ShishirPatil/gorilla
- Reference doc: `ref_docs/BFCL_REFERENCE.md`
- Parallel tool calls: `docs/decisions/005-parallel-tool-calls.md`
- Stateful scenarios (pattern source): `docs/decisions/008-stateful-eval-scenarios.md`
