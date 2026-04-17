# Changelog

All notable changes to forge are documented here.

## [0.4.3] — 2026-04-17

### Added
- **Qwen Coder XML rescue parsing** — rescue_tool_call now recognizes `<function=name><parameter=key>value</parameter></function>` format emitted by Qwen3-Coder and similar models (issue #55). Regex patterns adapted from Qwen's reference parser.

## [0.4.2] — 2026-04-10

### Added
- **28-model eval dataset** — 137K rows across Gemma4, Qwen3.5, Devstral, Mistral Small 3.2, Claude Opus/Sonnet 4.6, and more
- **32GB eval tier** — dual RTX 5070 Ti results; Gemma4 31B and Qwen3.5 27B hit 100% self-hosted
- **Git LFS tracking** — eval_results.jsonl tracked via LFS for cross-rig sharing
- **forge-proxy CLI** — `forge-proxy` entry point in `[project.scripts]`
- **codecov integration** — CI coverage reporting and badge

### Changed
- MODEL_GUIDE rewritten for 28-model dataset with 12GB and 32GB VRAM tiers
- Removed BFCL benchmark (historical reference in EVAL_GUIDE, last commit: a9b0257)
- Eval scenarios trimmed from 22 to 22 (removed 7 redundant, kept 18 copy-paste + 4 chain)

### Fixed
- Server readiness: `_wait_healthy()` polls `/props` instead of `/health` to eliminate 503 race on startup (v0.4.1)
- Null byte corruption in eval_results.jsonl from interrupted write

## [0.4.0] — 2026-04-02

### Added
- **SlotWorker** — priority-queued shared slot access for multi-slot llama-server configurations
- **Tool prerequisites** — conditional tool dependencies (`ToolDef.prerequisites`) with arg-matched enforcement
- **Workflow cancellation** — `cancel_event` parameter on `WorkflowRunner.run()` with `WorkflowCancelledError`
- **Multiple terminal tools** — `Workflow.terminal_tools` accepts a set of tool names
- **Custom retry nudges** — `WorkflowRunner` and `Guardrails` accept caller-provided nudge text
- **KV unified support** — `--kv-unified` flag passthrough, FORGE_FAST multi-slot budget fix
- **Compaction chain eval** — 4-scenario degradation curve (baseline/P1/P2/P3) for 10-step dependency chains
- **Proxy unit tests** — 54 tests covering handler, convert, and server modules
- **Real token counting** — backends report actual token usage for compaction decisions

### Changed
- Removed `trust_text_intent` and `TextResponse.intentional` — respond tool pattern supersedes
- Eval scenarios trimmed from 29 to 22 (removed redundant compaction variants)
- Long-running session advisory added to User Guide (transient message filtering)

### Fixed
- FORGE_FAST double-divides context when `n_slots > 1`
- Replaced `FORGE_MODELS_DIR` env var with `--models-dir` CLI arg

## [0.3.0] — 2026-03-12

### Added
- **Proxy server** — OpenAI-compatible drop-in proxy with automatic respond tool injection
- **Guardrails middleware** — composable middleware for foreign orchestration loops
- **Anthropic client** — frontier baseline backend
- **Eval harness** — 22 scenarios, batch runner, BFCL benchmark integration
- **Context thresholds** — configurable warning callbacks at budget percentages
- **TieredCompact** — three-phase compaction strategy (truncate → drop results → sliding window)

### Changed
- Context management rewritten with VRAM-aware budget resolution via `setup_backend()`

## [0.2.0] — 2026-02-15

### Added
- **WorkflowRunner** — agentic tool-calling loop with retry logic
- **ResponseValidator** — rescue parsing for malformed tool calls
- **StepEnforcer** — required step and terminal tool enforcement
- **OllamaClient** and **LlamafileClient** — local model backends
- **ServerManager** — automatic llama-server lifecycle management

## [0.1.0] — 2026-01-20

- Initial release — core framework with tool-calling loop, basic guardrails, Ollama backend
