# Changelog

All notable changes to forge are documented here.

## [0.6.0] — 2026-04-29

### Added
- **Per-model sampling defaults** — `forge.clients.sampling_defaults` ships a verified per-model recommendations map (Qwen3/3.5/3.6, Qwen3-Coder, Gemma 4, Mistral Small 3.2, Devstral Small 2, Ministral 3 Instruct + Reasoning, Mistral Nemo, Granite 4.0). Each row carries an inline HuggingFace card URL; values are verified one entry at a time, no extrapolation. Opt in via `recommended_sampling=True` on `OllamaClient` / `LlamafileClient`. Closes #58, #59, #61.
- **`UnsupportedModelError`** — `recommended_sampling=True` against a model not in the map raises rather than falling through to backend defaults silently.
- **Per-call sampling overrides** — `send()` and `send_stream()` accept a `sampling: dict | None` kwarg that merges field-by-field with the client's instance-level sampling without mutating it. Caller's explicit non-None fields win.
- **Proxy sampling pass-through** — proxy plumbs OpenAI-compatible body fields (`temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `seed`) through to the backend per request, never mutating the proxy's pre-built client. To get card-recommended sampling in proxy mode, the calling client looks up `forge.clients.get_sampling_defaults(model)` and includes the values in the request body.
- **Advanced reasoning eval suite** — 8 new scenarios under the `advanced_reasoning` tag (lambda + stateful pairs): `data_gap_recovery_extended`, `argument_transformation`, `inconsistent_api_recovery`, `grounded_synthesis`. Designed as top-tier separators after the sampling-params fix lifted 8B-class to 100% on the OG-18 suite.
- **Multi-rig eval dataset** — 119,600 rows across 46 configs × 26 scenarios × 2 ablations × 50 runs, consolidated from 4 rigs (rig-00..rig-03). Each row carries a `rig` field for hardware provenance; rig topology in `eval_rigs.json` at repo root.
- **Dashboard Suite scope** — orthogonal to the statefulness scope; slice between `all` / `og18` / `advanced_reasoning` with all aggregates recomputed from per-scenario data.
- **Granite 4.0 support** — `granite-4.0:h-micro-q4_K_M` and `granite-4.0:h-tiny-q4_K_M` in the sampling-defaults map (greedy decoding, T=0, secondary source citing IBM).

### Changed
- **Dropped hardcoded `temperature=0.7`** — `OllamaClient` and `LlamafileClient` no longer ship a hardcoded sampling default. With `recommended_sampling=False` (default), forge sends nothing and the backend's default applies. Caller-supplied kwargs always win.
- **`AnthropicClient`** — no longer sends a hardcoded temperature; the API's own defaults apply (Claude is frontier-optimized; forge uses it as a baseline-comparison tool).
- **Eval scenarios trimmed** — 18 → 26 across the consolidated dataset (18 OG + 8 advanced_reasoning).
- **MODEL_GUIDE** — restructured around three difficulty tiers (mechanical / mid / hard), with hard ≡ advanced_reasoning. Top recommendations updated to reflect 119K-row consolidated dataset; Ministral-3 dominates the 12GB tier, all top-10 configs run on llama-server.
- **`__version__`** — exposed on the `forge` package via `importlib.metadata`.

### Fixed
- Stray `git add .` could pick up rig-local `eval_results_rig*.jsonl` files; explicit LFS pattern keeps them tracked when intentional, ignored otherwise.

## [0.5.0] — 2026-04-19

### Added
- **Ablation study runner** — `scripts/run_ablation.py` runs models × guardrail presets sequentially with retry logic; designed for unattended overnight or travel runs.
- **N=50 ablation rollout** — full ablation study expanded to N=50, generating the IEEE preprint dataset.
- **Three-screen dashboard** — restructured around three audiences:
  - *Reforged* — one row per config ("which model do I run?")
  - *Reforged vs Bare* — paired per config ("how much does forge lift it?")
  - *Full Ablation* — 7 ablation variants per deep-ablated config ("which guardrail is doing the work?")
  - Three-screen split is also structurally necessary: reforged + bare is collected universally, while the full 7-way sweep only exists for one best-backend-per-model config.
- **12GB tier coverage extension** — 8 new 12GB configs (Ministral 8B Instruct Q4/Q8, 8B Reasoning Q8, 14B Reasoning Q4, Llama 3.1 8B Q4/Q8, Mistral 7B Q4/Q8) × 5 presets, N=50.
- **Granite 4.0 support** — `extract_tool_call` accepts OpenAI-style `{"name": ..., "arguments": ...}` keys (Granite emits this wrapped in `<tool_call>` tags). h-micro and h-tiny configs added to GGUF_MAP.
- **Statistical significance script** — `tests/eval/significance.py` computes pooled McNemar's test + Wilson 95% CI per ablation cell, paired on (scenario, run) against the reforged baseline. Intended for paper-table validation.
- **Batch eval timeout** — 300s wall-clock cap per scenario at all 3 call sites; on timeout the run is recorded as `completeness=False, error_type='Timeout'` and the batch keeps moving. 4 timeouts across 40,500 runs in the full study — safety net, not a scoring factor.
- **`--models-dir` CLI flag** on the ablation runner; replaces the hardcoded path.

### Changed
- **Markdown report layout** — split into `reforged/` subdir (all, by-family, by-backend), plus `reforged-vs-bare.md`, `ablation.md` (rewritten as 7-row grouped towers), `native-vs-prompt.md`, `budget.md`. Per-backend files dropped — duplicated dashboard work.
- **Dashboard sample-data fallback removed** — production builds always inject `window.__FORGE_DATA__` via `report.py`; the dev hot-reload path is gone.

### Fixed
- **llama.cpp reasoning budget hang** (issue #54) — builds after April 10 2026 activate an unbounded reasoning budget sampler for Gemma 4, Qwen 3.5, and Ministral Reasoning models, causing silent hangs. Document `--reasoning-budget 0` workaround in BACKEND_SETUP.md and MODEL_GUIDE.md.

### Removed
- **Granite 3.3** configs from GGUF_MAP — native FC is broken on llama.cpp for that version. Granite 4.0 (h-micro, h-tiny) retained.

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
- Eval scenarios consolidated to 22 (18 copy-paste + 4 compaction-chain), pruned of redundant variants

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
