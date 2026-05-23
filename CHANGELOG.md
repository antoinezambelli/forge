# Changelog

All notable changes to forge are documented here.

## [Unreleased]

### Added
- **OpenAI-compatible hosted-provider client (`OpenAICompatClient`).** Adapter for any backend exposing `/v1/chat/completions` with optional bearer auth — covers Cloudflare Workers AI, Fireworks, OpenRouter, OpenAI itself, and similar. Native function calling, SSE streaming, full `LLMClient` protocol surface. Supports `extra_headers` for provider quirks (e.g. OpenRouter's `HTTP-Referer` / `X-Title`) without a per-provider registry. Exported from `forge` and `forge.clients`. #88.
- **Hosted-providers section in [Backend Setup](docs/BACKEND_SETUP.md)** covering bearer-auth setup, the `get_context_length() → None` contract, and the per-model nature of function-calling support on hosted providers.

## [0.7.2] — 2026-05-24

vLLM backend support — serve AWQ/GPTQ and other vLLM-hosted models behind forge's guardrails, in both proxy modes and via `WorkflowRunner`.

### Added
- **vLLM backend (`VLLMClient`).** OpenAI-compatible client for a vLLM server, consuming vLLM's server-side `tool_calls` and `reasoning` (vLLM 0.21) fields. Native function calling only — vLLM parses tools server-side via `--enable-auto-tool-choice --tool-call-parser`, so there is no prompt-injection mode. Exported from `forge` and `forge.clients`.
- **vLLM in managed + external proxy modes.** `--backend vllm --model-path <dir|hf-repo-id>` launches and manages a vLLM server; `--backend-url <url> --backend vllm` proxies an externally managed one. `setup_backend()` / `ServerManager` gain a `model_path` parameter (the vLLM identity, distinct from `gguf_path`).
- **vLLM served-model-name discovery in external mode.** vLLM validates the request `model` field against its `--served-model-name` and 404s on a mismatch (unlike llama.cpp, which ignores the field). The proxy discovers the served name from `/v1/models` instead of sending a placeholder. #74 (thanks @srinathh).
- **vLLM section in [Backend Setup](docs/BACKEND_SETUP.md)** covering the server flags and `VLLMClient` usage.

### Changed
- **Proxy managed mode now delegates to `setup_backend()`** instead of reimplementing the server-start/budget dance, so every managed backend (including vLLM) shares one path. No public API change — `ProxyServer` and the `forge.proxy` CLI keep their v0.7.1 signatures, with `model_path` / `--model-path` and the `vllm` backend added.
- **External mode fails fast when a backend reports no context length** and no `--budget-tokens` is set, instead of silently falling back to an 8192-token budget that could truncate context. Anthropic-protocol downstreams are unaffected.

### Known limitations
- **The vLLM backend is unit-validated but was not exercised against a live vLLM server in this release cycle.** Its client and server-management code carry full unit coverage, and the proxy's protocol translation is verified end-to-end against llama.cpp (the proxy layer is backend-agnostic). `scripts/integration_test_proxy.py --vllm-url <url>` runs the full request battery against a real vLLM server when one is available.

## [0.7.1] — 2026-05-24

Proxy hardening: forge now works with Claude Code. First PyPI release to include the Docker, model-pass-through, and token-usage work that landed on `main` after v0.7.0.

### Added
- **Anthropic Messages API on the proxy (`POST /v1/messages`).** Point Claude Code — or any Anthropic-protocol client — at a forge-guarded model. Two downstream shapes: **Path 2** (default, `--backend-protocol openai`) translates Anthropic ↔ OpenAI for local llama.cpp / Ollama and emits Anthropic SSE back; **Path 1** (`--backend-protocol anthropic`, external mode) forwards to an Anthropic-shape downstream (LiteLLM, the Anthropic API, a self-hosted proxy), passing unknown fields through verbatim. Adds a `base_url` kwarg on `AnthropicClient`. See the new "Using forge with Claude Code" section in the User Guide.
- **`--mode {native,prompt}` proxy flag** — run prompt-injected function-calling through the proxy for OpenAI-compatible backends that lack a native tool-calling template, not just native FC. Closes #53.
- **Real token-usage reporting through the proxy** — responses carry actual prompt/completion counts (previously hardcoded zeros), in both OpenAI (`usage.prompt_tokens/...`) and Anthropic (`usage.input_tokens/output_tokens`) shapes, streaming and non-streaming. #81 (thanks @mhajder).
- **Per-request model-name pass-through for external backends** — the proxy honors the inbound `model` against external OpenAI-compatible backends. #80 (thanks @mhajder).
- **Dockerfile** for running the proxy as a container. #79 (thanks @mhajder).

### Changed
- **`last_usage` unified on slot-keyed `{slot_id: TokenUsage}` across all clients.** `AnthropicClient` previously stored a flat `{input_tokens, output_tokens}` dict; it now uses the slot-0 convention `LlamafileClient` / `OllamaClient` already follow, so usage extraction has one contract.
- **Inbound `model` rides the proxy's passthrough/extras channel** rather than the sampling map — a cleaner replacement for the #80 mechanism that keeps `model` out of `sampling`.

### Fixed
- **Proxy no longer hard-imports the optional `anthropic` SDK at load.** A plain `forge-guardrails` install (without the `[anthropic]` extra) can now start the proxy for local / OpenAI-shape backends; the SDK is imported lazily and only required for `--backend-protocol anthropic`.
- **Proxy router tolerates query strings.** Requests like Claude Code's `POST /v1/messages?beta=true` route correctly instead of returning 404.
- **`eval_runner` token accounting for local backends** — was silently counting zero tokens because it read the flat `last_usage` keys; now reads the slot-keyed `TokenUsage` (fixed by the unification above).

### Known limitations
- **`cache_control` is not preserved on Path 2.** OpenAI Chat Completions has no analog, so prompt-cache hints are dropped when the downstream is a local OpenAI-shape backend. Path 1 (Anthropic-shape downstream) preserves `cache_control` on clean turns. See ADR-015.
- **Prompt-mode multi-turn tool convergence is model-dependent.** Some models reliably consume prompt-injected tool results across turns; others re-call the same tool. Native FC is the more robust default for heavy multi-turn tool use (e.g. Claude Code).

## [0.7.0] — 2026-05-22

### Added
- **Granite 4.1 8B + Gemma-4-E4B + phi-4** — added to the eval lineup. Granite 4.1 mirrors the IBM greedy-decoding convention pending formal published sampling guidance; phi-4 has no formal sampling recommendation and falls through to backend defaults.
- **`_PROMPT_ONLY_MODELS` in `batch_eval`** — skips native FC for models lacking training for the OpenAI `tool_calls` schema (currently: phi-4, verified via curl 2026-05-14).
- **`_NO_RECOMMENDED_SAMPLING_MODELS` in `batch_eval`** — runs `recommended_sampling=False` for models without formal sampling guidance from any official source, so the eval doesn't raise `UnsupportedModelError` on them.
- **`MODEL_REGISTRY.md`** — new doc enumerating every model forge knows about, classified as Current (in v0.7.0 eval), Retired (cut from current eval), or Unpublished (sampling params staged, no published eval). Sampling values, source links, identity-key conventions.
- **Versioned eval datasets** — committed dataset files renamed to `eval_results_vX.Y.Z.jsonl`. Prior versions kept in LFS for reproducibility.
- **`report.py` `--html` + `--markdown` flags surfaced** in README and EVAL_GUIDE examples.

### Changed
- **Step enforcement + prerequisite violations surface on the tool channel.** Previously, `WorkflowRunner` emitted these as trailing `role="user"` nudges after the assistant `tool_call`. v0.7.0 emits one `role="tool"` message per blocked call with `[StepEnforcementError]` / `[PrereqError]` prefixes — the canonical "tool call failed, try again" wire shape OpenAI-tool-trained models are pretrained on. Surfaced by v4 forge-code dogfooding (gpt-oss-120b reliably exhausted prerequisite-violation budget under the old shape).
- **Unknown-tool retry on the tool channel.** Same refactor applied to `ResponseValidator` unknown-tool path: `[UnknownToolError]` tool-error reply instead of a user nudge.
- **Eval lineup refresh** — cut Llama 3.1 8B, Mistral 7B v0.3, Mistral Nemo 12B, Granite 4.0 (h-micro / h-tiny). All scored bare <30% on the v0.6.0 dataset — too weak to be informative, superseded by Ministral-3 / Granite 4.1 / phi-4. Sampling defaults retained in `sampling_defaults.py` for backward compatibility (see MODEL_REGISTRY Retired tier).
- **Eval dataset** — `eval_results_v0.7.0.jsonl` (96,200 rows, 74 cells; rig-01). Apples-to-apples delta on 21 common configs vs v0.6.0: +0.7pt overall, -1.2pt advanced_reasoning — both within CI. Published-leaderboard floor lifts +16.9pt via composition (weak-model cuts).
- **Dashboard + markdown views regenerated** against v0.7.0 dataset. Top of leaderboard reshuffled: Ministral-3 14B Reasoning Q4 LS/N now #1 at 84.5% (was Ministral-3 8B Instruct Q8 LS/P at 86.5% in v0.6.0; now #3 at 84.4%).
- **MODEL_GUIDE rewrite** — trimmed to opinions + rationale (333 → 145 lines). Full leaderboard, OG-18 100% list, hard suite top-5, models-to-avoid tables moved to the dashboard / markdown views. Sampling-parameters and "backend matters" sections retained. Native-vs-prompt heuristic corrected: not workload-driven, sensitivity is per-family.
- **ARCHITECTURE rebuild** — cut signature restating (1701 → 165 lines); the doc now covers design principles, surface modes, guardrail rationale, compaction priority rationale, respond-tool rationale, sampling opt-in semantics. Source is authoritative for class signatures; WORKFLOW.md owns the diagrams; ADRs own past decisions.
- **BACKEND_SETUP rewrite** — cut model-pick prose, Windows-specific install steps, Ollama Modelfile tutorial, llamafile distribution explainer, per-backend "run the eval" subsections, VRAM tables (360 → 135 lines). Per-backend section now: boot command + flag table + curl smoke-test + forge client snippet. Added Anthropic section using `pip install "forge-guardrails[anthropic]"`.
- **README opener** — leads with the contract (any tools, any order; structure opt-in via `required_steps`/`prerequisites`/`terminal_tool`) before the eval pitch. New "What forge isn't" (not an agent orchestrator, not a coding harness) preempts the conflations that surfaced on HN. Three-ways list reordered with proxy first (most popular entry point). Quick Start swapped from Ollama to llama-server.

### Fixed
- **WorkflowRunner docstring + tree** — added missing `retry_nudge` kwarg, `cancel_event` parameter on `run()`, `PREREQUISITE_NUDGE` + `CONTEXT_WARNING` message types, `MaxIterationsError` / `PrerequisiteError` / `StepEnforcementError` / `WorkflowCancelledError` in Raises lists across docs.
- **CompactStrategy + ContextManager signatures in docs** — `trigger_tokens` → `budget_tokens` (the strategy owns its own threshold logic now); `compact_threshold` → `context_thresholds` + `on_context_threshold` callbacks.
- **`LlamafileClient` constructor docs** — added missing sampling kwargs (`top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`), `chat_template_kwargs`, `slot_id`.
- **MODEL_FAMILIES in `report.py`** — added entries for `granite-4.1-8b` (Q4/Q8) and `phi-4-Q4_K_M` so cross-backend rollups in `by-backend.md` group these new models correctly.
- **WORKFLOW.md agentic-loop flowchart** — node names + edges updated to reflect the tool-error wire shape (`STEP_TOOL_ERROR`, `PREREQ_TOOL_ERROR`, `UNKNOWN_TOOL_ERROR`); compaction-priority table fixed (`step_nudge` and `prerequisite_nudge` are `role=tool`, `retry_nudge` remains `role=user`).
- **Stale `bfcl/` reference** removed from WORKFLOW.md module diagram (directory was removed pre-v0.7.0; ADR-009 retained as historical artifact).

### Known limitations
- **Anthropic numbers not re-measured in v0.7.0.** The Anthropic ablation matrix (~$272 to run) was not re-executed for v0.7.0. Numbers cited in any v0.7.0 doc are from the v0.6.0 dataset (`eval_results_v0.6.0.jsonl`). Tool-error-channel changes affect frontier models' wire on guardrail-fire paths too, but expected movement is small.

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
