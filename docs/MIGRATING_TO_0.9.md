# Migrating to Forge 0.9

Forge 0.9.0 is one intentional breaking update to the Proxy contract. Native
`WorkflowRunner`, `setup_backend()`, `ContextManager`, and compaction strategies
remain available; most migration work applies to `forge.proxy` deployments.

The tables below describe pre-0.9 behavior, the 0.9.0+ contract, the
compatibility class, and the required consumer action. Start with the ordinary
deployment changes. Specialized valid configurations follow, and rejected
contradictory or previously ineffective inputs are listed separately.

## Common migrations

Start here for common Proxy deployments. Commands omit unrelated flags for
clarity; Python callers make the equivalent keyword substitution.

| Before 0.9 | Forge 0.9 |
|---|---|
| **Generic external OpenAI, implicit profile**<br>`python -m forge.proxy --backend-url http://localhost:8080` | **No command change.** Omitting `--backend` still selects generic OpenAI compatibility. |
| **Generic external OpenAI, explicit protocol**<br>`python -m forge.proxy --backend-url http://localhost:8080 --backend-protocol openai` | `python -m forge.proxy --backend-url http://localhost:8080 --backend openai`<br>You may also omit `--backend`. |
| **Anthropic-compatible downstream (for example, LiteLLM)**<br>`python -m forge.proxy --backend-url http://litellm:4000 --backend-protocol anthropic` | `python -m forge.proxy --backend-url http://litellm:4000 --backend anthropic`<br>LiteLLM remains responsible for conversion to its target provider. |
| **Forge process liveness**<br>`curl http://localhost:8081/health` | `curl http://localhost:8081/forge/health`<br>`/health` now reports the backend's own readiness response. |
| **Unpinned external vLLM**<br>`python -m forge.proxy --backend-url http://localhost:8000 --backend vllm` | **No command change.** Served-model discovery now occurs on the first inference request rather than at startup. |
| **Pinned external vLLM**<br>`python -m forge.proxy --backend-url http://localhost:8000 --backend vllm --model my-model` | **No command change.** `--model` remains the authoritative wire-model pin; reporting-budget configuration is independent. |
| **External vLLM with the previously documented manual budget mode**<br>`python -m forge.proxy --backend-url http://localhost:8000 --backend vllm --budget-mode manual --budget-tokens 8192` | `python -m forge.proxy --backend-url http://localhost:8000 --backend vllm --budget-tokens 8192`<br>For unmanaged backends, the value is a reporting denominator; Forge does not allocate or compact the backend. |
| **Managed llama-server with extra backend flags**<br>`--extra-flags` used ordinary option parsing, so normal dash-prefixed llama-server arguments could not be expressed reliably. | `python -m forge.proxy --backend llamaserver --gguf model.gguf --port 8081 --extra-flags --jinja -ngl 99`<br>`--extra-flags` now starts the terminal backend argv tail; put every Forge option before it. |

Ordinary managed `llamaserver` and `ollama` startup commands also remain valid.
Even when the command is unchanged, review the health, model-catalog, and
response-identity rows below for observable behavior changes.

## Ordinary Proxy deployments

| Pre-0.9 invocation or behavior | 0.9.0+ invocation or behavior | Class | Consumer action |
|---|---|---|---|
| External Anthropic used `--backend-protocol anthropic`. Explicit OpenAI could use `--backend-protocol openai`. | The public protocol switch is removed. With `--backend-url`, use `--backend anthropic` or `--backend openai`; omitting `--backend` still selects generic OpenAI compatibility. | Breaking | Replace the Anthropic selector; remove an explicit OpenAI protocol selector or use `--backend openai`. There is no deprecated alias. |
| External `--model` was described as one uniform pin. | Generic OpenAI/llama profiles use it only as a fallback when an inbound request omits `model`; the inbound value wins. Anthropic and vLLM profiles use it as a wire-model pin. | Documentation correction | Decide whether callers should route each generic request or rely on the configured fallback. Use a pin only with the specialized profiles. |
| `GET /health` was unconditional Forge-local liveness. | `GET /health` is forwarded backend readiness, including an honest `404` from a backend that does not implement it (notably Ollama). `GET /forge/health` is unauthenticated Forge process liveness. | Breaking + additive | Move container/orchestrator liveness probes to `/forge/health`. Keep `/health` only when backend readiness is intended and the selected backend implements it. |
| `GET /v1/models` returned a synthesized zero/one-model Forge catalog and could mutate private vLLM identity. | `/v1/models` forwards the backend's exact status, body, and content type and has no private identity side effect. | Breaking | Accept the backend's honest catalog shape and errors. Do not depend on a Forge-created one-model list. |
| Response `model` could echo a caller alias or use a fabricated Forge value. | Response `model` is the effective pinned, discovered, configured, or request-routed wire model. | Breaking | Treat response identity as authoritative; update assertions that expect an input alias. |
| Tool-bearing Proxy requests could compact or normalize caller history when a budget threshold was crossed. | Proxy always uses `NoCompact`; it never deletes or compacts caller messages. Preserved llama/OpenAI adapter normalization may merge consecutive visible same-role messages; retry correction messages are still appended. | Breaking | Budget and trim history in the caller or backend. Do not confuse adapter wire normalization with budget-driven compaction. |
| Missing backend context metadata could block startup or first inference. | Unmanaged startup is metadata-side-effect-free, and managed `budget_mode=backend` also continues when its window cannot be discovered. Missing window facts make reporting unavailable; inference continues unless required vLLM identity is missing. Explicit managed `manual`, `forge-full`, and `forge-fast` allocation failures still fail. | Breaking | Remove readiness assumptions based only on optional window metadata. Monitor the backend for overflow and allocation errors; keep handling failures from explicit managed allocation modes. |
| A static credential could trigger unpinned vLLM discovery at startup. | All unpinned unmanaged vLLM identity discovery occurs on the first inference request. Pins dispatch directly. | Breaking | Allow the first unpinned request to pay discovery latency or return an identity error. A later request retries failed discovery. |

## Valid specialized configurations

| Pre-0.9 invocation or behavior | 0.9.0+ invocation or behavior | Class | Consumer action |
|---|---|---|---|
| `/v1/health`, `/models`, and `/props` were local 404s. | These read-only GET paths are forwarded. Exact path/query, a resolved credential when present, backend status/body/content type, recalculated content length, and Forge CORS are preserved; a backend's own 404 remains 404 and transport failures are 502. | Additive | Call these paths through Forge when backend metadata is needed, but do not assume every backend implements every allowed route. Other GETs, management mutations, `/models/sse`, and unknown `/forge/*` remain closed. |
| `GET /forge/usage` was absent. | It returns the one last-completed eligible process-local snapshot, or `204 No Content`. It is not a live meter, history ledger, durable session API, or cross-process store. | Additive | Poll only for observability. Expect 204 initially, after restart, or whenever complete trustworthy reporting is unavailable. |
| No session was attributed to context reporting. | A non-empty, non-whitespace `X-Claude-Code-Session-Id` wins over a top-level string `litellm_session_id` with the same constraint; sources are `claude_code` and `litellm`. Forge generates no identity. A non-empty, non-whitespace `X-Claude-Code-Agent-Id` or `X-Claude-Code-Parent-Agent-Id` makes a subagent request ineligible to replace the snapshot. | Additive | Send a recognized carrier when attribution is wanted. The LiteLLM body value is still forwarded unchanged even when the Claude header wins. |
| Context budgets could be treated as Proxy compaction authority. | Managed `backend`, `manual`, `forge-full`, and `forge-fast` remain allocation modes. In unmanaged mode, positive `budget_tokens` is only the `operator_config` reporting denominator. | Breaking | For managed manual allocation, use `--budget-mode manual --budget-tokens N`. For unmanaged mode, use `--budget-tokens N` only to report a known window. |
| Explicit `backend_port` was ignored for Ollama and unmanaged URLs. | An explicit port replaces only the URL authority port while preserving the normalized prefix/path. A terminal `/v1` is normalized once. | Breaking | Remove accidental port values or set the actual target port. Check prefixed gateway URLs. |
| An unpinned vLLM ID and context window were discovered as one coupled fact. A pin or explicit budget could suppress unrelated work. | Required identity and optional reporting window are independent. An ID without `max_model_len` is enough for inference. A pin settles identity only; a reporting budget settles the denominator only. | Breaking | Pin with `--model` when identity is known. Add `--budget-tokens` when the window is operator-known; neither substitutes for the other. |
| Clean vLLM calls accepted some passthrough fields/raw tools but dropped or rewrote them. | Clean OpenAI-shaped vLLM attempts preserve caller passthrough fields, raw tools, and `tool_choice`, while Forge still owns effective model and streaming behavior. | Breaking (accepted arguments now take effect) | Review fields already sent by callers; they now reach vLLM. Do not describe this as merely additive. |
| `litellm_session_id` could be dropped by Anthropic conversion, vLLM, Ollama, or applicable downstream paths. | It is observed without being consumed and forwarded unchanged on applicable paths. | Breaking where previously dropped | Ensure the downstream accepts the field or stop sending it. |
| CLI `--extra-flags` used `nargs="*"`, which could not naturally carry dash-prefixed backend arguments. | `--extra-flags` consumes the terminal argv remainder. | Breaking syntax | Put every Forge option first, then `--extra-flags` and the exact backend argv tail. Example: `... --port 8081 --extra-flags --reasoning-budget 0 -ngl 99`. |
| The Proxy's synthetic `respond` behavior was sometimes documented as automatic. | `--inject-respond-tool` remains explicit and defaults off. | Preserved contract, documentation correction | Add the flag only when this policy is wanted. Native callers can still use `respond_tool()` directly. |

## Context and operator ownership

`/forge/usage` is published only after a response was fully delivered and
the request has trustworthy final input occupancy, an effective model, and an
exact window. `current_usage_tokens` is final prompt/input occupancy (including
Anthropic cache write/read input); it is not completion-token usage. The
denominator source is one of `operator_config`, `managed_backend`, or
`backend_metadata`.

A failed request or partial delivery retains the previous snapshot. A delivered
eligible request whose reporting is unavailable clears it. Subagent/background
requests do not replace it. Overlapping requests publish in their natural
completion order. Same-model backend windows may reuse the current snapshot;
switching away and back refreshes rather than maintaining a persistent
per-model cache.

In unmanaged mode, Forge is not the backend operator. The operator owns model
allocation and swaps, oversized-context rejection/overflow, availability, and
backend failures. Metadata and `budget_tokens` report; they do not enforce,
compact, recover, supervise, or hot-swap.

## Authentication and deployment boundary

Forge Proxy is a per-operator sidecar. It does not authenticate callers and is
not a centralized multi-tenant authorization gateway. `--backend-api-key` is an
operator convenience for a credential Forge sends to the backend, not
caller authorization. Centralized multi-tenant authentication is unsupported;
put an authenticating gateway in front when that deployment is required.

Forwarded metadata uses the same one-credential rule as inference. Local
`/forge/health` and `/forge/usage` do not authenticate. Metadata forwarding
does not change the trust boundary or make Forge an auth gateway.

## Rejected contradictory or ineffective inputs

These are semver-breaking validation changes, but they were malformed,
contradictory, or had no coherent effect before 0.9.0.

| Pre-0.9 input | 0.9.0+ behavior | Class | Consumer action |
|---|---|---|---|
| Identity fields on a profile that does not own them; unmanaged `gguf`/`model_path`. | Rejected during pure normalization before side effects. | Breaking | Keep only the selected profile's identity field. |
| Unmanaged `budget_mode`. | Rejected because Forge has no allocation authority. | Breaking | Omit it; optionally supply positive `budget_tokens` for reporting only. |
| Managed `budget_tokens` outside manual mode, or missing/nonpositive manual tokens. | Rejected before startup. | Breaking | Pair a positive value only with `budget_mode=manual`. |
| `extra_flags` for Ollama/unmanaged, Ollama process/KV controls (`cache_type_k`, `cache_type_v`, `n_slots`, `kv_unified`), or flags conflicting with Forge-owned identity, port, or context options. | Rejected. | Breaking | Remove ineffective flags or pass them to the actual backend operator. |
| Both `--serialize` and `--no-serialize`; negative retry/tool-error limits. | Rejected by argparse/normalization. | Breaking | Select one serialization policy and use nonnegative limits. |
| Invalid CLI/Python configuration failed late or with mixed exception forms. | Python construction raises `ValueError`; CLI exits 2 through argparse, before side effects. | Breaking | Fix callers/tests to expect early validation. |

## Direct Python and native API changes

| Pre-0.9 API or behavior | 0.9.0+ API or behavior | Class | Consumer action |
|---|---|---|---|
| `ProxyServer` exposed concrete constructor defaults and accepted `backend_protocol`. | Omission may be represented by `None`; `backend_protocol` is removed. | Breaking | Do not introspect old concrete defaults; use `backend=` for the wire/profile selector. |
| `LLMClient` and built-in clients exposed combined `discover_backend_metadata()`. | The combined public method is removed. `get_context_length()` remains; vLLM also retains `get_served_model_name()`. | Breaking | Call the method matching the fact required, or both for both facts. |
| vLLM context lookup could use an arbitrary first catalog entry. | `get_context_length()` exact-matches the configured/adopted model. Direct vLLM passthrough/raw-tool arguments now take effect. | Breaking | Ensure the selected model exists in the catalog and review accepted passthrough. |
| Anthropic context lookup fabricated 200K/8192-style defaults. | A pinned model uses trustworthy exact-model metadata when available; an unpinned/request-routed or unsupported gateway returns `None`. | Breaking | Handle `None` and explicit discovery/auth/transport outcomes. |
| Native initial accounting used only prior response totals; `CompactEvent.tokens_after` could be stale. | Native compaction prefers current backend usage. After compaction invalidates that observation, `CompactEvent.tokens_after` reports a fresh character/4 heuristic until another backend result supplies current usage. | Breaking | Update usage/event assertions to expect a fresh post-compaction heuristic rather than stale pre-compaction usage. |
| `setup_backend()` accepted ineffective Ollama process/KV knobs and nonpositive manual allocations. | Ollama rejects `extra_flags`, `cache_type_k`, `cache_type_v`, `n_slots`, and `kv_unified`; invalid manual allocations are also rejected. Established common-call no-ops remain. | Breaking | Remove those Ollama-only invalid combinations. |

Native `setup_backend()`, `ContextManager`, `WorkflowRunner`, and custom/built-in
compaction remain distinct from Proxy: their compaction behavior is preserved.
The optional Anthropic integration now requires `anthropic>=0.86.0`.
