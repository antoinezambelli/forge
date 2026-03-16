# ADR-002: Anthropic Baseline Client

**Status:** Implemented (az/ablation branch, Feb 2026)

## Context

The eval harness measured self-hosted models against each other, but had no frontier reference point. We needed to answer: how do local models + forge compare to Claude on identical tool-calling tasks?

## Decision

Add `AnthropicClient` implementing the existing `LLMClient` protocol. Use `api_format = "openai"` — the runner serializes OpenAI-style messages, and the client converts to Anthropic format internally. This avoids a third serialization path in the runner.

**Key format conversion:** System message extraction to `system=` kwarg, `tool_calls` to `tool_use` content blocks, `role="tool"` to `tool_result` user blocks, unpaired `tool_use` (from step/unknown-tool nudges) get synthetic error `tool_result` blocks injected, consecutive same-role messages merged.

**Compaction scenarios skipped** — Claude has 200K context; our eval budgets are ~4-8K, so compaction never triggers.

## Alternatives Considered

- **`api_format = "anthropic"`** with a new serialization path in the runner. Rejected — adds a third format branch to `to_api_dict()` for one client. The conversion cost in the client is small and contained.
- **Anthropic Batch API** (50% cost reduction). Not usable — our eval loop is iterative (send → get response → execute tool → send again), so each API call depends on the previous response.

## Consequences

- `AnthropicClient` in `src/forge/clients/anthropic.py` (~200 lines)
- Dependency: `anthropic` SDK (dev extra)
- Haiku/Sonnet eval results provide the frontier ceiling in the README table
- Cost: ~$4.50/50 runs (Haiku), ~$13.50 (Sonnet)
- Cross-reference with ablation: Claude bare vs Claude + forge quantifies guardrail value at the frontier

## References

- Full design doc: git history, `dev_docs/ANTHROPIC_BASELINE.md` (pre-cleanup)
- Implementation: `src/forge/clients/anthropic.py`
- Unit tests: `tests/unit/test_anthropic_client.py` (18 tests for format conversion)
