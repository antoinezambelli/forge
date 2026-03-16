# ADR-001: Ablation Framework

**Status:** Implemented (az/ablation branch, Feb 2026)

## Context

Forge provides six guardrail layers on top of the raw agentic loop: rescue, retry nudge, unknown tool nudge, step enforcement, tool error recovery, and context compaction. We needed to quantify the impact of each guardrail — both for development (which guardrails matter most?) and for external credibility (does forge actually help?).

## Decision

Selective guardrail disabling via `AblationConfig` presets, run through the same eval harness as normal benchmarks.

**7 presets:** `full` (baseline), `no_rescue`, `no_nudge` (implies no_rescue), `no_steps`, `no_recovery`, `no_compact`, `bare` (all off).

**Implementation approach:** Map each preset to existing runner parameters (`rescue_enabled`, `max_retries_per_step=0`, `max_tool_errors=0`, `NoCompact()`, `required_steps=[]`). Only one new parameter added to WorkflowRunner: `rescue_enabled: bool`.

**Compaction scenarios** (`compaction_stress`, `phase2_compaction`) are skipped for any preset that disables compaction — they fail by definition without it. Excluded from averaging to avoid inflating the "bare is bad" signal artificially.

## Consequences

- JSONL records include an `ablation` field; existing records without it are treated as `full`
- `--ablation` CLI flag on both `eval_runner` and `batch_eval`
- Key finding: Haiku bare drops from 100% to 43% (completeness), recovers to 100% with full forge. Sonnet bare drops to 89%. Confirms forge adds value even at the frontier.
- `bare+any` variant (bare + `tool_choice: "any"`) tested on Claude models — Haiku recovers to 89%, showing `tool_choice` helps but doesn't replace structural guardrails

## References

- Full design doc: git history (pre-cleanup `dev_docs/ABLATION_STUDY.md`, since deleted)
- Implementation: `tests/eval/ablation.py`, `src/forge/core/runner.py` (`rescue_enabled`)
- Results: README.md eval table (bare/bare+any rows)
