# ADR-017: Reasoning replay is a bounded policy, default `none`

**Status:** accepted (unreleased)

## Context

Reasoning-capable backends (Ministral Reasoning, Qwen3 thinking, gemma 4, …) return hidden reasoning alongside tool calls. forge captures that reasoning for observability (`MessageType.REASONING`), and historically re-serialized **all** of it into backend-facing history on every later turn — unbounded accumulation, with no way to turn it off.

Two problems motivated bounding this:

- **Convergence.** A proxy non-convergence investigation traced runaway context growth to captured reasoning being replayed back to the backend each turn. Frontier labs practice *scoped* reasoning retention, not replay-everything.
- **Cost.** Replayed reasoning grows the prompt every turn. On long multi-step workflows it competes with real history for the context budget and inflates per-turn token cost.

A serializer reality check sharpened the question: even the legacy behavior was not a faithful 1:1 re-send — `fold_and_serialize` collapses consecutive reasoning blocks (only the one preceding a tool call survives), so only ~29% of generated reasoning reached the wire on real transcripts. "Replay everything" was already an approximation, not a ground truth worth preserving by default.

## Decision

One knob, `reasoning_replay ∈ {"full", "keep-last", "none"}`, shared by `WorkflowRunner` and the proxy (`--reasoning-replay`), **default `"none"`**.

- **`none` (default)** — captured reasoning never enters backend-facing history.
- **`keep-last`** — only the most recent captured reasoning block is replayed.
- **`full`** — legacy behavior; every captured reasoning block is replayed. Pre-knob forge ≡ `full`.

The policy affects **backend-facing serialization only**. Reasoning is still captured, still surfaces in `on_message` and internal history, and still lands in eval transcripts — observability is unchanged.

Proxy response shaping follows the policy: under `keep-last` current reasoning is exposed as `reasoning_content` (so clients that preserve reasoning fields can replay just the latest block); under `full` it rides assistant `content`; under `none` it is omitted. Anthropic-protocol responses emit reasoning text only under `full`; forge does not synthesize signed Anthropic thinking blocks.

## Evidence

The default was chosen from a dedicated re-sweep (the v0.7.5 grid): 14 models × {none, keep-last, full} × {bare, reforged} × {native, prompt}, 50 runs × 26 scenarios per cell, 170k runs total. Scoring treats the **scenario** as the sampling unit (runs cluster hard within scenarios), paired against the v0.7.0 legacy/`full` baseline.

- **`full` reproduces the pre-knob baseline** on all reasoning models (n.s. everywhere) — the knob is a clean superset of legacy behavior; the message-processing refactor did not regress the legacy path.
- **`none` is statistically indistinguishable from legacy overall** (+0.49pp, p=0.17), and in the reforged-only read (−0.35pp, p=0.45). Bounding replay is a free token saving on this suite.
- **`none` edges out `keep-last` overall** (+0.86pp, p=0.007); the two are indistinguishable reforged-only.
- **No robust per-config downside survives multiple-comparison correction.** The closest is the Ministral-14B-Reasoning-Q4 family (reforged-only raw drop ~1.5pp, p≈0.04–0.06, with `none` ≈ `keep-last`) — a family/quantization caveat, not a blocker.
- **Wire-level validation:** `none` → exactly 0 reasoning on the wire across every row; `keep-last` ∈ {0, 1}; per-transcript ordering full ≥ keep-last ≥ none holds by construction.

Full per-config tables: [results/raw/reasoning-replay.md](../results/raw/reasoning-replay.md).

## Consequences

- **Behavioral change for reasoning-capable backends.** Upgraders who want the old behavior pin `--reasoning-replay full` (proxy) or `WorkflowRunner(reasoning_replay="full")`. For non-reasoning/instruct models the knob is inert and nothing changes.
- **Token savings by default.** Backend-facing history stops accumulating reasoning; `full` remains the cost wildcard (context grows with run length).
- **Eval surface.** `reasoning_replay` is part of the eval resume key and a first-class report/dashboard dimension; rows predating the knob count as `full` (that is what they ran).
- **Claude rows are unaffected.** The Anthropic client drops returned thinking blocks rather than capturing them into history, so the knob is request-inert there; carrying thinking across turns natively is deferred pending evidence it moves scores.

## Alternatives considered

- **Default `keep-last`** (the knob's initial default while evidence was pending). A reasonable middle ground — but it measured slightly *below* `none` overall, still pays a replay cost, and busts rolling prompt-cache prefixes (earlier messages re-serialize differently each turn). Rejected once the grid showed `none` is quality-free.
- **Default `full` (legacy).** Preserves bug-for-bug continuity, but it is the most expensive policy, delivers no measured score benefit, and is the very accumulation pathology that motivated the knob.
- **Drop replay entirely (no knob).** Simplest, but unfalsifiable — `full`/`keep-last` exist precisely so the policy stays a measured variable and per-model exceptions (e.g. the Ministral-Q4 caveat) remain one flag away.
