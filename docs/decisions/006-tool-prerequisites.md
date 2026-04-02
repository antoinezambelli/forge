# ADR-006: Tool Prerequisites

**Status:** Accepted and implemented (March 2026)

## Problem

Forge has `required_steps` for "you must call X before finishing," but no way to express "if you call B, you must have called A first." This matters for tool pairs like `read_file`/`edit_file` — the model should read before editing, but not every workflow involves editing. Making `read_file` a required step would break investigation-only workflows.

Small self-hosted models (8B, 14B) routinely skip reads and hallucinate file contents. Even frontier models do this occasionally.

## Decision

**Conditional tool dependencies via `ToolDef.prerequisites`.** Checked at the runner level, enforced via nudge-and-retry (same pattern as step enforcement and unknown-tool nudges).

### Key design choices

1. **Name-only and arg-matched in one pass.** No phased rollout. Strings for name-only (`"read_file"`), dicts for arg-matched (`{"tool": "read_file", "match_arg": "path"}`). Mixed in one list. Workflow author picks what they need.

2. **`executed_tools` replaces `completed_steps`.** Unified `dict[str, list[dict]]` in StepTracker — records tool name + args for each successful execution. Both required-step checking (`tool_name in executed_tools`) and prereq checking (name-only or arg-matched) consume this one structure. Named `executed_tools` because a failed tool *call* isn't a successful *execution*.

3. **Blocked calls: `TOOL_CALL` + `PREREQUISITE_NUDGE`.** The model's attempted call is emitted as a `TOOL_CALL` (the model *did* make the call), paired with a `PREREQUISITE_NUDGE` (not a fake `TOOL_RESULT`). Model sees its attempt + "blocked, not executed." Compaction drops the pair as a unit.

4. **Parallel batch: whole-batch blocking.** Prereqs evaluated against pre-batch state. Any violation in the batch → entire batch blocked, single nudge. No partial execution, no reordering. (Phase 2 explores smarter handling.)

5. **Not in tool schema.** The LLM discovers the constraint via nudge, same as step enforcement. No noise in every prompt.

6. **Separate retry counter.** `consecutive_prereq_violations` is distinct from formatting retries and tool errors. Default `max_prereq_retries = 2`.

### Alternatives considered

- **Required steps only** — unconditional, breaks optional-tool workflows
- **System prompt instructions** — "always read before editing" — models ignore these, especially small ones
- **Include prereqs in tool schema** — adds prompt noise, no clear benefit over nudge-on-violation

## Implementation

See [TOOL_PREREQUISITES.md](TOOL_PREREQUISITES.md) for the full design. Summary:

### API surface

```python
ToolDef(
    name="edit_file",
    prerequisites=["read_file"],                              # name-only
)
ToolDef(
    name="edit_file",
    prerequisites=[{"tool": "read_file", "match_arg": "path"}],  # arg-matched
)
ToolDef(
    name="edit_file",
    prerequisites=["authenticate", {"tool": "read_file", "match_arg": "path"}],  # mixed
)
```

### Touch points (9 components)

| Component | Change | Size |
|-----------|--------|------|
| `ToolDef` | Add `prerequisites` field | Trivial |
| `StepTracker` | Replace `completed_steps` with `executed_tools`, add `check_prerequisites()` | Medium |
| `WorkflowRunner.run()` | Prereq check before execution, pass args to `record()` | Small |
| `nudges.py` | `prerequisite_nudge()` template | Trivial |
| `MessageType` | Add `PREREQUISITE_NUDGE` | Trivial |
| `TieredCompact` | Treat `PREREQUISITE_NUDGE` + blocked call as expendable pair | Trivial |
| `errors.py` | Add `PrerequisiteError` | Trivial |
| `AblationConfig` | Add `prerequisites_enabled`, `no_prereq` preset | Trivial |
| Unit tests | Prereq scenarios + update existing StepTracker tests | Medium |

### What doesn't change

- `Workflow.required_steps` — orthogonal, consumes `executed_tools` instead of `completed_steps`
- Client protocol — prereqs are runner-side, clients unaware
- `ToolSpec` — prereqs not sent to the LLM

## Phase 2: Smart Batch Prereq Handling (Future)

Phase 1 blocks the entire parallel batch on any prereq violation. Two future refinements:

- **Partial execution** — execute satisfied calls, block only unsatisfied ones
- **Auto-reordering** — topological sort within the batch to satisfy prereq chains

Both deferred until eval data shows models frequently batch prereq + dependent tool together.

## Eval scenario

`prerequisite_enforcement`:
- Tools: `read_file`, `edit_file` (prereq: `read_file`), `summarize` (terminal)
- User: "Fix the bug in utils.py"
- Validator: `edit_file` called AND `read_file` called before it
- System prompt does NOT mention the prerequisite — forge enforces it
- Ablation (`no_prereq`): model may edit without reading → hallucinated fix

## Sequencing

Independent of parallel tool calls, BFCL, and ablation study. The `executed_tools` StepTracker refactor could land as a standalone change first.

## References

- Full design doc: [TOOL_PREREQUISITES.md](TOOL_PREREQUISITES.md) (same directory)
- README roadmap item 19
- Related: [ADR-005 Parallel Tool Calls](005-parallel-tool-calls.md) (batch interaction)
