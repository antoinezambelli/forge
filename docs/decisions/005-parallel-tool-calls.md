# ADR-005: Parallel Tool Calling

**Status:** Done (branch `az/parallel_tools`, commit `cd2bd69`)

## Problem

All three clients previously discarded parallel tool calls. When Claude (or any model) returned multiple `tool_use` blocks in one response, forge kept the first and silently dropped the rest. This cost an extra iteration per dropped call, inflated wasted-call metrics, and blocked BFCL parallel FC evaluation.

## Decision

**Passthrough — no guardrails.** Stop discarding parallel calls, make `list[ToolCall]` the universal response type. If a model emits N valid tool calls, forge executes N tool calls.

This is consistent with forge's actual guardrail philosophy. Forge's guardrails are about **structural correctness**, not intent validation:

- **Rescue/retry** — "you didn't produce a valid tool call" (format correction)
- **Unknown tool nudge** — "that tool doesn't exist" (format correction)
- **Step enforcement** — "you haven't completed required steps before terminating" (completeness gate)
- **Error recovery** — "that tool threw an exception" (execution feedback)

None of these validate *which* tool the model chose, *how many times* it calls a tool, or *in what order*. A model can call `get_weather` three times sequentially with the same args — forge doesn't care. Parallel tool calling is the same: if the model batches valid calls, execute them.

### Alternatives rejected

The original RFC explored enforcement guardrails (count declaration, fan-out planning). These are rejected, not deferred:

- **Static fan-out** (`min_calls={"get_weather": 3}`) — narrow use case, adds workflow-level configuration for a problem forge doesn't own
- **Dynamic count declaration** (extra LLM round-trip to declare intent) — adds latency and complexity to solve a model quality problem, not a framework problem
- **Full planning phase** — even more surface area, no improvement over count declaration

The "model forgot the second call" failure mode is a model quality problem. Forge already has `required_steps` for "you must call X before terminating." Enforcing *how many times* a tool is called is a different class of constraint that doesn't fit forge's structural correction model.

## Implementation

### Core type change

```python
# Before
type LLMResponse = ToolCall | TextResponse

# After
type LLMResponse = list[ToolCall] | TextResponse
```

### Message dataclass

Replaced three singular fields (`tool_call_name`, `tool_call_args`, `tool_call_id`) with a `ToolCallInfo` frozen dataclass and a list field:

```python
@dataclass(frozen=True)
class ToolCallInfo:
    name: str
    args: dict[str, Any]
    call_id: str

@dataclass
class Message:
    ...
    tool_calls: list[ToolCallInfo] | None = None  # 1+ entries for assistant tool-call messages
```

`to_api_dict()` emits the full `tool_calls` list (1+ entries) in both Ollama and OpenAI formats.

### Components changed

| Component | Change |
|-----------|--------|
| `LLMResponse` type (`workflow.py`) | `ToolCall` → `list[ToolCall]` |
| `OllamaClient.send()` + `_iter_stream()` | Return all tool_calls, not just `[0]` |
| `LlamafileClient._send_native()` + `send_stream()` | Return all tool_calls |
| `AnthropicClient._parse_response()` + `send_stream()` | Collect all tool_use blocks |
| `Message` dataclass (`messages.py`) | `tool_calls: list[ToolCallInfo] \| None` replaces 3 singular fields |
| `WorkflowRunner.run()` (`runner.py`) | Batch validate, batch execute, emit 1 TOOL_CALL + N TOOL_RESULT messages |
| `extract_tool_call()` (`templates.py`) | Returns `list[ToolCall]` (all matches, not first-only) |
| `rescue_tool_call()` (`templates.py`) | Returns `list[ToolCall]` (empty list on failure, not `None`) |
| `StreamChunk.response` (`base.py`) | FINAL carries `list[ToolCall]` |

### Runner batch execution semantics

When the response is `list[ToolCall]`:
1. Validate all tool names exist (any unknown → nudge for first unknown)
2. If terminal tool in batch + steps not satisfied → step nudge (same escalation as before)
3. Emit one TOOL_CALL message with N `ToolCallInfo` entries, then execute each tool sequentially and emit N individual TOOL_RESULT messages
4. Reasoning (from `tool_calls[0].reasoning`) emitted as separate REASONING message before the TOOL_CALL message
5. If any tool errors in batch, `batch_had_error` flag → one consecutive error increment (not N)
6. If terminal tool was in batch and succeeded → return result
7. `last_error` tracks the most recent error from any tool in the batch (not just terminal)

### What didn't change

- `TextResponse` — still singular, no list wrapping
- `Workflow` / `ToolDef` / `ToolSpec` — tool definitions unchanged
- `StepTracker` — `record()` called per tool in a loop (trivial)
- `TieredCompact` — already uses `step_index` for iteration boundaries, so 1 TOOL_CALL + N TOOL_RESULT messages with the same `step_index` form an atomic group naturally
- Existing scenarios — produce singleton lists, behavior identical

### Observed in the wild

Ministral 8B Reasoning emits parallel tool calls at ~10% rate on `conditional_routing` and `phase2_compaction` scenarios (e.g., `check_metrics, check_logs, check_deployment` in one response). Most models and most runs still emit one tool call at a time.

## References

- Full design doc: [PARALLEL_TOOL_CALLS.md](PARALLEL_TOOL_CALLS.md) (same directory)
- README roadmap item #4
- BFCL integration concept: [BFCL_INTEGRATION_CONCEPT.md](BFCL_INTEGRATION_CONCEPT.md)
