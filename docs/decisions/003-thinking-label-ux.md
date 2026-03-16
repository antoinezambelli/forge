# ADR-003: `[thinking]` Label UX — Reasoning Capture Gating

**Status:** Done (implemented on `az/async_think` branch)

## Problem

With `--think false`, the eval verbose printer still shows `[thinking]` lines for models like Qwen3 that embed `<think>` tags in `content`. This is confusing — the user explicitly opted out of thinking output.

**Root cause:** OllamaClient captures reasoning from `content` regardless of the `_think` flag. LlamafileClient already gates this correctly via `_resolve_reasoning()`.

## Current Behavior

### LlamafileClient — correctly gated

`src/forge/clients/llamafile.py:145-168` — single gate method:

```python
def _resolve_reasoning(self, accumulated_reasoning, accumulated_content):
    if not self._think:
        return None          # ← _think=False → discard all reasoning
    if accumulated_reasoning:
        return accumulated_reasoning
    if accumulated_content:
        think_text, _ = _extract_think_tags(accumulated_content)
        if think_text:
            return think_text
        return accumulated_content or None
    return None
```

Every ToolCall creation in LlamafileClient routes through `_resolve_reasoning()`. When `_think=False`, reasoning is always `None`.

### OllamaClient — NOT gated

**Batch path** (`src/forge/clients/ollama.py:112`):

```python
reasoning = msg.get("thinking") or msg.get("content", "") or None
```

**Streaming path** (`src/forge/clients/ollama.py:201-206`):

```python
reasoning = (
    accumulated_thinking
    or accumulated_content          # ← captures everything
    or msg.get("content", "")
    or None
)
```

`_think` is only checked at request time (line 83: `if self._think: body["think"] = True`). It never gates the response-side capture.

### What happens with Qwen3 + `--think false` on Ollama

1. Request omits `think: true` → Ollama doesn't produce `msg.thinking`
2. Qwen3 still emits `<think>...</think>` inside `msg.content`
3. `or msg.get("content", "")` captures it as `reasoning`
4. Runner emits `MessageType.REASONING` message (5 sites in `runner.py`: lines 184, 226, 262, 296, 327)
5. Eval printer shows `[thinking]` (`tests/eval/eval_runner.py:172`)

## Fix

### Step 1: Add `_resolve_reasoning()` to OllamaClient

Add a method matching LlamafileClient's pattern. Simpler because Ollama doesn't need `[THINK]` tag extraction (Ollama's server handles that via the `thinking` field).

**Add to `src/forge/clients/ollama.py` after `_build_options()` (after line 69):**

```python
def _resolve_reasoning(
    self,
    thinking: str,
    content: str,
) -> str | None:
    """Gate reasoning capture on _think flag.

    When _think is False, discard all reasoning.
    When True: prefer thinking field, fall back to content.
    """
    if not self._think:
        return None
    return thinking or content or None
```

### Step 2: Use it in `send()` (batch path)

**`src/forge/clients/ollama.py:112` — replace:**

```python
reasoning = msg.get("thinking") or msg.get("content", "") or None
```

**with:**

```python
reasoning = self._resolve_reasoning(
    msg.get("thinking", ""), msg.get("content", ""),
)
```

### Step 3: Use it in `_iter_stream()` (streaming path)

**`src/forge/clients/ollama.py:201-206` — replace:**

```python
reasoning = (
    accumulated_thinking
    or accumulated_content
    or msg.get("content", "")
    or None
)
```

**with:**

```python
reasoning = self._resolve_reasoning(
    accumulated_thinking,
    accumulated_content or msg.get("content", ""),
)
```

### Step 4: Add unit tests

**File:** `tests/unit/test_ollama_client.py`

Two new tests, one per path. Both use `think=False` with content present alongside tool_calls:

**Test A — batch path** (add after `test_content_used_when_no_thinking`, ~line 201):

```python
@pytest.mark.asyncio
async def test_think_false_discards_reasoning(self) -> None:
    """think=False + content alongside tool_calls → reasoning is None."""
    client = _make_client(think=False)
    client._http.post.return_value = _mock_response({
        "message": {
            "role": "assistant",
            "content": "<think>Let me reason about this</think>",
            "tool_calls": [
                {"function": {"name": "get_pricing", "arguments": {"part": "X"}}}
            ],
        }
    })
    result = await client.send(
        [{"role": "user", "content": "test"}], tools=[_make_spec()]
    )
    assert isinstance(result, ToolCall)
    assert result.reasoning is None
```

**Test B — streaming path** (add after `test_streaming_think_false_explicit`, ~line 523):

```python
@pytest.mark.asyncio
async def test_streaming_think_false_discards_reasoning(self) -> None:
    """think=False + streamed content before tool_call → reasoning is None."""
    client = _make_client(think=False)
    lines = [
        json.dumps({"message": {"role": "assistant", "content": "Let me "}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "think..."}, "done": False}),
        json.dumps({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_pricing", "arguments": {"part": "X"}}}
                ],
            },
            "done": True,
        }),
    ]
    client._http.stream.return_value = _MockStreamResponse(lines)

    chunks = []
    async for chunk in client.send_stream(
        [{"role": "user", "content": "test"}], tools=[_make_spec()]
    ):
        chunks.append(chunk)

    final = [c for c in chunks if c.type == ChunkType.FINAL][0]
    assert isinstance(final.response, ToolCall)
    assert final.response.reasoning is None
```

### Step 5: Update README roadmap

Mark item 11 as done. The `[thinking]` label itself (`eval_runner.py:172`) stays as-is — the label is fine when reasoning is legitimately present. The fix is that reasoning is no longer present when it shouldn't be.

## What NOT to change

- **`eval_runner.py:172` label** — `[thinking]` is fine. The issue was never the label text; it was reasoning appearing when `--think false`. Once gated, REASONING messages won't exist when `_think=False`.
- **LlamafileClient** — already correct, no changes needed.
- **`runner.py` emission sites** — all 5 sites (lines 184, 226, 262, 296, 327) are already gated on `if tool_call.reasoning:`. Once the client returns `None`, they do nothing.
- **Existing tests** — `test_captures_reasoning_with_tool_call` (line 145) uses default `think=None` which resolves to `False` for non-matching model names. This test currently passes because `_think=False` doesn't gate reasoning. **This test will break** after the fix. Update it to use `think=True` explicitly, or accept that default-think clients don't capture reasoning from content. See decision below.

## Test impact: `test_captures_reasoning_with_tool_call`

This test (`test_ollama_client.py:145`) creates a client with default `think=None` + model `"test-model"` (no heuristic match → `_think=False`), then asserts `reasoning == "I should look up the price for this part."`.

After the fix, `_think=False` means `_resolve_reasoning()` returns `None`, so this assertion will fail.

**Options:**

1. **Change the test to `think=True`** — makes the intent explicit: "reasoning is captured when thinking is enabled"
2. **Change `_make_client` default model** to include a heuristic keyword (e.g. `"test-reasoning-model"`) — fragile, ties tests to heuristic

Recommend option 1. Also update `test_content_used_when_no_thinking` (line 184) the same way — it has the same implicit dependency on `_think` not gating.

Similarly for streaming: `test_streaming_captures_reasoning_from_deltas` (line 437) uses default `think=None` → `_think=False`. Update to `think=True`.

**Tests to update:**

| Test | Line | Current | Change to |
|------|------|---------|-----------|
| `test_captures_reasoning_with_tool_call` | 145 | `_make_client()` | `_make_client(think=True)` |
| `test_content_used_when_no_thinking` | 184 | `_make_client()` | `_make_client(think=True)` |
| `test_streaming_captures_reasoning_from_deltas` | 437 | `_make_client()` | `_make_client(think=True)` |

## Scope

- **Files modified:** 2 (`ollama.py`, `test_ollama_client.py`)
- **New code:** ~10 lines (method + two call sites)
- **New tests:** 2
- **Modified tests:** 3 (add `think=True`)
- **Risk:** Low — response-side only, no change to request behavior or protocol

## References

- README roadmap item 11
- LlamafileClient `_resolve_reasoning()`: `src/forge/clients/llamafile.py:145-168`
- Runner REASONING emission: `src/forge/core/runner.py:184,226,262,296,327`
- Eval printer: `tests/eval/eval_runner.py:172`
