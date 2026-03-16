# ADR-004: Async `on_chunk` Callback

**Status:** Done (implemented on `az/async_think` branch)

## Problem

`WorkflowRunner._send_streaming()` calls `on_chunk` synchronously inside the `async for` loop that reads SSE chunks from the backend:

`src/forge/core/runner.py:354-371`:

```python
async def _send_streaming(
    self,
    api_messages: list[dict[str, str]],
    tool_specs: list[ToolSpec],
) -> ToolCall | TextResponse:
    """Send via streaming, forwarding chunks to on_chunk callback."""
    response = None
    async for chunk in self.client.send_stream(api_messages, tools=tool_specs):
        if self.on_chunk is not None:
            self.on_chunk(chunk)            # ← synchronous, blocks SSE loop
        if chunk.type == ChunkType.FINAL:
            response = chunk.response
    if response is None:
        raise StreamError(...)
    return response
```

The type signature (`runner.py:42`):

```python
on_chunk: Callable[[StreamChunk], None] | None = None
```

A UI consumer wanting to `await websocket.send(chunk.content)` can't — the callback isn't awaitable. Since `run()` is already async, every real caller is in async context, and the sync constraint is artificial.

## Current Usage

Only **one caller** passes `on_chunk` today:

**`tests/unit/test_runner.py:727-740`** — unit test using `list.append`:

```python
async def test_on_chunk_callback_receives_chunks(self):
    received_chunks: list[StreamChunk] = []
    client = MockClient([
        ToolCall(tool="fetch", args={}),
        ToolCall(tool="submit", args={}),
    ])
    runner = _make_runner(client, stream=True, on_chunk=received_chunks.append)
    await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
    assert len(received_chunks) == 4
    assert received_chunks[0].type == ChunkType.TEXT_DELTA
    assert received_chunks[1].type == ChunkType.FINAL
```

The eval runner (`tests/eval/eval_runner.py:231-240`) does **not** pass `on_chunk`. It uses `on_message` for verbose printing and history collection.

## Decision

**Option C — async-only callbacks.** Change `on_chunk` to accept only async callables.

### Alternatives considered

| Option | Description | Rejected because |
|--------|-------------|------------------|
| A — Dual sync/async | Accept both, `inspect.isawaitable()` per chunk | Muddled type signature, runtime detection overhead |
| B — asyncio.Queue | Push to queue, consumer reads separately | Over-engineered, changes API shape, harder to test |
| D — Document only | Note "callbacks must be fast" | Punts the problem, anyone building UI figures it out alone |

Option C is clean, has exactly one test to update, and every caller is already in async context.

## Fix

### Step 1: Change type signature

**`src/forge/core/runner.py:42` — replace:**

```python
on_chunk: Callable[[StreamChunk], None] | None = None,
```

**with:**

```python
on_chunk: Callable[[StreamChunk], Awaitable[None]] | None = None,
```

Add `Awaitable` to the imports at the top of the file (from `collections.abc`).

### Step 2: Await the callback in `_send_streaming()`

**`src/forge/core/runner.py:362-363` — replace:**

```python
if self.on_chunk is not None:
    self.on_chunk(chunk)
```

**with:**

```python
if self.on_chunk is not None:
    await self.on_chunk(chunk)
```

### Step 3: Update `_make_runner` helper type hint

**`tests/unit/test_runner.py:101` — replace:**

```python
on_chunk=None,
```

**with:**

```python
on_chunk: Callable[[StreamChunk], Awaitable[None]] | None = None,
```

Add imports at top of test file:

```python
from collections.abc import Awaitable, Callable
```

### Step 4: Update the test

**`tests/unit/test_runner.py:727-740` — replace:**

```python
async def test_on_chunk_callback_receives_chunks(self):
    """on_chunk callback receives all chunks from the stream."""
    received_chunks: list[StreamChunk] = []
    client = MockClient([
        ToolCall(tool="fetch", args={}),
        ToolCall(tool="submit", args={}),
    ])
    runner = _make_runner(client, stream=True, on_chunk=received_chunks.append)
    await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

    # Each send_stream yields 2 chunks (TEXT_DELTA + FINAL), called twice
    assert len(received_chunks) == 4
    assert received_chunks[0].type == ChunkType.TEXT_DELTA
    assert received_chunks[1].type == ChunkType.FINAL
```

**with:**

```python
async def test_on_chunk_callback_receives_chunks(self):
    """on_chunk callback receives all chunks from the stream."""
    received_chunks: list[StreamChunk] = []

    async def collect(chunk: StreamChunk) -> None:
        received_chunks.append(chunk)

    client = MockClient([
        ToolCall(tool="fetch", args={}),
        ToolCall(tool="submit", args={}),
    ])
    runner = _make_runner(client, stream=True, on_chunk=collect)
    await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

    # Each send_stream yields 2 chunks (TEXT_DELTA + FINAL), called twice
    assert len(received_chunks) == 4
    assert received_chunks[0].type == ChunkType.TEXT_DELTA
    assert received_chunks[1].type == ChunkType.FINAL
```

### Step 5: Update docstring

**`src/forge/core/runner.py:59`** — update the `on_chunk` docstring:

```python
on_chunk: Async callback for each StreamChunk (awaited per chunk).
    Ignored if stream=False.
```

### Step 6: Update README roadmap

Mark item 13 as done.

## What NOT to change

- **`on_message` stays sync** — `on_message` is called once per message append (not in a hot SSE loop), and has two real callers in eval_runner (`_verbose_printer`, `collected_messages.append`). Converting it is a separate, lower-priority item. Note it as a follow-up if we want full async consistency.
- **Client `send_stream()` methods** — these are async generators already, no changes needed.
- **`StreamChunk` / `ChunkType`** — data types are unchanged.
- **`__init__.py` exports** — no new public types added.

## Follow-up consideration: `on_message`

`on_message` has the same sync pattern (`runner.py:43,95-96`):

```python
on_message: Callable[[Message], None] | None = None
# ...
if self.on_message is not None:
    self.on_message(msg)
```

It's used by the eval runner for `_verbose_printer` and `collected_messages.append` — both sync. Unlike `on_chunk` (hot SSE loop), `on_message` fires once per runner iteration, so the blocking cost is negligible. Convert to async for consistency if/when a real async consumer emerges, but not required now.

## Scope

- **Files modified:** 2 (`runner.py`, `test_runner.py`)
- **Lines changed:** ~10
- **New tests:** 0 (existing test updated)
- **Risk:** Low — one caller, one test, public API surface is the `WorkflowRunner` constructor

## References

- README roadmap item 13
- `WorkflowRunner.__init__()`: `src/forge/core/runner.py:34-73`
- `_send_streaming()`: `src/forge/core/runner.py:354-371`
- `StreamChunk` / `ChunkType`: `src/forge/clients/base.py:40-61`
- `LLMClient.send_stream()` protocol: `src/forge/clients/base.py:95-109`
- Unit test: `tests/unit/test_runner.py:727-740`
