"""Unit tests for forge.core.runner.WorkflowRunner."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from forge.clients.base import ChunkType, LLMClient, StreamChunk
from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from forge.core.runner import WorkflowRunner
from pydantic import BaseModel
from forge.core.workflow import (
    LLMResponse,
    TextResponse,
    ToolCall,
    ToolDef,
    ToolSpec,
    Workflow,
)
from forge.errors import MaxIterationsError, PrerequisiteError, StepEnforcementError, StreamError, ToolCallError, ToolExecutionError, ToolResolutionError, WorkflowCancelledError


class EmptyParams(BaseModel):
    pass


# ── Helpers ──────────────────────────────────────────────────────


class MockClient:
    """Mock LLMClient that returns scripted responses.

    Accepts ToolCall or TextResponse in the response list. Single ToolCall
    entries are automatically wrapped in a list to match the runner's
    expected LLMResponse = list[ToolCall] | TextResponse.
    """

    def __init__(self, responses: list[ToolCall | TextResponse]):
        self.responses = list(responses)
        self._call_index = 0
        self.send_calls: list[tuple[list[dict], list[ToolSpec] | None]] = []
        self.send_stream_calls: list[tuple[list[dict], list[ToolSpec] | None]] = []

    def _next(self) -> LLMResponse:
        resp = self.responses[self._call_index]
        self._call_index += 1
        if isinstance(resp, ToolCall):
            return [resp]
        return resp

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        self.send_calls.append((messages, tools))
        return self._next()

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.send_stream_calls.append((messages, tools))
        resp = self._next()
        yield StreamChunk(type=ChunkType.TEXT_DELTA, content="partial...")
        yield StreamChunk(type=ChunkType.FINAL, response=resp)

    async def get_context_length(self) -> int | None:
        return None


def _make_tool(name: str, fn=None) -> ToolDef:
    """Create a minimal ToolDef for testing."""
    if fn is None:
        fn = lambda **kwargs: f"{name}_result"
    return ToolDef(
        spec=ToolSpec(name=name, description=f"Tool {name}", parameters=EmptyParams),
        callable=fn,
    )


def _make_workflow(
    tools: dict[str, ToolDef] | None = None,
    required_steps: list[str] | None = None,
    terminal_tool: str = "submit",
) -> Workflow:
    """Create a test workflow with sensible defaults."""
    if tools is None:
        tools = {
            "fetch": _make_tool("fetch"),
            "submit": _make_tool("submit"),
        }
    if required_steps is None:
        required_steps = ["fetch"]
    return Workflow(
        name="test_wf",
        description="A test workflow",
        tools=tools,
        required_steps=required_steps,
        terminal_tool=terminal_tool,
        system_prompt_template="You are a {role}.",
    )


def _make_runner(
    client: MockClient,
    max_iterations: int = 10,
    max_retries_per_step: int = 3,
    max_tool_errors: int = 2,
    stream: bool = False,
    on_chunk=None,
    budget_tokens: int = 100_000,
) -> WorkflowRunner:
    """Create a WorkflowRunner with NoCompact strategy and generous budget."""
    ctx = ContextManager(strategy=NoCompact(), budget_tokens=budget_tokens)
    return WorkflowRunner(
        client=client,
        context_manager=ctx,
        max_iterations=max_iterations,
        max_retries_per_step=max_retries_per_step,
        max_tool_errors=max_tool_errors,
        stream=stream,
        on_chunk=on_chunk,
    )


# ── Happy path ───────────────────────────────────────────────────


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_simple_workflow(self):
        """One required step + terminal tool → returns terminal result."""
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        result = await runner.run(_make_workflow(), "do something", prompt_vars={"role": "tester"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_multi_step_workflow(self):
        """Multiple required steps all completed before terminal tool."""
        tools = {
            "step_a": _make_tool("step_a"),
            "step_b": _make_tool("step_b"),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["step_a", "step_b"])
        client = MockClient([
            ToolCall(tool="step_a", args={}),
            ToolCall(tool="step_b", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_tool_args_forwarded(self):
        """tool_call.args are passed as **kwargs to the tool callable."""
        received_args = {}

        def capture_tool(**kwargs):
            received_args.update(kwargs)
            return "captured"

        tools = {
            "fetch": _make_tool("fetch", fn=capture_tool),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={"key": "value", "count": 42}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert received_args == {"key": "value", "count": 42}


# ── Retry logic ──────────────────────────────────────────────────


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_text_response_triggers_retry(self):
        """TextResponse triggers retry nudge, then ToolCall succeeds."""
        client = MockClient([
            TextResponse(content="I don't know"),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises_tool_call_error(self):
        """All retries exhausted → ToolCallError with last raw response."""
        client = MockClient([
            TextResponse(content="nope1"),
            TextResponse(content="nope2"),
            TextResponse(content="nope3"),
            TextResponse(content="final_nope"),
        ])
        runner = _make_runner(client, max_retries_per_step=3)
        with pytest.raises(ToolCallError) as exc_info:
            await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert exc_info.value.raw_response == "final_nope"

    @pytest.mark.asyncio
    async def test_retry_counter_resets_on_tool_call(self):
        """Consecutive retry counter resets when a ToolCall is returned."""
        tools = {
            "step_a": _make_tool("step_a"),
            "step_b": _make_tool("step_b"),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["step_a", "step_b"])
        client = MockClient([
            # 2 retries then success
            TextResponse(content="fail1"),
            TextResponse(content="fail2"),
            ToolCall(tool="step_a", args={}),
            # 2 more retries then success — counter reset by previous ToolCall
            TextResponse(content="fail3"),
            TextResponse(content="fail4"),
            ToolCall(tool="step_b", args={}),
            # Terminal
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client, max_retries_per_step=3)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_retries_consume_iterations(self):
        """Each TextResponse retry consumes an iteration from max_iterations."""
        client = MockClient([
            TextResponse(content="fail1"),  # iteration 0
            TextResponse(content="fail2"),  # iteration 1
            ToolCall(tool="fetch", args={}),  # iteration 2
            ToolCall(tool="submit", args={}),  # iteration 3 — would exceed max_iterations=3
        ])
        runner = _make_runner(client, max_iterations=3, max_retries_per_step=3)
        # 2 retries + 1 fetch = 3 iterations used, no room for submit
        with pytest.raises(MaxIterationsError) as exc_info:
            await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert exc_info.value.iterations == 3

    @pytest.mark.asyncio
    async def test_max_iterations_bounds_total_llm_calls(self):
        """max_iterations is the hard ceiling on total LLM calls including retries."""
        client = MockClient([
            TextResponse(content="nope"),  # iteration 0
            TextResponse(content="nope"),  # iteration 1
            TextResponse(content="nope"),  # iteration 2
        ])
        # max_retries_per_step=5 (high), but max_iterations=3 (low) — hits iterations first
        runner = _make_runner(client, max_iterations=3, max_retries_per_step=5)
        with pytest.raises(MaxIterationsError):
            await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert len(client.send_calls) == 3


# ── Step enforcement ─────────────────────────────────────────────


class TestStepEnforcement:
    @pytest.mark.asyncio
    async def test_terminal_before_required_steps_injects_nudge(self):
        """Premature terminal call → step nudge injected, terminal NOT executed."""
        call_count = 0

        def counting_submit(**kwargs):
            nonlocal call_count
            call_count += 1
            return "submitted"

        tools = {
            "fetch": _make_tool("fetch"),
            "submit": _make_tool("submit", fn=counting_submit),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="submit", args={}),   # premature
            ToolCall(tool="fetch", args={}),     # do the required step
            ToolCall(tool="submit", args={}),    # now satisfied
        ])
        runner = _make_runner(client)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submitted"
        # submit callable should only be called once (the satisfied call)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_premature_terminal_resets_retry_counter(self):
        """Premature terminal (valid ToolCall) resets consecutive_retries."""
        client = MockClient([
            TextResponse(content="garbage"),      # retry 1
            TextResponse(content="garbage"),      # retry 2
            ToolCall(tool="submit", args={}),     # premature terminal — resets counter
            TextResponse(content="garbage"),      # retry 1 (reset)
            TextResponse(content="garbage"),      # retry 2
            ToolCall(tool="fetch", args={}),      # success
            ToolCall(tool="submit", args={}),     # terminal, satisfied
        ])
        # max_retries_per_step=3 — without reset, retries 1+2+1+2 = would exceed
        runner = _make_runner(client, max_retries_per_step=3)
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_step_nudge_allows_recovery(self):
        """After nudge, model completes required step, then terminal succeeds."""
        client = MockClient([
            ToolCall(tool="submit", args={}),   # nudged
            ToolCall(tool="fetch", args={}),     # required step
            ToolCall(tool="submit", args={}),    # now OK
        ])
        runner = _make_runner(client)
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_escalating_nudge_tiers(self):
        """3 premature terminal calls get tier 1, 2, 3 nudge messages."""
        client = MockClient([
            ToolCall(tool="submit", args={}),   # tier 1
            ToolCall(tool="submit", args={}),   # tier 2
            ToolCall(tool="submit", args={}),   # tier 3
            ToolCall(tool="fetch", args={}),     # required step
            ToolCall(tool="submit", args={}),    # now OK
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        original_compact = ctx.maybe_compact
        captured_messages: list[list[Message]] = []

        def spy_compact(messages, step_index=0, step_hint=""):
            captured_messages.append(list(messages))
            return original_compact(messages, step_index=step_index, step_hint=step_hint)

        ctx.maybe_compact = spy_compact

        runner = WorkflowRunner(client=client, context_manager=ctx)
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

        # Collect all step nudge messages
        all_nudges = []
        for msgs in captured_messages:
            for m in msgs:
                if m.metadata.type == MessageType.STEP_NUDGE and m not in all_nudges:
                    all_nudges.append(m)

        assert len(all_nudges) == 3
        # Tier 1: polite
        assert "cannot call submit yet" in all_nudges[0].content.lower()
        # Tier 2: direct
        assert "must call one of these tools now" in all_nudges[1].content.lower()
        # Tier 3: aggressive
        assert "STOP" in all_nudges[2].content

    @pytest.mark.asyncio
    async def test_premature_terminal_exhausted_raises_step_enforcement_error(self):
        """4th premature terminal call raises StepEnforcementError."""
        client = MockClient([
            ToolCall(tool="submit", args={}),   # tier 1
            ToolCall(tool="submit", args={}),   # tier 2
            ToolCall(tool="submit", args={}),   # tier 3
            ToolCall(tool="submit", args={}),   # raises
        ])
        runner = _make_runner(client)
        with pytest.raises(StepEnforcementError) as exc_info:
            await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert exc_info.value.terminal_tool == "submit"
        assert exc_info.value.attempts == 4
        assert exc_info.value.pending_steps == ["fetch"]

    @pytest.mark.asyncio
    async def test_premature_terminal_counter_resets_on_progress(self):
        """Successful tool call resets the premature terminal counter."""
        client = MockClient([
            ToolCall(tool="submit", args={}),   # premature 1
            ToolCall(tool="submit", args={}),   # premature 2
            ToolCall(tool="submit", args={}),   # premature 3
            # If counter didn't reset, the next premature call would be attempt 4 → raise
            ToolCall(tool="fetch", args={}),     # success — resets counter
            ToolCall(tool="submit", args={}),   # premature 1 (reset)
            ToolCall(tool="submit", args={}),   # premature 2 (reset)
            ToolCall(tool="submit", args={}),   # premature 3 (reset)
            # Still alive after 3 more because counter was reset
            ToolCall(tool="submit", args={}),    # now satisfied (fetch was done)
        ])
        # Need enough iterations for all 8 calls
        runner = _make_runner(client, max_iterations=10)
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"


# ── Error handling ───────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_unknown_tool_nudges_then_recovers(self):
        """Unknown tool name → nudge, model corrects → workflow completes."""
        client = MockClient([
            ToolCall(tool="get_pricing", args={}),  # wrong name
            ToolCall(tool="fetch", args={}),          # corrected
            ToolCall(tool="submit", args={}),         # terminal
        ])
        runner = _make_runner(client)
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_unknown_tool_nudge_lists_available_tools(self):
        """Nudge message for unknown tool contains available tool names."""
        client = MockClient([
            ToolCall(tool="nonexistent", args={}),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        # Second send call has the nudge message appended
        second_call_msgs = client.send_calls[1][0]
        nudge_content = second_call_msgs[-1]["content"]
        assert "nonexistent" in nudge_content
        assert "fetch" in nudge_content
        assert "submit" in nudge_content
        assert "does not exist" in nudge_content

    @pytest.mark.asyncio
    async def test_unknown_tool_consumes_iteration(self):
        """Unknown tool nudge consumes an iteration from max_iterations."""
        client = MockClient([
            ToolCall(tool="bad_name", args={}),   # iteration 0 (nudge)
            ToolCall(tool="fetch", args={}),        # iteration 1
            ToolCall(tool="submit", args={}),       # iteration 2 — exceeds max_iterations=2
        ])
        runner = _make_runner(client, max_iterations=2)
        with pytest.raises(MaxIterationsError):
            await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

    @pytest.mark.asyncio
    async def test_unknown_tool_exhausts_retries(self):
        """Repeated unknown tool names exhaust max_retries_per_step → ToolCallError."""
        client = MockClient([
            ToolCall(tool="bad1", args={}),
            ToolCall(tool="bad2", args={}),
            ToolCall(tool="bad3", args={}),
            ToolCall(tool="bad4", args={}),
        ])
        runner = _make_runner(client, max_retries_per_step=3)
        with pytest.raises(ToolCallError, match="Retries exhausted"):
            await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

    @pytest.mark.asyncio
    async def test_unknown_tool_and_text_response_share_retry_counter(self):
        """TextResponse and unknown tool retries accumulate on the same counter."""
        client = MockClient([
            TextResponse(content="garbage"),         # retry 1
            ToolCall(tool="nonexistent", args={}),   # retry 2
            TextResponse(content="more garbage"),    # retry 3
            ToolCall(tool="still_wrong", args={}),   # retry 4 → exceeds max_retries=3
        ])
        runner = _make_runner(client, max_retries_per_step=3)
        with pytest.raises(ToolCallError, match="Retries exhausted"):
            await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

    @pytest.mark.asyncio
    async def test_tool_error_feeds_back_then_recovers(self):
        """Tool raises once, error fed back, model retries with correct args."""
        call_count = 0

        def flaky_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("expected int, got str for argument 'count'")
            return "fetch_result"

        tools = {
            "fetch": _make_tool("fetch", fn=flaky_fetch),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={"count": "five"}),   # will raise
            ToolCall(tool="fetch", args={"count": 5}),        # corrected
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_tool_error_message_contains_exception_info(self):
        """Error fed back to model contains exception type and message."""

        def bad_tool(**kwargs):
            raise ValueError("something went wrong")

        tools = {
            "fetch": _make_tool("fetch", fn=bad_tool),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),   # error 1
            ToolCall(tool="fetch", args={}),   # error 2
            ToolCall(tool="fetch", args={}),   # error 3 → exceeds max_tool_errors=2
        ])
        runner = _make_runner(client, max_tool_errors=2)
        with pytest.raises(ToolExecutionError):
            await runner.run(wf, "go", prompt_vars={"role": "agent"})

        # Check that the error was fed back in the second send call
        second_call_msgs = client.send_calls[1][0]
        error_msg = second_call_msgs[-1]["content"]
        assert "[ToolError]" in error_msg
        assert "ValueError" in error_msg
        assert "something went wrong" in error_msg

    @pytest.mark.asyncio
    async def test_tool_errors_exhaust_max_tool_errors(self):
        """Consecutive tool errors exceeding max_tool_errors → ToolExecutionError."""

        def bad_tool(**kwargs):
            raise ValueError("always fails")

        tools = {
            "fetch": _make_tool("fetch", fn=bad_tool),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),  # error 1
            ToolCall(tool="fetch", args={}),  # error 2
            ToolCall(tool="fetch", args={}),  # error 3 → exceeds max_tool_errors=2
        ])
        runner = _make_runner(client, max_tool_errors=2)
        with pytest.raises(ToolExecutionError) as exc_info:
            await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert exc_info.value.tool_name == "fetch"
        assert isinstance(exc_info.value.cause, ValueError)

    @pytest.mark.asyncio
    async def test_tool_error_counter_resets_on_success(self):
        """Successful tool execution resets the consecutive tool error counter."""
        call_count = 0

        def sometimes_fails(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count in (1, 3):
                raise ValueError(f"fail #{call_count}")
            return "ok"

        tools = {
            "fetch": _make_tool("fetch", fn=sometimes_fails),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),  # error 1 (call_count=1)
            ToolCall(tool="fetch", args={}),  # success  (call_count=2) → resets counter
            ToolCall(tool="fetch", args={}),  # error 1 again (call_count=3) → counter reset
            ToolCall(tool="fetch", args={}),  # success  (call_count=4)
            ToolCall(tool="submit", args={}),
        ])
        # max_tool_errors=1 — without reset, second error would raise
        runner = _make_runner(client, max_tool_errors=1)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_terminal_tool_error_recovery(self):
        """Terminal tool error is fed back, model retries terminal → succeeds."""
        call_count = 0

        def flaky_submit(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("bad args")
            return "submitted"

        tools = {
            "fetch": _make_tool("fetch"),
            "submit": _make_tool("submit", fn=flaky_submit),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={"bad": True}),   # terminal raises
            ToolCall(tool="submit", args={}),               # terminal succeeds
        ])
        runner = _make_runner(client)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submitted"

    @pytest.mark.asyncio
    async def test_failed_required_step_not_recorded(self):
        """A required step that raises is NOT recorded — model must succeed to satisfy it."""
        call_count = 0

        def flaky_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("file not found")
            return "fetch_result"

        tools = {
            "fetch": _make_tool("fetch", fn=flaky_fetch),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),    # fails — should NOT satisfy required step
            ToolCall(tool="submit", args={}),   # premature — fetch not satisfied yet
            ToolCall(tool="fetch", args={}),    # succeeds — now satisfied
            ToolCall(tool="submit", args={}),   # terminal — OK
        ])
        runner = _make_runner(client)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"
        # submit after failed fetch should have been blocked by step nudge
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_tool_error_consumes_iteration(self):
        """Tool error consumes an iteration from max_iterations."""

        def bad_tool(**kwargs):
            raise ValueError("fails")

        tools = {
            "fetch": _make_tool("fetch", fn=bad_tool),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),  # iteration 0 (error)
            ToolCall(tool="fetch", args={}),  # iteration 1 (error)
            ToolCall(tool="fetch", args={}),  # iteration 2 — exceeds max_iterations=2
        ])
        runner = _make_runner(client, max_iterations=2, max_tool_errors=5)
        with pytest.raises(MaxIterationsError):
            await runner.run(wf, "go", prompt_vars={"role": "agent"})

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self):
        """Non-terminal ToolCalls exhaust max_iterations → MaxIterationsError."""
        client = MockClient([
            ToolCall(tool="fetch", args={}) for _ in range(5)
        ])
        runner = _make_runner(client, max_iterations=3)
        with pytest.raises(MaxIterationsError) as exc_info:
            await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert exc_info.value.iterations == 3
        assert "fetch" in exc_info.value.completed_steps


# ── Context management integration ──────────────────────────────


class TestContextManagement:
    @pytest.mark.asyncio
    async def test_compaction_called_each_iteration(self):
        """maybe_compact called once per iteration with correct args."""
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        original_compact = ctx.maybe_compact
        compact_calls = []

        def spy_compact(messages, step_index=0, step_hint=""):
            compact_calls.append({"step_index": step_index, "step_hint": step_hint})
            return original_compact(messages, step_index=step_index, step_hint=step_hint)

        ctx.maybe_compact = spy_compact

        runner = WorkflowRunner(client=client, context_manager=ctx)
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        assert len(compact_calls) == 2
        assert compact_calls[0]["step_index"] == 0
        assert compact_calls[1]["step_index"] == 1

    @pytest.mark.asyncio
    async def test_messages_grow_correctly(self):
        """After each tool call, messages contain correct types in order."""
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        # Inspect what was sent to the client on the second call (after fetch executed)
        # The second send call should include system, user, tool_call, tool_result
        second_call_msgs = client.send_calls[1][0]
        assert len(second_call_msgs) == 4
        assert second_call_msgs[0]["role"] == "system"
        assert second_call_msgs[1]["role"] == "user"
        assert second_call_msgs[2]["role"] == "assistant"
        assert second_call_msgs[3]["role"] == "tool"


# ── Streaming ────────────────────────────────────────────────────


class TestStreaming:
    @pytest.mark.asyncio
    async def test_stream_mode_uses_send_stream(self):
        """With stream=True, runner calls send_stream() instead of send()."""
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client, stream=True)
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert len(client.send_stream_calls) == 2
        assert len(client.send_calls) == 0

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_stream_extracts_final_response(self):
        """Streaming extracts ToolCall from FINAL chunk and acts on it."""
        client = MockClient([
            ToolCall(tool="fetch", args={"x": 1}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client, stream=True)
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_stream_without_final_chunk_raises_stream_error(self):
        """Stream that ends without FINAL chunk raises StreamError."""

        class NoFinalClient:
            """Mock client whose send_stream yields deltas but no FINAL."""

            async def send(self, messages, tools=None):
                return [ToolCall(tool="fetch", args={})]

            async def send_stream(self, messages, tools=None):
                yield StreamChunk(type=ChunkType.TEXT_DELTA, content="partial")

            async def get_context_length(self):
                return None

        runner = _make_runner(MockClient([]), stream=True)
        runner.client = NoFinalClient()
        with pytest.raises(StreamError, match="FINAL chunk"):
            await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})


# ── Async tool support ───────────────────────────────────────────


class TestAsyncToolSupport:
    @pytest.mark.asyncio
    async def test_async_tool_callable(self):
        """Async tool callable is awaited correctly."""
        async def async_fetch(**kwargs):
            return "async_result"

        tools = {
            "fetch": _make_tool("fetch", fn=async_fetch),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_sync_tool_callable(self):
        """Sync tool callable is called correctly."""
        def sync_fetch(**kwargs):
            return "sync_result"

        tools = {
            "fetch": _make_tool("fetch", fn=sync_fetch),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"


# ── Message structure ────────────────────────────────────────────


class TestMessageStructure:
    @pytest.mark.asyncio
    async def test_initial_messages_correct(self):
        """Initial messages: system prompt + user input with correct metadata."""
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        await runner.run(_make_workflow(), "do something", prompt_vars={"role": "tester"})

        # First send call should have exactly 2 messages
        first_call_msgs = client.send_calls[0][0]
        assert len(first_call_msgs) == 2
        assert first_call_msgs[0]["role"] == "system"
        assert first_call_msgs[0]["content"] == "You are a tester."
        assert first_call_msgs[1]["role"] == "user"
        assert first_call_msgs[1]["content"] == "do something"

    @pytest.mark.asyncio
    async def test_tool_call_message_format(self):
        """Tool call message contains tool name and args, result is stringified."""
        client = MockClient([
            ToolCall(tool="fetch", args={"key": "val"}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        # Second call has messages including the tool_call and tool_result
        second_msgs = client.send_calls[1][0]
        # assistant message (structured tool_call)
        tc = second_msgs[2]
        assert tc["role"] == "assistant"
        assert tc["tool_calls"][0]["function"]["name"] == "fetch"
        assert tc["tool_calls"][0]["function"]["arguments"] == {"key": "val"}
        # tool message (tool_result)
        assert second_msgs[3]["content"] == "fetch_result"

    @pytest.mark.asyncio
    async def test_text_response_emits_assistant_before_retry_nudge(self):
        """TextResponse content is recorded as ASSISTANT/TEXT_RESPONSE before the retry nudge."""
        client = MockClient([
            TextResponse(content="I'm not sure what to do"),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        # Second send call should have: system, user, assistant(reasoning), user(nudge)
        second_call_msgs = client.send_calls[1][0]
        assert len(second_call_msgs) == 4
        assert second_call_msgs[2]["role"] == "assistant"
        assert second_call_msgs[2]["content"] == "I'm not sure what to do"
        assert second_call_msgs[3]["role"] == "user"
        assert "not a valid tool call" in second_call_msgs[3]["content"]

    @pytest.mark.asyncio
    async def test_unknown_tool_emits_assistant_before_nudge(self):
        """Unknown tool call is recorded as ASSISTANT messages before the nudge."""
        client = MockClient([
            ToolCall(tool="nonexistent", args={"x": 1}),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        # Second send call should have: system, user, assistant(tool_call), user(nudge)
        second_call_msgs = client.send_calls[1][0]
        assert len(second_call_msgs) == 4
        assert second_call_msgs[2]["role"] == "assistant"
        assert second_call_msgs[2]["tool_calls"][0]["function"]["name"] == "nonexistent"
        assert second_call_msgs[3]["role"] == "user"
        assert "does not exist" in second_call_msgs[3]["content"]

    @pytest.mark.asyncio
    async def test_step_nudge_emits_assistant_before_nudge(self):
        """Premature terminal tool call is recorded as ASSISTANT before the step nudge."""
        client = MockClient([
            ToolCall(tool="submit", args={}),   # premature
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        # Second send call should have: system, user, assistant(tool_call), user(nudge)
        second_call_msgs = client.send_calls[1][0]
        assert len(second_call_msgs) == 4
        assert second_call_msgs[2]["role"] == "assistant"
        assert second_call_msgs[2]["tool_calls"][0]["function"]["name"] == "submit"
        assert second_call_msgs[3]["role"] == "user"
        assert "cannot call submit yet" in second_call_msgs[3]["content"].lower()

    @pytest.mark.asyncio
    async def test_retry_nudge_message_metadata(self):
        """Retry nudge has MessageType.RETRY_NUDGE metadata."""
        client = MockClient([
            TextResponse(content="bad output"),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        original_compact = ctx.maybe_compact
        captured_messages: list[list[Message]] = []

        def spy_compact(messages, step_index=0, step_hint=""):
            captured_messages.append(list(messages))
            return original_compact(messages, step_index=step_index, step_hint=step_hint)

        ctx.maybe_compact = spy_compact

        runner = WorkflowRunner(client=client, context_manager=ctx)
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        # After the retry, the second iteration's messages should contain the nudge
        second_iteration_msgs = captured_messages[1]
        nudge_msgs = [m for m in second_iteration_msgs if m.metadata.type == MessageType.RETRY_NUDGE]
        assert len(nudge_msgs) == 1
        assert "not a valid tool call" in nudge_msgs[0].content

    @pytest.mark.asyncio
    async def test_step_nudge_message_metadata(self):
        """Step nudge has MessageType.STEP_NUDGE metadata."""
        client = MockClient([
            ToolCall(tool="submit", args={}),   # premature
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        original_compact = ctx.maybe_compact
        captured_messages: list[list[Message]] = []

        def spy_compact(messages, step_index=0, step_hint=""):
            captured_messages.append(list(messages))
            return original_compact(messages, step_index=step_index, step_hint=step_hint)

        ctx.maybe_compact = spy_compact

        runner = WorkflowRunner(client=client, context_manager=ctx)
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        # After the premature terminal call, the second iteration should have step nudge
        second_iteration_msgs = captured_messages[1]
        nudge_msgs = [m for m in second_iteration_msgs if m.metadata.type == MessageType.STEP_NUDGE]
        assert len(nudge_msgs) == 1
        assert "cannot call submit yet" in nudge_msgs[0].content.lower()


# ── Rescue tool calls ─────────────────────────────────────────────


class TestRescueToolCalls:
    @pytest.mark.asyncio
    async def test_rescue_json_from_text_response(self):
        """TextResponse with valid JSON tool call is rescued — no retry nudge."""
        client = MockClient([
            TextResponse(content='{"tool": "fetch", "args": {"key": "val"}}'),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"
        # Only 2 LLM calls — no retry iteration wasted
        assert len(client.send_calls) == 2

    @pytest.mark.asyncio
    async def test_rescue_rehearsal_syntax(self):
        """TextResponse with rehearsal syntax is rescued."""
        client = MockClient([
            TextResponse(content='fetch[ARGS]{"key": "val"}'),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"
        assert len(client.send_calls) == 2

    @pytest.mark.asyncio
    async def test_rescue_resets_retry_counter(self):
        """Rescued tool call resets consecutive_retries like a normal ToolCall."""
        client = MockClient([
            TextResponse(content="plain garbage"),            # retry 1
            TextResponse(content="more garbage"),             # retry 2
            TextResponse(content='{"tool": "fetch", "args": {}}'),  # rescued → resets
            TextResponse(content="garbage again"),            # retry 1 (reset)
            TextResponse(content="still garbage"),            # retry 2
            TextResponse(content='{"tool": "fetch", "args": {}}'),  # rescued → resets
            ToolCall(tool="submit", args={}),
        ])
        # max_retries=3 — without reset from rescue, retries would exhaust
        runner = _make_runner(client, max_retries_per_step=3, max_iterations=10)
        result = await runner.run(
            _make_workflow(required_steps=[]),
            "go", prompt_vars={"role": "agent"},
        )
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_rescue_unknown_tool_falls_through(self):
        """Rescued ToolCall with unknown tool name goes to the unknown tool nudge path."""
        client = MockClient([
            TextResponse(content='{"tool": "nonexistent", "args": {}}'),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        # rescue_tool_call returns None for unknown tools → normal retry nudge
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_rescue_still_executes_tool(self):
        """Rescued tool call is actually executed (not just recorded)."""
        received_args = {}

        def capture_tool(**kwargs):
            received_args.update(kwargs)
            return "captured"

        tools = {
            "fetch": _make_tool("fetch", fn=capture_tool),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            TextResponse(content='{"tool": "fetch", "args": {"count": 42}}'),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert received_args == {"count": 42}

    @pytest.mark.asyncio
    async def test_rescue_satisfies_required_step(self):
        """Rescued tool call satisfies a required step."""
        client = MockClient([
            TextResponse(content='{"tool": "fetch", "args": {}}'),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        # submit succeeds because rescued fetch satisfied the required step
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_non_rescuable_text_still_nudges(self):
        """Plain text that can't be rescued still triggers retry nudge."""
        client = MockClient([
            TextResponse(content="Let me think about this..."),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"
        # 3 LLM calls — text response + retry nudge + fetch + submit
        assert len(client.send_calls) == 3


# ── Reasoning capture ─────────────────────────────────────────────


class TestReasoningCapture:
    @pytest.mark.asyncio
    async def test_reasoning_message_appended_before_tool_call(self):
        """ToolCall with reasoning → REASONING message before TOOL_CALL in history."""
        client = MockClient([
            ToolCall(tool="fetch", args={}, reasoning="I need to fetch the data first."),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        original_compact = ctx.maybe_compact
        captured_messages: list[list[Message]] = []

        def spy_compact(messages, step_index=0, step_hint=""):
            captured_messages.append(list(messages))
            return original_compact(messages, step_index=step_index, step_hint=step_hint)

        ctx.maybe_compact = spy_compact

        runner = WorkflowRunner(client=client, context_manager=ctx)
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

        # Second iteration's messages should have: system, user, REASONING, TOOL_CALL, TOOL_RESULT
        second_iteration_msgs = captured_messages[1]
        reasoning_msgs = [m for m in second_iteration_msgs if m.metadata.type == MessageType.REASONING]
        assert len(reasoning_msgs) == 1
        assert reasoning_msgs[0].content == "I need to fetch the data first."
        assert reasoning_msgs[0].role == MessageRole.ASSISTANT

        # REASONING should come before TOOL_CALL
        types = [m.metadata.type for m in second_iteration_msgs]
        reasoning_idx = types.index(MessageType.REASONING)
        tool_call_idx = types.index(MessageType.TOOL_CALL)
        assert reasoning_idx < tool_call_idx

    @pytest.mark.asyncio
    async def test_no_reasoning_message_when_reasoning_is_none(self):
        """ToolCall without reasoning → no REASONING message in history."""
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        original_compact = ctx.maybe_compact
        captured_messages: list[list[Message]] = []

        def spy_compact(messages, step_index=0, step_hint=""):
            captured_messages.append(list(messages))
            return original_compact(messages, step_index=step_index, step_hint=step_hint)

        ctx.maybe_compact = spy_compact

        runner = WorkflowRunner(client=client, context_manager=ctx)
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        # No REASONING messages should exist in any iteration
        for msgs in captured_messages:
            reasoning_msgs = [m for m in msgs if m.metadata.type == MessageType.REASONING]
            assert len(reasoning_msgs) == 0

    @pytest.mark.asyncio
    async def test_reasoning_preserved_on_tool_error(self):
        """Tool error path preserves reasoning before the TOOL_CALL + error result."""
        call_count = 0

        def flaky_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("bad input")
            return "ok"

        tools = {
            "fetch": _make_tool("fetch", fn=flaky_fetch),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}, reasoning="Let me try fetching."),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        original_compact = ctx.maybe_compact
        captured_messages: list[list[Message]] = []

        def spy_compact(messages, step_index=0, step_hint=""):
            captured_messages.append(list(messages))
            return original_compact(messages, step_index=step_index, step_hint=step_hint)

        ctx.maybe_compact = spy_compact

        runner = WorkflowRunner(client=client, context_manager=ctx)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

        # Second iteration should see: ..., REASONING, TOOL_CALL, TOOL_RESULT(error)
        second_iteration_msgs = captured_messages[1]
        reasoning_msgs = [m for m in second_iteration_msgs if m.metadata.type == MessageType.REASONING]
        assert len(reasoning_msgs) == 1
        assert reasoning_msgs[0].content == "Let me try fetching."

        # REASONING must come immediately before its paired TOOL_CALL
        types = [m.metadata.type for m in second_iteration_msgs]
        reasoning_idx = types.index(MessageType.REASONING)
        assert types[reasoning_idx + 1] == MessageType.TOOL_CALL

    @pytest.mark.asyncio
    async def test_reasoning_preserved_on_terminal_tool_error(self):
        """Terminal tool error path preserves reasoning."""
        call_count = 0

        def flaky_submit(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("bad args")
            return "submitted"

        tools = {
            "fetch": _make_tool("fetch"),
            "submit": _make_tool("submit", fn=flaky_submit),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}, reasoning="Time to submit the result."),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        original_compact = ctx.maybe_compact
        captured_messages: list[list[Message]] = []

        def spy_compact(messages, step_index=0, step_hint=""):
            captured_messages.append(list(messages))
            return original_compact(messages, step_index=step_index, step_hint=step_hint)

        ctx.maybe_compact = spy_compact

        runner = WorkflowRunner(client=client, context_manager=ctx)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submitted"

        # Third iteration should see the reasoning from the failed terminal call
        third_iteration_msgs = captured_messages[2]
        reasoning_msgs = [m for m in third_iteration_msgs if m.metadata.type == MessageType.REASONING]
        assert len(reasoning_msgs) == 1
        assert reasoning_msgs[0].content == "Time to submit the result."

        # REASONING must come immediately before its paired TOOL_CALL
        types = [m.metadata.type for m in third_iteration_msgs]
        reasoning_idx = types.index(MessageType.REASONING)
        assert types[reasoning_idx + 1] == MessageType.TOOL_CALL

    @pytest.mark.asyncio
    async def test_reasoning_folded_into_tool_call_on_wire(self):
        """Reasoning is folded into the tool_call message's content on the wire."""
        client = MockClient([
            ToolCall(tool="fetch", args={}, reasoning="Thinking about this..."),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        # Second send call: reasoning folded into tool_call content
        second_call_msgs = client.send_calls[1][0]
        # system, user, tool_call(assistant with content), tool_result(tool)
        assert len(second_call_msgs) == 4
        assert second_call_msgs[2]["role"] == "assistant"
        assert second_call_msgs[2]["content"] == "Thinking about this..."
        assert "tool_calls" in second_call_msgs[2]

    @pytest.mark.asyncio
    async def test_text_response_not_folded_into_tool_call(self):
        """TEXT_RESPONSE is not folded into the next tool_call's content (unlike REASONING)."""
        client = MockClient([
            TextResponse(content="Let me think about this..."),
            ToolCall(tool="fetch", args={}, reasoning="Now I know what to do"),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        # Third send call: after text_response+nudge recovery, then reasoning+fetch
        third_call_msgs = client.send_calls[2][0]
        # Find the tool_call message for fetch
        tc_msgs = [m for m in third_call_msgs if "tool_calls" in m]
        assert len(tc_msgs) == 1
        # Its content should be the reasoning, NOT the text response prose
        assert tc_msgs[0]["content"] == "Now I know what to do"


# ── on_message callback ──────────────────────────────────────────


class TestOnMessageCallback:
    @pytest.mark.asyncio
    async def test_on_message_receives_all_messages(self):
        """on_message fires for every message: system, user, tool_call, tool_result, terminal."""
        collected: list[Message] = []
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=client, context_manager=ctx, on_message=collected.append,
        )
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        types = [m.metadata.type for m in collected]
        assert types == [
            MessageType.SYSTEM_PROMPT,
            MessageType.USER_INPUT,
            MessageType.TOOL_CALL,
            MessageType.TOOL_RESULT,
            MessageType.TOOL_CALL,
            MessageType.TOOL_RESULT,
        ]

    @pytest.mark.asyncio
    async def test_on_message_none_is_safe(self):
        """on_message=None (default) does not cause errors."""
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)  # on_message not set
        result = await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_on_message_captures_retry_nudge(self):
        """on_message fires for retry nudge messages."""
        collected: list[Message] = []
        client = MockClient([
            TextResponse(content="bad"),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=client, context_manager=ctx, on_message=collected.append,
        )
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        types = [m.metadata.type for m in collected]
        assert MessageType.RETRY_NUDGE in types

    @pytest.mark.asyncio
    async def test_on_message_captures_step_nudge(self):
        """on_message fires for step nudge messages."""
        collected: list[Message] = []
        client = MockClient([
            ToolCall(tool="submit", args={}),   # premature
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=client, context_manager=ctx, on_message=collected.append,
        )
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        types = [m.metadata.type for m in collected]
        assert MessageType.STEP_NUDGE in types

    @pytest.mark.asyncio
    async def test_on_message_captures_tool_errors(self):
        """on_message fires for tool error messages (TOOL_RESULT with [ToolError])."""
        collected: list[Message] = []

        def bad_fetch(**kwargs):
            raise ValueError("broken")

        tools = {
            "fetch": _make_tool("fetch", fn=bad_fetch),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),   # error
            ToolCall(tool="fetch", args={}),   # error
            ToolCall(tool="fetch", args={}),   # exceeds max_tool_errors
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=client, context_manager=ctx,
            max_tool_errors=2, on_message=collected.append,
        )
        with pytest.raises(ToolExecutionError):
            await runner.run(wf, "go", prompt_vars={"role": "agent"})

        error_msgs = [m for m in collected if "[ToolError]" in m.content]
        assert len(error_msgs) == 3

    @pytest.mark.asyncio
    async def test_on_message_captures_reasoning(self):
        """on_message fires for reasoning messages."""
        collected: list[Message] = []
        client = MockClient([
            ToolCall(tool="fetch", args={}, reasoning="Thinking..."),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=client, context_manager=ctx, on_message=collected.append,
        )
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        types = [m.metadata.type for m in collected]
        assert MessageType.REASONING in types
        reasoning = [m for m in collected if m.metadata.type == MessageType.REASONING]
        assert reasoning[0].content == "Thinking..."

    @pytest.mark.asyncio
    async def test_on_message_captures_rescued_tool_call(self):
        """Rescued tool call emits TOOL_CALL + TOOL_RESULT, no TEXT_RESPONSE or RETRY_NUDGE."""
        collected: list[Message] = []
        client = MockClient([
            TextResponse(content='{"tool": "fetch", "args": {"key": "val"}}'),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=client, context_manager=ctx, on_message=collected.append,
        )
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        types = [m.metadata.type for m in collected]
        assert MessageType.RETRY_NUDGE not in types
        assert MessageType.TEXT_RESPONSE not in types
        assert MessageType.TOOL_CALL in types
        assert MessageType.TOOL_RESULT in types

    @pytest.mark.asyncio
    async def test_on_message_with_streaming(self):
        """on_message works correctly when stream=True."""
        collected: list[Message] = []
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=client, context_manager=ctx,
            stream=True, on_message=collected.append,
        )
        await runner.run(_make_workflow(), "go", prompt_vars={"role": "agent"})

        types = [m.metadata.type for m in collected]
        assert types == [
            MessageType.SYSTEM_PROMPT,
            MessageType.USER_INPUT,
            MessageType.TOOL_CALL,
            MessageType.TOOL_RESULT,
            MessageType.TOOL_CALL,
            MessageType.TOOL_RESULT,
        ]


# ── initial_messages ──────────────────────────────────────────


class TestInitialMessages:
    """Test the initial_messages parameter on run()."""

    @pytest.mark.asyncio
    async def test_initial_messages_skips_system_and_user_init(self):
        """When initial_messages is provided, no system/user messages are emitted."""
        collected: list[Message] = []
        client = MockClient([
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=client, context_manager=ctx,
            on_message=collected.append,
        )

        # Seed with pre-built history (system + user + prior assistant/tool)
        seed = [
            Message(MessageRole.SYSTEM, "You are a tester.", MessageMeta(MessageType.SYSTEM_PROMPT)),
            Message(MessageRole.USER, "do something", MessageMeta(MessageType.USER_INPUT)),
        ]

        wf = _make_workflow(required_steps=[])
        await runner.run(wf, "do something", initial_messages=seed)

        # on_message should NOT have fired for the seed messages
        # It should only have the new messages from the loop
        types = [m.metadata.type for m in collected]
        assert MessageType.SYSTEM_PROMPT not in types
        assert MessageType.USER_INPUT not in types
        # But we should still get the tool call + result from the loop
        assert MessageType.TOOL_CALL in types
        assert MessageType.TOOL_RESULT in types

    @pytest.mark.asyncio
    async def test_initial_messages_included_in_api_call(self):
        """Seed messages are sent to the LLM (visible in the API messages)."""
        client = MockClient([
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(client=client, context_manager=ctx)

        seed = [
            Message(MessageRole.SYSTEM, "You are a tester.", MessageMeta(MessageType.SYSTEM_PROMPT)),
            Message(MessageRole.USER, "first question", MessageMeta(MessageType.USER_INPUT)),
            Message(MessageRole.ASSISTANT, "", MessageMeta(MessageType.TOOL_CALL),
                    tool_calls=[ToolCallInfo(name="fetch", args={}, call_id="c0")]),
            Message(MessageRole.TOOL, "fetch_result", MessageMeta(MessageType.TOOL_RESULT),
                    tool_name="fetch", tool_call_id="c0"),
            Message(MessageRole.USER, "follow-up", MessageMeta(MessageType.USER_INPUT)),
        ]

        wf = _make_workflow(required_steps=[])
        await runner.run(wf, "follow-up", initial_messages=seed)

        # The API call should include all seed messages + new ones
        api_msgs, _ = client.send_calls[0]
        assert api_msgs[0]["role"] == "system"
        assert api_msgs[0]["content"] == "You are a tester."
        assert api_msgs[1]["role"] == "user"
        assert api_msgs[1]["content"] == "first question"

    @pytest.mark.asyncio
    async def test_none_initial_messages_is_default(self):
        """Passing initial_messages=None behaves identically to not passing it."""
        collected: list[Message] = []
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=client, context_manager=ctx,
            on_message=collected.append,
        )

        wf = _make_workflow()
        await runner.run(wf, "go", prompt_vars={"role": "agent"}, initial_messages=None)

        # Should behave exactly like existing tests — system + user emitted
        types = [m.metadata.type for m in collected]
        assert types[0] == MessageType.SYSTEM_PROMPT
        assert types[1] == MessageType.USER_INPUT


# ── ToolResolutionError ──────────────────────────────────────────


class TestToolResolutionError:
    @pytest.mark.asyncio
    async def test_resolution_error_feeds_back_and_recovers(self):
        """ToolResolutionError is fed back, model retries with different args → succeeds."""
        call_count = 0

        def lookup(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ToolResolutionError("No entry found for 'capital of France'. Try another key.")
            return "The capital of France is Paris."

        tools = {
            "fetch": _make_tool("fetch", fn=lookup),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={"query": "capital of France"}),
            ToolCall(tool="fetch", args={"query": "france"}),
            ToolCall(tool="submit", args={}),
        ])
        runner = _make_runner(client)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_resolution_error_message_fed_back_to_model(self):
        """ToolResolutionError message appears in the next LLM call as a tool result."""

        def always_miss(**kwargs):
            raise ToolResolutionError("No entry found for 'bad key'. Try another key.")

        tools = {
            "fetch": _make_tool("fetch", fn=always_miss),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="fetch", args={}),
        ])
        runner = _make_runner(client, max_iterations=3)
        with pytest.raises(MaxIterationsError):
            await runner.run(wf, "go", prompt_vars={"role": "agent"})

        # Second send call should have the resolution error fed back
        second_call_msgs = client.send_calls[1][0]
        error_msg = second_call_msgs[-1]["content"]
        assert "[ToolResolutionError]" in error_msg
        assert "No entry found" in error_msg

    @pytest.mark.asyncio
    async def test_resolution_error_does_not_increment_consecutive_tool_errors(self):
        """ToolResolutionError does NOT count toward consecutive_tool_errors."""

        def always_miss(**kwargs):
            raise ToolResolutionError("miss")

        tools = {
            "fetch": _make_tool("fetch", fn=always_miss),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}) for _ in range(5)
        ])
        # max_tool_errors=1 — a single hard error would raise ToolExecutionError.
        # But 5 consecutive ToolResolutionErrors should NOT trigger it.
        runner = _make_runner(client, max_iterations=5, max_tool_errors=1)
        with pytest.raises(MaxIterationsError):
            await runner.run(wf, "go", prompt_vars={"role": "agent"})
        # If consecutive_tool_errors were incremented, we'd get ToolExecutionError
        # instead of MaxIterationsError — the test passes if we reach here.

    @pytest.mark.asyncio
    async def test_resolution_error_does_not_record_step(self):
        """ToolResolutionError does NOT mark the step as completed."""
        call_count = 0

        def miss_then_hit(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ToolResolutionError("miss")
            return "hit"

        tools = {
            "fetch": _make_tool("fetch", fn=miss_then_hit),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),    # ToolResolutionError — step NOT recorded
            ToolCall(tool="submit", args={}),   # premature — fetch not satisfied
            ToolCall(tool="fetch", args={}),    # succeeds — step recorded
            ToolCall(tool="submit", args={}),   # now OK
        ])
        runner = _make_runner(client)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_resolution_error_bounded_by_max_iterations(self):
        """ToolResolutionError retries are bounded by max_iterations."""

        def always_miss(**kwargs):
            raise ToolResolutionError("miss")

        tools = {
            "fetch": _make_tool("fetch", fn=always_miss),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}) for _ in range(4)
        ])
        runner = _make_runner(client, max_iterations=3)
        with pytest.raises(MaxIterationsError) as exc_info:
            await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert exc_info.value.iterations == 3

    @pytest.mark.asyncio
    async def test_resolution_error_then_hard_error_counts_correctly(self):
        """ToolResolutionError followed by hard error: only hard error increments counter."""
        call_count = 0

        def mixed_errors(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ToolResolutionError("soft miss")
            if call_count == 2:
                raise ValueError("hard crash")
            return "ok"

        tools = {
            "fetch": _make_tool("fetch", fn=mixed_errors),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),   # ToolResolutionError (no increment)
            ToolCall(tool="fetch", args={}),   # ValueError (increment to 1)
            ToolCall(tool="fetch", args={}),   # succeeds
            ToolCall(tool="submit", args={}),
        ])
        # max_tool_errors=1 — the one hard error hits 1, which is <=, not >
        runner = _make_runner(client, max_tool_errors=1)
        result = await runner.run(wf, "go", prompt_vars={"role": "agent"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_resolution_error_on_message_callback(self):
        """on_message fires for ToolResolutionError tool result."""
        collected: list[Message] = []

        def miss(**kwargs):
            raise ToolResolutionError("No entry found for 'x'. Try another key.")

        tools = {
            "fetch": _make_tool("fetch", fn=miss),
            "submit": _make_tool("submit"),
        }
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="fetch", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=client, context_manager=ctx,
            max_iterations=2, on_message=collected.append,
        )
        with pytest.raises(MaxIterationsError):
            await runner.run(wf, "go", prompt_vars={"role": "agent"})

        error_msgs = [m for m in collected if "[ToolResolutionError]" in m.content]
        assert len(error_msgs) == 2
        assert all(m.metadata.type == MessageType.TOOL_RESULT for m in error_msgs)

    @pytest.mark.asyncio
    async def test_resolution_error_is_not_forge_error(self):
        """ToolResolutionError is a plain Exception, not a ForgeError."""
        from forge.errors import ForgeError
        err = ToolResolutionError("test")
        assert isinstance(err, Exception)
        assert not isinstance(err, ForgeError)


# ── Prerequisite enforcement ─────────────────────────────────────


class TestPrerequisiteEnforcement:
    """Runner-level prerequisite enforcement."""

    @pytest.mark.asyncio
    async def test_prereq_nudge_then_success(self):
        """Model calls edit without read, gets nudged, reads, then edits."""
        tools = {
            "read_file": _make_tool("read_file"),
            "edit_file": ToolDef(
                spec=ToolSpec(name="edit_file", description="Edit", parameters=EmptyParams),
                callable=lambda **kwargs: "edited",
                prerequisites=[{"tool": "read_file", "match_arg": "path"}],
            ),
            "submit": _make_tool("submit"),
        }
        client = MockClient([
            # 1: model tries edit_file first → blocked
            ToolCall(tool="edit_file", args={"path": "foo.py"}),
            # 2: model corrects, calls read_file
            ToolCall(tool="read_file", args={"path": "foo.py"}),
            # 3: model calls edit_file again → now allowed
            ToolCall(tool="edit_file", args={"path": "foo.py"}),
            # 4: terminal
            ToolCall(tool="submit", args={}),
        ])
        wf = _make_workflow(tools=tools, required_steps=[], terminal_tool="submit")
        runner = _make_runner(client)
        result = await runner.run(wf, "fix foo.py", prompt_vars={"role": "dev"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_prereq_nudge_emits_correct_message_type(self):
        """PREREQUISITE_NUDGE message type is emitted on violation."""
        collected = []
        tools = {
            "read_file": _make_tool("read_file"),
            "edit_file": ToolDef(
                spec=ToolSpec(name="edit_file", description="Edit", parameters=EmptyParams),
                callable=lambda **kwargs: "edited",
                prerequisites=["read_file"],
            ),
            "submit": _make_tool("submit"),
        }
        client = MockClient([
            ToolCall(tool="edit_file", args={}),
            ToolCall(tool="read_file", args={}),
            ToolCall(tool="edit_file", args={}),
            ToolCall(tool="submit", args={}),
        ])
        wf = _make_workflow(tools=tools, required_steps=[], terminal_tool="submit")
        runner = _make_runner(client)
        runner.on_message = collected.append
        await runner.run(wf, "go", prompt_vars={"role": "dev"})

        prereq_nudges = [m for m in collected if m.metadata.type == MessageType.PREREQUISITE_NUDGE]
        assert len(prereq_nudges) == 1
        assert "read_file" in prereq_nudges[0].content

    @pytest.mark.asyncio
    async def test_prereq_exhaustion_raises(self):
        """PrerequisiteError raised after max consecutive violations."""
        tools = {
            "read_file": _make_tool("read_file"),
            "edit_file": ToolDef(
                spec=ToolSpec(name="edit_file", description="Edit", parameters=EmptyParams),
                callable=lambda **kwargs: "edited",
                prerequisites=["read_file"],
            ),
            "submit": _make_tool("submit"),
        }
        client = MockClient([
            ToolCall(tool="edit_file", args={}),
            ToolCall(tool="edit_file", args={}),
            ToolCall(tool="edit_file", args={}),
            ToolCall(tool="edit_file", args={}),
        ])
        wf = _make_workflow(tools=tools, required_steps=[], terminal_tool="submit")
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(client=client, context_manager=ctx, max_iterations=10)
        with pytest.raises(PrerequisiteError, match="read_file"):
            await runner.run(wf, "go", prompt_vars={"role": "dev"})

    @pytest.mark.asyncio
    async def test_no_prereqs_no_interference(self):
        """Workflows without prerequisites behave identically to before."""
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        wf = _make_workflow()
        runner = _make_runner(client)
        result = await runner.run(wf, "go", prompt_vars={"role": "dev"})
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_args_recorded_for_prereq_tracking(self):
        """Tool args are passed through to record() for prereq matching."""
        collected = []
        tools = {
            "read_file": _make_tool("read_file"),
            "edit_file": ToolDef(
                spec=ToolSpec(name="edit_file", description="Edit", parameters=EmptyParams),
                callable=lambda **kwargs: "edited",
                prerequisites=[{"tool": "read_file", "match_arg": "path"}],
            ),
            "submit": _make_tool("submit"),
        }
        client = MockClient([
            # Read a.py, then try to edit b.py → blocked (arg mismatch)
            ToolCall(tool="read_file", args={"path": "a.py"}),
            ToolCall(tool="edit_file", args={"path": "b.py"}),
            # Read b.py, then edit b.py → allowed
            ToolCall(tool="read_file", args={"path": "b.py"}),
            ToolCall(tool="edit_file", args={"path": "b.py"}),
            ToolCall(tool="submit", args={}),
        ])
        wf = _make_workflow(tools=tools, required_steps=[], terminal_tool="submit")
        runner = _make_runner(client)
        runner.on_message = collected.append
        result = await runner.run(wf, "go", prompt_vars={"role": "dev"})
        assert result == "submit_result"

        prereq_nudges = [m for m in collected if m.metadata.type == MessageType.PREREQUISITE_NUDGE]
        assert len(prereq_nudges) == 1  # only the b.py mismatch


# ── Multiple terminal tools ──────────────────────────────────────


class TestMultipleTerminalTools:
    """Runner-level multiple terminal tool support."""

    @pytest.mark.asyncio
    async def test_first_terminal_exits(self):
        """Workflow exits on the first terminal tool called."""
        tools = {
            "gather": _make_tool("gather"),
            "set_ac": _make_tool("set_ac", fn=lambda **kw: "ac_on"),
            "no_action": _make_tool("no_action", fn=lambda **kw: "skipped"),
        }
        client = MockClient([
            ToolCall(tool="gather", args={}),
            ToolCall(tool="set_ac", args={}),
        ])
        wf = _make_workflow(
            tools=tools, required_steps=["gather"],
            terminal_tool=["set_ac", "no_action"],
        )
        runner = _make_runner(client)
        result = await runner.run(wf, "manage ac", prompt_vars={"role": "agent"})
        assert result == "ac_on"

    @pytest.mark.asyncio
    async def test_second_terminal_exits(self):
        """Workflow also exits on the other terminal tool."""
        tools = {
            "gather": _make_tool("gather"),
            "set_ac": _make_tool("set_ac", fn=lambda **kw: "ac_on"),
            "no_action": _make_tool("no_action", fn=lambda **kw: "skipped"),
        }
        client = MockClient([
            ToolCall(tool="gather", args={}),
            ToolCall(tool="no_action", args={}),
        ])
        wf = _make_workflow(
            tools=tools, required_steps=["gather"],
            terminal_tool=["set_ac", "no_action"],
        )
        runner = _make_runner(client)
        result = await runner.run(wf, "manage ac", prompt_vars={"role": "agent"})
        assert result == "skipped"

    @pytest.mark.asyncio
    async def test_premature_terminal_blocked_for_all(self):
        """Both terminal tools are blocked before required steps."""
        collected = []
        tools = {
            "gather": _make_tool("gather"),
            "set_ac": _make_tool("set_ac"),
            "no_action": _make_tool("no_action"),
        }
        client = MockClient([
            ToolCall(tool="set_ac", args={}),       # premature
            ToolCall(tool="no_action", args={}),     # also premature
            ToolCall(tool="gather", args={}),         # required step
            ToolCall(tool="set_ac", args={}),         # now allowed
        ])
        wf = _make_workflow(
            tools=tools, required_steps=["gather"],
            terminal_tool=["set_ac", "no_action"],
        )
        runner = _make_runner(client)
        runner.on_message = collected.append
        await runner.run(wf, "go", prompt_vars={"role": "agent"})

        step_nudges = [m for m in collected if m.metadata.type == MessageType.STEP_NUDGE]
        assert len(step_nudges) == 2  # both premature attempts nudged


# ── Cancellation ─────────────────────────────────────────────────


class TestCancellation:
    """WorkflowRunner cancellation via cancel_event."""

    @pytest.mark.asyncio
    async def test_cancel_before_start(self):
        """Cancel event set before run() starts raises immediately."""
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        wf = _make_workflow()
        runner = _make_runner(client)
        cancel = asyncio.Event()
        cancel.set()

        with pytest.raises(WorkflowCancelledError) as exc_info:
            await runner.run(wf, "go", prompt_vars={"role": "dev"}, cancel_event=cancel)

        assert exc_info.value.iteration == 0
        assert exc_info.value.completed_steps == {}

    @pytest.mark.asyncio
    async def test_cancel_mid_workflow(self):
        """Cancel event set after first tool fires on next iteration."""
        cancel = asyncio.Event()

        def cancel_after_fetch(**kwargs):
            cancel.set()
            return "fetch_result"

        tools = {
            "fetch": _make_tool("fetch", fn=cancel_after_fetch),
            "submit": _make_tool("submit"),
        }
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        runner = _make_runner(client)

        with pytest.raises(WorkflowCancelledError) as exc_info:
            await runner.run(wf, "go", prompt_vars={"role": "dev"}, cancel_event=cancel)

        assert "fetch" in exc_info.value.completed_steps
        assert exc_info.value.iteration == 1

    @pytest.mark.asyncio
    async def test_cancel_preserves_messages(self):
        """Cancelled error includes conversation history."""
        cancel = asyncio.Event()

        def cancel_after_fetch(**kwargs):
            cancel.set()
            return "fetch_result"

        tools = {
            "fetch": _make_tool("fetch", fn=cancel_after_fetch),
            "submit": _make_tool("submit"),
        }
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        wf = _make_workflow(tools=tools, required_steps=["fetch"])
        runner = _make_runner(client)

        with pytest.raises(WorkflowCancelledError) as exc_info:
            await runner.run(wf, "go", prompt_vars={"role": "dev"}, cancel_event=cancel)

        messages = exc_info.value.messages
        assert len(messages) > 0
        # Should have system prompt, user input, tool call, tool result
        types = [m.metadata.type for m in messages]
        assert MessageType.SYSTEM_PROMPT in types
        assert MessageType.USER_INPUT in types
        assert MessageType.TOOL_RESULT in types

    @pytest.mark.asyncio
    async def test_no_cancel_event_runs_normally(self):
        """None cancel_event has no effect."""
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        wf = _make_workflow()
        runner = _make_runner(client)
        result = await runner.run(wf, "go", prompt_vars={"role": "dev"}, cancel_event=None)
        assert result == "submit_result"

    @pytest.mark.asyncio
    async def test_unset_cancel_event_runs_normally(self):
        """Cancel event that is never set doesn't interfere."""
        cancel = asyncio.Event()  # never set
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        wf = _make_workflow()
        runner = _make_runner(client)
        result = await runner.run(wf, "go", prompt_vars={"role": "dev"}, cancel_event=cancel)
        assert result == "submit_result"


# ── Custom retry nudge ───────────────────────────────────────────


class TestCustomRetryNudge:
    """WorkflowRunner custom retry nudge support."""

    @pytest.mark.asyncio
    async def test_custom_nudge_string(self):
        """String retry_nudge is used as static message."""
        collected = []
        client = MockClient([
            TextResponse(content="bare text"),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        wf = _make_workflow()
        runner = _make_runner(client)
        runner.on_message = collected.append
        runner._retry_nudge_fn = lambda _raw: "Wrap in respond tool."
        result = await runner.run(wf, "go", prompt_vars={"role": "dev"})

        nudges = [m for m in collected if m.metadata.type == MessageType.RETRY_NUDGE]
        assert len(nudges) == 1
        assert nudges[0].content == "Wrap in respond tool."

    @pytest.mark.asyncio
    async def test_custom_nudge_callable(self):
        """Callable retry_nudge receives raw response."""
        collected = []
        client = MockClient([
            TextResponse(content="my response"),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        wf = _make_workflow()
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=client, context_manager=ctx,
            retry_nudge=lambda raw: f"Please use a tool. You said: {raw[:10]}",
        )
        runner.on_message = collected.append
        result = await runner.run(wf, "go", prompt_vars={"role": "dev"})

        nudges = [m for m in collected if m.metadata.type == MessageType.RETRY_NUDGE]
        assert len(nudges) == 1
        assert "Please use a tool. You said: my respons" in nudges[0].content

    @pytest.mark.asyncio
    async def test_string_retry_nudge_constructor(self):
        """String passed to constructor is wrapped into callable."""
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=MockClient([]),
            context_manager=ctx,
            retry_nudge="Use the respond tool.",
        )
        assert runner._retry_nudge_fn is not None
        assert runner._retry_nudge_fn("anything") == "Use the respond tool."

    @pytest.mark.asyncio
    async def test_none_retry_nudge_uses_default(self):
        """None retry_nudge falls back to default."""
        collected = []
        client = MockClient([
            TextResponse(content="bare text"),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        wf = _make_workflow()
        runner = _make_runner(client)
        runner.on_message = collected.append
        await runner.run(wf, "go", prompt_vars={"role": "dev"})

        nudges = [m for m in collected if m.metadata.type == MessageType.RETRY_NUDGE]
        assert len(nudges) == 1
        assert "tool call" in nudges[0].content.lower()
