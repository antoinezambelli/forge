"""Tests for SlotWorker — serialized access to WorkflowRunner with priority queuing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.messages import MessageType
from forge.core.runner import WorkflowRunner
from forge.core.slot_worker import SlotWorker
from forge.core.workflow import (
    TextResponse,
    ToolCall,
    ToolDef,
    ToolSpec,
    Workflow,
)
from forge.errors import WorkflowCancelledError
from pydantic import BaseModel


class EmptyParams(BaseModel):
    pass


# ── Helpers ──────────────────────────────────────────────────


def _make_tool(name: str, fn=None) -> ToolDef:
    if fn is None:
        fn = lambda **kwargs: f"{name}_result"
    return ToolDef(
        spec=ToolSpec(name=name, description=f"Tool {name}", parameters=EmptyParams),
        callable=fn,
    )


def _make_workflow(terminal_tool: str = "submit") -> Workflow:
    return Workflow(
        name="test_wf",
        description="A test workflow",
        tools={
            "fetch": _make_tool("fetch"),
            terminal_tool: _make_tool(terminal_tool),
        },
        required_steps=["fetch"],
        terminal_tool=terminal_tool,
        system_prompt_template="You are a {role}.",
    )


class MockClient:
    """Mock LLMClient that returns scripted responses."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self._call_index = 0

    async def send(self, messages, tools=None, sampling=None):
        resp = self.responses[self._call_index]
        self._call_index += 1
        if isinstance(resp, ToolCall):
            return [resp]
        return resp

    async def send_stream(self, messages, tools=None, sampling=None):
        raise NotImplementedError

    async def get_context_length(self):
        return None


def _make_worker(client: MockClient) -> SlotWorker:
    ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
    runner = WorkflowRunner(client=client, context_manager=ctx)
    return SlotWorker(runner)


# ── Basic operation ──────────────────────────────────────────


class TestBasicOperation:

    @pytest.mark.asyncio
    async def test_submit_returns_result(self):
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        worker = _make_worker(client)
        await worker.start()
        try:
            result = await worker.submit(
                _make_workflow(), "go", prompt_vars={"role": "dev"},
            )
            assert result == "submit_result"
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_fifo_ordering(self):
        """Tasks submitted with equal priority run in FIFO order."""
        order = []

        def make_client(tag):
            def fn(**kw):
                order.append(tag)
                return f"{tag}_done"
            client = MockClient([
                ToolCall(tool="fetch", args={}),
                ToolCall(tool="submit", args={}),
            ])
            return client, fn

        # We need a single client that can handle multiple sequential workflows.
        # Use a client with enough responses for 2 sequential workflows.
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])

        def track_first(**kw):
            order.append("first")
            return "first_result"

        def track_second(**kw):
            order.append("second")
            return "second_result"

        wf1 = Workflow(
            name="wf1", description="first",
            tools={"fetch": _make_tool("fetch"), "submit": _make_tool("submit", fn=track_first)},
            required_steps=["fetch"], terminal_tool="submit",
            system_prompt_template="You are a {role}.",
        )
        wf2 = Workflow(
            name="wf2", description="second",
            tools={"fetch": _make_tool("fetch"), "submit": _make_tool("submit", fn=track_second)},
            required_steps=["fetch"], terminal_tool="submit",
            system_prompt_template="You are a {role}.",
        )

        worker = _make_worker(client)
        await worker.start()
        try:
            r1 = await worker.submit(wf1, "first", prompt_vars={"role": "dev"})
            r2 = await worker.submit(wf2, "second", prompt_vars={"role": "dev"})
            assert r1 == "first_result"
            assert r2 == "second_result"
            assert order == ["first", "second"]
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_idle_properties(self):
        client = MockClient([])
        worker = _make_worker(client)
        assert worker.running_priority is None
        assert worker.pending == 0

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        client = MockClient([])
        worker = _make_worker(client)
        await worker.start()
        task1 = worker._worker_task
        await worker.start()  # should not create a second task
        assert worker._worker_task is task1
        await worker.stop()


# ── Priority ─────────────────────────────────────────────────


class TestPriority:

    @pytest.mark.asyncio
    async def test_higher_priority_runs_first_when_queued(self):
        """When multiple tasks are queued, lower int runs first."""
        order = []

        # We'll submit two tasks while the worker is blocked on a third.
        # Use an event to control when the first task completes.
        gate = asyncio.Event()

        def blocking_fetch(**kw):
            # We need this to be async to block
            return "fetch_result"

        def track_high(**kw):
            order.append("high")
            return "high_result"

        def track_low(**kw):
            order.append("low")
            return "low_result"

        # First workflow blocks, then two more queue up
        client = MockClient([
            # First workflow (blocker)
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
            # High priority (should run second)
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
            # Low priority (should run third)
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])

        wf_blocker = _make_workflow()
        wf_high = Workflow(
            name="high", description="high",
            tools={"fetch": _make_tool("fetch"), "submit": _make_tool("submit", fn=track_high)},
            required_steps=["fetch"], terminal_tool="submit",
            system_prompt_template="You are a {role}.",
        )
        wf_low = Workflow(
            name="low", description="low",
            tools={"fetch": _make_tool("fetch"), "submit": _make_tool("submit", fn=track_low)},
            required_steps=["fetch"], terminal_tool="submit",
            system_prompt_template="You are a {role}.",
        )

        worker = _make_worker(client)
        await worker.start()
        try:
            # Run blocker first, then queue low and high
            r_block = await worker.submit(wf_blocker, "block", prompt_vars={"role": "dev"})

            # Now queue low (priority=10) then high (priority=1)
            # High should run before low despite being submitted second
            task_low = asyncio.create_task(
                worker.submit(wf_low, "low", priority=10, prompt_vars={"role": "dev"})
            )
            task_high = asyncio.create_task(
                worker.submit(wf_high, "high", priority=1, prompt_vars={"role": "dev"})
            )

            r_high = await task_high
            r_low = await task_low

            assert r_high == "high_result"
            assert r_low == "low_result"
            assert order == ["high", "low"]
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_default_priority_is_zero(self):
        """Default priority is 0 (equal for all tasks)."""
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        worker = _make_worker(client)
        await worker.start()
        try:
            # Submit without explicit priority
            result = await worker.submit(
                _make_workflow(), "go", prompt_vars={"role": "dev"},
            )
            assert result == "submit_result"
        finally:
            await worker.stop()


# ── Preemption ───────────────────────────────────────────────


class TestPreemption:

    @pytest.mark.asyncio
    async def test_cancel_current(self):
        """cancel_current() cancels the running workflow."""
        gate = asyncio.Event()

        async def slow_fetch(**kw):
            await gate.wait()
            return "fetch_result"

        client = MockClient([
            ToolCall(tool="fetch", args={}),
        ])
        wf = Workflow(
            name="slow", description="slow",
            tools={
                "fetch": ToolDef(
                    spec=ToolSpec(name="fetch", description="fetch", parameters=EmptyParams),
                    callable=slow_fetch,
                ),
                "submit": _make_tool("submit"),
            },
            required_steps=["fetch"], terminal_tool="submit",
            system_prompt_template="You are a {role}.",
        )

        worker = _make_worker(client)
        await worker.start()
        try:
            task = asyncio.create_task(
                worker.submit(wf, "go", prompt_vars={"role": "dev"})
            )
            await asyncio.sleep(0.05)  # let the worker start
            worker.cancel_current()
            gate.set()  # unblock the tool so the runner can check cancel

            with pytest.raises(WorkflowCancelledError):
                await task
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_auto_preempt_lower_priority(self):
        """Higher-priority submit auto-cancels lower-priority running task."""
        gate = asyncio.Event()

        async def slow_fetch(**kw):
            await gate.wait()
            return "fetch_result"

        # Slow workflow on slot (priority 10)
        client = MockClient([
            ToolCall(tool="fetch", args={}),  # slow task
            ToolCall(tool="fetch", args={}),  # high-priority task
            ToolCall(tool="submit", args={}),
        ])
        wf_slow = Workflow(
            name="slow", description="slow",
            tools={
                "fetch": ToolDef(
                    spec=ToolSpec(name="fetch", description="fetch", parameters=EmptyParams),
                    callable=slow_fetch,
                ),
                "submit": _make_tool("submit"),
            },
            required_steps=["fetch"], terminal_tool="submit",
            system_prompt_template="You are a {role}.",
        )
        wf_fast = _make_workflow()

        worker = _make_worker(client)
        await worker.start()
        try:
            slow_task = asyncio.create_task(
                worker.submit(wf_slow, "slow", priority=10, prompt_vars={"role": "dev"})
            )
            await asyncio.sleep(0.05)  # let slow task start running

            # Submit higher priority — should auto-cancel the slow task
            fast_task = asyncio.create_task(
                worker.submit(wf_fast, "fast", priority=1, prompt_vars={"role": "dev"})
            )
            gate.set()  # unblock so cancel can propagate

            result = await fast_task
            assert result == "submit_result"

            with pytest.raises(WorkflowCancelledError):
                await slow_task
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_no_preempt_equal_priority(self):
        """Equal priority does not trigger preemption."""
        client = MockClient([
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        worker = _make_worker(client)
        await worker.start()
        try:
            r1 = await worker.submit(
                _make_workflow(), "first", priority=5, prompt_vars={"role": "dev"},
            )
            r2 = await worker.submit(
                _make_workflow(), "second", priority=5, prompt_vars={"role": "dev"},
            )
            assert r1 == "submit_result"
            assert r2 == "submit_result"
        finally:
            await worker.stop()


# ── Error handling ───────────────────────────────────────────


class TestErrorHandling:

    @pytest.mark.asyncio
    async def test_exception_propagates_to_caller(self):
        """Exceptions from the runner propagate to the submit() caller."""
        from forge.errors import MaxIterationsError

        client = MockClient([
            TextResponse(content="bare text"),
            TextResponse(content="bare text again"),
            TextResponse(content="bare text third"),
            TextResponse(content="bare text fourth"),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=client, context_manager=ctx,
            max_iterations=3, rescue_enabled=False,
        )
        worker = SlotWorker(runner)
        await worker.start()
        try:
            with pytest.raises(Exception):
                await worker.submit(
                    _make_workflow(), "go", prompt_vars={"role": "dev"},
                )
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_worker_continues_after_error(self):
        """Worker processes next task after a failed one."""
        client = MockClient([
            # First workflow: will fail (not enough responses to complete)
            TextResponse(content="fail"),
            TextResponse(content="fail"),
            TextResponse(content="fail"),
            TextResponse(content="fail"),
            # Second workflow: succeeds
            ToolCall(tool="fetch", args={}),
            ToolCall(tool="submit", args={}),
        ])
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=100_000)
        runner = WorkflowRunner(
            client=client, context_manager=ctx,
            max_iterations=3, rescue_enabled=False,
        )
        worker = SlotWorker(runner)
        await worker.start()
        try:
            # First task fails
            with pytest.raises(Exception):
                await worker.submit(
                    _make_workflow(), "fail", prompt_vars={"role": "dev"},
                )
            # Second task succeeds
            result = await worker.submit(
                _make_workflow(), "succeed", prompt_vars={"role": "dev"},
            )
            assert result == "submit_result"
        finally:
            await worker.stop()
