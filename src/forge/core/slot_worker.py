"""SlotWorker — serialized access to a WorkflowRunner with priority queuing."""

from __future__ import annotations

import asyncio
from typing import Any

from forge.core.workflow import Workflow
from forge.core.messages import Message
from forge.core.runner import WorkflowRunner


class SlotWorker:
    """Serializes workflow execution on a single inference slot.

    Wraps a WorkflowRunner with a priority queue. Each ``submit()`` call
    waits for its turn, runs to completion, and returns the result.
    Cancel events are wired automatically.

    Priority is an int — lower values run first. The consumer defines
    what the levels mean (forge imposes no semantics). Default priority
    is 0 for all tasks (pure FIFO).

    Auto-preemption: if a submitted task has strictly higher priority
    (lower int) than the currently running task, the running task is
    cancelled via ``cancel_event`` and the higher-priority task takes over.

    Args:
        runner: The WorkflowRunner to serialize access to.
    """

    def __init__(self, runner: WorkflowRunner) -> None:
        self.runner = runner
        self._queue: asyncio.PriorityQueue[
            tuple[int, int, Workflow, str, dict[str, str] | None, asyncio.Future[Any]]
        ] = asyncio.PriorityQueue()
        self._counter = 0
        self._cancel_event: asyncio.Event | None = None
        self._current_priority: int | None = None
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the worker loop."""
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Stop the worker loop. Pending tasks receive CancelledError."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def submit(
        self,
        workflow: Workflow,
        user_message: str,
        priority: int = 0,
        prompt_vars: dict[str, str] | None = None,
    ) -> Any:
        """Submit a workflow for execution and wait for the result.

        Args:
            workflow: The workflow to execute.
            user_message: The user's input message.
            priority: Execution priority (lower = higher priority).
                Default 0. Consumer defines the semantics.
            prompt_vars: Variables for the system prompt template.

        Returns:
            The terminal tool's result.

        Raises:
            WorkflowCancelledError: If the task was preempted by a
                higher-priority submission.
            Any exception raised by the WorkflowRunner.
        """
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._counter += 1
        await self._queue.put(
            (priority, self._counter, workflow, user_message, prompt_vars, future)
        )

        # Auto-preempt: if a higher-priority task arrives while a
        # lower-priority task is running, cancel the running task.
        if (
            self._current_priority is not None
            and priority < self._current_priority
            and self._cancel_event is not None
        ):
            self._cancel_event.set()

        return await future

    def cancel_current(self) -> None:
        """Cancel the currently running workflow, if any."""
        if self._cancel_event is not None:
            self._cancel_event.set()

    @property
    def running_priority(self) -> int | None:
        """Priority of the currently running task, or None if idle."""
        return self._current_priority

    @property
    def pending(self) -> int:
        """Number of tasks waiting in the queue."""
        return self._queue.qsize()

    async def _worker(self) -> None:
        """Process tasks from the queue sequentially."""
        while True:
            priority, _counter, workflow, user_message, prompt_vars, future = (
                await self._queue.get()
            )

            # Skip cancelled futures (consumer cancelled before we got to it)
            if future.cancelled():
                continue

            self._cancel_event = asyncio.Event()
            self._current_priority = priority

            try:
                result = await self.runner.run(
                    workflow,
                    user_message,
                    prompt_vars=prompt_vars,
                    cancel_event=self._cancel_event,
                )
                if not future.done():
                    future.set_result(result)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                self._cancel_event = None
                self._current_priority = None
