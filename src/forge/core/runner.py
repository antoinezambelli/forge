"""WorkflowRunner — the agentic tool-calling loop."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from forge.clients.base import LLMClient, RequestMetadata, StreamChunk
from forge.context.manager import ContextManager
from forge.core.inference import run_inference
from forge.core.messages import (
    ConversationState,
    MessageRole,
    ReasoningPart,
    RichMessage,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ValidArguments,
)
from forge.core.reasoning import DEFAULT_REASONING_REPLAY, ReasoningReplay, validate_reasoning_replay
from forge.core.workflow import Workflow
from forge.errors import MaxIterationsError, PrerequisiteError, StepEnforcementError, ToolCallError, ToolExecutionError, ToolResolutionError, WorkflowCancelledError
from forge.guardrails import ErrorTracker, ResponseValidator, StepEnforcer


class WorkflowRunner:
    """Executes a Workflow against an LLMClient with context management.

    1. Builds the initial message list (system prompt + user input)
    2. Sends messages to the LLM via the client (streaming or batch)
    3. Inspects the response — retry malformed/refusal replies with a nudge
    4. Validates and executes returned tool calls (batch-aware)
    5. Manages context budget via ContextManager
    6. Enforces required steps via StepEnforcer
    7. Terminates on terminal tool or max iterations

    Retry logic lives here, not on the client.
    """

    def __init__(
        self,
        client: LLMClient,
        context_manager: ContextManager,
        max_iterations: int = 10,
        max_retries_per_step: int = 3,
        max_tool_errors: int = 2,
        stream: bool = False,
        on_chunk: Callable[[StreamChunk], Awaitable[None]] | None = None,
        on_message: Callable[[RichMessage], None] | None = None,
        rescue_enabled: bool = True,
        retry_nudge: Callable[[str], str] | str | None = None,
        reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
    ):
        """
        Args:
            client: The LLM client to send messages through.
            context_manager: Manages context budget and triggers compaction.
            max_iterations: Hard ceiling on total LLM round trips. Retries
                consume iterations.
            max_retries_per_step: Consecutive formatting failures before
                raising ToolCallError. Resets on valid calls.
            max_tool_errors: Consecutive tool execution errors before raising
                ToolExecutionError. Errors are fed back to the model for
                self-correction. Resets on successful execution.
            stream: If True, uses send_stream(). Streaming is a side channel
                — the runner still waits for the FINAL chunk before acting.
            on_chunk: Async callback for each StreamChunk (awaited per chunk).
                Ignored if stream=False.
            on_message: Callback fired for each new RichMessage, including
                transient context warnings. Shared replies are emitted once.
            rescue_enabled: If False, skip rescue_tool_call() — bare text
                goes straight to retry nudge (or failure if retries=0).
            retry_nudge: Custom nudge for bare text responses. Pass a string
                for a static message, or a callable ``(raw_response) -> str``
                for dynamic nudges. If None, uses the default.
            reasoning_replay: How much captured reasoning to replay to the
                backend on later turns: ``full``, ``keep-last``, or ``none``.
        """
        self.client = client
        self.context_manager = context_manager
        self.max_iterations = max_iterations
        self.max_retries_per_step = max_retries_per_step
        self.max_tool_errors = max_tool_errors
        self.stream = stream
        self.on_chunk = on_chunk
        self.on_message = on_message
        self.rescue_enabled = rescue_enabled
        self.reasoning_replay = validate_reasoning_replay(reasoning_replay)
        if isinstance(retry_nudge, str):
            self._retry_nudge_fn: Callable[[str], str] | None = lambda _raw, _msg=retry_nudge: _msg
        else:
            self._retry_nudge_fn = retry_nudge

    async def run(
        self,
        workflow: Workflow,
        user_message: str,
        prompt_vars: dict[str, str] | None = None,
        initial_messages: list[RichMessage] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> Any:
        """Execute the workflow and return the terminal tool's result.

        Args:
            workflow: The workflow to execute.
            user_message: The user's input message.
            prompt_vars: Variables for the system prompt template.
            initial_messages: A list of RichMessage objects with typed parts;
                strings and legacy messages are rejected. Shallow-list seeds
                the conversation instead of building a fresh system prompt + user
                input. The on_message callback fires only for NEW messages
                created during this run, not the replayed history. The caller
                must include the system prompt and new user message in the
                seed.
            cancel_event: If provided and set, the runner will raise
                WorkflowCancelledError at the start of the next iteration.
                Checked once per loop, before the inference call.

        Raises:
            MaxIterationsError: If max_iterations exceeded without terminal tool.
            ToolCallError: If max_retries_per_step exhausted on a single step.
            ToolExecutionError: If a tool callable raised and the model failed
                to self-correct after max_tool_errors consecutive attempts.
            WorkflowCancelledError: If cancel_event was set during execution.
        """
        if initial_messages is not None:
            if not isinstance(initial_messages, list) or any(
                not isinstance(message, RichMessage)
                or not isinstance(message.parts, list)
                or any(
                    not isinstance(part, (TextPart, ReasoningPart, ToolCallPart, ToolResultPart))
                    for part in message.parts
                )
                for message in initial_messages
            ):
                raise TypeError("initial_messages must be a list of RichMessage objects with typed parts")
            state = ConversationState(list(initial_messages))
        else:
            state = ConversationState()

        def _notify(messages: list[RichMessage]) -> None:
            if self.on_message is not None:
                for message in messages:
                    self.on_message(message)

        def _emit(message: RichMessage) -> None:
            state.messages.append(message)
            _notify([message])

        if initial_messages is None:
            rendered_prompt = workflow.build_system_prompt(**(prompt_vars or {}))
            _emit(RichMessage(MessageRole.SYSTEM, [TextPart(rendered_prompt)]))
            _emit(RichMessage(MessageRole.USER, [TextPart(user_message)]))

        def _emit_result(
            call: ToolCallPart, text: str, outcome: Literal["success", "error"],
            purpose: str | None = None,
        ) -> None:
            _emit(RichMessage(
                MessageRole.TOOL,
                [ToolResultPart(call.call_id, [TextPart(text)], outcome)],
                step_index=iteration,
                forge_purpose=purpose,
                extra_fields={"name": call.name},
            ))

        # Step 2 — Initialize guardrail middleware
        tool_names = list(workflow.tools.keys())
        validator = ResponseValidator(
            tool_names, rescue_enabled=self.rescue_enabled,
            retry_nudge_fn=self._retry_nudge_fn,
        )
        tool_prerequisites = workflow.prerequisites
        step_enforcer = StepEnforcer(
            required_steps=workflow.required_steps,
            terminal_tools=workflow.terminal_tools,
            tool_prerequisites=tool_prerequisites,
        )
        error_tracker = ErrorTracker(
            max_retries=self.max_retries_per_step,
            max_tool_errors=self.max_tool_errors,
        )

        # Step 3 — Main loop (one LLM call per iteration, retries consume iterations)
        tool_definitions = workflow.get_tool_definitions()
        tool_call_counter = 0
        iteration = 0

        while iteration < self.max_iterations:
            # 3.0 — Check for cancellation
            if cancel_event is not None and cancel_event.is_set():
                raise WorkflowCancelledError(
                    messages=state.messages,
                    completed_steps=step_enforcer.completed_steps,
                    iteration=iteration,
                )

            # 3a — Inference: compact, render, send, validate, retry
            result = await run_inference(
                state=state,
                client=self.client,
                context_manager=self.context_manager,
                validator=validator,
                error_tracker=error_tracker,
                tool_definitions=tool_definitions,
                tool_call_counter=tool_call_counter,
                step_index=iteration,
                step_hint=step_enforcer.summary_hint(),
                max_attempts=self.max_iterations - iteration,
                stream=self.stream,
                on_chunk=self.on_chunk,
                metadata=RequestMetadata(reasoning_replay=self.reasoning_replay),
                _on_exhausted=_notify,
            )
            # max_attempts exhausted — iteration budget spent
            if result is None:
                break
            # Retries consume iterations (preserves pre-extraction semantics)
            iteration += result.attempts
            # Inference owns recording and call identity. Group the same accepted
            # reply with this completed iteration's corrections and results.
            result.response.step_index = iteration
            _notify(result.new_messages)
            tool_call_counter = result.tool_call_counter
            tool_calls = result.response.tool_calls
            if not tool_calls:
                continue

            # 3b — Check for premature terminal
            step_check = step_enforcer.check(tool_calls)

            if step_check.needs_nudge:
                if step_enforcer.premature_exhausted:
                    attempted = next(
                        tc.name for tc in tool_calls
                        if tc.name in workflow.terminal_tools
                    )
                    raise StepEnforcementError(
                        terminal_tool=attempted,
                        attempts=step_enforcer.premature_attempts,
                        pending_steps=step_enforcer.pending(),
                    )
                nudge = step_check.nudge
                for call in tool_calls:
                    _emit_result(call, f"[StepEnforcementError] {nudge.content}", "error", nudge.kind)
                continue

            # 3b.2 — Check prerequisites
            prereq_check = step_enforcer.check_prerequisites(tool_calls)

            if prereq_check.needs_nudge:
                if step_enforcer.prereq_exhausted:
                    # Find the first violating tool for the error
                    for tc in tool_calls:
                        prereqs = tool_prerequisites.get(tc.name)
                        if prereqs:
                            result = step_enforcer._tracker.check_prerequisites(
                                tc.name, tc.arguments.value, prereqs,
                            )
                            if not result.satisfied:
                                raise PrerequisiteError(
                                    tool_name=tc.name,
                                    violations=step_enforcer.prereq_violations,
                                    missing_prereqs=result.missing,
                                )
                nudge = prereq_check.nudge
                for call in tool_calls:
                    _emit_result(call, f"[PrerequisiteError] {nudge.content}", "error", nudge.kind)
                continue

            # 3c — Execute all tool calls in the batch
            # Execute each tool and emit results
            batch_had_error = False
            last_error: tuple[str, Exception] | None = None
            terminal_result = None
            for tc in tool_calls:
                assert isinstance(tc.arguments, ValidArguments)
                fn = workflow.handlers[tc.name]
                try:
                    if inspect.iscoroutinefunction(fn):
                        result_val = await fn(**tc.arguments.value)
                    else:
                        result_val = fn(**tc.arguments.value)
                except ToolResolutionError as exc:
                    _emit_result(tc, f"[ToolResolutionError] {exc}", "error")
                    if tc.name in workflow.terminal_tools:
                        terminal_result = exc
                    continue
                except Exception as exc:
                    batch_had_error = True
                    last_error = (tc.name, exc)
                    _emit_result(tc, f"[ToolError] {type(exc).__name__}: {exc}", "error")
                    if tc.name in workflow.terminal_tools:
                        terminal_result = exc
                    continue

                # Success
                step_enforcer.record(tc.name, tc.arguments.value)
                result_str = result_val if isinstance(result_val, str) else json.dumps(result_val)
                _emit_result(tc, result_str, "success")

                if tc.name in workflow.terminal_tools:
                    terminal_result = result_val

            # 3d — Post-batch bookkeeping
            if batch_had_error:
                error_tracker.record_result(success=False)
                if error_tracker.tool_errors_exhausted:
                    assert last_error is not None
                    raise ToolExecutionError(
                        last_error[0],
                        cause=last_error[1],
                    )
            else:
                error_tracker.reset_errors()
                step_enforcer.reset_premature()
                step_enforcer.reset_prereq_violations()

            # 3e — If terminal tool was in the batch and succeeded, return
            if terminal_result is not None and not isinstance(terminal_result, Exception):
                return terminal_result

        # Step 4 — Max iterations exceeded
        raise MaxIterationsError(
            self.max_iterations, step_enforcer.completed_steps, step_enforcer.pending()
        )
