"""WorkflowRunner — the agentic tool-calling loop."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from forge.clients.base import LLMClient, StreamChunk
from forge.context.manager import ContextManager
from forge.core.inference import _NUDGE_KIND_TO_TYPE, _build_tool_call_infos, run_inference
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from forge.core.workflow import ToolCall, TextResponse, Workflow, ToolSpec
from forge.errors import MaxIterationsError, StepEnforcementError, ToolCallError, ToolExecutionError, ToolResolutionError
from forge.guardrails import ErrorTracker, ResponseValidator, StepEnforcer


class WorkflowRunner:
    """Executes a Workflow against an LLMClient with context management.

    1. Builds the initial message list (system prompt + user input)
    2. Sends messages to the LLM via the client (streaming or batch)
    3. Inspects the response — if TextResponse (malformed/refusal), retries with nudge
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
        on_message: Callable[[Message], None] | None = None,
        rescue_enabled: bool = True,
    ):
        """
        Args:
            client: The LLM client to send messages through.
            context_manager: Manages context budget and triggers compaction.
            max_iterations: Hard ceiling on total LLM round trips. Retries
                consume iterations.
            max_retries_per_step: Consecutive formatting failures before
                raising ToolCallError. Resets on any valid ToolCall.
            max_tool_errors: Consecutive tool execution errors before raising
                ToolExecutionError. Errors are fed back to the model for
                self-correction. Resets on successful execution.
            stream: If True, uses send_stream(). Streaming is a side channel
                — the runner still waits for the FINAL chunk before acting.
            on_chunk: Async callback for each StreamChunk (awaited per chunk).
                Ignored if stream=False.
            on_message: Callback fired when a Message is appended to history.
                Does not affect runner behavior.
            rescue_enabled: If False, skip rescue_tool_call() — TextResponse
                goes straight to retry nudge (or failure if retries=0).
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

    async def run(
        self,
        workflow: Workflow,
        user_message: str,
        prompt_vars: dict[str, str] | None = None,
        initial_messages: list[Message] | None = None,
    ) -> Any:
        """Execute the workflow and return the terminal tool's result.

        Args:
            workflow: The workflow to execute.
            user_message: The user's input message.
            prompt_vars: Variables for the system prompt template.
            initial_messages: If provided, seeds the conversation with these
                messages instead of building a fresh system prompt + user
                input. The on_message callback fires only for NEW messages
                created during this run, not the replayed history. The caller
                must include the system prompt and new user message in the
                seed.

        Raises:
            MaxIterationsError: If max_iterations exceeded without terminal tool.
            ToolCallError: If max_retries_per_step exhausted on a single step.
            ToolExecutionError: If a tool callable raised and the model failed
                to self-correct after max_tool_errors consecutive attempts.
        """
        # Step 1 — Build initial messages
        if initial_messages is not None:
            messages: list[Message] = list(initial_messages)

            def _emit(msg: Message) -> None:
                messages.append(msg)
                if self.on_message is not None:
                    self.on_message(msg)
        else:
            rendered_prompt = workflow.build_system_prompt(**(prompt_vars or {}))
            messages: list[Message] = []

            def _emit(msg: Message) -> None:
                messages.append(msg)
                if self.on_message is not None:
                    self.on_message(msg)

            _emit(Message(MessageRole.SYSTEM, rendered_prompt, MessageMeta(MessageType.SYSTEM_PROMPT)))
            _emit(Message(MessageRole.USER, user_message, MessageMeta(MessageType.USER_INPUT)))

        # Step 2 — Initialize guardrail middleware
        tool_names = list(workflow.tools.keys())
        validator = ResponseValidator(tool_names, rescue_enabled=self.rescue_enabled)
        step_enforcer = StepEnforcer(
            required_steps=workflow.required_steps,
            terminal_tool=workflow.terminal_tool,
        )
        error_tracker = ErrorTracker(
            max_retries=self.max_retries_per_step,
            max_tool_errors=self.max_tool_errors,
        )

        # Step 3 — Main loop (one LLM call per iteration, retries consume iterations)
        tool_specs = workflow.get_tool_specs()
        tool_call_counter = 0
        iteration = 0

        while iteration < self.max_iterations:
            # 3a — Inference: compact, fold, serialize, send, validate, retry
            result = await run_inference(
                messages=messages,
                client=self.client,
                context_manager=self.context_manager,
                validator=validator,
                error_tracker=error_tracker,
                tool_specs=tool_specs,
                tool_call_counter=tool_call_counter,
                step_index=iteration,
                step_hint=step_enforcer.summary_hint(),
                max_attempts=self.max_iterations - iteration,
                stream=self.stream,
                on_chunk=self.on_chunk,
            )
            # max_attempts exhausted — iteration budget spent
            if result is None:
                break
            # Retries consume iterations (preserves pre-extraction semantics)
            iteration += result.attempts
            # Emit new messages from retries (assistant text, nudges)
            for msg in result.new_messages:
                if self.on_message is not None:
                    self.on_message(msg)
            tool_call_counter = result.tool_call_counter
            tool_calls = result.response

            # 3b — Check for premature terminal
            step_check = step_enforcer.check(tool_calls)

            if step_check.needs_nudge:
                if step_enforcer.premature_exhausted:
                    raise StepEnforcementError(
                        terminal_tool=workflow.terminal_tool,
                        attempts=step_enforcer.premature_attempts,
                        pending_steps=step_enforcer.pending(),
                    )
                if tool_calls[0].reasoning:
                    _emit(Message(
                        MessageRole.ASSISTANT,
                        tool_calls[0].reasoning,
                        MessageMeta(MessageType.REASONING, step_index=iteration),
                    ))
                tc_infos, tool_call_counter = _build_tool_call_infos(tool_calls, tool_call_counter)
                _emit(Message(
                    MessageRole.ASSISTANT,
                    "",
                    MessageMeta(MessageType.TOOL_CALL, step_index=iteration),
                    tool_calls=tc_infos,
                ))
                nudge = step_check.nudge
                nudge_type = _NUDGE_KIND_TO_TYPE[nudge.kind]
                _emit(Message(
                    MessageRole.USER,
                    nudge.content,
                    MessageMeta(nudge_type, step_index=iteration),
                ))
                continue

            # 3c — Execute all tool calls in the batch
            tc_infos, tool_call_counter = _build_tool_call_infos(tool_calls, tool_call_counter)
            call_ids = [tc.call_id for tc in tc_infos]

            # Emit reasoning (from first call) and assistant message
            if tool_calls[0].reasoning:
                _emit(Message(
                    MessageRole.ASSISTANT,
                    tool_calls[0].reasoning,
                    MessageMeta(MessageType.REASONING, step_index=iteration),
                ))
            _emit(Message(
                MessageRole.ASSISTANT,
                "",
                MessageMeta(MessageType.TOOL_CALL, step_index=iteration),
                tool_calls=tc_infos,
            ))

            # Execute each tool and emit results
            batch_had_error = False
            last_error: tuple[str, Exception] | None = None
            terminal_result = None
            for i, tc in enumerate(tool_calls):
                tc_id = call_ids[i]
                fn = workflow.get_callable(tc.tool)
                try:
                    if asyncio.iscoroutinefunction(fn):
                        result_val = await fn(**tc.args)
                    else:
                        result_val = fn(**tc.args)
                except ToolResolutionError as exc:
                    _emit(Message(
                        MessageRole.TOOL,
                        f"[ToolResolutionError] {exc}",
                        MessageMeta(MessageType.TOOL_RESULT, step_index=iteration),
                        tool_name=tc.tool,
                        tool_call_id=tc_id,
                    ))
                    if tc.tool == workflow.terminal_tool:
                        terminal_result = exc
                    continue
                except Exception as exc:
                    batch_had_error = True
                    last_error = (tc.tool, exc)
                    _emit(Message(
                        MessageRole.TOOL,
                        f"[ToolError] {type(exc).__name__}: {exc}",
                        MessageMeta(MessageType.TOOL_RESULT, step_index=iteration),
                        tool_name=tc.tool,
                        tool_call_id=tc_id,
                    ))
                    if tc.tool == workflow.terminal_tool:
                        terminal_result = exc
                    continue

                # Success
                step_enforcer.record(tc.tool)
                result_str = result_val if isinstance(result_val, str) else json.dumps(result_val)
                _emit(Message(
                    MessageRole.TOOL,
                    result_str,
                    MessageMeta(MessageType.TOOL_RESULT, step_index=iteration),
                    tool_name=tc.tool,
                    tool_call_id=tc_id,
                ))

                if tc.tool == workflow.terminal_tool:
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

            # 3e — If terminal tool was in the batch and succeeded, return
            if terminal_result is not None and not isinstance(terminal_result, Exception):
                return terminal_result

        # Step 4 — Max iterations exceeded
        raise MaxIterationsError(
            self.max_iterations, step_enforcer.completed_steps, step_enforcer.pending()
        )
