"""WorkflowRunner — the agentic tool-calling loop."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from forge.clients.base import LLMClient, StreamChunk, ChunkType
from forge.context.manager import ContextManager
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from forge.core.steps import StepTracker
from forge.core.workflow import LLMResponse, ToolCall, TextResponse, Workflow, ToolSpec
from forge.errors import MaxIterationsError, StepEnforcementError, StreamError, ToolCallError, ToolExecutionError, ToolResolutionError
from forge.prompts.nudges import retry_nudge, step_nudge, unknown_tool_nudge
from forge.prompts.templates import rescue_tool_call


class WorkflowRunner:
    """Executes a Workflow against an LLMClient with context management.

    1. Builds the initial message list (system prompt + user input)
    2. Sends messages to the LLM via the client (streaming or batch)
    3. Inspects the response — if TextResponse (malformed/refusal), retries with nudge
    4. Validates and executes returned tool calls (batch-aware)
    5. Manages context budget via ContextManager
    6. Enforces required steps via StepTracker
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

        # Step 2 — Initialize StepTracker
        step_tracker = StepTracker(required_steps=workflow.required_steps)

        # Step 3 — Main loop (one LLM call per iteration)
        tool_specs = workflow.get_tool_specs()
        consecutive_retries = 0
        consecutive_tool_errors = 0
        premature_terminal_attempts = 0
        tool_call_counter = 0

        for iteration in range(self.max_iterations):
            # 3a — Compact context
            messages = self.context_manager.maybe_compact(
                messages, step_index=iteration, step_hint=step_tracker.summary_hint()
            )

            # 3b — Serialize and send
            # Fold REASONING messages into the following TOOL_CALL message's
            # content field so the wire format has one assistant message with
            # both content and tool_calls (valid OpenAI format, invisible to
            # Jinja parity checker). Internal Message list stays separate for
            # compaction.
            fmt = getattr(self.client, "api_format", "ollama")
            api_messages: list[dict[str, Any]] = []
            pending_reasoning: str | None = None
            for m in messages:
                if m.metadata.type == MessageType.REASONING and m.role == MessageRole.ASSISTANT:
                    pending_reasoning = m.content
                    continue
                d = m.to_api_dict(format=fmt)
                if pending_reasoning is not None and m.tool_calls is not None:
                    d["content"] = pending_reasoning
                    pending_reasoning = None
                elif pending_reasoning is not None:
                    # REASONING not followed by TOOL_CALL — emit it standalone
                    api_messages.append({"role": "assistant", "content": pending_reasoning})
                    pending_reasoning = None
                api_messages.append(d)
            if pending_reasoning is not None:
                api_messages.append({"role": "assistant", "content": pending_reasoning})
            if self.stream:
                response = await self._send_streaming(api_messages, tool_specs)
            else:
                response = await self.client.send(api_messages, tools=tool_specs)

            # 3c — TextResponse: try to rescue tool calls, else retry nudge
            if isinstance(response, TextResponse):
                rescued = rescue_tool_call(response.content, list(workflow.tools.keys())) if self.rescue_enabled else []
                if not rescued:
                    consecutive_retries += 1
                    if consecutive_retries > self.max_retries_per_step:
                        raise ToolCallError(
                            f"Retries exhausted after {self.max_retries_per_step} "
                            "consecutive failed attempts",
                            raw_response=response.content,
                        )
                    _emit(Message(
                        MessageRole.ASSISTANT,
                        response.content,
                        MessageMeta(MessageType.TEXT_RESPONSE, step_index=iteration),
                    ))
                    _emit(Message(
                        MessageRole.USER,
                        retry_nudge(response.content),
                        MessageMeta(MessageType.RETRY_NUDGE, step_index=iteration),
                    ))
                    continue
                response = rescued

            # 3d — Got list[ToolCall]
            tool_calls = response

            # 3e — Validate all tool names exist
            unknown = [tc for tc in tool_calls if tc.tool not in workflow.tools]
            if unknown:
                consecutive_retries += 1
                if consecutive_retries > self.max_retries_per_step:
                    raise ToolCallError(
                        f"Retries exhausted after {self.max_retries_per_step} "
                        "consecutive failed attempts",
                        raw_response=str([(tc.tool, tc.args) for tc in tool_calls]),
                    )
                # Emit reasoning if present (from first tool call)
                if tool_calls[0].reasoning:
                    _emit(Message(
                        MessageRole.ASSISTANT,
                        tool_calls[0].reasoning,
                        MessageMeta(MessageType.REASONING, step_index=iteration),
                    ))
                # Emit the assistant message with all tool calls
                tc_infos = []
                for tc in tool_calls:
                    tc_id = f"call_{tool_call_counter:09d}"
                    tool_call_counter += 1
                    tc_infos.append(ToolCallInfo(name=tc.tool, args=tc.args, call_id=tc_id))
                _emit(Message(
                    MessageRole.ASSISTANT,
                    "",
                    MessageMeta(MessageType.TOOL_CALL, step_index=iteration),
                    tool_calls=tc_infos,
                ))
                # Nudge for the first unknown tool
                _emit(Message(
                    MessageRole.USER,
                    unknown_tool_nudge(unknown[0].tool, list(workflow.tools.keys())),
                    MessageMeta(MessageType.RETRY_NUDGE, step_index=iteration),
                ))
                continue

            # 3f — All tools valid: reset consecutive retry counter
            consecutive_retries = 0

            # 3g — Check for terminal tool in batch
            terminal_idx = None
            for i, tc in enumerate(tool_calls):
                if tc.tool == workflow.terminal_tool:
                    terminal_idx = i
                    break

            if terminal_idx is not None and not step_tracker.is_satisfied():
                # Premature terminal — nudge
                premature_terminal_attempts += 1
                if premature_terminal_attempts > 3:
                    raise StepEnforcementError(
                        terminal_tool=workflow.terminal_tool,
                        attempts=premature_terminal_attempts,
                        pending_steps=step_tracker.pending(),
                    )
                if tool_calls[0].reasoning:
                    _emit(Message(
                        MessageRole.ASSISTANT,
                        tool_calls[0].reasoning,
                        MessageMeta(MessageType.REASONING, step_index=iteration),
                    ))
                tc_infos = []
                for tc in tool_calls:
                    tc_id = f"call_{tool_call_counter:09d}"
                    tool_call_counter += 1
                    tc_infos.append(ToolCallInfo(name=tc.tool, args=tc.args, call_id=tc_id))
                _emit(Message(
                    MessageRole.ASSISTANT,
                    "",
                    MessageMeta(MessageType.TOOL_CALL, step_index=iteration),
                    tool_calls=tc_infos,
                ))
                _emit(Message(
                    MessageRole.USER,
                    step_nudge(workflow.terminal_tool, step_tracker.pending(), tier=premature_terminal_attempts),
                    MessageMeta(MessageType.STEP_NUDGE, step_index=iteration),
                ))
                continue

            # 3h — Execute all tool calls in the batch
            # Assign call IDs up front so the assistant message is complete.
            tc_infos = []
            call_ids: list[str] = []
            for tc in tool_calls:
                tc_id = f"call_{tool_call_counter:09d}"
                tool_call_counter += 1
                tc_infos.append(ToolCallInfo(name=tc.tool, args=tc.args, call_id=tc_id))
                call_ids.append(tc_id)

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
                        result = await fn(**tc.args)
                    else:
                        result = fn(**tc.args)
                except ToolResolutionError as exc:
                    # Data didn't resolve — feed message back to model
                    # but don't count toward consecutive_tool_errors.
                    # Step is NOT recorded; bounded by max_iterations.
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
                    # If this was the terminal tool, stash the exception
                    if tc.tool == workflow.terminal_tool:
                        terminal_result = exc
                    continue

                # Success
                step_tracker.record(tc.tool)
                result_str = result if isinstance(result, str) else json.dumps(result)
                _emit(Message(
                    MessageRole.TOOL,
                    result_str,
                    MessageMeta(MessageType.TOOL_RESULT, step_index=iteration),
                    tool_name=tc.tool,
                    tool_call_id=tc_id,
                ))

                # If this is the terminal tool and it succeeded, stash result
                if tc.tool == workflow.terminal_tool:
                    terminal_result = result

            # 3i — Post-batch bookkeeping
            if batch_had_error:
                consecutive_tool_errors += 1
                if consecutive_tool_errors > self.max_tool_errors:
                    assert last_error is not None
                    raise ToolExecutionError(
                        last_error[0],
                        cause=last_error[1],
                    )
            else:
                consecutive_tool_errors = 0
                premature_terminal_attempts = 0

            # 3j — If terminal tool was in the batch and succeeded, return
            if terminal_idx is not None and not isinstance(terminal_result, Exception):
                return terminal_result

        # Step 4 — Max iterations exceeded
        raise MaxIterationsError(
            self.max_iterations, step_tracker.completed_steps, step_tracker.pending()
        )

    async def _send_streaming(
        self,
        api_messages: list[dict[str, str]],
        tool_specs: list[ToolSpec],
    ) -> LLMResponse:
        """Send via streaming, forwarding chunks to on_chunk callback."""
        response = None
        async for chunk in self.client.send_stream(api_messages, tools=tool_specs):
            if self.on_chunk is not None:
                await self.on_chunk(chunk)
            if chunk.type == ChunkType.FINAL:
                response = chunk.response
        if response is None:
            raise StreamError(
                "Stream ended without FINAL chunk — the client adapter "
                "may be malformed or the connection was interrupted"
            )
        return response
