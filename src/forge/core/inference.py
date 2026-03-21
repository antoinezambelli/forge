"""Inference loop — compact, fold, serialize, send, validate, retry.

Extracted from WorkflowRunner so both the runner and the proxy can share
the same input-processing and validation logic. This is the "front half"
of the agentic loop: everything up to and including getting a clean
response from the LLM. The "back half" (step enforcement, tool execution,
terminal check) stays in the caller.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from forge.clients.base import ChunkType, LLMClient, StreamChunk
from forge.context.manager import ContextManager
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from forge.errors import StreamError, ToolCallError
from forge.guardrails import ErrorTracker, ResponseValidator

# Maps Nudge.kind → MessageType for message emission.
_NUDGE_KIND_TO_TYPE: dict[str, MessageType] = {
    "retry": MessageType.RETRY_NUDGE,
    "unknown_tool": MessageType.RETRY_NUDGE,
    "step": MessageType.STEP_NUDGE,
}


@dataclass
class InferenceResult:
    """Result of a single inference call (may include transparent retries).

    Attributes:
        response: The validated LLM response — tool calls or text.
        new_messages: Messages generated during this call (assistant text from
            failed attempts, nudges, and the final assistant response). The
            caller should append these to their message history.
        tool_call_counter: Updated counter for generating unique call IDs.
    """

    response: list[ToolCall] | TextResponse
    new_messages: list[Message] = field(default_factory=list)
    tool_call_counter: int = 0
    attempts: int = 1


def fold_and_serialize(
    messages: list[Message],
    api_format: str,
) -> list[dict[str, Any]]:
    """Reasoning-fold and serialize forge Messages to API dicts.

    Folds REASONING messages into the following TOOL_CALL message's content
    field so the wire format has one assistant message with both content and
    tool_calls (valid OpenAI format). Internal Message list stays separate
    for compaction.
    """
    api_messages: list[dict[str, Any]] = []
    pending_reasoning: str | None = None

    for m in messages:
        if m.metadata.type == MessageType.REASONING and m.role == MessageRole.ASSISTANT:
            pending_reasoning = m.content
            continue
        d = m.to_api_dict(format=api_format)
        if pending_reasoning is not None and m.tool_calls is not None:
            d["content"] = pending_reasoning
            pending_reasoning = None
        elif pending_reasoning is not None:
            api_messages.append({"role": "assistant", "content": pending_reasoning})
            pending_reasoning = None
        api_messages.append(d)

    if pending_reasoning is not None:
        api_messages.append({"role": "assistant", "content": pending_reasoning})

    return api_messages


def _build_tool_call_infos(
    tool_calls: list[ToolCall],
    tool_call_counter: int,
) -> tuple[list[ToolCallInfo], int]:
    """Assign call IDs to tool calls, returning infos and updated counter."""
    tc_infos = []
    for tc in tool_calls:
        tc_id = f"call_{tool_call_counter:09d}"
        tool_call_counter += 1
        tc_infos.append(ToolCallInfo(name=tc.tool, args=tc.args, call_id=tc_id))
    return tc_infos, tool_call_counter


async def run_inference(
    messages: list[Message],
    client: LLMClient,
    context_manager: ContextManager,
    validator: ResponseValidator,
    error_tracker: ErrorTracker,
    tool_specs: list[ToolSpec],
    tool_call_counter: int = 0,
    step_index: int = 0,
    step_hint: str = "",
    max_attempts: int | None = None,
    trust_text_intent: bool = False,
    stream: bool = False,
    on_chunk: Callable[[StreamChunk], Awaitable[None]] | None = None,
) -> InferenceResult | None:
    """Send messages to the LLM with compaction, folding, validation, and retry.

    Retries are handled internally — the caller gets a clean response or an
    exception. On each retry, the model's failed output and a corrective
    nudge are appended to the message list and returned in
    ``InferenceResult.new_messages`` so the caller can track history.

    Args:
        messages: The current conversation history (forge Messages). This
            list is mutated during compaction and retry (messages may be
            removed by compaction and added by retries).
        client: The LLM backend client.
        context_manager: For context budget compaction.
        validator: For rescue parsing, retry nudges, unknown tool checks.
        error_tracker: Tracks consecutive retry budget. The caller owns
            this object and passes it in so budget persists across calls.
        tool_specs: Available tools to send to the LLM.
        tool_call_counter: Current counter for generating unique call IDs.
            The updated value is returned in the result.
        step_index: Current iteration index (for compaction and message metadata).
        step_hint: Hint for compaction summarization.
        max_attempts: Maximum LLM calls this invocation may make (including
            retries). When None, bounded only by max_retries. The runner
            passes remaining iteration budget here so retries don't exceed
            max_iterations.
        trust_text_intent: If True, trust the backend's intentional flag
            on TextResponse and skip retry. Proxy sets this to True;
            WorkflowRunner leaves it False (default).
        stream: If True, use send_stream() instead of send().
        on_chunk: Async callback for streaming chunks.

    Returns:
        InferenceResult with validated tool calls, new messages, and
        updated tool_call_counter. Returns None if max_attempts is
        exhausted without a valid response (caller should treat this
        as iteration budget spent).

    Raises:
        ToolCallError: If retry budget (max_retries) is exhausted.
        StreamError: If streaming ends without a FINAL chunk.
    """
    api_format = getattr(client, "api_format", "ollama")
    new_messages: list[Message] = []
    max_retries = error_tracker.max_retries
    attempt_limit = max_retries + 1
    if max_attempts is not None:
        attempt_limit = min(attempt_limit, max_attempts)
    attempts = 0

    for _attempt in range(attempt_limit):
        attempts += 1

        # Compact
        compacted = context_manager.maybe_compact(
            messages, step_index=step_index, step_hint=step_hint,
        )
        # Update the caller's list in-place if compaction changed it
        if compacted is not messages:
            messages.clear()
            messages.extend(compacted)

        # Fold and serialize
        api_messages = fold_and_serialize(messages, api_format)

        # Send
        if stream:
            response = await _send_streaming(client, api_messages, tool_specs, on_chunk)
        else:
            response = await client.send(api_messages, tools=tool_specs)

        # Validate
        validation = validator.validate(response, trust_text_intent=trust_text_intent)

        if not validation.needs_retry:
            error_tracker.reset_retries()
            # Intentional text response or validated tool calls
            validated = validation.text_response or validation.tool_calls
            return InferenceResult(
                response=validated,
                new_messages=new_messages,
                tool_call_counter=tool_call_counter,
                attempts=attempts,
            )

        # Retry path
        error_tracker.record_retry()
        if error_tracker.retries_exhausted:
            raw = response.content if isinstance(response, TextResponse) else str(
                [(tc.tool, tc.args) for tc in response]
            )
            raise ToolCallError(
                f"Retries exhausted after {max_retries} consecutive failed attempts",
                raw_response=raw,
            )

        # Emit the assistant's failed output
        if isinstance(response, TextResponse):
            msg = Message(
                MessageRole.ASSISTANT,
                response.content,
                MessageMeta(MessageType.TEXT_RESPONSE, step_index=step_index),
            )
            messages.append(msg)
            new_messages.append(msg)
        else:
            # Unknown tool — emit reasoning + tool call messages
            tool_calls = response
            if tool_calls[0].reasoning:
                reasoning_msg = Message(
                    MessageRole.ASSISTANT,
                    tool_calls[0].reasoning,
                    MessageMeta(MessageType.REASONING, step_index=step_index),
                )
                messages.append(reasoning_msg)
                new_messages.append(reasoning_msg)
            tc_infos, tool_call_counter = _build_tool_call_infos(tool_calls, tool_call_counter)
            tc_msg = Message(
                MessageRole.ASSISTANT,
                "",
                MessageMeta(MessageType.TOOL_CALL, step_index=step_index),
                tool_calls=tc_infos,
            )
            messages.append(tc_msg)
            new_messages.append(tc_msg)

        # Emit nudge
        nudge = validation.nudge
        nudge_type = _NUDGE_KIND_TO_TYPE[nudge.kind]
        nudge_msg = Message(
            MessageRole.USER,
            nudge.content,
            MessageMeta(nudge_type, step_index=step_index),
        )
        messages.append(nudge_msg)
        new_messages.append(nudge_msg)

    # max_attempts exhausted without valid response — signal to caller
    return None


async def _send_streaming(
    client: LLMClient,
    api_messages: list[dict[str, Any]],
    tool_specs: list[ToolSpec],
    on_chunk: Callable[[StreamChunk], Awaitable[None]] | None = None,
) -> LLMResponse:
    """Send via streaming, forwarding chunks to on_chunk callback."""
    response = None
    async for chunk in client.send_stream(api_messages, tools=tool_specs):
        if on_chunk is not None:
            await on_chunk(chunk)
        if chunk.type == ChunkType.FINAL:
            response = chunk.response
    if response is None:
        raise StreamError(
            "Stream ended without FINAL chunk — the client adapter "
            "may be malformed or the connection was interrupted"
        )
    return response
