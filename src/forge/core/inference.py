"""Shared canonical inference: render, send, validate, record once, retry."""

from __future__ import annotations
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from forge.clients.base import (
    ChunkType,
    LLMClient,
    InferenceRequest,
    RequestMetadata,
    StreamChunk,
    TokenUsage,
    _capture_usage,
)
from forge.context.manager import ContextManager
from forge.core.messages import (
    ConversationState,
    RichMessage,
    MessageRole,
    TextPart,
    ToolResultPart,
)
from forge.core.codecs import argument_value
from forge.core.workflow import ToolDefinition
from forge.errors import StreamError, ToolCallError
from forge.guardrails import ErrorTracker, ResponseValidator
from forge.guardrails.nudge import TOOL_ERROR_KINDS


class _ToolCallExhausted(ToolCallError):
    def __init__(
        self, message: str, raw_response: str | None, usage: TokenUsage | None
    ) -> None:
        super().__init__(message, raw_response=raw_response)
        self.usage = usage


def _sync_token_count(
    usage: TokenUsage | None, context_manager: ContextManager
) -> None:
    if usage is None:
        context_manager.invalidate_usage()
    else:
        context_manager.update_token_count(
            usage.prompt_tokens
            + usage.cache_creation_input_tokens
            + usage.cache_read_input_tokens
        )


@dataclass
class InferenceResult:
    response: RichMessage
    new_messages: list[RichMessage] = field(default_factory=list)
    usage: TokenUsage | None = None
    tool_call_counter: int = 0
    attempts: int = 1


async def run_inference(
    state: ConversationState,
    client: LLMClient,
    context_manager: ContextManager,
    validator: ResponseValidator,
    error_tracker: ErrorTracker,
    tool_definitions: list[ToolDefinition],
    metadata: RequestMetadata | None = None,
    tool_call_counter: int = 0,
    step_index: int = 0,
    step_hint: str = "",
    max_attempts: int | None = None,
    stream: bool = False,
    on_chunk: Callable[[StreamChunk], Awaitable[None]] | None = None,
    _on_exhausted: Callable[[list[RichMessage]], None] | None = None,
) -> InferenceResult | None:
    # Runner normally delivers InferenceResult.new_messages after step grouping.
    # Exhaustion has no result, so flush those same references on these exits.
    metadata = metadata or RequestMetadata()
    new_messages = []
    limit = error_tracker.max_retries + 1
    if max_attempts is not None:
        limit = min(limit, max_attempts)
    for attempt in range(limit):
        compacted = context_manager.maybe_compact(
            state, step_index=step_index, step_hint=step_hint
        )
        if compacted is not state:
            state.messages[:] = compacted.messages
        warning = context_manager.check_thresholds(state)
        outgoing = state
        if warning:
            message = RichMessage(
                MessageRole.USER, [TextPart(warning)], step_index, "context_warning"
            )
            new_messages.append(message)
            outgoing = ConversationState([*state.messages, message])
        request = client.render_request(
            outgoing, tool_definitions, metadata, stream=stream
        )
        with _capture_usage() as capture:
            response = (
                await _send_streaming(client, request, on_chunk)
                if stream
                else await client.send(request)
            )
        usage = capture.usage
        _sync_token_count(usage, context_manager)
        validation = validator.validate(response)
        response.step_index = step_index
        calls = response.tool_calls
        for call in calls:
            call.call_id = f"call_{tool_call_counter:09d}"
            tool_call_counter += 1
        state.messages.append(response)
        new_messages.append(response)
        if not validation.needs_retry:
            error_tracker.reset_retries()
            return InferenceResult(
                response, new_messages, usage, tool_call_counter, attempt + 1
            )
        nudge = validation.nudge
        if nudge.kind in TOOL_ERROR_KINDS:
            error_tracker.record_result(success=False)
            exhausted = error_tracker.tool_errors_exhausted
            budget = f"max_tool_errors={error_tracker.max_tool_errors}"
        else:
            error_tracker.record_retry()
            exhausted = error_tracker.retries_exhausted
            budget = f"max_retries={error_tracker.max_retries}"
        if exhausted:
            raw = (
                str([(p.name, argument_value(p)) for p in calls])
                if calls
                else "\n".join(
                    p.text for p in response.parts if isinstance(p, TextPart)
                )
            )
            if _on_exhausted is not None:
                _on_exhausted(new_messages)
            raise _ToolCallExhausted(
                f"Exhausted after {budget} consecutive failed attempts ({nudge.kind})",
                raw,
                usage,
            )
        if attempt + 1 == limit:
            if _on_exhausted is not None:
                _on_exhausted(new_messages)
            return None
        if calls:
            prefix = (
                "[ToolArgValidationError]"
                if nudge.kind == "tool_arg_validation"
                else "[UnknownTool]"
            )
            corrections = [
                RichMessage(
                    MessageRole.TOOL,
                    [
                        ToolResultPart(
                            p.call_id, [TextPart(f"{prefix} {nudge.content}")], "error"
                        )
                    ],
                    step_index,
                    nudge.kind,
                )
                for p in calls
            ]
        else:
            corrections = [
                RichMessage(
                    MessageRole.USER, [TextPart(nudge.content)], step_index, nudge.kind
                )
            ]
        state.messages.extend(corrections)
        new_messages.extend(corrections)
        context_manager.invalidate_usage()
    return None


async def _send_streaming(
    client: LLMClient,
    request: InferenceRequest,
    on_chunk: Callable[[StreamChunk], Awaitable[None]] | None = None,
) -> RichMessage:
    response = None
    async for chunk in client.send_stream(request):
        if on_chunk is not None:
            await on_chunk(chunk)
        if chunk.type == ChunkType.FINAL:
            response = chunk.response
    if response is None:
        raise StreamError(
            "Stream ended without FINAL chunk — the client adapter may be malformed or the connection was interrupted"
        )
    return response
