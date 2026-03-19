"""Chat completion request handler with guardrail validation.

Converts OpenAI messages to forge types on entry. All internal state
is forge Messages. Converts back to OpenAI format only at the two
exit boundaries: sending to the backend and sending to the client.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from forge.context.manager import ContextManager
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType
from forge.core.workflow import TextResponse, ToolCall
from forge.guardrails.error_tracker import ErrorTracker
from forge.guardrails.nudge import Nudge
from forge.guardrails.response_validator import ResponseValidator
from forge.proxy.convert import (
    forge_messages_to_openai,
    openai_messages_to_forge,
    parse_batch_response,
    parse_streamed_response,
    synthesize_batch_tool_calls,
    synthesize_sse_tool_calls,
)
from forge.proxy.stream import buffer_sse_stream, iter_sse_bytes

logger = logging.getLogger("forge.proxy")

# Maps Nudge.kind to MessageType (same mapping as WorkflowRunner).
_NUDGE_KIND_TO_TYPE: dict[str, MessageType] = {
    "retry": MessageType.RETRY_NUDGE,
    "unknown_tool": MessageType.RETRY_NUDGE,
    "step": MessageType.STEP_NUDGE,
}


class ChatHandler:
    """Handles /v1/chat/completions requests with guardrail validation.

    Internally operates on forge Message objects. OpenAI wire format
    is only used at the edges (inbound conversion, backend serialization,
    client response).

    Args:
        backend_url: Base URL of the model server (e.g. "http://localhost:8080").
        context_manager: ContextManager for compaction. None disables compaction.
        max_retries: Max guardrail retry attempts per request.
        rescue_enabled: Attempt to parse tool calls from plain text.
        timeout: HTTP timeout for backend requests in seconds.
    """

    def __init__(
        self,
        backend_url: str,
        context_manager: ContextManager | None = None,
        max_retries: int = 3,
        rescue_enabled: bool = True,
        timeout: float = 300.0,
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.context_manager = context_manager
        self.max_retries = max_retries
        self.rescue_enabled = rescue_enabled
        self._client = httpx.AsyncClient(timeout=timeout)

    async def handle(self, body: dict[str, Any]) -> tuple[list[bytes], int]:
        """Process a chat completion request with guardrail validation.

        Args:
            body: Parsed JSON request body from the client.

        Returns:
            Tuple of (response chunks as bytes, HTTP status code).
        """
        is_streaming = body.get("stream", False)
        tool_names = self._extract_tool_names(body)

        if not tool_names:
            # No tools -- nothing to validate, pass through
            return await self._forward_raw(body, is_streaming)

        # Convert inbound messages to forge types
        messages = openai_messages_to_forge(body.get("messages", []))

        # Compact context if approaching budget
        if self.context_manager is not None:
            messages = self.context_manager.maybe_compact(messages)

        # Set up per-request guardrails
        validator = ResponseValidator(
            tool_names=tool_names,
            rescue_enabled=self.rescue_enabled,
        )
        errors = ErrorTracker(max_retries=self.max_retries)

        model = body.get("model", "forge-proxy")

        # Guardrail retry loop
        last_chunks: list[bytes] = []
        last_status: int = 200

        for attempt in range(self.max_retries + 1):
            # Serialize forge Messages to OpenAI format for the backend
            request_body = {**body, "messages": forge_messages_to_openai(messages)}

            chunks, status, response = await self._request(request_body, is_streaming)

            last_chunks = chunks
            last_status = status

            if status != 200 or response is None:
                return chunks, status

            # Validate through ResponseValidator
            validation = validator.validate(response)

            if not validation.needs_retry:
                # Clean response
                if validation.tool_calls and isinstance(response, TextResponse):
                    # Rescued from text -- synthesize proper format
                    logger.info("Rescued tool call from text on attempt %d", attempt + 1)
                    if is_streaming:
                        return synthesize_sse_tool_calls(validation.tool_calls, model), 200
                    else:
                        return [synthesize_batch_tool_calls(validation.tool_calls, model)], 200
                # Original response was valid -- return raw chunks
                return chunks, status

            # Needs retry
            errors.record_retry()
            if errors.retries_exhausted:
                logger.warning(
                    "Retries exhausted after %d attempts, returning last response",
                    attempt + 1,
                )
                return last_chunks, last_status

            nudge = validation.nudge
            logger.info(
                "Retry %d/%d (%s): %s",
                attempt + 1,
                self.max_retries,
                nudge.kind,
                nudge.content[:80],
            )

            # Append the model's bad response as a forge Message
            if isinstance(response, TextResponse):
                messages.append(Message(
                    MessageRole.ASSISTANT,
                    response.content,
                    MessageMeta(MessageType.TEXT_RESPONSE),
                ))
            else:
                # Unknown tool -- emit as TOOL_CALL message
                from forge.core.messages import ToolCallInfo
                tc_infos = [
                    ToolCallInfo(
                        name=tc.tool,
                        args=tc.args,
                        call_id=f"call_retry_{attempt}_{i}",
                    )
                    for i, tc in enumerate(response)
                ]
                messages.append(Message(
                    MessageRole.ASSISTANT,
                    "",
                    MessageMeta(MessageType.TOOL_CALL),
                    tool_calls=tc_infos,
                ))

            # Append the nudge as a forge Message
            nudge_type = _NUDGE_KIND_TO_TYPE.get(nudge.kind, MessageType.RETRY_NUDGE)
            messages.append(Message(
                MessageRole.USER,
                nudge.content,
                MessageMeta(nudge_type),
            ))

        return last_chunks, last_status

    async def _request(
        self, body: dict[str, Any], is_streaming: bool
    ) -> tuple[list[bytes], int, Any]:
        """Send request to backend, parse response into forge types."""
        url = f"{self.backend_url}/v1/chat/completions"

        if is_streaming:
            async with self._client.stream(
                "POST", url, json=body, headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    return [error_body], resp.status_code, None
                buf = await buffer_sse_stream(resp)

            chunks = iter_sse_bytes(buf)
            try:
                response, _usage = parse_streamed_response(buf.chunks)
            except Exception as e:
                logger.warning("Failed to parse streamed response: %s", e)
                return chunks, 200, None

            return chunks, 200, response
        else:
            resp = await self._client.post(
                url, json=body, headers={"Content-Type": "application/json"}
            )
            if resp.status_code != 200:
                return [resp.content], resp.status_code, None
            try:
                response = parse_batch_response(resp.json())
            except Exception as e:
                logger.warning("Failed to parse batch response: %s", e)
                return [resp.content], 200, None

            return [resp.content], 200, response

    async def _forward_raw(
        self, body: dict[str, Any], is_streaming: bool
    ) -> tuple[list[bytes], int]:
        """Forward request without guardrails (no tools in request)."""
        url = f"{self.backend_url}/v1/chat/completions"

        if is_streaming:
            async with self._client.stream(
                "POST", url, json=body, headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    return [error_body], resp.status_code
                buf = await buffer_sse_stream(resp)
            return iter_sse_bytes(buf), 200
        else:
            resp = await self._client.post(
                url, json=body, headers={"Content-Type": "application/json"}
            )
            return [resp.content], resp.status_code

    @staticmethod
    def _extract_tool_names(body: dict[str, Any]) -> list[str]:
        """Extract tool names from the OpenAI tools array."""
        tools = body.get("tools", [])
        names = []
        for tool in tools:
            fn = tool.get("function", {})
            name = fn.get("name")
            if name:
                names.append(name)
        return names

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
