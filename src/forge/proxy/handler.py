"""Chat completion request handler with guardrail validation.

Buffers the backend response, runs ResponseValidator (rescue, retry,
unknown tool check), and retries with nudge injection on failure.
The client only sees the final clean response.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from forge.core.workflow import TextResponse, ToolCall
from forge.guardrails.error_tracker import ErrorTracker
from forge.guardrails.response_validator import ResponseValidator
from forge.proxy.convert import (
    parse_batch_response,
    parse_streamed_response,
    synthesize_sse_tool_calls,
)
from forge.proxy.stream import buffer_sse_stream, iter_sse_bytes

logger = logging.getLogger("forge.proxy")


class ChatHandler:
    """Handles /v1/chat/completions requests with guardrail validation.

    Args:
        backend_url: Base URL of the model server (e.g. "http://localhost:8080").
        max_retries: Max guardrail retry attempts per request.
        rescue_enabled: Attempt to parse tool calls from plain text.
        timeout: HTTP timeout for backend requests in seconds.
    """

    def __init__(
        self,
        backend_url: str,
        max_retries: int = 3,
        rescue_enabled: bool = True,
        timeout: float = 300.0,
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.max_retries = max_retries
        self.rescue_enabled = rescue_enabled
        self._client = httpx.AsyncClient(timeout=timeout)

    async def handle(self, body: dict[str, Any]) -> tuple[list[bytes], int]:
        """Process a chat completion request with guardrail validation.

        Forwards to the backend, validates the response, and retries
        with nudge injection if the model misbehaves. The client only
        sees the final clean response.

        Args:
            body: Parsed JSON request body from the client.

        Returns:
            Tuple of (response chunks as bytes, HTTP status code).
        """
        is_streaming = body.get("stream", False)

        # Extract tool names from the request for validation
        tool_names = self._extract_tool_names(body)

        if not tool_names:
            # No tools in request -- nothing to validate, pass through
            if is_streaming:
                return await self._forward_streaming(body)
            else:
                return await self._forward_batch(body)

        # Set up per-request guardrails
        validator = ResponseValidator(
            tool_names=tool_names,
            rescue_enabled=self.rescue_enabled,
        )
        errors = ErrorTracker(max_retries=self.max_retries)

        # Messages are mutable -- we may append nudges for retries
        messages = list(body.get("messages", []))
        model = body.get("model", "forge-proxy")

        # Guardrail retry loop
        last_chunks: list[bytes] = []
        last_status: int = 200

        for attempt in range(self.max_retries + 1):
            # Build request with current messages
            request_body = {**body, "messages": messages}

            if is_streaming:
                chunks, status, response = await self._request_streaming(request_body)
            else:
                chunks, status, response = await self._request_batch(request_body)

            last_chunks = chunks
            last_status = status

            if status != 200:
                # Backend error -- return as-is, don't retry
                return chunks, status

            if response is None:
                # Couldn't parse response -- return raw chunks
                logger.warning("Could not parse backend response on attempt %d", attempt + 1)
                return chunks, status

            # Validate through ResponseValidator
            validation = validator.validate(response)

            if not validation.needs_retry:
                # Clean response
                if validation.tool_calls and self._was_rescued(response, validation.tool_calls):
                    # Response was rescued from text -- synthesize proper SSE
                    logger.info("Rescued tool call from text response on attempt %d", attempt + 1)
                    if is_streaming:
                        return synthesize_sse_tool_calls(validation.tool_calls, model), 200
                    else:
                        return [self._synthesize_batch_tool_calls(validation.tool_calls, model)], 200
                # Original response was valid -- return as-is
                return chunks, status

            # Needs retry
            errors.record_retry()
            if errors.retries_exhausted:
                logger.warning(
                    "Retries exhausted after %d attempts, returning last response",
                    attempt + 1,
                )
                return last_chunks, last_status

            # Inject nudge and retry
            nudge = validation.nudge
            logger.info(
                "Retry %d/%d (%s): %s",
                attempt + 1,
                self.max_retries,
                nudge.kind,
                nudge.content[:80],
            )

            # Append the model's bad response and the nudge to messages
            if isinstance(response, TextResponse):
                messages.append({"role": "assistant", "content": response.content})
            else:
                # Unknown tool -- append the tool call attempt
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_retry_{attempt}_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.tool,
                                "arguments": json.dumps(tc.args),
                            },
                        }
                        for i, tc in enumerate(response)
                    ],
                })
            messages.append({"role": "user", "content": nudge.content})

        return last_chunks, last_status

    async def _request_streaming(
        self, body: dict[str, Any]
    ) -> tuple[list[bytes], int, Any]:
        """Send a streaming request, buffer, parse."""
        url = f"{self.backend_url}/v1/chat/completions"

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

        logger.debug(
            "Buffered %d SSE chunks, parsed %s",
            len(buf.chunks),
            type(response).__name__,
        )

        return chunks, 200, response

    async def _request_batch(
        self, body: dict[str, Any]
    ) -> tuple[list[bytes], int, Any]:
        """Send a non-streaming request, parse."""
        url = f"{self.backend_url}/v1/chat/completions"

        resp = await self._client.post(
            url, json=body, headers={"Content-Type": "application/json"}
        )

        if resp.status_code != 200:
            return [resp.content], resp.status_code, None

        try:
            data = resp.json()
            response = parse_batch_response(data)
        except Exception as e:
            logger.warning("Failed to parse batch response: %s", e)
            return [resp.content], 200, None

        return [resp.content], 200, response

    async def _forward_streaming(
        self, body: dict[str, Any]
    ) -> tuple[list[bytes], int]:
        """Forward streaming request without guardrails."""
        url = f"{self.backend_url}/v1/chat/completions"

        async with self._client.stream(
            "POST", url, json=body, headers={"Content-Type": "application/json"}
        ) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                return [error_body], resp.status_code
            buf = await buffer_sse_stream(resp)

        return iter_sse_bytes(buf), 200

    async def _forward_batch(
        self, body: dict[str, Any]
    ) -> tuple[list[bytes], int]:
        """Forward batch request without guardrails."""
        url = f"{self.backend_url}/v1/chat/completions"
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

    @staticmethod
    def _was_rescued(
        original_response: Any, validated_calls: list[ToolCall]
    ) -> bool:
        """True if the original was a TextResponse but validation produced tool calls."""
        return isinstance(original_response, TextResponse)

    @staticmethod
    def _synthesize_batch_tool_calls(
        tool_calls: list[ToolCall], model: str
    ) -> bytes:
        """Synthesize an OpenAI batch response from rescued tool calls."""
        return json.dumps({
            "id": "chatcmpl-forge-rescued",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_rescued_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.tool,
                                "arguments": json.dumps(tc.args),
                            },
                        }
                        for i, tc in enumerate(tool_calls)
                    ],
                },
                "finish_reason": "tool_calls",
            }],
        }).encode()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
