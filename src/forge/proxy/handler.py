"""Chat completion request handler.

Phase 1: Pass-through. Forwards the request to the backend, buffers
the full streamed response, and replays it to the client.

Phase 2 will add guardrail validation and retry logic here.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from forge.proxy.stream import BufferedStream, buffer_sse_stream, iter_sse_bytes

logger = logging.getLogger("forge.proxy")


class ChatHandler:
    """Handles /v1/chat/completions requests.

    Args:
        backend_url: Base URL of the model server (e.g. "http://localhost:8080").
        max_retries: Max guardrail retry attempts per request. (Phase 2)
        timeout: HTTP timeout for backend requests in seconds.
    """

    def __init__(
        self,
        backend_url: str,
        max_retries: int = 3,
        timeout: float = 300.0,
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout)

    async def handle(self, body: dict[str, Any]) -> tuple[list[bytes], int]:
        """Process a chat completion request.

        Args:
            body: Parsed JSON request body from the client.

        Returns:
            Tuple of (SSE chunks as bytes, HTTP status code).
            On success, status is 200 and chunks are the SSE stream.
            On failure, status is the backend's error code and chunks
            contain a single JSON error response.
        """
        # Phase 1: pass-through. Forward to backend, buffer, replay.
        is_streaming = body.get("stream", False)

        if is_streaming:
            return await self._handle_streaming(body)
        else:
            return await self._handle_batch(body)

    async def _handle_streaming(
        self, body: dict[str, Any]
    ) -> tuple[list[bytes], int]:
        """Forward a streaming request, buffer the response, replay."""
        url = f"{self.backend_url}/v1/chat/completions"

        async with self._client.stream(
            "POST", url, json=body, headers={"Content-Type": "application/json"}
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                return [error_body], response.status_code

            buf = await buffer_sse_stream(response)

        logger.debug(
            "Buffered %d SSE chunks (complete=%s)", len(buf.chunks), buf.complete
        )

        return iter_sse_bytes(buf), 200

    async def _handle_batch(
        self, body: dict[str, Any]
    ) -> tuple[list[bytes], int]:
        """Forward a non-streaming request, return the response."""
        url = f"{self.backend_url}/v1/chat/completions"

        response = await self._client.post(
            url, json=body, headers={"Content-Type": "application/json"}
        )

        return [response.content], response.status_code

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
