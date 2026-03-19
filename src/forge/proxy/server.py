"""Raw ASGI application for the forge proxy server.

No framework dependencies — just an async function that speaks HTTP.

Endpoints:
    POST /v1/chat/completions  -- Core proxy (guardrails + forward)
    GET  /v1/models            -- Pass-through to backend
    GET  /health               -- Proxy health check
    *    /*                    -- Pass-through to backend
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from forge.proxy.handler import ChatHandler

logger = logging.getLogger("forge.proxy")


def create_app(
    backend_url: str,
    max_retries: int = 3,
    rescue_enabled: bool = True,
    timeout: float = 300.0,
) -> Any:
    """Create the proxy ASGI application.

    Args:
        backend_url: Base URL of the model server (e.g. "http://localhost:8080").
        max_retries: Max guardrail retry attempts per request.
        rescue_enabled: Attempt to parse tool calls from plain text.
        timeout: HTTP timeout for backend requests in seconds.

    Returns:
        An ASGI application callable.
    """
    handler = ChatHandler(
        backend_url=backend_url,
        max_retries=max_retries,
        rescue_enabled=rescue_enabled,
        timeout=timeout,
    )
    passthrough_client = httpx.AsyncClient(timeout=timeout)
    backend = backend_url.rstrip("/")

    async def app(scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            await _handle_lifespan(scope, receive, send)
            return

        if scope["type"] != "http":
            return

        path = scope["path"]
        method = scope["method"]

        if path == "/health" and method == "GET":
            await _send_json(send, 200, {"status": "ok"})

        elif path == "/v1/chat/completions" and method == "POST":
            body = await _read_body(receive)
            parsed = json.loads(body)
            is_streaming = parsed.get("stream", False)

            chunks, status = await handler.handle(parsed)

            if status != 200 or not is_streaming:
                await _send_response(
                    send, status, chunks[0],
                    content_type=b"application/json",
                )
            else:
                await _send_sse(send, chunks)

        elif path == "/v1/models" and method == "GET":
            response = await passthrough_client.get(f"{backend}/v1/models")
            await _send_response(
                send, response.status_code, response.content,
                content_type=b"application/json",
            )

        else:
            # Pass-through for any unmatched endpoint
            body = await _read_body(receive)
            headers = _extract_headers(scope)
            response = await passthrough_client.request(
                method=method,
                url=f"{backend}{path}",
                content=body,
                headers=headers,
            )
            ct = response.headers.get("content-type", "application/json")
            await _send_response(
                send, response.status_code, response.content,
                content_type=ct.encode(),
            )

    async def _handle_lifespan(scope: dict, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await handler.close()
                await passthrough_client.aclose()
                await send({"type": "lifespan.shutdown.complete"})
                return

    return app


async def _read_body(receive: Any) -> bytes:
    """Read the full request body from ASGI receive."""
    body = b""
    while True:
        msg = await receive()
        body += msg.get("body", b"")
        if not msg.get("more_body", False):
            break
    return body


def _extract_headers(scope: dict) -> dict[str, str]:
    """Extract headers from ASGI scope, dropping hop-by-hop headers."""
    skip = {b"host", b"content-length", b"transfer-encoding"}
    return {
        k.decode(): v.decode()
        for k, v in scope.get("headers", [])
        if k.lower() not in skip
    }


async def _send_json(send: Any, status: int, data: Any) -> None:
    """Send a JSON response."""
    body = json.dumps(data).encode()
    await _send_response(send, status, body, content_type=b"application/json")


async def _send_response(
    send: Any, status: int, body: bytes, content_type: bytes = b"application/json"
) -> None:
    """Send a complete HTTP response."""
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            [b"content-type", content_type],
            [b"content-length", str(len(body)).encode()],
        ],
    })
    await send({
        "type": "http.response.body",
        "body": body,
    })


async def _send_sse(send: Any, chunks: list[bytes]) -> None:
    """Send an SSE stream response."""
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [
            [b"content-type", b"text/event-stream"],
            [b"cache-control", b"no-cache"],
            [b"connection", b"keep-alive"],
        ],
    })
    for chunk in chunks:
        await send({
            "type": "http.response.body",
            "body": chunk,
            "more_body": True,
        })
    await send({
        "type": "http.response.body",
        "body": b"",
    })
