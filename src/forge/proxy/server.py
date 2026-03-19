"""Raw asyncio HTTP server for the proxy.

No framework dependencies — uses asyncio.start_server directly.
Handles routing, request serialization (single-GPU lock), health
checks, and SSE streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from forge.clients.base import LLMClient
from forge.context.manager import ContextManager
from forge.proxy.handler import handle_chat_completions

logger = logging.getLogger("forge.proxy")

# Maximum request body size (16 MB)
_MAX_BODY = 16 * 1024 * 1024


class HTTPServer:
    """Raw asyncio HTTP server with OpenAI-compatible routing."""

    def __init__(
        self,
        client: LLMClient,
        context_manager: ContextManager,
        host: str = "127.0.0.1",
        port: int = 8081,
        serialize_requests: bool = True,
        max_retries: int = 3,
        rescue_enabled: bool = True,
    ) -> None:
        self._client = client
        self._context_manager = context_manager
        self._host = host
        self._port = port
        self._max_retries = max_retries
        self._rescue_enabled = rescue_enabled
        self._server: asyncio.Server | None = None
        self._lock: asyncio.Lock | None = asyncio.Lock() if serialize_requests else None

    async def start(self) -> None:
        """Start listening for connections."""
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port,
        )
        logger.info("Proxy listening on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the server."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single HTTP connection."""
        try:
            # Read request line
            request_line = await asyncio.wait_for(
                reader.readline(), timeout=30.0,
            )
            if not request_line:
                return

            request_str = request_line.decode("utf-8", errors="replace").strip()
            parts = request_str.split(" ", 2)
            if len(parts) < 2:
                await self._send_error(writer, 400, "Bad request")
                return

            method, path = parts[0], parts[1]
            logger.info(">> %s %s", method, path)

            # Read headers
            headers = await self._read_headers(reader)
            content_length = int(headers.get("content-length", "0"))

            # Read body
            body_bytes = b""
            if content_length > 0:
                if content_length > _MAX_BODY:
                    await self._send_error(writer, 413, "Request too large")
                    return
                body_bytes = await asyncio.wait_for(
                    reader.readexactly(content_length), timeout=60.0,
                )

            # Route
            if method == "GET" and path == "/health":
                await self._handle_health(writer)
            elif method == "GET" and path == "/v1/models":
                await self._handle_models(writer)
            elif method == "POST" and path == "/v1/chat/completions":
                await self._handle_completions(writer, body_bytes)
            elif method == "OPTIONS":
                await self._send_cors_preflight(writer)
            else:
                await self._send_error(writer, 404, "Not found")

        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception:
            logger.exception("Unhandled error in connection handler")
            try:
                await self._send_error(writer, 500, "Internal server error")
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_headers(self, reader: asyncio.StreamReader) -> dict[str, str]:
        """Read HTTP headers until blank line."""
        headers: dict[str, str] = {}
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=30.0)
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                break
            if ":" in decoded:
                key, value = decoded.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        return headers

    async def _handle_health(self, writer: asyncio.StreamWriter) -> None:
        """GET /health — returns OK."""
        body = json.dumps({"status": "ok"})
        await self._send_json(writer, 200, body)

    async def _handle_models(self, writer: asyncio.StreamWriter) -> None:
        """GET /v1/models — returns a minimal model list."""
        body = json.dumps({
            "object": "list",
            "data": [{"id": "forge", "object": "model"}],
        })
        await self._send_json(writer, 200, body)

    async def _handle_completions(
        self,
        writer: asyncio.StreamWriter,
        body_bytes: bytes,
    ) -> None:
        """POST /v1/chat/completions — the main proxy endpoint."""
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            await self._send_error(writer, 400, "Invalid JSON")
            return

        is_stream = body.get("stream", False)
        msg_count = len(body.get("messages", []))
        tool_count = len(body.get("tools", []))
        logger.info(
            "   stream=%s messages=%d tools=%d model=%s",
            is_stream, msg_count, tool_count, body.get("model", "?"),
        )

        # Serialize requests for single-GPU backends
        if self._lock is not None:
            async with self._lock:
                result = await self._run_handler(body)
        else:
            result = await self._run_handler(body)

        if isinstance(result, Exception):
            error_msg = str(result)
            logger.info("<< ERROR: %s", error_msg[:120])
            if is_stream:
                events = [{"error": error_msg}]
                await self._send_sse(writer, events)
            else:
                await self._send_error(writer, 502, error_msg)
            return

        if is_stream:
            logger.info("<< SSE %d events", len(result))
            await self._send_sse(writer, result)
        else:
            logger.info("<< JSON 200")
            await self._send_json(writer, 200, json.dumps(result))

    async def _run_handler(
        self, body: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]] | Exception:
        """Run the handler, catching errors."""
        try:
            return await handle_chat_completions(
                body=body,
                client=self._client,
                context_manager=self._context_manager,
                max_retries=self._max_retries,
                rescue_enabled=self._rescue_enabled,
            )
        except Exception as exc:
            logger.exception("Handler error")
            return exc

    async def _send_json(
        self, writer: asyncio.StreamWriter, status: int, body: str,
    ) -> None:
        """Send a JSON HTTP response."""
        response = (
            f"HTTP/1.1 {status} {_status_text(status)}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body.encode())}\r\n"
            f"Connection: close\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"\r\n"
            f"{body}"
        )
        writer.write(response.encode())
        await writer.drain()

    async def _send_sse(
        self, writer: asyncio.StreamWriter, events: list[dict[str, Any]],
    ) -> None:
        """Send an SSE streaming response with chunked transfer encoding."""
        header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Transfer-Encoding: chunked\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: keep-alive\r\n"
            "\r\n"
        )
        writer.write(header.encode())
        await writer.drain()

        for event in events:
            if writer.is_closing():
                return
            data = f"data: {json.dumps(event)}\n\n".encode()
            writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            await writer.drain()

        done = b"data: [DONE]\n\n"
        writer.write(f"{len(done):x}\r\n".encode() + done + b"\r\n")
        # Terminating zero-length chunk
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        logger.info("<< SSE complete, [DONE] sent")

    async def _send_error(
        self, writer: asyncio.StreamWriter, status: int, message: str,
    ) -> None:
        """Send an error JSON response."""
        body = json.dumps({"error": {"message": message, "type": "proxy_error"}})
        await self._send_json(writer, status, body)

    async def _send_cors_preflight(self, writer: asyncio.StreamWriter) -> None:
        """Handle CORS preflight."""
        response = (
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        writer.write(response.encode())
        await writer.drain()


def _status_text(code: int) -> str:
    """HTTP status code to text."""
    return {
        200: "OK",
        204: "No Content",
        400: "Bad Request",
        404: "Not Found",
        413: "Payload Too Large",
        500: "Internal Server Error",
        502: "Bad Gateway",
    }.get(code, "Error")
