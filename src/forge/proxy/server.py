"""Raw asyncio HTTP server for the proxy.

No framework dependencies — uses asyncio.start_server directly.
Handles routing, request queuing (single-GPU serialization), health
checks, SSE streaming, and client disconnect detection.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from forge.clients.base import AUTH_HEADER_NAMES, LLMClient
from forge.context.manager import ContextManager
from forge.core.reasoning import DEFAULT_REASONING_REPLAY, ReasoningReplay, validate_reasoning_replay
from forge.errors import (
    BackendDiscoveryError,
    BackendError,
    MissingCredentialError,
    MultipleCredentialsError,
)
from forge.proxy.auth import DUPLICATE_AUTH_MARKER, resolve_inbound_credential
from forge.proxy.handler import (
    LazyDiscovery,
    handle_chat_completions,
    run_lazy_discovery,
)

logger = logging.getLogger("forge.proxy")

# Maximum request body size (16 MB)
_MAX_BODY = 16 * 1024 * 1024


@dataclass
class _QueueItem:
    """A request waiting to be processed by the inference worker."""

    body: dict[str, Any]
    protocol: str = "openai"
    # Per-request inbound headers (lowercased). Carries the inbound credential
    # the handler relocates to the backend. Per-item, never shared.
    headers: dict[str, str] = field(default_factory=dict)
    future: asyncio.Future = field(default=None)  # type: ignore[assignment]
    cancelled: bool = False

    def __post_init__(self) -> None:
        if self.future is None:
            self.future = asyncio.get_running_loop().create_future()


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
        max_tool_errors: int = 2,
        rescue_enabled: bool = True,
        native_passthrough: bool = True,
        inject_respond_tool: bool = False,
        reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
        backend_protocol: str = "openai",
        backend_api_key_present: bool = False,
        lazy_discovery: LazyDiscovery | None = None,
    ) -> None:
        self._client = client
        self._context_manager = context_manager
        self._lazy_discovery = lazy_discovery
        self._host = host
        self._port = port
        self._max_retries = max_retries
        self._max_tool_errors = max_tool_errors
        self._rescue_enabled = rescue_enabled
        self._native_passthrough = native_passthrough
        self._inject_respond_tool = inject_respond_tool
        self._reasoning_replay = validate_reasoning_replay(reasoning_replay)
        # Target wire protocol of the backend (relocation target) and whether a
        # static --backend-api-key is configured (for the two-source check).
        # The raw key itself never reaches the handler — it is baked into the
        # backend client at construction; the handler only needs to know it
        # exists to refuse an inbound credential alongside it.
        self._backend_protocol = backend_protocol
        self._backend_api_key_present = backend_api_key_present
        self._server: asyncio.Server | None = None
        self._serialize = serialize_requests
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start listening for connections."""
        if self._serialize:
            self._worker_task = asyncio.create_task(self._inference_worker())
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port,
        )
        logger.info("Proxy listening on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the server."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _inference_worker(self) -> None:
        """Single worker that pulls requests off the queue and processes them.

        Ensures only one inference runs at a time (single-GPU constraint).
        """
        while True:
            item = await self._queue.get()
            try:
                if item.cancelled or item.future.cancelled():
                    logger.info("   Skipping cancelled request")
                    continue
                result = await self._run_handler(item.body, item.protocol, item.headers)
                if not item.future.done():
                    item.future.set_result(result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not item.future.done():
                    item.future.set_result(exc)
            finally:
                self._queue.task_done()

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

            method, raw_path = parts[0], parts[1]
            logger.info(">> %s %s", method, raw_path)
            # Strip the query string before routing. Real clients append
            # query params (e.g. Claude Code POSTs /v1/messages?beta=true);
            # exact-matching the raw target would 404 every such request.
            path = raw_path.split("?", 1)[0]

            # Read headers
            headers = await self._read_headers(reader)
            try:
                content_length = int(headers.get("content-length", "0"))
            except ValueError:
                await self._send_error(writer, 400, "Invalid Content-Length")
                return

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
                await self._handle_completions(
                    writer, body_bytes, protocol="openai", headers=headers,
                )
            elif method == "POST" and path == "/v1/messages":
                await self._handle_completions(
                    writer, body_bytes, protocol="anthropic", headers=headers,
                )
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
                key = key.strip().lower()
                # A repeated auth header name would collapse to last-wins in a
                # plain dict — forge must never silently pick a credential
                # winner, so flag it for the credential resolver to refuse.
                if key in AUTH_HEADER_NAMES and key in headers:
                    headers[DUPLICATE_AUTH_MARKER] = "1"
                headers[key] = value.strip()
        return headers

    async def _handle_health(self, writer: asyncio.StreamWriter) -> None:
        """GET /health — returns OK."""
        body = json.dumps({"status": "ok"})
        await self._send_json(writer, 200, body)

    async def _handle_models(self, writer: asyncio.StreamWriter) -> None:
        """GET /v1/models — report the backend model the proxy is fronting."""
        body = json.dumps({
            "object": "list",
            "data": [{"id": self._client.model, "object": "model"}],
        })
        await self._send_json(writer, 200, body)

    async def _handle_completions(
        self,
        writer: asyncio.StreamWriter,
        body_bytes: bytes,
        protocol: str = "openai",
        headers: dict[str, str] | None = None,
    ) -> None:
        """POST /v1/chat/completions (or /v1/messages) — the main proxy endpoint."""
        headers = headers or {}
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            await self._send_error(writer, 400, "Invalid JSON")
            return

        if not isinstance(body, dict):
            await self._send_error(writer, 400, "Request body must be a JSON object")
            return

        is_stream = body.get("stream", False)
        msg_count = len(body.get("messages", []))
        tool_count = len(body.get("tools", []))
        logger.info(
            "   proto=%s stream=%s messages=%d tools=%d model=%s",
            protocol, is_stream, msg_count, tool_count, body.get("model", "?"),
        )

        # Streaming responses flush a 200 + SSE header before the handler runs
        # (so a queued client knows the connection is alive). Run the checks that
        # must be able to fail with a real HTTP status — credential resolution
        # and the first-request discovery probe — BEFORE that flush, so a bad/
        # missing/duplicate credential returns 400/401 rather than 200 + an SSE
        # error event. On success they latch, so the handler skips re-running
        # discovery; its credential resolution is pure and repeats harmlessly.
        # Non-streaming needs no pre-check — it never flushes early, so its
        # errors already carry a real status.
        if is_stream:
            predispatch_error = await self._predispatch(protocol, headers)
            if predispatch_error is not None:
                await self._send_exception(
                    writer, predispatch_error, protocol, as_stream=False,
                )
                return

        if self._serialize:
            # Queue the request and wait for the worker to process it
            item = _QueueItem(body=body, protocol=protocol, headers=headers)
            queue_depth = self._queue.qsize()
            if queue_depth > 0:
                logger.info("   Queued (depth=%d)", queue_depth + 1)

            # For streaming requests, send SSE headers immediately so the
            # client knows we're alive while waiting in the queue
            if is_stream:
                await self._send_sse_header(writer)

            self._queue.put_nowait(item)

            # Wait for result, monitoring for client disconnect
            result = await self._await_with_disconnect(item, writer)
        else:
            if is_stream:
                await self._send_sse_header(writer)
            result = await self._run_handler(body, protocol, headers)

        if result is None:
            # Client disconnected
            logger.info("<< Client disconnected, discarding result")
            return

        if isinstance(result, Exception):
            await self._send_exception(writer, result, protocol, as_stream=is_stream)
            return

        if is_stream:
            logger.info("<< SSE %d events", len(result))
            await self._send_sse_body(writer, result, protocol=protocol)
        else:
            logger.info("<< JSON 200")
            await self._send_json(writer, 200, json.dumps(result))

    async def _predispatch(
        self, protocol: str, headers: dict[str, str],
    ) -> Exception | None:
        """Pre-flush validation for a streaming request.

        Resolves the inbound credential and runs first-request backend discovery
        — the checks that must be able to fail with a real HTTP status — and
        returns the Exception to surface, or None to proceed. Idempotent: the
        handler re-resolves the (pure) credential and sees discovery latched, so
        running this first changes nothing on the success path.
        """
        try:
            extra_headers = resolve_inbound_credential(
                headers,
                source_protocol=protocol,
                target_protocol=self._backend_protocol,
                backend_api_key_present=self._backend_api_key_present,
            )
            await run_lazy_discovery(
                self._client, self._context_manager, self._lazy_discovery, extra_headers,
            )
            return None
        except Exception as exc:
            return exc

    async def _send_exception(
        self,
        writer: asyncio.StreamWriter,
        exc: Exception,
        protocol: str,
        as_stream: bool,
    ) -> None:
        """Send an exception as the response.

        ``as_stream`` True → an SSE error event (the 200 + SSE header was already
        flushed, e.g. a backend fault mid-generation); False → a real HTTP error
        status. Exception messages are safe to log/return by construction —
        forge never authors a secret into one, and ``BackendError`` keeps the raw
        backend body off its message (on ``exc.body`` instead).
        """
        error_msg = str(exc)
        logger.info("<< ERROR: %s", error_msg[:120])
        # Credential problems are client errors (two credentials / one colliding
        # with --backend-api-key → 400, or none to an auth-required backend →
        # 401), not backend failures. These messages carry only slot names.
        if isinstance(exc, MultipleCredentialsError):
            status = 400
        elif isinstance(exc, MissingCredentialError):
            status = 401
        elif isinstance(exc, BackendDiscoveryError):
            # Deferred discovery failed: a backend auth rejection is the caller's
            # 401; any other cause (backend down, bad shape) is a 502.
            status = 401 if exc.status_code in (401, 403) else 502
        elif isinstance(exc, BackendError) and exc.status_code in (401, 403):
            # A backend auth rejection during normal dispatch (a later zero-cred
            # request to a gated backend, or a bad inbound key) is the caller's
            # 401, not a forge/backend fault.
            status = 401
        else:
            status = 502
        if as_stream:
            await self._send_sse_body(writer, [{"error": error_msg}], protocol=protocol)
        else:
            await self._send_error(writer, status, error_msg)

    async def _await_with_disconnect(
        self,
        item: _QueueItem,
        writer: asyncio.StreamWriter,
    ) -> dict[str, Any] | list[dict[str, Any]] | Exception | None:
        """Wait for a queued item's result, checking for client disconnect.

        Returns None if the client disconnected.
        """
        while not item.future.done():
            if writer.is_closing():
                item.cancelled = True
                logger.info("   Client disconnected, cancelling queued request")
                return None
            try:
                await asyncio.wait_for(
                    asyncio.shield(item.future), timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
        return item.future.result()

    async def _run_handler(
        self,
        body: dict[str, Any],
        protocol: str = "openai",
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]] | Exception:
        """Run the handler, catching errors."""
        try:
            return await handle_chat_completions(
                body=body,
                client=self._client,
                context_manager=self._context_manager,
                max_retries=self._max_retries,
                max_tool_errors=self._max_tool_errors,
                rescue_enabled=self._rescue_enabled,
                native_passthrough=self._native_passthrough,
                inject_respond_tool=self._inject_respond_tool,
                protocol=protocol,
                reasoning_replay=self._reasoning_replay,
                headers=headers,
                backend_protocol=self._backend_protocol,
                backend_api_key_present=self._backend_api_key_present,
                lazy_discovery=self._lazy_discovery,
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

    async def _send_sse_header(self, writer: asyncio.StreamWriter) -> None:
        """Send SSE response headers immediately (before body is ready)."""
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

    async def _send_sse_body(
        self,
        writer: asyncio.StreamWriter,
        events: list[dict[str, Any]],
        protocol: str = "openai",
    ) -> None:
        """Send SSE event data and terminator. Headers must already be sent.

        OpenAI wire format: ``data: {json}\\n\\n`` per event, terminated by
        ``data: [DONE]\\n\\n``.

        Anthropic wire format: ``event: <type>\\ndata: {json}\\n\\n`` per
        event (type read from the event's top-level ``type`` field). No
        ``[DONE]`` terminator — the ``message_stop`` event ends the stream.
        """
        for event in events:
            if writer.is_closing():
                return
            if protocol == "anthropic":
                event_type = event.get("type", "")
                payload = f"event: {event_type}\ndata: {json.dumps(event)}\n\n".encode()
            else:
                payload = f"data: {json.dumps(event)}\n\n".encode()
            writer.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            await writer.drain()

        if protocol == "openai":
            done = b"data: [DONE]\n\n"
            writer.write(f"{len(done):x}\r\n".encode() + done + b"\r\n")

        # Terminating zero-length chunk
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        logger.info("<< SSE complete (%s)", protocol)

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
            # x-api-key is a first-class inbound credential slot (Anthropic-wire);
            # browser clients must be allowed to preflight it. anthropic-version /
            # anthropic-beta are standard Anthropic client headers.
            "Access-Control-Allow-Headers: Content-Type, Authorization, X-Api-Key, "
            "anthropic-version, anthropic-beta\r\n"
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
        401: "Unauthorized",
        404: "Not Found",
        413: "Payload Too Large",
        500: "Internal Server Error",
        502: "Bad Gateway",
    }.get(code, "Error")
