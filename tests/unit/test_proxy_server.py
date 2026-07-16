"""Tests for proxy HTTP server."""

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.workflow import TextResponse, ToolCall
from forge.errors import BackendError
from forge.proxy.handler import LazyDiscovery
from forge.proxy.server import HTTPServer


# ── Helpers ──────────────────────────────────────────────────


def _mock_client(response):
    """Create a mock LLMClient that returns the given response."""
    client = AsyncMock()
    client.api_format = "ollama"
    client.model = "mock-model"
    client.send = AsyncMock(return_value=response)
    # No raw HTTP backend to forward to (passthrough tests override this).
    # forward_request is called WITHOUT await — it returns an async context
    # manager, not a coroutine — so it needs a sync mock, or the None check
    # misfires on the auto-created coroutine.
    client.forward_request = MagicMock(return_value=None)
    return client


@pytest.fixture
async def server_factory():
    """Factory fixture that creates an HTTPServer on a random port."""
    servers = []

    async def _make(response, serialize=False):
        client = _mock_client(response)
        ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
        srv = HTTPServer(
            client=client,
            context_manager=ctx,
            host="127.0.0.1",
            port=0,  # OS picks a free port
            serialize_requests=serialize,
        )
        await srv.start()
        # Get the actual port from the server
        sock = srv._server.sockets[0]
        port = sock.getsockname()[1]
        servers.append(srv)
        return srv, port

    yield _make

    for srv in servers:
        await srv.stop()


async def _http_request(port, method, path, body=None):
    """Send an HTTP request and return (status, headers_dict, body_str)."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        if body is not None:
            body_bytes = json.dumps(body).encode()
            request = (
                f"{method} {path} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
                f"\r\n"
            ).encode() + body_bytes
        else:
            request = (
                f"{method} {path} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"\r\n"
            ).encode()

        writer.write(request)
        await writer.drain()

        # Read response
        response_data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
        response_str = response_data.decode("utf-8", errors="replace")

        # Parse status line
        lines = response_str.split("\r\n")
        status = int(lines[0].split(" ", 2)[1])

        # Find body (after blank line)
        body_start = response_str.find("\r\n\r\n")
        response_body = response_str[body_start + 4:] if body_start >= 0 else ""

        return status, response_body
    finally:
        writer.close()
        await writer.wait_closed()


async def _sse_request(port, body):
    """Send a streaming request and return list of SSE data lines."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        body_bytes = json.dumps(body).encode()
        request = (
            f"POST /v1/chat/completions HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"\r\n"
        ).encode() + body_bytes

        writer.write(request)
        await writer.drain()

        response_data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
        response_str = response_data.decode("utf-8", errors="replace")

        # Extract SSE data lines from chunked transfer encoding
        data_lines = []
        for line in response_str.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                data_lines.append(line[6:])

        return data_lines
    finally:
        writer.close()
        await writer.wait_closed()


# ── Health & Models ──────────────────────────────────────────


class TestHealthAndModels:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        status, body = await _http_request(port, "GET", "/health")
        assert status == 200
        data = json.loads(body)
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_v1_health_alias(self, server_factory):
        # Mirrors llama-server's own /v1/health alias of its public health
        # endpoint; same fallback as /health.
        srv, port = await server_factory(TextResponse(content=""))
        status, body = await _http_request(port, "GET", "/v1/health")
        assert status == 200
        assert json.loads(body)["status"] == "ok"

    @pytest.mark.asyncio
    async def test_models_endpoint(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        status, body = await _http_request(port, "GET", "/v1/models")
        assert status == 200
        data = json.loads(body)
        assert data["object"] == "list"
        assert data["data"][0]["id"] == "mock-model"

    @pytest.mark.asyncio
    async def test_not_found(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        status, body = await _http_request(port, "GET", "/nonexistent")
        assert status == 404

    @pytest.mark.asyncio
    async def test_cors_preflight(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        status, _ = await _http_request(port, "OPTIONS", "/v1/chat/completions")
        assert status == 204


# ── Chat Completions ────────────────────────────────────────


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_no_tools_text_response(self, server_factory):
        srv, port = await server_factory(TextResponse(content="Hello!"))
        body = {"messages": [{"role": "user", "content": "hi"}]}
        status, response_body = await _http_request(
            port, "POST", "/v1/chat/completions", body,
        )
        assert status == 200
        data = json.loads(response_body)
        assert data["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_tool_call_response(self, server_factory):
        srv, port = await server_factory(
            [ToolCall(tool="search", args={"q": "test"})],
        )
        body = {
            "messages": [{"role": "user", "content": "search for test"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
        }
        status, response_body = await _http_request(
            port, "POST", "/v1/chat/completions", body,
        )
        assert status == 200
        data = json.loads(response_body)
        tc = data["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            bad_body = b"not json"
            request = (
                f"POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(bad_body)}\r\n"
                f"\r\n"
            ).encode() + bad_body
            writer.write(request)
            await writer.drain()
            response_data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            assert b"400" in response_data
        finally:
            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_invalid_content_length_returns_400(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            request = (
                f"POST /v1/chat/completions HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: abc\r\n"
                f"\r\n"
            ).encode()
            writer.write(request)
            await writer.drain()
            response_data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            assert b"400" in response_data
        finally:
            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_non_object_body_returns_400(self, server_factory):
        srv, port = await server_factory(TextResponse(content=""))
        # Valid JSON but not an object (array) must be rejected before the
        # handler calls body.get(...), which would otherwise raise.
        status, _ = await _http_request(
            port, "POST", "/v1/chat/completions", body=[1, 2, 3]
        )
        assert status == 400


# ── SSE Streaming ───────────────────────────────────────────


class TestSSEStreaming:
    @pytest.mark.asyncio
    async def test_streaming_text_response(self, server_factory):
        srv, port = await server_factory(TextResponse(content="Hello!"))
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        data_lines = await _sse_request(port, body)
        # Should have content events and [DONE]
        assert "[DONE]" in data_lines
        json_events = [json.loads(d) for d in data_lines if d != "[DONE]"]
        assert len(json_events) > 0

    @pytest.mark.asyncio
    async def test_streaming_tool_call(self, server_factory):
        srv, port = await server_factory(
            [ToolCall(tool="search", args={"q": "x"})],
        )
        body = {
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            "stream": True,
        }
        data_lines = await _sse_request(port, body)
        assert "[DONE]" in data_lines
        json_events = [json.loads(d) for d in data_lines if d != "[DONE]"]
        # Should have tool call deltas
        has_tool_call = any(
            "tool_calls" in e["choices"][0].get("delta", {})
            for e in json_events
        )
        assert has_tool_call


# ── Serialization ───────────────────────────────────────────


class TestSerialization:
    @pytest.mark.asyncio
    async def test_serialized_requests_processed(self, server_factory):
        """Serialized mode processes requests through the queue."""
        srv, port = await server_factory(
            TextResponse(content="ok"), serialize=True,
        )
        body = {"messages": [{"role": "user", "content": "hi"}]}
        status, response_body = await _http_request(
            port, "POST", "/v1/chat/completions", body,
        )
        assert status == 200
        data = json.loads(response_body)
        assert data["choices"][0]["message"]["content"] == "ok"


# ── Inbound credential threading (v0.8.0) ────────────────────


async def _http_request_with_auth(port, body, auth_header):
    """POST /v1/chat/completions with an extra Authorization header."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        body_bytes = json.dumps(body).encode()
        request = (
            f"POST /v1/chat/completions HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Authorization: {auth_header}\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"\r\n"
        ).encode() + body_bytes
        writer.write(request)
        await writer.drain()
        await asyncio.wait_for(reader.read(65536), timeout=10.0)
    finally:
        writer.close()
        await writer.wait_closed()


async def _auth_server(serialize, backend_api_key_present=False):
    """An HTTPServer fronting an Anthropic-wire backend, with a mock client."""
    client = _mock_client(TextResponse(content="ok"))
    ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
    srv = HTTPServer(
        client=client,
        context_manager=ctx,
        host="127.0.0.1",
        port=0,
        serialize_requests=serialize,
        backend_protocol="anthropic",
        backend_api_key_present=backend_api_key_present,
    )
    await srv.start()
    port = srv._server.sockets[0].getsockname()[1]
    return srv, port, client


async def _raw_request(port, header_lines, body):
    """POST with arbitrary extra header lines; return (status, body_str)."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        body_bytes = json.dumps(body).encode()
        extra = "".join(f"{line}\r\n" for line in header_lines)
        request = (
            f"POST /v1/chat/completions HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\n"
            f"{extra}"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"\r\n"
        ).encode() + body_bytes
        writer.write(request)
        await writer.drain()
        data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
        text = data.decode("utf-8", errors="replace")
        status = int(text.split("\r\n", 1)[0].split(" ", 2)[1])
        start = text.find("\r\n\r\n")
        return status, (text[start + 4:] if start >= 0 else "")
    finally:
        writer.close()
        await writer.wait_closed()


class TestInboundCredentialThreading:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("serialize", [False, True])
    async def test_inbound_auth_relocated_to_backend_client(self, serialize):
        # Both dispatch paths (direct and the serialized queue worker) must
        # carry the inbound header to the handler. Source openai endpoint →
        # anthropic backend: Bearer stripped, relocated to x-api-key.
        srv, port, client = await _auth_server(serialize)
        try:
            await _http_request_with_auth(
                port,
                {"messages": [{"role": "user", "content": "hi"}]},
                "Bearer INBOUND",
            )
            assert client.send.await_count == 1
            assert client.send.call_args.kwargs["extra_headers"] == {"x-api-key": "INBOUND"}
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_duplicate_auth_header_refused_400_no_secret(self):
        # Two same-name Authorization headers must be refused (never pick a
        # winner), as a 400 client error, with no secret in the response body.
        srv, port, client = await _auth_server(serialize=False)
        try:
            status, resp_body = await _raw_request(
                port,
                ["Authorization: Bearer SECRET-ONE", "Authorization: Bearer SECRET-TWO"],
                {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 400
            assert "SECRET-ONE" not in resp_body
            assert "SECRET-TWO" not in resp_body
            client.send.assert_not_awaited()
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_inbound_plus_static_key_refused_400(self):
        # Inbound credential + configured --backend-api-key → 400 client error.
        srv, port, client = await _auth_server(
            serialize=False, backend_api_key_present=True,
        )
        try:
            status, resp_body = await _raw_request(
                port,
                ["Authorization: Bearer SECRET-INBOUND"],
                {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 400
            assert "SECRET-INBOUND" not in resp_body
            client.send.assert_not_awaited()
        finally:
            await srv.stop()


# ── Deferred discovery → status mapping (finding #2) ─────────


async def _discovery_server(*, side_effect=None, budget=50000, apply_budget=True):
    """An OpenAI-wire server whose first request triggers deferred discovery."""
    client = _mock_client(TextResponse(content="ok"))
    if side_effect is not None:
        client.discover_backend_metadata = AsyncMock(side_effect=side_effect)
    else:
        client.discover_backend_metadata = AsyncMock(return_value=budget)
    ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
    srv = HTTPServer(
        client=client,
        context_manager=ctx,
        host="127.0.0.1",
        port=0,
        serialize_requests=False,
        backend_protocol="openai",
        lazy_discovery=LazyDiscovery(deferred=True, apply_budget=apply_budget),
    )
    await srv.start()
    port = srv._server.sockets[0].getsockname()[1]
    return srv, port, client, ctx


class TestDeferredDiscoveryStatusMapping:
    @pytest.mark.asyncio
    async def test_auth_rejection_maps_401(self):
        srv, port, client, _ = await _discovery_server(
            side_effect=BackendError(401, "unauthorized"),
        )
        try:
            status, _ = await _raw_request(
                port, [], {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 401  # backend rejected the probe credential
            client.send.assert_not_awaited()
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_backend_fault_maps_502(self):
        srv, port, client, _ = await _discovery_server(
            side_effect=BackendError(502, "backend unreachable"),
        )
        try:
            status, _ = await _raw_request(
                port, [], {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 502
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_success_applies_budget_and_serves_200(self):
        srv, port, client, ctx = await _discovery_server(budget=50000)
        try:
            status, _ = await _raw_request(
                port, [], {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 200
            assert ctx.budget_tokens == 50000  # discovered budget latched
            client.discover_backend_metadata.assert_awaited_once()
        finally:
            await srv.stop()


# ── Codex review hardening (backend-error status + secret hygiene + CORS) ──


async def _error_server(exc):
    """An openai-wire server whose backend client.send raises ``exc``."""
    client = _mock_client(TextResponse(content="x"))
    client.send = AsyncMock(side_effect=exc)
    ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
    srv = HTTPServer(
        client=client, context_manager=ctx, host="127.0.0.1", port=0,
        serialize_requests=False, backend_protocol="openai",
    )
    await srv.start()
    return srv, srv._server.sockets[0].getsockname()[1]


async def _raw_response(port, method, path):
    """Return the full raw HTTP response string for a bodyless request."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(
            f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode()
        )
        await writer.drain()
        data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
        return data.decode("utf-8", errors="replace")
    finally:
        writer.close()
        await writer.wait_closed()


class TestBackendErrorStatusMapping:
    @pytest.mark.asyncio
    async def test_backend_401_during_dispatch_maps_401(self):
        srv, port = await _error_server(BackendError(401, "unauthorized"))
        try:
            status, _ = await _raw_request(
                port, [], {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 401  # backend auth rejection is the caller's 401
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_backend_403_maps_401(self):
        srv, port = await _error_server(BackendError(403, "forbidden"))
        try:
            status, _ = await _raw_request(
                port, [], {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 401
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_backend_500_still_maps_502(self):
        srv, port = await _error_server(BackendError(500, "boom"))
        try:
            status, _ = await _raw_request(
                port, [], {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 502  # non-auth backend fault stays 502
        finally:
            await srv.stop()


class TestSecretHygiene:
    @pytest.mark.asyncio
    async def test_backend_error_body_secret_not_leaked_to_client(self):
        # A backend that echoes the inbound auth header in its raw error body
        # must not leak it: the raw body rides exc.body (off the message), so the
        # proxy returns only the safe "Backend returned <status>" summary.
        srv, port = await _error_server(
            BackendError(500, raw_body="rejected Authorization: Bearer sk-leak-7777"),
        )
        try:
            _, body = await _raw_request(
                port, [], {"messages": [{"role": "user", "content": "hi"}]},
            )
            assert "sk-leak-7777" not in body
            assert "Backend returned 500" in body  # safe status summary only
        finally:
            await srv.stop()


class TestStreamingErrorStatus:
    """Pre-dispatch errors on a streaming request return a real HTTP status,
    not a 200 + SSE error event (the SSE header is flushed only after the
    credential + first-request discovery checks pass)."""

    @pytest.mark.asyncio
    async def test_streaming_duplicate_auth_returns_400_not_200(self):
        srv, port, client = await _auth_server(serialize=False)
        try:
            status, body = await _raw_request(
                port,
                ["Authorization: Bearer SECRET-ONE", "Authorization: Bearer SECRET-TWO"],
                {"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert status == 400  # real status, not 200 + an SSE error event
            assert "SECRET-ONE" not in body and "SECRET-TWO" not in body
            client.send.assert_not_awaited()
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_streaming_discovery_failure_returns_401_not_200(self):
        srv, port, client, _ = await _discovery_server(
            side_effect=BackendError(401, "unauthorized"),
        )
        try:
            status, _ = await _raw_request(
                port, [],
                {"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert status == 401  # deferred-discovery 401 surfaces before the stream
            client.send.assert_not_awaited()
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_streaming_success_still_sse_200(self):
        # The success path is unchanged: header flushes and SSE events stream.
        srv, port, client = await _auth_server(serialize=False)
        try:
            status, body = await _raw_request(
                port,
                ["Authorization: Bearer GOODKEY"],
                {"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert status == 200
            assert "data:" in body
        finally:
            await srv.stop()


# ── Verbatim passthrough (llama.cpp-native surface + /v1/models) ──


class _FakeBackendResponse:
    """Streaming-mode httpx.Response stand-in: status_code / aread / aiter_raw."""

    def __init__(self, status, chunks):
        self.status_code = status
        self._chunks = chunks

    async def aread(self):
        return b"".join(self._chunks)

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk


class _FakeForwardCM:
    """Async CM standing in for LLMClient.forward_request()."""

    def __init__(self, resp=None, enter_exc=None):
        self._resp = resp
        self._enter_exc = enter_exc

    async def __aenter__(self):
        if self._enter_exc is not None:
            raise self._enter_exc
        return self._resp

    async def __aexit__(self, *args):
        return False


async def _passthrough_server(
    *,
    resp=None,
    enter_exc=None,
    forward_none=False,
    backend_protocol="openai",
    backend_api_key_present=False,
):
    client = _mock_client(TextResponse(content="ok"))
    if forward_none:
        client.forward_request = MagicMock(return_value=None)
    else:
        client.forward_request = MagicMock(
            return_value=_FakeForwardCM(resp, enter_exc),
        )
    ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
    srv = HTTPServer(
        client=client, context_manager=ctx, host="127.0.0.1", port=0,
        serialize_requests=False, backend_protocol=backend_protocol,
        backend_api_key_present=backend_api_key_present,
    )
    await srv.start()
    return srv, srv._server.sockets[0].getsockname()[1], client


async def _request(port, method, path, body=None, header_lines=()):
    """Any-method HTTP request; returns (status, body_str, raw_response)."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        body_bytes = b"" if body is None else json.dumps(body).encode()
        head = (
            f"{method} {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            + "".join(f"{line}\r\n" for line in header_lines)
        )
        if body_bytes:
            head += (
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
            )
        writer.write(head.encode() + b"\r\n" + body_bytes)
        await writer.drain()
        data = await asyncio.wait_for(reader.read(-1), timeout=10.0)
        raw = data.decode("utf-8", errors="replace")
        status = int(raw.split("\r\n", 1)[0].split(" ", 2)[1])
        sep = raw.find("\r\n\r\n")
        return status, (raw[sep + 4:] if sep >= 0 else ""), raw
    finally:
        writer.close()
        await writer.wait_closed()


# The full llama.cpp-native surface the proxy forwards — one case per route,
# (method, inbound path[+query], request body or None). Mirrors
# _PASSTHROUGH_ROUTES plus the /v1/models family (_handle_models) and the
# health endpoints (_handle_health).
PASSTHROUGH_CASES = [
    ("GET", "/health", None),
    ("GET", "/v1/health", None),
    ("GET", "/props?model=Bonsai-27B&autoload=false", None),
    ("POST", "/props", {"foo": 1}),
    ("GET", "/v1/models", None),
    ("GET", "/models", None),
    ("POST", "/models", {"model": "x"}),
    ("DELETE", "/models", {"model": "x"}),
    ("POST", "/models/load", {"model": "Bonsai-27B"}),
    ("POST", "/models/unload", {"model": "Bonsai-27B"}),
]

_MODELS_LISTINGS = ("/v1/models", "/models")
_HEALTH_PATHS = ("/health", "/v1/health")


class TestPassthroughForwarding:
    """The whole passthrough family relays the backend verbatim.

    One parameterized contract instead of a class per endpoint: the backend's
    status and body are the answer (its own 404 for an endpoint it doesn't
    serve, 400 "model is not loaded" from /props?model=), the raw method /
    path / query / body reach the client's forward_request untouched, and
    credential handling matches the completions path.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method,target,body", PASSTHROUGH_CASES)
    async def test_answer_forwarded_verbatim(self, method, target, body):
        payload = b'{"answer": 42}'
        srv, port, client = await _passthrough_server(
            resp=_FakeBackendResponse(200, [payload]),
        )
        try:
            status, resp_body, _ = await _request(port, method, target, body)
            assert status == 200
            assert resp_body == payload.decode()
            # Method, path + raw query, and raw body reach the client
            # untouched (the last call — /v1/models probes capability first).
            call = client.forward_request.call_args
            assert call.args == (method, target)
            expected_body = b"" if body is None else json.dumps(body).encode()
            assert call.kwargs["body"] == expected_body
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method,target,body", PASSTHROUGH_CASES)
    async def test_backend_status_is_the_answer(self, method, target, body):
        # Error statuses ride through unmapped — never collapsed to 502.
        err = b'{"error": {"code": 404, "message": "File Not Found"}}'
        srv, port, _ = await _passthrough_server(
            resp=_FakeBackendResponse(404, [err]),
        )
        try:
            status, resp_body, _ = await _request(port, method, target, body)
            assert status == 404
            assert resp_body == err.decode()
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method,target,body", PASSTHROUGH_CASES)
    async def test_no_raw_backend_404_except_local_fallbacks(self, method, target, body):
        # forward_request → None (Anthropic SDK path): forge's own 404 —
        # except the models listing (legacy synthesized single-identity
        # entry) and health (the proxy's own liveness).
        srv, port, _ = await _passthrough_server(forward_none=True)
        try:
            status, resp_body, _ = await _request(port, method, target, body)
            # Only the GET spellings are the models LISTING; POST/DELETE
            # /models are router management and 404 like the rest.
            if method == "GET" and target in _MODELS_LISTINGS:
                assert status == 200
                assert json.loads(resp_body)["data"][0]["id"] == "mock-model"
            elif method == "GET" and target in _HEALTH_PATHS:
                assert status == 200
                assert json.loads(resp_body)["status"] == "ok"
            else:
                assert status == 404
                assert "Not found" in resp_body
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_unreachable_backend_maps_502(self):
        srv, port, _ = await _passthrough_server(
            enter_exc=BackendError(502, "backend unreachable"),
        )
        try:
            status, _, _ = await _request(port, "GET", "/props")
            assert status == 502
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_inbound_credential_relocated(self):
        # openai inbound → anthropic backend: Bearer stripped, token moved to
        # x-api-key — identical to the completions path.
        srv, port, client = await _passthrough_server(
            resp=_FakeBackendResponse(200, [b"{}"]),
            backend_protocol="anthropic",
        )
        try:
            status, _, _ = await _request(
                port, "GET", "/props", header_lines=["Authorization: Bearer INBOUND"],
            )
            assert status == 200
            assert client.forward_request.call_args.kwargs["extra_headers"] == {
                "x-api-key": "INBOUND",
            }
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_duplicate_auth_refused_400_no_secret(self):
        srv, port, client = await _passthrough_server(
            resp=_FakeBackendResponse(200, [b"{}"]),
        )
        try:
            status, resp_body, _ = await _request(
                port, "GET", "/props",
                header_lines=[
                    "Authorization: Bearer SECRET-ONE",
                    "Authorization: Bearer SECRET-TWO",
                ],
            )
            assert status == 400
            assert "SECRET-ONE" not in resp_body and "SECRET-TWO" not in resp_body
            client.forward_request.assert_not_called()
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_inbound_plus_static_key_refused_400(self):
        srv, port, client = await _passthrough_server(
            resp=_FakeBackendResponse(200, [b"{}"]),
            backend_api_key_present=True,
        )
        try:
            status, resp_body, _ = await _request(
                port, "GET", "/props",
                header_lines=["Authorization: Bearer SECRET-INBOUND"],
            )
            assert status == 400
            assert "SECRET-INBOUND" not in resp_body
            client.forward_request.assert_not_called()
        finally:
            await srv.stop()


class TestModelsSseRelay:
    """GET /models/sse is the one streaming row of the passthrough table:
    chunks relay unbuffered with SSE headers; a refused subscription buffers
    and forwards its real status like every other passthrough answer."""

    @pytest.mark.asyncio
    async def test_stream_relayed_with_sse_headers(self):
        events = [b'event: models\ndata: {"loading": 1}\n\n', b"data: done\n\n"]
        srv, port, client = await _passthrough_server(
            resp=_FakeBackendResponse(200, events),
        )
        try:
            _, _, raw = await _request(port, "GET", "/models/sse?api_key=sk-placeholder")
            assert raw.startswith("HTTP/1.1 200 OK")
            assert "text/event-stream" in raw
            for event in events:
                assert event.decode() in raw
            assert raw.rstrip().endswith("0")  # chunked terminator sent
            call = client.forward_request.call_args
            assert call.args == ("GET", "/models/sse?api_key=sk-placeholder")
            assert call.kwargs["stream"] is True  # read timeout disabled
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_refused_subscription_forwards_status_and_body(self):
        err = json.dumps({"error": {"code": 403, "message": "nope"}}).encode()
        srv, port, _ = await _passthrough_server(
            resp=_FakeBackendResponse(403, [err]),
        )
        try:
            _, _, raw = await _request(port, "GET", "/models/sse")
            assert raw.startswith("HTTP/1.1 403")
            assert '"nope"' in raw
            assert "text/event-stream" not in raw
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_empty_chunk_does_not_end_stream(self):
        # A zero-length backend chunk must not become the chunked-encoding
        # terminator: the event after it still arrives.
        events = [b"data: first\n\n", b"", b"data: second\n\n"]
        srv, port, _ = await _passthrough_server(
            resp=_FakeBackendResponse(200, events),
        )
        try:
            _, _, raw = await _request(port, "GET", "/models/sse")
            assert "data: first" in raw
            assert "data: second" in raw
        finally:
            await srv.stop()


class TestCorsAllowsApiKey:
    @pytest.mark.asyncio
    async def test_preflight_allows_x_api_key(self):
        srv, _, _ = await _auth_server(serialize=False)
        port = srv._server.sockets[0].getsockname()[1]
        try:
            raw = await _raw_response(port, "OPTIONS", "/v1/chat/completions")
            assert raw.startswith("HTTP/1.1 204")
            allow = next(
                l for l in raw.splitlines()
                if l.lower().startswith("access-control-allow-headers")
            )
            assert "x-api-key" in allow.lower()
        finally:
            await srv.stop()
