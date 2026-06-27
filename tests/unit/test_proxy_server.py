"""Tests for proxy HTTP server."""

import asyncio
import json

import pytest
from unittest.mock import AsyncMock

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
