"""Tests for proxy HTTP server."""

import asyncio
import json

import pytest
from unittest.mock import AsyncMock

from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.workflow import TextResponse, ToolCall
from forge.proxy.server import HTTPServer


# ── Helpers ──────────────────────────────────────────────────


def _mock_client(response):
    """Create a mock LLMClient that returns the given response."""
    client = AsyncMock()
    client.api_format = "ollama"
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
        assert len(data["data"]) > 0

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
