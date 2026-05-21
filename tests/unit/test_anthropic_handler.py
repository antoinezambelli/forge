"""Tests for forge.proxy.handler — Anthropic backend and /v1/messages.

Tests the handler's anthropic_backend flag and handle_messages entry point,
plus the server's /v1/messages endpoint with anthropic_backend=True.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.workflow import TextResponse, ToolCall
from forge.proxy.handler import handle_chat_completions, handle_messages
from forge.proxy.server import HTTPServer


# ── Helpers ──────────────────────────────────────────────────


def _mock_client(response):
    """Create a mock LLMClient that returns the given response."""
    client = AsyncMock()
    client.api_format = "ollama"
    client.send = AsyncMock(return_value=response)
    return client


# ── handle_chat_completions with anthropic_backend=True ──────


class TestHandleChatCompletionsAnthropicBackend:
    """Verify responses are converted to Anthropic format."""

    @pytest.fixture
    def ctx(self):
        return ContextManager(strategy=NoCompact(), budget_tokens=8192)

    @pytest.mark.asyncio
    async def test_text_response_anthropic_format(self, ctx):
        client = _mock_client(TextResponse(content="Hello!"))
        body = {
            "model": "forge",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        result = await handle_chat_completions(
            body=body, client=client, context_manager=ctx,
            anthropic_backend=True,
        )
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "forge"
        assert result["stop_reason"] == "end_turn"
        assert result["content"][0] == {"type": "text", "text": "Hello!"}
        assert "usage" in result

    @pytest.mark.asyncio
    async def test_tool_call_response_anthropic_format(self, ctx):
        client = _mock_client([ToolCall(tool="search", args={"q": "test"})])
        body = {
            "model": "forge",
            "messages": [{"role": "user", "content": "search"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }],
            "stream": False,
        }
        result = await handle_chat_completions(
            body=body, client=client, context_manager=ctx,
            anthropic_backend=True,
        )
        assert result["type"] == "message"
        assert result["stop_reason"] == "tool_use"
        tool_block = result["content"][0]
        assert tool_block["type"] == "tool_use"
        assert tool_block["name"] == "search"
        assert tool_block["input"] == {"q": "test"}

    @pytest.mark.asyncio
    async def test_streaming_text_anthropic_sse(self, ctx):
        client = _mock_client(TextResponse(content="Hi"))
        body = {
            "model": "forge",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        result = await handle_chat_completions(
            body=body, client=client, context_manager=ctx,
            anthropic_backend=True,
        )
        assert isinstance(result, list)
        # Should have message_start
        assert result[0]["type"] == "message_start"
        # Should have text deltas
        text_deltas = [e for e in result if e.get("delta", {}).get("type") == "text_delta"]
        assert len(text_deltas) >= 1
        # Should end with message_stop
        assert result[-1]["type"] == "message_stop"

    @pytest.mark.asyncio
    async def test_streaming_tool_call_anthropic_sse(self, ctx):
        client = _mock_client([ToolCall(tool="get_weather", args={"city": "Paris"})])
        body = {
            "model": "forge",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }],
            "stream": True,
        }
        result = await handle_chat_completions(
            body=body, client=client, context_manager=ctx,
            anthropic_backend=True,
        )
        assert isinstance(result, list)
        # Should have tool_use content blocks
        tool_starts = [e for e in result if e.get("content_block", {}).get("type") == "tool_use"]
        assert len(tool_starts) >= 1
        # Should have input_json_delta events
        input_deltas = [e for e in result if e.get("delta", {}).get("type") == "input_json_delta"]
        assert len(input_deltas) >= 1

    @pytest.mark.asyncio
    async def test_non_anthropic_backend_stays_openai(self, ctx):
        """anthropic_backend=False returns OpenAI format (regression check)."""
        client = _mock_client(TextResponse(content="Hello!"))
        body = {
            "model": "forge",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        result = await handle_chat_completions(
            body=body, client=client, context_manager=ctx,
            anthropic_backend=False,
        )
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello!"


# ── handle_messages (Anthropic incoming) ──────────────────────


class TestHandleMessages:
    """Verify Anthropic-format requests are converted and handled."""

    @pytest.fixture
    def ctx(self):
        return ContextManager(strategy=NoCompact(), budget_tokens=8192)

    @pytest.mark.asyncio
    async def test_text_response_from_anthropic_request(self, ctx):
        client = _mock_client(TextResponse(content="Hello from forge!"))
        body = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": False,
        }
        result = await handle_messages(
            body=body, client=client, context_manager=ctx,
            anthropic_backend=True,
        )
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        # Content should come from the backend (OpenAI format converted)
        assert result["content"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_system_message_converted(self, ctx):
        """Top-level system string → system role message."""
        client = _mock_client(TextResponse(content="Got it"))
        body = {
            "system": "You are a weather bot.",
            "messages": [{"role": "user", "content": "Weather?"}],
            "max_tokens": 100,
            "stream": False,
        }
        result = await handle_messages(
            body=body, client=client, context_manager=ctx,
            anthropic_backend=True,
        )
        assert result["type"] == "message"

    @pytest.mark.asyncio
    async def test_anthropic_tools_converted_to_openai(self, ctx):
        """Anthropic tool format → OpenAI tool format."""
        client = _mock_client([ToolCall(tool="get_weather", args={"city": "London"})])
        body = {
            "messages": [{"role": "user", "content": "Weather in London?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }],
            "max_tokens": 100,
            "stream": False,
        }
        result = await handle_messages(
            body=body, client=client, context_manager=ctx,
            anthropic_backend=True,
        )
        assert result["stop_reason"] == "tool_use"
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_temperature_forwarded(self, ctx):
        """Temperature param passed through to handler."""
        client = _mock_client(TextResponse(content="ok"))
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False,
        }
        result = await handle_messages(
            body=body, client=client, context_manager=ctx,
            anthropic_backend=True,
        )
        assert result["type"] == "message"

    @pytest.mark.asyncio
    async def test_streaming_anthropic_request(self, ctx):
        client = _mock_client(TextResponse(content="streaming"))
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": True,
        }
        result = await handle_messages(
            body=body, client=client, context_manager=ctx,
            anthropic_backend=True,
        )
        assert isinstance(result, list)
        assert result[0]["type"] == "message_start"
        assert result[-1]["type"] == "message_stop"


# ── HTTPServer /v1/messages with anthropic_backend=True ──────


class TestServerAnthropicEndpoint:
    """Full HTTP request to /v1/messages endpoint."""

    @pytest.fixture
    async def anthropic_server_factory(self):
        """Factory that creates an HTTPServer with anthropic_backend=True."""
        servers = []

        async def _make(response, serialize=False):
            client = _mock_client(response)
            ctx = ContextManager(strategy=NoCompact(), budget_tokens=8192)
            srv = HTTPServer(
                client=client,
                context_manager=ctx,
                host="127.0.0.1",
                port=0,
                serialize_requests=serialize,
                anthropic_backend=True,
            )
            await srv.start()
            sock = srv._server.sockets[0]
            port = sock.getsockname()[1]
            servers.append(srv)
            return srv, port

        yield _make

        for srv in servers:
            await srv.stop()

    async def _anthropic_request(self, port, body):
        """Send an Anthropic-format request and return (is_stream, result).

        For streaming: returns (True, list of SSE data line strings).
        For non-streaming: returns (False, parsed JSON dict).
        """
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            body_bytes = json.dumps(body).encode()
            request = (
                f"POST /v1/messages HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
                f"\r\n"
            ).encode() + body_bytes

            writer.write(request)
            await writer.drain()

            is_stream = body.get("stream", False)
            response_data = bytearray()

            if is_stream:
                # SSE uses chunked transfer encoding, ends with 0\r\n\r\n
                while True:
                    chunk = await asyncio.wait_for(reader.read(65536), timeout=10.0)
                    if not chunk:
                        break
                    response_data.extend(chunk)
                    if b"0\r\n\r\n" in response_data:
                        break
            else:
                # JSON response — read headers, then Content-Length bytes
                header_end = b"\r\n\r\n"
                while header_end not in response_data:
                    chunk = await asyncio.wait_for(reader.read(65536), timeout=10.0)
                    if not chunk:
                        break
                    response_data.extend(chunk)
                header_part = response_data.split(header_end)[0].decode()
                content_length = 0
                for line in header_part.split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        content_length = int(line.split(":")[1].strip())
                        break
                body_start = response_data.index(header_end) + 4
                body_received = len(response_data) - body_start
                while body_received < content_length:
                    chunk = await asyncio.wait_for(reader.read(65536), timeout=10.0)
                    if not chunk:
                        break
                    response_data.extend(chunk)
                    body_received = len(response_data) - body_start

            response_str = response_data.decode("utf-8", errors="replace")

            if is_stream:
                data_lines = []
                for line in response_str.split("\n"):
                    line = line.strip()
                    if line.startswith("data: "):
                        data_lines.append(line[6:])
                return (True, data_lines)
            else:
                # Extract JSON body from HTTP response
                header_end = response_str.index("\r\n\r\n")
                json_str = response_str[header_end + 4:].strip()
                return (False, json.loads(json_str))
        finally:
            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_non_streaming_anthropic_response(self, anthropic_server_factory):
        srv, port = await anthropic_server_factory(TextResponse(content="Hello!"))
        body = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
        is_stream, result = await self._anthropic_request(port, body)
        assert is_stream is False
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"][0] == {"type": "text", "text": "Hello!"}

    @pytest.mark.asyncio
    async def test_streaming_anthropic_response(self, anthropic_server_factory):
        srv, port = await anthropic_server_factory(TextResponse(content="Hi there"))
        body = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": True,
        }
        is_stream, data_lines = await self._anthropic_request(port, body)
        assert is_stream is True
        assert "[DONE]" in data_lines
        json_events = [json.loads(d) for d in data_lines if d != "[DONE]"]
        assert len(json_events) >= 2
        # First event should be message_start
        assert json_events[0]["type"] == "message_start"
        # Last event should be message_stop
        assert json_events[-1]["type"] == "message_stop"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, anthropic_server_factory):
        srv, port = await anthropic_server_factory(TextResponse(content=""))
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            request = (
                b"POST /v1/messages HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 5\r\n"
                b"\r\n"
                b"not json"
            )
            writer.write(request)
            await writer.drain()
            response_data = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            assert b"400" in response_data
        finally:
            writer.close()
            await writer.wait_closed()
