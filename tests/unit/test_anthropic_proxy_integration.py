"""Integration tests for the Anthropic proxy endpoint.

Uses the real Anthropic SDK to send requests to a running proxy instance,
verifying the full round-trip: SDK → HTTP → handler → mock backend → response.

These are the closest thing we have to external client tests — they validate
that the proxy correctly implements the Anthropic Messages API contract.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from anthropic import Anthropic

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
async def anthropic_proxy_server():
    """Factory that creates an HTTPServer with anthropic_backend=True on a random port."""
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
        await asyncio.sleep(0)  # let accept loop start
        sock = srv._server.sockets[0]
        port = sock.getsockname()[1]
        servers.append(srv)
        return srv, port

    yield _make

    for srv in servers:
        await srv.stop()


def _sdk_client(port):
    """Create an Anthropic SDK client pointing at the proxy."""
    return Anthropic(
        base_url=f"http://127.0.0.1:{port}",
        api_key="test-key",
        # Short timeout for tests
        timeout=10.0,
    )


async def _sync_sdk_call(fn, *args, **kwargs):
    """Run a sync SDK call in a thread so the event loop stays free for the server."""
    return await asyncio.to_thread(fn, *args, **kwargs)


# ── Non-streaming: text response ─────────────────────────────


class TestTextResponse:
    @pytest.mark.asyncio
    async def test_simple_text_via_sdk(self, anthropic_proxy_server):
        """SDK sends text request → proxy returns Anthropic text response."""
        _, port = await anthropic_proxy_server(TextResponse(content="Hello from proxy!"))
        client = _sdk_client(port)
        response = await _sync_sdk_call(
            client.messages.create,
            model="claude-sonnet-4",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response.type == "message"
        assert response.role == "assistant"
        assert response.content[0].type == "text"
        assert response.content[0].text == "Hello from proxy!"
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_system_message_forwarded(self, anthropic_proxy_server):
        """System prompt in Anthropic body reaches backend."""
        _, port = await anthropic_proxy_server(TextResponse(content="Got it"))
        client = _sdk_client(port)
        response = await _sync_sdk_call(
            client.messages.create,
            model="claude-sonnet-4",
            max_tokens=1024,
            system="You are a helpful bot.",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert response.type == "message"
        assert response.content[0].text == "Got it"

    @pytest.mark.asyncio
    async def test_temperature_forwarded(self, anthropic_proxy_server):
        """Temperature param is passed through the handler chain."""
        _, port = await anthropic_proxy_server(TextResponse(content="ok"))
        client = _sdk_client(port)
        response = await _sync_sdk_call(
            client.messages.create,
            model="claude-sonnet-4",
            max_tokens=1024,
            temperature=0.7,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response.content[0].text == "ok"


# ── Non-streaming: tool calls ─────────────────────────────────


class TestToolCalls:
    @pytest.mark.asyncio
    async def test_single_tool_call_via_sdk(self, anthropic_proxy_server):
        """SDK sends request with tools → proxy returns tool_use block."""
        _, port = await anthropic_proxy_server(
            [ToolCall(tool="get_weather", args={"city": "Paris"})],
        )
        client = _sdk_client(port)
        response = await _sync_sdk_call(
            client.messages.create,
            model="claude-sonnet-4",
            max_tokens=1024,
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                }
            ],
            messages=[{"role": "user", "content": "Weather in Paris?"}],
        )
        assert response.type == "message"
        assert response.stop_reason == "tool_use"
        assert len(response.content) == 1
        assert response.content[0].type == "tool_use"
        assert response.content[0].name == "get_weather"
        assert response.content[0].input == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_via_sdk(self, anthropic_proxy_server):
        """Multiple tool calls in one response."""
        _, port = await anthropic_proxy_server(
            [
                ToolCall(tool="f1", args={"a": 1}),
                ToolCall(tool="f2", args={"b": 2}),
            ]
        )
        client = _sdk_client(port)
        response = await _sync_sdk_call(
            client.messages.create,
            model="claude-sonnet-4",
            max_tokens=1024,
            tools=[
                {
                    "name": "f1",
                    "description": "First tool",
                    "input_schema": {"type": "object", "properties": {"a": {"type": "integer"}}},
                },
                {
                    "name": "f2",
                    "description": "Second tool",
                    "input_schema": {"type": "object", "properties": {"b": {"type": "integer"}}},
                },
            ],
            messages=[{"role": "user", "content": "Do both"}],
        )
        assert response.stop_reason == "tool_use"
        assert len(response.content) == 2
        assert response.content[0].name == "f1"
        assert response.content[0].input == {"a": 1}
        assert response.content[1].name == "f2"
        assert response.content[1].input == {"b": 2}


# ── Streaming: text ──────────────────────────────────────────


class TestStreamingText:
    @pytest.mark.asyncio
    async def test_text_stream_via_sdk(self, anthropic_proxy_server):
        """SDK streaming request → proxy returns SSE events."""
        _, port = await anthropic_proxy_server(TextResponse(content="Hi there"))
        client = _sdk_client(port)

        def _stream():
            events = []
            with client.messages.stream(
                model="claude-sonnet-4",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hi"}],
            ) as stream:
                for event in stream:
                    events.append(event)
            return events

        events = await _sync_sdk_call(_stream)

        # Should have message_start, content blocks, message_stop
        types = [type(e).__name__ for e in events]
        assert any("MessageStart" in t for t in types)
        assert any("MessageStop" in t for t in types)
        # Should have text content (TextEvent has .text, RawContentBlockDeltaEvent has .delta.text)
        text_events = [e for e in events if hasattr(e, "text") and e.text]
        assert len(text_events) >= 1

    @pytest.mark.asyncio
    async def test_stream_accumulates_full_text(self, anthropic_proxy_server):
        """Streaming text accumulates to the full response."""
        _, port = await anthropic_proxy_server(TextResponse(content="Hello world"))
        client = _sdk_client(port)

        def _stream():
            text_parts = []
            with client.messages.stream(
                model="claude-sonnet-4",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hi"}],
            ) as stream:
                for event in stream:
                    # TextEvent has .text directly (the SDK's parsed text delta)
                    if getattr(event, "type", None) == "text":
                        text_parts.append(event.text)
            return text_parts

        text_parts = await _sync_sdk_call(_stream)
        assert "".join(text_parts) == "Hello world"


# ── Streaming: tool calls ─────────────────────────────────────


class TestStreamingTools:
    @pytest.mark.asyncio
    async def test_tool_stream_via_sdk(self, anthropic_proxy_server):
        """SDK streaming tool call → proxy returns tool_use SSE events."""
        _, port = await anthropic_proxy_server(
            [ToolCall(tool="fetch", args={"url": "http://example.com"})],
        )
        client = _sdk_client(port)

        def _stream():
            tool_blocks = []
            with client.messages.stream(
                model="claude-sonnet-4",
                max_tokens=1024,
                tools=[
                    {
                        "name": "fetch",
                        "description": "Fetch a URL",
                        "input_schema": {
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                        },
                    }
                ],
                messages=[{"role": "user", "content": "Fetch this"}],
            ) as stream:
                for event in stream:
                    # Only collect from content_block_start (not delta)
                    if (
                        getattr(event, "type", None) == "content_block_start"
                        and hasattr(event, "content_block")
                        and event.content_block is not None
                        and event.content_block.type == "tool_use"
                    ):
                        tool_blocks.append(event.content_block)
            return tool_blocks

        tool_blocks = await _sync_sdk_call(_stream)
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "fetch"
        assert tool_blocks[0].input == {"url": "http://example.com"}


# ── Edge cases ───────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_content_response(self, anthropic_proxy_server):
        """Empty text response is handled gracefully."""
        _, port = await anthropic_proxy_server(TextResponse(content=""))
        client = _sdk_client(port)
        response = await _sync_sdk_call(
            client.messages.create,
            model="claude-sonnet-4",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response.type == "message"
        assert response.content[0].text == ""

    @pytest.mark.asyncio
    async def test_multiple_messages_conversation(self, anthropic_proxy_server):
        """Multi-turn message array is handled."""
        _, port = await anthropic_proxy_server(TextResponse(content="Response"))
        client = _sdk_client(port)
        response = await _sync_sdk_call(
            client.messages.create,
            model="claude-sonnet-4",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Reply"},
                {"role": "user", "content": "Second"},
            ],
        )
        assert response.content[0].text == "Response"

    @pytest.mark.asyncio
    async def test_serialized_mode(self, anthropic_proxy_server):
        """Serialize mode processes requests through the queue."""
        _, port = await anthropic_proxy_server(
            TextResponse(content="serialized"),
            serialize=True,
        )
        client = _sdk_client(port)
        response = await _sync_sdk_call(
            client.messages.create,
            model="claude-sonnet-4",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response.content[0].text == "serialized"
