"""Full handler path tests for encoder/decoder combinations.

Tests the complete handler pipeline (decoder → guardrails → encoder)
for each encoder/decoder combo by mocking backend responses.

| Decoder | Encoder | Endpoint | anthropic_backend |
|---------|---------|----------|-------------------|
| OpenAI  | OpenAI  | /v1/chat/completions | False |
| Anthropic | Anthropic | /v1/messages | True |
| Anthropic | OpenAI | /v1/messages | False |
| OpenAI  | Anthropic | /v1/chat/completions | True |

Existing tests in test_anthropic_handler.py cover the anthropic_backend=True
path. This file focuses on the Anthropic→OpenAI combo (anthropic_backend=False
with handle_messages) and cross-verification that all combos produce the
correct output format.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.workflow import TextResponse, ToolCall
from forge.proxy.handler import handle_chat_completions, handle_messages

# ── Helpers ──────────────────────────────────────────────────


def _mock_client(response):
    client = AsyncMock()
    client.api_format = "ollama"
    client.send = AsyncMock(return_value=response)
    return client


def _ctx():
    return ContextManager(strategy=NoCompact(), budget_tokens=8192)


def _anthropic_body(messages=None, tools=None, stream=False, model="claude-sonnet-4"):
    """Build a minimal Anthropic-format request body."""
    b = {"messages": messages or [{"role": "user", "content": "hi"}], "model": model}
    if tools is not None:
        b["tools"] = tools
    if stream:
        b["stream"] = True
    return b


def _openai_body(messages=None, tools=None, stream=False, model="forge"):
    """Build a minimal OpenAI-format request body."""
    b = {"messages": messages or [{"role": "user", "content": "hi"}], "model": model}
    if tools is not None:
        b["tools"] = tools
    if stream:
        b["stream"] = True
    return b


# ── Anthropic → OpenAI (handle_messages with anthropic_backend=False) ──


class TestAnthropicToOpenAI:
    """handle_messages with anthropic_backend=False: Anthropic input → OpenAI output."""

    @pytest.mark.asyncio
    async def test_text_response(self):
        """Anthropic request → OpenAI text response."""
        client = _mock_client(TextResponse(content="Hello!"))
        body = _anthropic_body()
        result = await handle_messages(
            body=body,
            client=client,
            context_manager=_ctx(),
            anthropic_backend=False,
        )
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        """Anthropic request → OpenAI tool call response."""
        client = _mock_client([ToolCall(tool="search", args={"q": "test"})])
        body = _anthropic_body(
            tools=[
                {
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            ],
        )
        result = await handle_messages(
            body=body,
            client=client,
            context_manager=_ctx(),
            anthropic_backend=False,
        )
        assert result["object"] == "chat.completion"
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search"
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_streaming_text(self):
        """Anthropic streaming request → OpenAI SSE text events."""
        client = _mock_client(TextResponse(content="Hi"))
        body = _anthropic_body(stream=True)
        result = await handle_messages(
            body=body,
            client=client,
            context_manager=_ctx(),
            anthropic_backend=False,
        )
        assert isinstance(result, list)
        assert result[-1]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_streaming_tool_call(self):
        """Anthropic streaming request → OpenAI SSE tool call events."""
        client = _mock_client([ToolCall(tool="fetch", args={"url": "http://x"})])
        body = _anthropic_body(
            tools=[
                {
                    "name": "fetch",
                    "description": "Fetch",
                    "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}},
                }
            ],
            stream=True,
        )
        result = await handle_messages(
            body=body,
            client=client,
            context_manager=_ctx(),
            anthropic_backend=False,
        )
        assert isinstance(result, list)
        assert result[-1]["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_system_message_passed_through(self):
        """System message in Anthropic body is converted and passed to backend."""
        client = _mock_client(TextResponse(content="Got it"))
        body = _anthropic_body(messages=[{"role": "user", "content": "hi"}])
        body["system"] = "You are a helpful bot."
        result = await handle_messages(
            body=body,
            client=client,
            context_manager=_ctx(),
            anthropic_backend=False,
        )
        assert result["choices"][0]["message"]["content"] == "Got it"


# ── OpenAI → OpenAI (handle_chat_completions with anthropic_backend=False) ──


class TestOpenaiToOpenAI:
    """handle_chat_completions with anthropic_backend=False: OpenAI in/out (regression)."""

    @pytest.mark.asyncio
    async def test_text_response(self):
        client = _mock_client(TextResponse(content="Hello!"))
        result = await handle_chat_completions(
            _openai_body(),
            client=client,
            context_manager=_ctx(),
            anthropic_backend=False,
        )
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        result = await handle_chat_completions(
            _openai_body(
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "search", "description": "S", "parameters": {}},
                    }
                ]
            ),
            client=client,
            context_manager=_ctx(),
            anthropic_backend=False,
        )
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_streaming_text(self):
        client = _mock_client(TextResponse(content="Hi"))
        result = await handle_chat_completions(
            _openai_body(stream=True),
            client=client,
            context_manager=_ctx(),
            anthropic_backend=False,
        )
        assert isinstance(result, list)
        assert result[-1]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_streaming_tool_call(self):
        client = _mock_client([ToolCall(tool="f", args={})])
        result = await handle_chat_completions(
            _openai_body(
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "f", "description": "S", "parameters": {}},
                    }
                ],
                stream=True,
            ),
            client=client,
            context_manager=_ctx(),
            anthropic_backend=False,
        )
        assert isinstance(result, list)
        assert result[-1]["choices"][0]["finish_reason"] == "tool_calls"


# ── OpenAI → Anthropic (handle_chat_completions with anthropic_backend=True) ──


class TestOpenaiToAnthropic:
    """handle_chat_completions with anthropic_backend=True: OpenAI in → Anthropic out."""

    @pytest.mark.asyncio
    async def test_text_response(self):
        client = _mock_client(TextResponse(content="Hello!"))
        result = await handle_chat_completions(
            _openai_body(),
            client=client,
            context_manager=_ctx(),
            anthropic_backend=True,
        )
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert result["content"][0] == {"type": "text", "text": "Hello!"}

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        result = await handle_chat_completions(
            _openai_body(
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "search", "description": "S", "parameters": {}},
                    }
                ]
            ),
            client=client,
            context_manager=_ctx(),
            anthropic_backend=True,
        )
        assert result["type"] == "message"
        assert result["stop_reason"] == "tool_use"
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_streaming_text(self):
        client = _mock_client(TextResponse(content="Hi"))
        result = await handle_chat_completions(
            _openai_body(stream=True),
            client=client,
            context_manager=_ctx(),
            anthropic_backend=True,
        )
        assert isinstance(result, list)
        assert result[0]["type"] == "message_start"
        assert result[-1]["type"] == "message_stop"
        text_deltas = [e for e in result if e.get("delta", {}).get("type") == "text_delta"]
        assert len(text_deltas) >= 1

    @pytest.mark.asyncio
    async def test_streaming_tool_call(self):
        client = _mock_client([ToolCall(tool="f", args={"x": 1})])
        result = await handle_chat_completions(
            _openai_body(
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "f", "description": "S", "parameters": {}},
                    }
                ],
                stream=True,
            ),
            client=client,
            context_manager=_ctx(),
            anthropic_backend=True,
        )
        assert isinstance(result, list)
        tool_starts = [e for e in result if e.get("content_block", {}).get("type") == "tool_use"]
        assert len(tool_starts) >= 1
        input_deltas = [e for e in result if e.get("delta", {}).get("type") == "input_json_delta"]
        assert len(input_deltas) >= 1
        assert result[-1]["type"] == "message_stop"


# ── Anthropic → Anthropic (handle_messages with anthropic_backend=True) ──


class TestAnthropicToAnthropic:
    """handle_messages with anthropic_backend=True: Anthropic in/out (regression)."""

    @pytest.mark.asyncio
    async def test_text_response(self):
        client = _mock_client(TextResponse(content="Hello!"))
        body = _anthropic_body()
        result = await handle_messages(
            body=body,
            client=client,
            context_manager=_ctx(),
            anthropic_backend=True,
        )
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"][0] == {"type": "text", "text": "Hello!"}

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        body = _anthropic_body(
            tools=[
                {
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            ]
        )
        result = await handle_messages(
            body=body,
            client=client,
            context_manager=_ctx(),
            anthropic_backend=True,
        )
        assert result["stop_reason"] == "tool_use"
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_streaming(self):
        client = _mock_client(TextResponse(content="Hi"))
        body = _anthropic_body(stream=True)
        result = await handle_messages(
            body=body,
            client=client,
            context_manager=_ctx(),
            anthropic_backend=True,
        )
        assert isinstance(result, list)
        assert result[0]["type"] == "message_start"
        assert result[-1]["type"] == "message_stop"


# ── Cross-combo verification: same backend response, different output formats ──


class TestFormatConsistency:
    """Same backend response should produce correctly formatted output for each combo."""

    @pytest.mark.asyncio
    async def test_text_response_format_differs_by_backend(self):
        """Text response should be OpenAI format when anthropic_backend=False,
        Anthropic format when anthropic_backend=True."""
        client = _mock_client(TextResponse(content="Hello!"))

        openai_result = await handle_chat_completions(
            _openai_body(),
            client=client,
            context_manager=_ctx(),
            anthropic_backend=False,
        )
        assert "object" in openai_result
        assert "choices" in openai_result
        assert "type" not in openai_result

        client.reset_mock()
        anthropic_result = await handle_chat_completions(
            _openai_body(),
            client=client,
            context_manager=_ctx(),
            anthropic_backend=True,
        )
        assert "type" in anthropic_result
        assert anthropic_result["type"] == "message"
        assert "choices" not in anthropic_result

    @pytest.mark.asyncio
    async def test_tool_call_format_differs_by_backend(self):
        """Tool call response should differ by backend format."""
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])

        openai_result = await handle_chat_completions(
            _openai_body(
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "search", "description": "S", "parameters": {}},
                    }
                ]
            ),
            client=client,
            context_manager=_ctx(),
            anthropic_backend=False,
        )
        assert openai_result["choices"][0]["finish_reason"] == "tool_calls"
        assert "tool_calls" in openai_result["choices"][0]["message"]

        client.reset_mock()
        anthropic_result = await handle_chat_completions(
            _openai_body(
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "search", "description": "S", "parameters": {}},
                    }
                ]
            ),
            client=client,
            context_manager=_ctx(),
            anthropic_backend=True,
        )
        assert anthropic_result["stop_reason"] == "tool_use"
        assert anthropic_result["content"][0]["type"] == "tool_use"
