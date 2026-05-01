"""Tests for proxy request handler."""

import pytest
from unittest.mock import AsyncMock

from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.workflow import TextResponse, ToolCall
from forge.proxy.handler import handle_chat_completions, _extract_tool_specs


# ── Helpers ──────────────────────────────────────────────────


def _mock_client(response):
    """Create a mock LLMClient that returns the given response."""
    client = AsyncMock()
    client.api_format = "ollama"
    client.send = AsyncMock(return_value=response)
    return client


def _context_manager():
    return ContextManager(strategy=NoCompact(), budget_tokens=8192)


def _body(messages=None, tools=None, stream=False, model="test"):
    """Build a minimal request body."""
    b = {"messages": messages or [{"role": "user", "content": "hi"}], "model": model}
    if tools is not None:
        b["tools"] = tools
    if stream:
        b["stream"] = True
    return b


def _tool_def(name="search", description="Search", parameters=None):
    """Build an OpenAI-format tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }


# ── _extract_tool_specs ──────────────────────────────────────


class TestExtractToolSpecs:
    def test_none_returns_empty(self):
        assert _extract_tool_specs(None) == []

    def test_empty_list_returns_empty(self):
        assert _extract_tool_specs([]) == []

    def test_extracts_function_tools(self):
        specs = _extract_tool_specs([_tool_def("search"), _tool_def("fetch")])
        assert len(specs) == 2
        assert specs[0].name == "search"
        assert specs[1].name == "fetch"

    def test_skips_non_function_types(self):
        tools = [{"type": "retrieval"}, _tool_def("search")]
        specs = _extract_tool_specs(tools)
        assert len(specs) == 1
        assert specs[0].name == "search"

    def test_extracts_parameters(self):
        params = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }
        specs = _extract_tool_specs([_tool_def("search", parameters=params)])
        assert specs[0].name == "search"


# ── No tools → passthrough ──────────────────────────────────


class TestNoToolsPassthrough:
    @pytest.mark.asyncio
    async def test_text_response_passthrough(self):
        client = _mock_client(TextResponse(content="Hello!"))
        result = await handle_chat_completions(
            _body(), client, _context_manager(),
        )
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_text_response_passthrough_stream(self):
        client = _mock_client(TextResponse(content="Hello!"))
        result = await handle_chat_completions(
            _body(stream=True), client, _context_manager(),
        )
        # SSE events list
        assert isinstance(result, list)
        assert result[-1]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_model_name_propagated(self):
        client = _mock_client(TextResponse(content="hi"))
        result = await handle_chat_completions(
            _body(model="my-model"), client, _context_manager(),
        )
        assert result["model"] == "my-model"


# ── With tools → guardrails ─────────────────────────────────


class TestWithTools:
    @pytest.mark.asyncio
    async def test_tool_call_returned(self):
        """Valid tool call is returned in OpenAI format."""
        client = _mock_client([ToolCall(tool="search", args={"q": "test"})])
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
        )
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search"
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_tool_call_stream(self):
        """Valid tool call returns SSE events."""
        client = _mock_client([ToolCall(tool="search", args={})])
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")], stream=True),
            client, _context_manager(),
        )
        assert isinstance(result, list)
        assert result[-1]["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_respond_tool_auto_injected(self):
        """Respond tool is injected — model calling respond returns text."""
        client = _mock_client([ToolCall(tool="respond", args={"message": "Hi!"})])
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
        )
        # respond is stripped — client sees text, not a tool call
        assert result["choices"][0]["message"]["content"] == "Hi!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert "tool_calls" not in result["choices"][0]["message"]

    @pytest.mark.asyncio
    async def test_respond_stripped_in_stream(self):
        """Respond call in stream mode returns text SSE events."""
        client = _mock_client([ToolCall(tool="respond", args={"message": "Hi!"})])
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")], stream=True),
            client, _context_manager(),
        )
        assert isinstance(result, list)
        assert result[-1]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_mixed_respond_and_tool_calls(self):
        """If respond is mixed with real tool calls, respond is dropped."""
        client = _mock_client([
            ToolCall(tool="search", args={"q": "test"}),
            ToolCall(tool="respond", args={"message": "also this"}),
        ])
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
        )
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_respond_not_double_injected(self):
        """If client already provides respond tool, don't inject again."""
        client = _mock_client([ToolCall(tool="respond", args={"message": "Hi!"})])
        tools = [_tool_def("search"), _tool_def("respond")]
        result = await handle_chat_completions(
            _body(tools=tools), client, _context_manager(),
        )
        # Should still work — respond stripped to text
        assert result["choices"][0]["message"]["content"] == "Hi!"


# ── Error paths ─────────────────────────────────────────────


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_retries_exhausted_returns_text(self):
        """When retries are exhausted, last text is returned to client."""
        # Model always returns text — will exhaust retries
        client = AsyncMock()
        client.api_format = "ollama"
        client.send = AsyncMock(return_value=TextResponse(content="I can't do that"))
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]),
            client, _context_manager(), max_retries=1,
        )
        # Should return the text rather than an error
        assert result["choices"][0]["message"]["content"] == "I can't do that"
        assert result["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_retries_exhausted_stream(self):
        """Retries exhausted in stream mode returns text SSE events."""
        client = AsyncMock()
        client.api_format = "ollama"
        client.send = AsyncMock(return_value=TextResponse(content="nope"))
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")], stream=True),
            client, _context_manager(), max_retries=1,
        )
        assert isinstance(result, list)
        # Should contain the text in SSE events
        content_events = [
            e for e in result
            if e["choices"][0].get("delta", {}).get("content")
        ]
        assert len(content_events) > 0


class TestSamplingPlumbing:
    """Issue A: inbound body sampling fields plumbed through to client.send."""

    @pytest.mark.asyncio
    async def test_no_tools_path_passes_sampling(self):
        """Inbound body sampling fields reach client.send on the no-tools path."""
        client = _mock_client(TextResponse(content="ok"))
        body = _body(messages=[{"role": "user", "content": "hi"}])
        body["temperature"] = 0.5
        body["top_p"] = 0.9

        await handle_chat_completions(body, client, _context_manager(), max_retries=1)

        client.send.assert_called_once()
        sampling = client.send.call_args.kwargs["sampling"]
        assert sampling == {"temperature": 0.5, "top_p": 0.9}

    @pytest.mark.asyncio
    async def test_no_tools_path_no_sampling_fields(self):
        """No sampling fields in body → sampling=None."""
        client = _mock_client(TextResponse(content="ok"))

        await handle_chat_completions(
            _body(), client, _context_manager(), max_retries=1,
        )

        sampling = client.send.call_args.kwargs["sampling"]
        assert sampling is None

    @pytest.mark.asyncio
    async def test_tools_path_passes_sampling_to_run_inference(self, monkeypatch):
        """With tools, sampling reaches run_inference (and through it the client)."""
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        captured: dict = {}

        async def fake_run_inference(**kwargs):
            captured["sampling"] = kwargs.get("sampling")
            from forge.core.inference import InferenceResult
            return InferenceResult(
                response=[ToolCall(tool="search", args={"q": "x"})],
                new_messages=[],
                tool_call_counter=0,
                attempts=1,
            )

        monkeypatch.setattr(
            "forge.proxy.handler.run_inference", fake_run_inference,
        )

        body = _body(tools=[_tool_def("search")])
        body["seed"] = 42
        body["temperature"] = 0.3

        await handle_chat_completions(body, client, _context_manager(), max_retries=1)

        assert captured["sampling"] == {"temperature": 0.3, "seed": 42}

    @pytest.mark.asyncio
    async def test_per_call_sampling_does_not_mutate_client(self):
        """Per-call sampling overrides do not leak into subsequent calls."""
        client = _mock_client(TextResponse(content="ok"))

        # First request: with temperature override.
        body1 = _body()
        body1["temperature"] = 0.99
        await handle_chat_completions(body1, client, _context_manager(), max_retries=1)
        first_sampling = client.send.call_args.kwargs["sampling"]
        assert first_sampling == {"temperature": 0.99}

        # Second request: no sampling fields.
        await handle_chat_completions(_body(), client, _context_manager(), max_retries=1)
        second_sampling = client.send.call_args.kwargs["sampling"]
        assert second_sampling is None
