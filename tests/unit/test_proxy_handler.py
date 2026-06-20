"""Tests for proxy request handler."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from forge.clients.ollama import OllamaClient
from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.workflow import TextResponse, ToolCall
from forge.clients.base import TokenUsage
from forge.proxy.handler import handle_chat_completions, _extract_tool_specs


# ── Helpers ──────────────────────────────────────────────────


def _mock_client(response):
    """Create a mock LLMClient that returns the given response."""
    client = AsyncMock()
    client.api_format = "ollama"
    client.send = AsyncMock(return_value=response)
    client.last_usage = {}
    client._slot_id = 0
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
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
        
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
        )
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search"
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_tool_call_stream(self):
        """Valid tool call returns SSE events."""
        client = _mock_client([ToolCall(tool="search", args={})])
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
        
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")], stream=True),
            client, _context_manager(),
        )
        assert isinstance(result, list)
        assert result[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert result[-1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_respond_tool_auto_injected(self):
        """With inject_respond_tool=True, a respond() call is stripped to text."""
        client = _mock_client([ToolCall(tool="respond", args={"message": "Hi!"})])
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}

        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
            inject_respond_tool=True,
        )
        # respond is stripped — client sees text, not a tool call
        assert result["choices"][0]["message"]["content"] == "Hi!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert "tool_calls" not in result["choices"][0]["message"]
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_respond_stripped_in_stream(self):
        """Respond call in stream mode returns text SSE events."""
        client = _mock_client([ToolCall(tool="respond", args={"message": "Hi!"})])
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
        
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")], stream=True),
            client, _context_manager(),
        )
        assert isinstance(result, list)
        assert result[-1]["choices"][0]["finish_reason"] == "stop"
        assert result[-1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_mixed_respond_and_tool_calls(self):
        """If respond is mixed with real tool calls, respond is dropped."""
        client = _mock_client([
            ToolCall(tool="search", args={"q": "test"}),
            ToolCall(tool="respond", args={"message": "also this"}),
        ])
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}

        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
            inject_respond_tool=True,
        )
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "search"
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_respond_not_double_injected(self):
        """If client already provides respond tool, don't inject again."""
        client = _mock_client([ToolCall(tool="respond", args={"message": "Hi!"})])
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
        
        tools = [_tool_def("search"), _tool_def("respond")]
        result = await handle_chat_completions(
            _body(tools=tools), client, _context_manager(),
            inject_respond_tool=True,
        )
        # Should still work — respond stripped to text (not double-injected)
        assert result["choices"][0]["message"]["content"] == "Hi!"
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


# ── Error paths ─────────────────────────────────────────────


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_retries_exhausted_returns_text(self):
        """When retries are exhausted, last text is returned to client."""
        # Model always returns text — will exhaust retries
        client = _mock_client(TextResponse(content="I can't do that"))
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]),
            client, _context_manager(), max_retries=1,
        )
        # Should return the text rather than an error
        assert result["choices"][0]["message"]["content"] == "I can't do that"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_retries_exhausted_stream(self):
        """Retries exhausted in stream mode returns text SSE events."""
        client = _mock_client(TextResponse(content="nope"))
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
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
        assert result[-1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_malformed_args_bounded_by_max_tool_errors(self):
        """Malformed tool args drain the tool-error budget, not retries — so
        max_tool_errors (not max_retries) bounds the loop. Proves the proxy's
        new max_tool_errors knob threads through to the ErrorTracker."""
        # Known tool, non-dict args → tool_arg_validation every turn.
        client = _mock_client([ToolCall(tool="search", args="bad")])  # type: ignore[arg-type]
        client.last_usage = {0: TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)}
        result = await handle_chat_completions(
            _body(tools=[_tool_def("search")]),
            client, _context_manager(), max_retries=5, max_tool_errors=1,
        )
        # Exhausted on the tool-error budget (1), not max_retries (5):
        # send #1 (error, budget→1) + send #2 (error, 2 > 1 → exhausted).
        assert client.send.call_count == 2
        assert result["choices"][0]["finish_reason"] == "stop"


class TestSamplingPlumbing:
    """Issue A: inbound body sampling fields plumbed through to client.send."""

    @pytest.mark.asyncio
    async def test_no_tools_path_passes_sampling(self):
        """Inbound body sampling fields reach client.send on the no-tools path."""
        client = _mock_client(TextResponse(content="ok"))
        client.last_usage = {0: TokenUsage(1, 1, 2)}
        body = _body(messages=[{"role": "user", "content": "hi"}])
        body["temperature"] = 0.5
        body["top_p"] = 0.9

        result = await handle_chat_completions(body, client, _context_manager(), max_retries=1)

        client.send.assert_called_once()
        sampling = client.send.call_args.kwargs["sampling"]
        assert sampling == {"temperature": 0.5, "top_p": 0.9}
        assert result["usage"] == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

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
                usage=TokenUsage(10, 5, 15),
                tool_call_counter=0,
                attempts=1,
            )

        monkeypatch.setattr(
            "forge.proxy.handler.run_inference", fake_run_inference,
        )

        body = _body(tools=[_tool_def("search")])
        body["seed"] = 42
        body["temperature"] = 0.3

        result = await handle_chat_completions(body, client, _context_manager(), max_retries=1)

        assert captured["sampling"] == {"temperature": 0.3, "seed": 42}
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

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

    @pytest.mark.asyncio
    async def test_passthrough_carries_unknown_body_fields(self):
        """Inbound body fields outside sampling/forge-owned flow through passthrough."""
        client = _mock_client(TextResponse(content="ok"))
        body = _body(messages=[{"role": "user", "content": "hi"}])
        body["max_tokens"] = 256
        body["tool_choice"] = "auto"

        await handle_chat_completions(body, client, _context_manager(), max_retries=1)

        passthrough = client.send.call_args.kwargs["passthrough"]
        assert passthrough == {
            "model": "test",
            "max_tokens": 256,
            "tool_choice": "auto",
        }


    @pytest.mark.asyncio
    async def test_stream_options_excluded_from_passthrough(self):
        """stream_options must not leak into passthrough.

        Forge controls streaming independently — when it makes non-streaming
        calls to the backend, a leaked stream_options causes validation
        errors on strict backends (e.g. vLLM rejects stream_options when
        stream is not True).
        """
        client = _mock_client(TextResponse(content="ok"))
        body = _body(messages=[{"role": "user", "content": "hi"}])
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        body["max_tokens"] = 256

        await handle_chat_completions(body, client, _context_manager(), max_retries=1)

        passthrough = client.send.call_args.kwargs["passthrough"]
        assert "stream_options" not in passthrough
        assert passthrough == {"model": "test", "max_tokens": 256}

# ── Anthropic protocol routing ───────────────────────────────


class TestAnthropicProtocol:
    """End-to-end handler tests for the /v1/messages (protocol="anthropic") path."""

    def _anthropic_body(self, messages=None, tools=None, system=None, **extra):
        body = {
            "model": "claude-3-5-sonnet",
            "messages": messages or [{"role": "user", "content": "hi"}],
            "max_tokens": 256,
        }
        if tools is not None:
            body["tools"] = tools
        if system is not None:
            body["system"] = system
        body.update(extra)
        return body

    @pytest.mark.asyncio
    async def test_no_tools_returns_anthropic_shape(self):
        client = _mock_client(TextResponse(content="hello"))
        body = self._anthropic_body()
        result = await handle_chat_completions(
            body, client, _context_manager(), max_retries=1, protocol="anthropic",
        )
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert result["content"] == [{"type": "text", "text": "hello"}]

    @pytest.mark.asyncio
    async def test_tool_call_returns_anthropic_shape(self, monkeypatch):
        from forge.core.inference import InferenceResult

        async def fake_run_inference(**kwargs):
            return InferenceResult(
                response=[ToolCall(tool="get_weather", args={"city": "Paris"})],
                new_messages=[],
                tool_call_counter=0,
                attempts=1,
            )
        monkeypatch.setattr("forge.proxy.handler.run_inference", fake_run_inference)

        client = _mock_client([ToolCall(tool="get_weather", args={"city": "Paris"})])
        body = self._anthropic_body(
            tools=[{
                "name": "get_weather",
                "description": "Weather.",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
        )
        result = await handle_chat_completions(
            body, client, _context_manager(), max_retries=1, protocol="anthropic",
        )
        assert result["type"] == "message"
        assert result["stop_reason"] == "tool_use"
        tu_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tu_blocks) == 1
        assert tu_blocks[0]["name"] == "get_weather"
        assert tu_blocks[0]["input"] == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_streaming_returns_anthropic_event_sequence(self):
        client = _mock_client(TextResponse(content="streamed"))
        body = self._anthropic_body(stream=True)
        events = await handle_chat_completions(
            body, client, _context_manager(), max_retries=1, protocol="anthropic",
        )
        assert isinstance(events, list)
        types = [e["type"] for e in events]
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"

    @pytest.mark.asyncio
    async def test_anthropic_passthrough_translates_to_openai_shape(self):
        """tool_choice and stop_sequences land in passthrough in OpenAI shape."""
        client = _mock_client(TextResponse(content="ok"))
        body = self._anthropic_body(
            stop_sequences=["</done>"],
            tool_choice={"type": "any"},
        )
        await handle_chat_completions(
            body, client, _context_manager(), max_retries=1, protocol="anthropic",
        )
        passthrough = client.send.call_args.kwargs["passthrough"]
        assert passthrough["stop"] == ["</done>"]
        assert passthrough["tool_choice"] == "required"
        assert passthrough["model"] == "claude-3-5-sonnet"
        assert passthrough["max_tokens"] == 256
        # Anthropic-only fields with no OpenAI analog don't appear.
        assert "thinking" not in passthrough
        assert "metadata" not in passthrough

    @pytest.mark.asyncio
    async def test_system_top_level_flows_into_messages(self):
        """Anthropic puts system at top level; forge prepends it as a SYSTEM message."""
        client = _mock_client(TextResponse(content="ok"))
        body = self._anthropic_body(system="You are helpful.")
        await handle_chat_completions(
            body, client, _context_manager(), max_retries=1, protocol="anthropic",
        )
        api_messages = client.send.call_args.args[0]
        assert api_messages[0]["role"] == "system"
        assert api_messages[0]["content"] == "You are helpful."


# ── Native transparent passthrough ──────────────────────────


class TestNativePassthrough:
    """Native proxy passthrough keeps raw tools by default; raw messages
    are forwarded only when full reasoning replay preserves old behavior."""

    @pytest.mark.asyncio
    async def test_default_reasoning_replay_filters_raw_reasoning_only(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        messages = [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "old",
                "tool_calls": [],
                "name": "a1",
                "vendor": {"kept": True},
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "latest",
                "tool_calls": [],
                "name": "a2",
            },
        ]
        await handle_chat_completions(
            _body(messages=messages, tools=[_tool_def("search")]),
            client, _context_manager(),
        )
        # Default policy is "none": every reasoning field is stripped, but the
        # rest of each raw message survives verbatim.
        sent_messages = client.send.call_args.args[0]
        assert sent_messages[0]["name"] == "a1"
        assert sent_messages[0]["vendor"] == {"kept": True}
        assert "reasoning_content" not in sent_messages[0]
        assert sent_messages[1]["name"] == "a2"
        assert "reasoning_content" not in sent_messages[1]

    @pytest.mark.asyncio
    async def test_keep_last_reasoning_replay_keeps_latest_only(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        messages = [
            {"role": "assistant", "content": None, "reasoning_content": "old", "tool_calls": [], "name": "a1"},
            {"role": "assistant", "content": None, "reasoning_content": "latest", "tool_calls": [], "name": "a2"},
        ]
        await handle_chat_completions(
            _body(messages=messages, tools=[_tool_def("search")]),
            client, _context_manager(), reasoning_replay="keep-last",
        )
        sent_messages = client.send.call_args.args[0]
        assert "reasoning_content" not in sent_messages[0]
        assert sent_messages[1]["reasoning_content"] == "latest"

    @pytest.mark.asyncio
    async def test_raw_tools_forwarded_verbatim(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        params = {
            "type": "object",
            "properties": {"q": {"type": "string", "description": "the query"}},
            "required": ["q"],
            "additionalProperties": False,
        }
        tools = [_tool_def("search", parameters=params)]
        await handle_chat_completions(
            _body(tools=tools), client, _context_manager(),
        )
        # The backend sees the client's exact tools array (full schema, no
        # name/schema drift), not forge's reconstructed format_tool output.
        sent = client.send.call_args.kwargs["raw_openai_tools"]
        assert sent == tools
        # Respond is NOT appended by default.
        assert [t["function"]["name"] for t in sent] == ["search"]
        # tool_specs (validation sidecar) still passed separately.
        assert client.send.call_args.kwargs["tools"][0].name == "search"

    @pytest.mark.asyncio
    async def test_raw_messages_forwarded_verbatim(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        # An extra non-standard key proves no normalization/folding happened.
        messages = [{"role": "user", "content": "hi", "name": "u1"}]
        await handle_chat_completions(
            _body(messages=messages, tools=[_tool_def("search")]),
            client, _context_manager(), reasoning_replay="full",
        )
        sent_messages = client.send.call_args.args[0]
        assert sent_messages == messages

    @pytest.mark.asyncio
    async def test_inbound_body_mutation_does_not_affect_sent(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        tools = [_tool_def("search")]
        body = _body(tools=tools)
        await handle_chat_completions(body, client, _context_manager())
        # Mutate the caller's body after the call — detached copy is unaffected.
        body["tools"][0]["function"]["name"] = "MUTATED"
        body["messages"][0]["content"] = "MUTATED"
        sent_tools = client.send.call_args.kwargs["raw_openai_tools"]
        sent_messages = client.send.call_args.args[0]
        assert sent_tools[0]["function"]["name"] == "search"
        assert sent_messages[0]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_respond_not_injected_by_default(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
        )
        sent = client.send.call_args.kwargs["raw_openai_tools"]
        names = [t["function"]["name"] for t in sent]
        assert "respond" not in names
        spec_names = [s.name for s in client.send.call_args.kwargs["tools"]]
        assert "respond" not in spec_names

    @pytest.mark.asyncio
    async def test_respond_injected_into_raw_tools_when_opted_in(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
            inject_respond_tool=True,
        )
        sent = client.send.call_args.kwargs["raw_openai_tools"]
        names = [t["function"]["name"] for t in sent]
        assert names == ["search", "respond"]


class TestOllamaProxyIntegration:
    """Proxy → REAL OllamaClient (mocked transport): the seam that let #111/#115
    ship. Every other handler test mocks the client, so it asserts what reaches
    the client, not what the client actually POSTs to /api/chat. These assert the
    on-the-wire body on the raw-passthrough first attempt."""

    def _real_ollama(self, response_data):
        client = OllamaClient(base_url="http://test:11434", model="m", think=False)
        mock_http = AsyncMock()
        mock_http.stream = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = response_data
        resp.text = json.dumps(response_data)
        mock_http.post.return_value = resp
        client._http = mock_http
        return client

    @pytest.mark.asyncio
    async def test_no_tools_array_content_normalized(self):
        """#115 on the no-tools chat path (also raw-passthrough)."""
        client = self._real_ollama({"message": {"role": "assistant", "content": "ok"}})
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        await handle_chat_completions(
            _body(messages=messages), client, _context_manager(),
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["messages"][0]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_tool_history_string_args_normalized(self):
        """#111: a replayed assistant tool_calls turn with JSON-string args is
        coerced to a dict on the wire (the multi-turn 400)."""
        client = self._real_ollama({
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "search", "arguments": {"q": "ok"}}}
                ],
            }
        })
        messages = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "search", "arguments": '{"q": "1"}'}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        await handle_chat_completions(
            _body(messages=messages, tools=[_tool_def("search")]),
            client, _context_manager(),
        )
        # First attempt is the raw-passthrough path where the bug lived.
        body = client._http.post.call_args_list[0].kwargs["json"]
        tc_msg = [m for m in body["messages"] if m.get("tool_calls")][0]
        assert tc_msg["tool_calls"][0]["function"]["arguments"] == {"q": "1"}


# ── Prompt capability handoff ───────────────────────────────


class TestPromptCapabilityHandoff:
    """In prompt capability (native_passthrough=False) the handler suppresses
    the verbatim passthrough so the request folds normally and the client's
    prompt path injects the tools. (The injection itself is covered by the
    LlamafileClient prompt-mode tests.)"""

    @pytest.mark.asyncio
    async def test_prompt_mode_suppresses_raw_tools(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        await handle_chat_completions(
            _body(tools=[_tool_def("search")]), client, _context_manager(),
            native_passthrough=False,
        )
        # No verbatim tools forwarded — the client's prompt path injects them.
        assert "raw_openai_tools" not in client.send.call_args.kwargs
        # tool_specs (the source for build_tool_prompt) are still passed.
        assert client.send.call_args.kwargs["tools"][0].name == "search"

    @pytest.mark.asyncio
    async def test_prompt_mode_folds_messages_not_verbatim(self):
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        # A non-standard key would survive verbatim passthrough but is dropped
        # by fold_and_serialize — proving the raw transcript was NOT forwarded.
        messages = [{"role": "user", "content": "hi", "name": "u1"}]
        await handle_chat_completions(
            _body(messages=messages, tools=[_tool_def("search")]),
            client, _context_manager(), native_passthrough=False,
        )
        sent_messages = client.send.call_args.args[0]
        assert sent_messages != messages
        assert "name" not in sent_messages[0]

    @pytest.mark.asyncio
    async def test_native_default_still_forwards_raw(self):
        # Sanity: default (native) path is unaffected by the new param.
        client = _mock_client([ToolCall(tool="search", args={"q": "x"})])
        tools = [_tool_def("search")]
        await handle_chat_completions(
            _body(tools=tools), client, _context_manager(),
        )
        assert client.send.call_args.kwargs["raw_openai_tools"] == tools
