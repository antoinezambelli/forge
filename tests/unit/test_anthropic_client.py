"""Anthropic conversion and HTTP client behavior, without external API calls."""

import asyncio
from collections.abc import AsyncIterator
import json
from typing import Literal

import pytest
import httpx
from pydantic import BaseModel, Field

from forge.clients.anthropic import AnthropicClient
from forge.clients.base import ChunkType, TokenUsage
from forge.core.workflow import TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError, MissingModelError


class CityParams(BaseModel):
    city: str = Field(description="City name")


def _make_spec(name: str = "get_weather") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Get {name}",
        parameters=CityParams,
    )


class _EventStream(httpx.AsyncByteStream):
    """Emit actual SSE bytes, optionally failing after the supplied events."""

    def __init__(
        self,
        events: list[dict],
        *,
        error: Exception | None = None,
    ) -> None:
        self._events = events
        self._error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self._events:
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
        if self._error is not None:
            raise self._error


def _stream_event(event_type: str, **fields: object) -> dict:
    return {"type": event_type, **fields}


def _text_delta(text: str, index: int = 0) -> dict:
    return _stream_event(
        "content_block_delta",
        index=index,
        delta={"type": "text_delta", "text": text},
    )


def _thinking_delta(thinking: str) -> dict:
    return _stream_event(
        "content_block_delta",
        index=0,
        delta={"type": "thinking_delta", "thinking": thinking},
    )


def _tool_start(name: str, index: int = 1) -> dict:
    return _stream_event(
        "content_block_start",
        index=index,
        content_block={
            "type": "tool_use", "id": f"toolu_{index}", "name": name, "input": {},
        },
    )


def _json_delta(partial_json: str, index: int = 1) -> dict:
    return _stream_event(
        "content_block_delta",
        index=index,
        delta={"type": "input_json_delta", "partial_json": partial_json},
    )


def _mock_http(client: AnthropicClient, handler) -> None:
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers=client._http.headers,
        base_url=client._http.base_url,
    )


def _mock_stream(
    client: AnthropicClient,
    events: list[dict],
    *,
    usage: dict | None = None,
    error: Exception | None = None,
) -> None:
    usage = usage or {"input_tokens": 1, "output_tokens": 1}
    if events and events[-1]["type"] == "message_stop":
        events = [
            *events[:-1],
            _stream_event("message_delta", usage={"output_tokens": usage["output_tokens"]}),
            events[-1],
        ]
    events = [
        _stream_event(
            "message_start",
            message={"content": [], "usage": {**usage, "output_tokens": 0}},
        ),
        *events,
    ]
    _mock_http(client, lambda _: httpx.Response(
        200, headers={"content-type": "text/event-stream"},
        stream=_EventStream(events, error=error),
    ))


def _text_response(text: str = "ok", **usage: int) -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 1, "output_tokens": 1, **usage},
    }


class SetUnitParams(BaseModel):
    unit: Literal["celsius", "fahrenheit"] = Field(description="Unit")


def _make_spec_with_enum() -> ToolSpec:
    return ToolSpec(
        name="set_unit",
        description="Set temperature unit",
        parameters=SetUnitParams,
    )


@pytest.mark.asyncio
async def test_direct_base_url_preserves_deployment_prefix() -> None:
    service_root = "https://anthropic.example/deployment/"
    client = AnthropicClient(
        model="claude-test", api_key="dummy", base_url=service_root,
    )
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=_text_response())

    _mock_http(client, respond)
    await client.send([{"role": "user", "content": "hi"}])
    assert str(requests[0].url) == service_root + "v1/messages"


# ── _convert_tools ───────────────────────────────────────────────


class TestConvertTools:
    def test_basic_tool(self) -> None:
        result = AnthropicClient._convert_tools([_make_spec()])
        assert len(result) == 1
        tool = result[0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get get_weather"
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "city" in schema["properties"]
        assert schema["required"] == ["city"]

    def test_enum_param(self) -> None:
        result = AnthropicClient._convert_tools([_make_spec_with_enum()])
        prop = result[0]["input_schema"]["properties"]["unit"]
        assert prop["enum"] == ["celsius", "fahrenheit"]

    def test_optional_param(self) -> None:
        class SearchWithLimitParams(BaseModel):
            query: str = Field(description="Query")
            limit: int | None = Field(default=None, description="Max results")

        spec = ToolSpec(
            name="search",
            description="Search",
            parameters=SearchWithLimitParams,
        )
        result = AnthropicClient._convert_tools([spec])
        assert "query" in result[0]["input_schema"]["required"]
        assert "limit" not in result[0]["input_schema"]["required"]

    def test_multiple_tools(self) -> None:
        specs = [_make_spec("tool_a"), _make_spec("tool_b")]
        result = AnthropicClient._convert_tools(specs)
        assert [t["name"] for t in result] == ["tool_a", "tool_b"]


# ── _convert_messages ────────────────────────────────────────────


class TestConvertMessages:
    def test_extracts_system(self) -> None:
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        system, converted = AnthropicClient._convert_messages(msgs)
        assert system == "You are helpful."
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_simple_user_assistant(self) -> None:
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        system, converted = AnthropicClient._convert_messages(msgs)
        assert system is None
        assert len(converted) == 2
        assert converted[0] == {"role": "user", "content": "Hi"}
        assert converted[1] == {"role": "assistant", "content": "Hello!"}

    def test_tool_call_conversion(self) -> None:
        """assistant tool_calls → tool_use content blocks."""
        msgs = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_001",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
        ]
        _, converted = AnthropicClient._convert_messages(msgs)
        assert len(converted) == 2
        assistant_msg = converted[1]
        assert assistant_msg["role"] == "assistant"
        blocks = assistant_msg["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["id"] == "call_001"
        assert blocks[0]["name"] == "get_weather"
        assert blocks[0]["input"] == {"city": "Paris"}

    def test_tool_call_with_reasoning(self) -> None:
        """Reasoning in content field becomes a text block before tool_use."""
        msgs = [
            {"role": "user", "content": "Task"},
            {
                "role": "assistant",
                "content": "Let me look that up.",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_001",
                        "function": {
                            "name": "search",
                            "arguments": '{"q": "test"}',
                        },
                    }
                ],
            },
        ]
        _, converted = AnthropicClient._convert_messages(msgs)
        blocks = converted[1]["content"]
        assert len(blocks) == 2
        assert blocks[0] == {"type": "text", "text": "Let me look that up."}
        assert blocks[1]["type"] == "tool_use"

    def test_tool_result_becomes_user(self) -> None:
        """role=tool → user message with tool_result content block."""
        msgs = [
            {"role": "user", "content": "Go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_001",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "result_data",
                "tool_call_id": "call_001",
                "name": "f",
            },
        ]
        _, converted = AnthropicClient._convert_messages(msgs)
        assert len(converted) == 3
        tool_result_msg = converted[2]
        assert tool_result_msg["role"] == "user"
        block = tool_result_msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call_001"
        assert block["content"] == "result_data"

    def test_unpaired_tool_use_gets_error_result(self) -> None:
        """Step nudge: tool_call without tool result → synthetic error."""
        msgs = [
            {"role": "user", "content": "Task"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_001",
                        "function": {"name": "submit", "arguments": "{}"},
                    }
                ],
            },
            {"role": "user", "content": "Complete required steps first."},
        ]
        _, converted = AnthropicClient._convert_messages(msgs)
        assert len(converted) == 3
        # The user message should have both a tool_result and the nudge text
        user_msg = converted[2]
        assert user_msg["role"] == "user"
        blocks = user_msg["content"]
        assert blocks[0]["type"] == "tool_result"
        assert blocks[0]["tool_use_id"] == "call_001"
        assert blocks[0]["is_error"] is True
        assert blocks[1]["type"] == "text"
        assert "required steps" in blocks[1]["text"]

    def test_consecutive_same_role_merged(self) -> None:
        """Consecutive user messages (tool_result + nudge) get merged."""
        msgs = [
            {"role": "user", "content": "Task"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_001",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "ok",
                "tool_call_id": "call_001",
                "name": "f",
            },
            # Another user message right after tool result (both become role=user)
            # This shouldn't happen normally but tests the merge logic
        ]
        _, converted = AnthropicClient._convert_messages(msgs)
        # tool result becomes user — no consecutive user since assistant is between
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"
        assert converted[2]["role"] == "user"

    def test_full_2step_scenario(self) -> None:
        """End-to-end: system + user + tool_call + result + terminal."""
        msgs = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Look up weather in Paris"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_000",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "22C sunny",
                "tool_call_id": "call_000",
                "name": "get_weather",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_001",
                        "function": {
                            "name": "submit",
                            "arguments": '{"answer": "22C"}',
                        },
                    }
                ],
            },
        ]
        system, converted = AnthropicClient._convert_messages(msgs)
        assert system == "You are a helper."
        assert len(converted) == 4
        assert [m["role"] for m in converted] == [
            "user", "assistant", "user", "assistant"
        ]
        # First assistant: tool_use
        assert converted[1]["content"][0]["type"] == "tool_use"
        assert converted[1]["content"][0]["name"] == "get_weather"
        # Tool result
        assert converted[2]["content"][0]["type"] == "tool_result"
        assert converted[2]["content"][0]["tool_use_id"] == "call_000"
        # Terminal
        assert converted[3]["content"][0]["type"] == "tool_use"
        assert converted[3]["content"][0]["name"] == "submit"

    def test_retry_scenario(self) -> None:
        """TextResponse + retry nudge: plain assistant + user, no tool_result."""
        msgs = [
            {"role": "user", "content": "Do the thing"},
            {"role": "assistant", "content": "I'm not sure how to proceed."},
            {"role": "user", "content": "You must call a tool."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_001",
                        "function": {"name": "do_thing", "arguments": "{}"},
                    }
                ],
            },
        ]
        _, converted = AnthropicClient._convert_messages(msgs)
        assert len(converted) == 4
        assert [m["role"] for m in converted] == [
            "user", "assistant", "user", "assistant"
        ]

    def test_arguments_as_dict(self) -> None:
        """Arguments already parsed as dict (Ollama format) still works."""
        msgs = [
            {"role": "user", "content": "Go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_001",
                        "function": {
                            "name": "f",
                            "arguments": {"key": "val"},
                        },
                    }
                ],
            },
        ]
        _, converted = AnthropicClient._convert_messages(msgs)
        assert converted[1]["content"][0]["input"] == {"key": "val"}


# ── _parse_response ──────────────────────────────────────────────


class TestParseResponse:
    def test_text_response(self) -> None:
        response = {"content": [{"type": "text", "text": "Hello!"}]}

        result = AnthropicClient._parse_response(response)
        assert isinstance(result, TextResponse)
        assert result.content == "Hello!"

    def test_tool_use_response(self) -> None:
        response = {"content": [{
            "type": "tool_use", "name": "get_weather", "input": {"city": "Paris"},
        }]}

        result = AnthropicClient._parse_response(response)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].tool == "get_weather"
        assert result[0].args == {"city": "Paris"}
        assert result[0].reasoning is None

    def test_tool_use_with_text_reasoning(self) -> None:
        response = {"content": [
            {"type": "text", "text": "Let me check the weather."},
            {"type": "tool_use", "name": "get_weather", "input": {"city": "Paris"}},
        ]}

        result = AnthropicClient._parse_response(response)
        assert isinstance(result, list)
        assert result[0].tool == "get_weather"
        assert result[0].reasoning == "Let me check the weather."

    def test_native_thinking_precedes_text_and_attaches_to_first_tool(self) -> None:
        response = {"content": [
            {"type": "thinking", "thinking": "I should check both tools."},
            {"type": "text", "text": "Visible preamble."},
            {"type": "tool_use", "name": "get_weather", "input": {"city": "Paris"}},
            {"type": "tool_use", "name": "set_unit", "input": {"unit": "celsius"}},
        ]}

        result = AnthropicClient._parse_response(response)

        assert result == [
            ToolCall(
                tool="get_weather",
                args={"city": "Paris"},
                reasoning="I should check both tools.",
            ),
            ToolCall(tool="set_unit", args={"unit": "celsius"}, reasoning=None),
        ]

    def test_text_response_excludes_native_thinking(self) -> None:
        response = {"content": [
            {"type": "thinking", "thinking": "Hidden reasoning."},
            {"type": "text", "text": "Visible answer."},
        ]}

        result = AnthropicClient._parse_response(response)

        assert result == TextResponse(content="Visible answer.")

    def test_empty_text_response(self) -> None:
        response = {"content": []}

        result = AnthropicClient._parse_response(response)
        assert isinstance(result, TextResponse)
        assert result.content == ""


# -- Streaming event parsing -----------------------------------------------


class TestStreaming:
    @pytest.mark.asyncio
    async def test_text_stream_emits_deltas_final_response_and_usage(self) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy")
        events = [
            _stream_event(
                "content_block_start", index=0,
                content_block={"type": "text", "text": ""},
            ),
            _text_delta("Hello, "),
            _text_delta("world!"),
            _stream_event("content_block_stop", index=0),
            _stream_event("message_delta", usage={
                "input_tokens": None, "output_tokens": 3,
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
            }),
            _stream_event("message_stop"),
        ]
        usage = {
            "input_tokens": 12,
            "output_tokens": 4,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 5,
        }
        _mock_stream(client, events, usage=usage)

        chunks = [
            chunk
            async for chunk in client.send_stream(
                [{"role": "user", "content": "hello"}],
            )
        ]

        assert [chunk.type for chunk in chunks] == [
            ChunkType.TEXT_DELTA,
            ChunkType.TEXT_DELTA,
            ChunkType.FINAL,
        ]
        assert [chunk.content for chunk in chunks[:-1]] == ["Hello, ", "world!"]
        assert chunks[-1].response == TextResponse(content="Hello, world!")
        assert client.last_usage == {
            0: TokenUsage(
                prompt_tokens=12,
                completion_tokens=4,
                total_tokens=16,
                cache_creation_input_tokens=3,
                cache_read_input_tokens=5,
            ),
        }

    @pytest.mark.asyncio
    async def test_multi_tool_stream_keeps_fragment_and_block_boundaries(self) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy")
        events = [
            _stream_event(
                "content_block_start", index=0,
                content_block={"type": "text", "text": ""},
            ),
            _text_delta("Let me check both."),
            _stream_event("content_block_stop", index=0),
            _tool_start("get_weather"),
            _json_delta('{"city":'),
            _json_delta('"Paris"}'),
            _stream_event("content_block_stop", index=1),
            _tool_start("set_unit", index=2),
            _json_delta('{"unit":"cel', index=2),
            _json_delta('sius"}', index=2),
            _stream_event("content_block_stop", index=2),
            _stream_event("message_stop"),
        ]
        _mock_stream(client, events)

        chunks = [
            chunk
            async for chunk in client.send_stream(
                [{"role": "user", "content": "weather and units"}],
            )
        ]

        assert [
            chunk.content
            for chunk in chunks
            if chunk.type == ChunkType.TOOL_CALL_DELTA
        ] == ['{"city":', '"Paris"}', '{"unit":"cel', 'sius"}']
        assert chunks[-1].type == ChunkType.FINAL
        assert chunks[-1].response == [
            ToolCall(
                tool="get_weather",
                args={"city": "Paris"},
                reasoning="Let me check both.",
            ),
            ToolCall(tool="set_unit", args={"unit": "celsius"}, reasoning=None),
        ]

    @pytest.mark.asyncio
    async def test_stream_native_thinking_precedes_visible_text(self) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy")
        events = [
            _stream_event(
                "content_block_start", index=0,
                content_block={"type": "thinking", "thinking": "", "signature": ""},
            ),
            _thinking_delta("I should "),
            _thinking_delta("search."),
            _stream_event("content_block_stop", index=0),
            _stream_event(
                "content_block_start", index=1,
                content_block={"type": "text", "text": ""},
            ),
            _text_delta("Visible preamble.", index=1),
            _stream_event("content_block_stop", index=1),
            _tool_start("search", index=2),
            _json_delta('{"q":"forge"}', index=2),
            _stream_event("content_block_stop", index=2),
            _stream_event("message_stop"),
        ]
        _mock_stream(client, events)

        chunks = [
            chunk
            async for chunk in client.send_stream(
                [{"role": "user", "content": "search for forge"}],
            )
        ]

        assert [
            chunk.content
            for chunk in chunks
            if chunk.type == ChunkType.TEXT_DELTA
        ] == ["Visible preamble."]
        assert chunks[-1].response == [
            ToolCall(
                tool="search",
                args={"q": "forge"},
                reasoning="I should search.",
            ),
        ]

    @pytest.mark.asyncio
    async def test_http_stream_failure_becomes_backend_error(self) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy", max_retries=0)
        http_error = httpx.ReadError(
            "stream failed",
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        _mock_stream(client, [], error=http_error)

        with pytest.raises(
            BackendError, match="Backend returned 0: Anthropic backend request failed",
        ) as exc:
            _ = [
                chunk
                async for chunk in client.send_stream(
                    [{"role": "user", "content": "hello"}],
                )
            ]

        assert exc.value.__cause__ is http_error
        assert exc.value.body == "stream failed"


# -- Usage reporting (slot-keyed last_usage) -------------------------------


class TestUsageReporting:
    @pytest.mark.asyncio
    async def test_send_records_slot_keyed_usage(self) -> None:
        """send() preserves the slot-keyed direct-client usage mirror."""
        client = AnthropicClient(model="claude-test", api_key="dummy")

        _mock_http(client, lambda _: httpx.Response(
            200, json=_text_response("hello", input_tokens=12, output_tokens=7),
        ))

        result = await client.send([{"role": "user", "content": "hi"}])

        assert isinstance(result, TextResponse)
        expected = TokenUsage(prompt_tokens=12, completion_tokens=7, total_tokens=19)
        assert client.last_usage == {0: expected}


# ── Prompt caching (static tools+system breakpoint) ──────────────


class TestPromptCaching:
    """Opt-in prompt caching marks a static breakpoint over tool defs + system
    in the rebuild path only; off by default; never touches the verbatim path."""

    _MESSAGES = [
        {"role": "system", "content": "stable system prompt"},
        {"role": "user", "content": "hi"},
    ]

    def test_static_cache_marks_tools_and_system(self) -> None:
        client = AnthropicClient(
            model="claude-test", api_key="dummy", prompt_caching=True
        )
        tools = [_make_spec("a"), _make_spec("b")]
        kwargs = client._build_kwargs(self._MESSAGES, tools)

        # Last tool carries the ephemeral breakpoint (caches the tool prefix).
        assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}
        # System is converted to a cached text block (caches tools+system).
        assert isinstance(kwargs["system"], list)
        assert kwargs["system"][0]["text"] == "stable system prompt"
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_no_cache_control_by_default(self) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy")
        tools = [_make_spec("a"), _make_spec("b")]
        kwargs = client._build_kwargs(self._MESSAGES, tools)

        assert "cache_control" not in kwargs["tools"][-1]
        # System stays a plain string when caching is off.
        assert kwargs["system"] == "stable system prompt"

    def test_cache_does_not_touch_verbatim_inbound(self) -> None:
        """prompt_caching must not mutate the path-1 verbatim body — that path
        carries the proxy's own cache_control and bypasses the rebuild."""
        client = AnthropicClient(
            model="claude-test", api_key="dummy", prompt_caching=True
        )
        inbound = {
            "max_tokens": 10,
            "system": "verbatim system",
            "messages": [{"role": "user", "content": "hi"}],
        }
        kwargs = client._build_kwargs([], None, None, inbound)

        # System stays the verbatim string (NOT converted to a cached block).
        assert kwargs["system"] == "verbatim system"

    @pytest.mark.asyncio
    async def test_send_records_cache_usage(self) -> None:
        client = AnthropicClient(
            model="claude-test", api_key="dummy", prompt_caching=True
        )
        _mock_http(client, lambda _: httpx.Response(
            200, json=_text_response(
                input_tokens=5, output_tokens=3,
                cache_creation_input_tokens=100, cache_read_input_tokens=200,
            ),
        ))

        await client.send([{"role": "user", "content": "hi"}])

        tu = client.last_usage[0]
        assert tu.prompt_tokens == 5
        assert tu.cache_creation_input_tokens == 100
        assert tu.cache_read_input_tokens == 200


class TestRequestRoutedModel:
    def test_dynamic_client_uses_verbatim_request_model(self) -> None:
        client = AnthropicClient(model=None, api_key="dummy")
        kwargs = client._build_kwargs(
            [], None, inbound_anthropic_body={"model": "route-a", "messages": []},
        )
        assert kwargs["model"] == "route-a"

    def test_dynamic_client_uses_rebuild_passthrough_model(self) -> None:
        client = AnthropicClient(model=None, api_key="dummy")
        kwargs = client._build_kwargs(
            [{"role": "user", "content": "hi"}], None,
            passthrough={"model": "route-b"},
        )
        assert kwargs["model"] == "route-b"

    @pytest.mark.parametrize("verbatim", [False, True])
    def test_dynamic_client_fails_without_request_model(self, verbatim) -> None:
        client = AnthropicClient(model=None, api_key="dummy")
        with pytest.raises(MissingModelError):
            client._build_kwargs(
                [{"role": "user", "content": "hi"}], None,
                inbound_anthropic_body={"messages": []} if verbatim else None,
            )

    def test_direct_fixed_client_overrides_passthrough_model(self) -> None:
        client = AnthropicClient(model="fixed-model", api_key="dummy")
        kwargs = client._build_kwargs(
            [{"role": "user", "content": "hi"}], None,
            passthrough={"model": "request-model"},
        )
        assert kwargs["model"] == "fixed-model"

    @pytest.mark.asyncio
    async def test_concurrent_clean_and_rebuilt_models_stay_isolated(self) -> None:
        client = AnthropicClient(model=None, api_key="dummy")
        both_arrived = asyncio.Event()
        release = asyncio.Event()
        recorded = []

        async def gated_create(request):
            recorded.append(json.loads(request.content))
            if len(recorded) == 2:
                both_arrived.set()
            await release.wait()
            return httpx.Response(200, json=_text_response())

        _mock_http(client, gated_create)
        clean = asyncio.create_task(client.send(
            [], inbound_anthropic_body={
                "model": "route-opus",
                "messages": [{"role": "user", "content": "one"}],
            },
        ))
        rebuilt = asyncio.create_task(client.send(
            [{"role": "user", "content": "two"}],
            passthrough={"model": "route-sonnet"},
        ))
        await both_arrived.wait()
        release.set()
        await asyncio.gather(clean, rebuilt)

        assert {kwargs["model"] for kwargs in recorded} == {
            "route-opus", "route-sonnet",
        }
        assert client.model is None


class TestThinking:
    """Adaptive extended-thinking request wiring (baseline rows). Request-only:
    thinking is merged into the rebuild path and forces tool_choice=auto."""

    _MESSAGES = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]

    def test_thinking_merged_into_kwargs(self) -> None:
        client = AnthropicClient(
            model="claude-test", api_key="dummy", thinking={"type": "adaptive"}
        )
        kwargs = client._build_kwargs(self._MESSAGES, [_make_spec("a")])
        assert kwargs["thinking"] == {"type": "adaptive"}

    def test_no_thinking_by_default(self) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy")
        kwargs = client._build_kwargs(self._MESSAGES, [_make_spec("a")])
        assert "thinking" not in kwargs

    def test_thinking_suppresses_forced_tool_choice(self) -> None:
        # Anthropic forbids a forced tool_choice with thinking on -> must drop it.
        client = AnthropicClient(
            model="claude-test", api_key="dummy",
            tool_choice="any", thinking={"type": "adaptive"},
        )
        kwargs = client._build_kwargs(self._MESSAGES, [_make_spec("a")])
        assert "tool_choice" not in kwargs
        assert kwargs["thinking"] == {"type": "adaptive"}

    def test_forced_tool_choice_kept_when_no_thinking(self) -> None:
        client = AnthropicClient(
            model="claude-test", api_key="dummy", tool_choice="any"
        )
        kwargs = client._build_kwargs(self._MESSAGES, [_make_spec("a")])
        assert kwargs["tool_choice"] == {"type": "any"}


class TestGetContextLength:
    @pytest.mark.asyncio
    async def test_official_pinned_model_uses_exact_http_metadata(self) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy")
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"max_input_tokens": 123456})

        _mock_http(client, respond)
        assert await client.get_context_length() == 123456
        assert len(requests) == 1
        assert requests[0].method == "GET"
        assert str(requests[0].url) == "https://api.anthropic.com/v1/models/claude-test"

    @pytest.mark.asyncio
    async def test_unpinned_official_client_returns_none(self) -> None:
        client = AnthropicClient(model=None, api_key="dummy")
        def unexpected_request(request):
            pytest.fail("unpinned clients must not request model metadata")

        _mock_http(client, unexpected_request)
        assert await client.get_context_length() is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url",
        [
            "https://gateway.example",
            "http://api.anthropic.com",
            "https://api.anthropic.com/deployment",
            "https://api.anthropic.com/?version=1",
        ],
    )
    async def test_untrusted_metadata_target_returns_none(self, base_url: str) -> None:
        client = AnthropicClient(
            model="claude-test", api_key="dummy", base_url=base_url,
        )

        def unexpected_request(request):
            pytest.fail("untrusted targets must not receive metadata requests")

        _mock_http(client, unexpected_request)
        assert await client.get_context_length() is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("window", [None, 0, -1, True, "200000"])
    async def test_unusable_metadata_window_returns_none(self, window: object) -> None:
        client = AnthropicClient(
            model="claude-test", api_key="dummy", base_url="https://api.anthropic.com:443/",
        )
        _mock_http(client, lambda _: httpx.Response(
            200, json={"max_input_tokens": window},
        ))
        assert await client.get_context_length() is None

    @pytest.mark.asyncio
    async def test_http_failure_maps_to_safe_backend_error(self) -> None:
        client = AnthropicClient(
            model="claude-test", api_key="dummy", max_retries=0,
        )

        def fail(request):
            raise httpx.ConnectError("metadata unavailable", request=request)

        _mock_http(client, fail)
        with pytest.raises(BackendError, match="metadata request failed"):
            await client.get_context_length()


@pytest.mark.asyncio
async def test_body_strips_caller_transport_controls_but_keeps_litellm_session():
    client = AnthropicClient(model="claude-test", api_key="")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=_text_response())

    _mock_http(client, respond)
    inbound = {
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_session_id": None,
        "extra_headers": {"Authorization": "Bearer attacker"},
        "extra_body": {"caller": "must-not-survive"},
        "extra_query": {"x": "y"},
        "timeout": 1,
    }

    await client.send(
        [], inbound_anthropic_body=inbound,
        extra_headers={"x-api-key": "secret"},
    )

    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["litellm_session_id"] is None
    assert not {"extra_headers", "extra_body", "extra_query", "timeout"} & body.keys()
    assert requests[0].headers["x-api-key"] == "secret"
    assert "authorization" not in requests[0].headers
    assert not requests[0].url.query
