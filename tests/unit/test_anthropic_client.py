"""Tests for forge.clients.anthropic — format conversion helpers.

All tests exercise the static conversion methods directly.
No API calls or mocks needed.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
import tomllib
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import anthropic
import httpx
from pydantic import BaseModel, Field

from forge.clients.anthropic import AnthropicClient
from forge.clients.base import ChunkType, TokenUsage
from forge.core.workflow import TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError, MissingModelError


def test_anthropic_dependency_floor_exposes_exact_context_metadata() -> None:
    config = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8"),
    )
    extras = config["project"]["optional-dependencies"]

    assert extras["anthropic"] == ["anthropic>=0.86.0"]
    assert "anthropic>=0.86.0" in extras["dev"]


class CityParams(BaseModel):
    city: str = Field(description="City name")


def _make_spec(name: str = "get_weather") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Get {name}",
        parameters=CityParams,
    )


class _FakeAnthropicStream:
    """Minimal SDK stream surface for deterministic event-parser tests."""

    def __init__(
        self,
        events: list[SimpleNamespace],
        *,
        usage: SimpleNamespace | None = None,
        error: Exception | None = None,
    ) -> None:
        self._events = events
        self._usage = usage or SimpleNamespace(
            input_tokens=1,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        self._error = error

    async def __aenter__(self) -> "_FakeAnthropicStream":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        async def iterate() -> AsyncIterator[SimpleNamespace]:
            for event in self._events:
                yield event
            if self._error is not None:
                raise self._error

        return iterate()

    async def get_final_message(self) -> SimpleNamespace:
        return SimpleNamespace(usage=self._usage)


def _stream_event(event_type: str, **fields: object) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **fields)


def _text_delta(text: str) -> SimpleNamespace:
    return _stream_event(
        "content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _tool_start(name: str) -> SimpleNamespace:
    return _stream_event(
        "content_block_start",
        content_block=SimpleNamespace(type="tool_use", name=name),
    )


def _json_delta(partial_json: str) -> SimpleNamespace:
    return _stream_event(
        "content_block_delta",
        delta=SimpleNamespace(type="input_json_delta", partial_json=partial_json),
    )


class SetUnitParams(BaseModel):
    unit: Literal["celsius", "fahrenheit"] = Field(description="Unit")


def _make_spec_with_enum() -> ToolSpec:
    return ToolSpec(
        name="set_unit",
        description="Set temperature unit",
        parameters=SetUnitParams,
    )


def test_direct_base_url_is_passed_to_sdk_unchanged() -> None:
    service_root = "https://anthropic.example/deployment/"
    with patch("forge.clients.anthropic.anthropic.AsyncAnthropic") as sdk:
        AnthropicClient(model="claude-test", api_key="dummy", base_url=service_root)
    assert sdk.call_args.kwargs["base_url"] == service_root


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
        response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello!"
        response.content = [text_block]

        result = AnthropicClient._parse_response(response)
        assert isinstance(result, TextResponse)
        assert result.content == "Hello!"

    def test_tool_use_response(self) -> None:
        response = MagicMock()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "get_weather"
        tool_block.input = {"city": "Paris"}
        response.content = [tool_block]

        result = AnthropicClient._parse_response(response)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].tool == "get_weather"
        assert result[0].args == {"city": "Paris"}
        assert result[0].reasoning is None

    def test_tool_use_with_text_reasoning(self) -> None:
        response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me check the weather."
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "get_weather"
        tool_block.input = {"city": "Paris"}
        response.content = [text_block, tool_block]

        result = AnthropicClient._parse_response(response)
        assert isinstance(result, list)
        assert result[0].tool == "get_weather"
        assert result[0].reasoning == "Let me check the weather."

    def test_empty_text_response(self) -> None:
        response = MagicMock()
        response.content = []

        result = AnthropicClient._parse_response(response)
        assert isinstance(result, TextResponse)
        assert result.content == ""


# -- Streaming event parsing -----------------------------------------------


class TestStreaming:
    @pytest.mark.asyncio
    async def test_text_stream_emits_deltas_final_response_and_usage(self) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy")
        events = [
            _text_delta("Hello, "),
            _text_delta("world!"),
            _stream_event("message_stop"),
        ]
        usage = SimpleNamespace(
            input_tokens=12,
            output_tokens=4,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=5,
        )
        client._client = MagicMock()
        client._client.messages.stream.return_value = _FakeAnthropicStream(
            events, usage=usage,
        )

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
            _text_delta("Let me check both."),
            _tool_start("get_weather"),
            _json_delta('{"city":'),
            _json_delta('"Paris"}'),
            _stream_event("content_block_stop"),
            _tool_start("set_unit"),
            _json_delta('{"unit":"cel'),
            _json_delta('sius"}'),
            _stream_event("content_block_stop"),
            _stream_event("message_stop"),
        ]
        client._client = MagicMock()
        client._client.messages.stream.return_value = _FakeAnthropicStream(events)

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
    async def test_sdk_stream_failure_becomes_backend_error(self) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy")
        sdk_error = anthropic.APIError(
            "stream failed",
            httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            body=None,
        )
        client._client = MagicMock()
        client._client.messages.stream.return_value = _FakeAnthropicStream(
            [], error=sdk_error,
        )

        with pytest.raises(BackendError, match="Backend returned 0: stream failed") as exc:
            _ = [
                chunk
                async for chunk in client.send_stream(
                    [{"role": "user", "content": "hello"}],
                )
            ]

        assert exc.value.__cause__ is sdk_error


# -- Usage reporting (slot-keyed last_usage) -------------------------------


class TestUsageReporting:
    @pytest.mark.asyncio
    async def test_send_records_slot_keyed_usage(self) -> None:
        """send() preserves the slot-keyed direct-client usage mirror."""
        client = AnthropicClient(model="claude-test", api_key="dummy")

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "hello"
        response = MagicMock()
        response.content = [text_block]
        response.usage.input_tokens = 12
        response.usage.output_tokens = 7
        # Real Anthropic Usage reports these as ints (0 without caching); set
        # them so the MagicMock doesn't auto-create truthy attrs.
        response.usage.cache_creation_input_tokens = 0
        response.usage.cache_read_input_tokens = 0

        async def fake_create(**kwargs):
            return response

        client._client.messages.create = fake_create

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
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "ok"
        response = MagicMock()
        response.content = [text_block]
        response.usage.input_tokens = 5
        response.usage.output_tokens = 3
        response.usage.cache_creation_input_tokens = 100
        response.usage.cache_read_input_tokens = 200
        client._client.messages.create = AsyncMock(return_value=response)

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

        async def gated_create(**kwargs):
            recorded.append(kwargs)
            if len(recorded) == 2:
                both_arrived.set()
            await release.wait()
            response = MagicMock()
            response.content = [MagicMock(type="text", text="ok")]
            response.usage.input_tokens = 1
            response.usage.output_tokens = 1
            response.usage.cache_creation_input_tokens = 0
            response.usage.cache_read_input_tokens = 0
            return response

        client._client.messages.create = AsyncMock(side_effect=gated_create)
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
    async def test_official_pinned_model_uses_exact_sdk_metadata(self) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy")
        retrieve = AsyncMock(return_value=SimpleNamespace(max_input_tokens=123456))
        client._client = SimpleNamespace(
            base_url="https://api.anthropic.com/",
            models=SimpleNamespace(retrieve=retrieve),
        )
        assert await client.get_context_length() == 123456
        retrieve.assert_awaited_once_with("claude-test")

    @pytest.mark.asyncio
    async def test_unpinned_official_client_returns_none(self) -> None:
        client = AnthropicClient(model=None, api_key="dummy")
        retrieve = AsyncMock()
        client._client = SimpleNamespace(
            base_url="https://api.anthropic.com",
            models=SimpleNamespace(retrieve=retrieve),
        )
        assert await client.get_context_length() is None
        retrieve.assert_not_awaited()

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
        client = AnthropicClient(model="claude-test", api_key="dummy")
        retrieve = AsyncMock()
        client._client = SimpleNamespace(
            base_url=base_url,
            models=SimpleNamespace(retrieve=retrieve),
        )
        assert await client.get_context_length() is None
        retrieve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_older_sdk_without_models_resource_returns_none(self) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy")
        client._client = SimpleNamespace(base_url="https://api.anthropic.com")
        assert await client.get_context_length() is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("window", [None, 0, -1, True, "200000"])
    async def test_unusable_metadata_window_returns_none(self, window: object) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy")
        client._client = SimpleNamespace(
            base_url="https://api.anthropic.com:443/",
            models=SimpleNamespace(
                retrieve=AsyncMock(
                    return_value=SimpleNamespace(max_input_tokens=window),
                ),
            ),
        )
        assert await client.get_context_length() is None

    @pytest.mark.asyncio
    async def test_sdk_failure_maps_to_safe_backend_error(self) -> None:
        client = AnthropicClient(model="claude-test", api_key="dummy")
        request = httpx.Request("GET", "https://api.anthropic.com/v1/models/x")
        client._client = SimpleNamespace(
            base_url="https://api.anthropic.com",
            models=SimpleNamespace(
                retrieve=AsyncMock(
                    side_effect=__import__("anthropic").APIConnectionError(
                        request=request,
                    ),
                ),
            ),
        )
        with pytest.raises(BackendError, match="metadata request failed"):
            await client.get_context_length()


def test_sdk_kwargs_owns_only_litellm_extra_body_and_strips_caller_controls():
    client = AnthropicClient.__new__(AnthropicClient)
    client._static_auth = False
    client._client = SimpleNamespace(api_key=None, auth_token=None)
    kwargs = {
        "litellm_session_id": None,
        "extra_body": {"caller": "must-not-survive"},
        "extra_query": {"x": "y"},
        "timeout": 1,
    }

    prepared = client._prepare_sdk_kwargs(kwargs, {"x-api-key": "secret"})

    assert prepared["extra_body"] == {"litellm_session_id": None}
    assert prepared["extra_headers"] == {"X-Api-Key": "secret"}
    assert "extra_query" not in prepared
    assert "timeout" not in prepared
