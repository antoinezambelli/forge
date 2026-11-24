from forge.core.messages import ReasoningPart, TextPart, ConversationState
from forge.clients.base import RequestMetadata
from forge.clients.requests import request_extras
from tests.canonical import assistant, call

"""Tests for proxy message conversion (Anthropic ↔ forge)."""
import json
import pytest
from forge.proxy.convert_anthropic import (
    anthropic_to_messages,
    anthropic_tools_to_definitions,
    reply_to_anthropic,
    text_response_to_anthropic,
    reply_to_anthropic_sse,
    text_to_anthropic_sse,
)


def test_litellm_session_id_preserved_by_key_presence():
    assert (
        request_extras(
            RequestMetadata(source_protocol="anthropic", extra_fields={"litellm_session_id": None}),
            "openai",
        )["litellm_session_id"]
        is None
    )


class TestAnthropicToolsToDefinitions:
    def test_basic_tool(self):
        specs = anthropic_tools_to_definitions(
            [
                {
                    "name": "get_weather",
                    "description": "Get weather.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ]
        )
        assert len(specs) == 1
        assert specs[0].name == "get_weather"
        assert specs[0].description == "Get weather."

    def test_empty_or_none(self):
        assert anthropic_tools_to_definitions(None) == []
        assert anthropic_tools_to_definitions([]) == []

    def test_rejects_nameless_tool(self):
        with pytest.raises(ValueError):
            anthropic_tools_to_definitions([{"description": "no name", "input_schema": {}}])


class TestAnthropicToolChoiceTranslation:
    def test_auto(self):
        assert (
            request_extras(
                RequestMetadata(
                    source_protocol="anthropic", extra_fields={"tool_choice": {"type": "auto"}}
                ),
                "openai",
            )["tool_choice"]
            == "auto"
        )

    def test_any_becomes_required(self):
        assert (
            request_extras(
                RequestMetadata(
                    source_protocol="anthropic", extra_fields={"tool_choice": {"type": "any"}}
                ),
                "openai",
            )["tool_choice"]
            == "required"
        )

    def test_none(self):
        assert (
            request_extras(
                RequestMetadata(
                    source_protocol="anthropic", extra_fields={"tool_choice": {"type": "none"}}
                ),
                "openai",
            )["tool_choice"]
            == "none"
        )

    def test_named_tool(self):
        result = request_extras(
            RequestMetadata(
                source_protocol="anthropic",
                extra_fields={"tool_choice": {"type": "tool", "name": "search"}},
            ),
            "openai",
        )["tool_choice"]
        assert result == {"type": "function", "function": {"name": "search"}}

    def test_passes_through_unknown_shape(self):
        assert (
            request_extras(
                RequestMetadata(source_protocol="anthropic", extra_fields={"tool_choice": "auto"}),
                "openai",
            )["tool_choice"]
            == "auto"
        )


class TestAnthropicToOpenAIPassthrough:
    def test_translates_known_fields(self):
        body = {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "hi"}],
            "system": "be helpful",
            "max_tokens": 1024,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["</done>"],
            "tool_choice": {"type": "any"},
        }
        out = request_extras(
            RequestMetadata(source_protocol="anthropic", extra_fields=body), "openai"
        )
        assert out["model"] == "claude-3-5-sonnet"
        assert out["max_tokens"] == 1024
        assert out["temperature"] == 0.5
        assert out["top_p"] == 0.9
        assert out["top_k"] == 40
        assert out["stop"] == ["</done>"]
        assert out["tool_choice"] == "required"

    def test_excludes_forge_owned(self):
        body = {"messages": [], "tools": [], "system": "x", "stream": True}
        out = request_extras(
            RequestMetadata(source_protocol="anthropic", extra_fields=body), "openai"
        )
        assert "messages" not in out
        assert "tools" not in out
        assert "system" not in out
        assert "stream" not in out

    def test_drops_anthropic_only_fields(self):
        """cache_control on blocks lives inside messages (filtered).
        thinking/metadata/service_tier have no OpenAI analog — dropped here,
        preserved via inbound_anthropic_body on path 1 (ADR-015)."""
        body = {
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "metadata": {"user_id": "u-123"},
            "service_tier": "auto",
        }
        out = request_extras(
            RequestMetadata(source_protocol="anthropic", extra_fields=body), "openai"
        )
        assert out == {}


class TestToolCallsToAnthropic:
    def test_shape(self):
        result = reply_to_anthropic(
            assistant(call(name="get_weather", arguments={"city": "Paris"})),
            model="claude-3-5-sonnet",
        )
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-3-5-sonnet"
        assert result["stop_reason"] == "tool_use"
        assert result["id"].startswith("msg_")
        tu_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tu_blocks) == 1
        assert tu_blocks[0]["name"] == "get_weather"
        assert tu_blocks[0]["input"] == {"city": "Paris"}
        assert tu_blocks[0]["id"].startswith("toolu_")

    def test_default_omits_reasoning_text_block(self):
        result = reply_to_anthropic(
            assistant(ReasoningPart("Let me search."), call(name="search", arguments={"q": "x"}))
        )
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        assert text_blocks == []

    def test_full_reasoning_replay_emits_reasoning_as_text_block(self):
        result = reply_to_anthropic(
            assistant(ReasoningPart("Let me search."), call(name="search", arguments={"q": "x"})),
            reasoning_replay="full",
        )
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        assert text_blocks and text_blocks[0]["text"] == "Let me search."

    def test_multiple_tool_calls(self):
        result = reply_to_anthropic(
            assistant(call(name="a", arguments={"x": 1}), call(name="b", arguments={"y": 2}))
        )
        tu_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tu_blocks) == 2
        assert tu_blocks[0]["name"] == "a"
        assert tu_blocks[1]["name"] == "b"


class TestTextResponseToAnthropic:
    def test_shape(self):
        result = text_response_to_anthropic("Hello!", model="claude-3-5-sonnet")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-3-5-sonnet"
        assert result["stop_reason"] == "end_turn"
        assert result["content"] == [{"type": "text", "text": "Hello!"}]


class TestToolCallsToAnthropicSSE:
    def test_event_sequence(self):
        events = reply_to_anthropic_sse(
            assistant(call(name="get_weather", arguments={"city": "Paris"}))
        )
        types = [e["type"] for e in events]
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"
        assert types.count("content_block_start") >= 1
        assert types.count("content_block_delta") >= 1
        assert types.count("content_block_stop") >= 1
        delta = next((e for e in events if e["type"] == "message_delta"))
        assert delta["delta"]["stop_reason"] == "tool_use"

    def test_default_omits_reasoning_stream_block(self):
        events = reply_to_anthropic_sse(
            assistant(ReasoningPart("Hmm."), call(name="search", arguments={"q": "x"}))
        )
        starts = [e for e in events if e["type"] == "content_block_start"]
        assert starts[0]["content_block"]["type"] == "tool_use"

    def test_full_reasoning_replay_streams_text_block_before_tool_use(self):
        events = reply_to_anthropic_sse(
            assistant(ReasoningPart("Hmm."), call(name="search", arguments={"q": "x"})),
            reasoning_replay="full",
        )
        starts = [e for e in events if e["type"] == "content_block_start"]
        assert starts[0]["content_block"]["type"] == "text"
        assert starts[1]["content_block"]["type"] == "tool_use"
        text_delta = next(
            (
                e
                for e in events
                if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
            )
        )
        assert text_delta["delta"]["text"] == "Hmm."

    def test_input_json_delta_is_parseable(self):
        events = reply_to_anthropic_sse(assistant(call(name="x", arguments={"a": 1, "b": "two"})))
        json_delta = next(
            (
                e
                for e in events
                if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta"
            )
        )
        assert json.loads(json_delta["delta"]["partial_json"]) == {"a": 1, "b": "two"}


class TestTextToAnthropicSSE:
    def test_event_sequence(self):
        events = text_to_anthropic_sse("Hi there.")
        types = [e["type"] for e in events]
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"
        delta = next((e for e in events if e["type"] == "message_delta"))
        assert delta["delta"]["stop_reason"] == "end_turn"
        text_delta = next((e for e in events if e["type"] == "content_block_delta"))
        assert text_delta["delta"]["text"] == "Hi there."


@pytest.mark.parametrize(
    "tools, message",
    [
        ({}, "list"),
        ([None], "function"),
        ([{}], "input_schema"),
        ([{"type": "computer_20250124", "name": "computer"}], "function"),
        ([{"input_schema": {}}], "name"),
        ([{"name": "", "input_schema": {}}], "name"),
        ([{"name": 3, "input_schema": {}}], "name"),
        ([{"name": "x"}], "input_schema"),
        ([{"name": "x", "input_schema": None}], "dictionary"),
        ([{"name": "x", "input_schema": []}], "dictionary"),
        ([{"name": "x", "input_schema": {}, "description": None}], "description"),
        ([{"name": "x", "input_schema": {}, "description": 2}], "description"),
        ([{"name": "x", "input_schema": {}}, {"name": "x", "input_schema": {}}], "Duplicate"),
    ],
)
def test_anthropic_definition_failures(tools, message):
    with pytest.raises(ValueError, match=message):
        anthropic_tools_to_definitions(tools)


def test_anthropic_defaults_and_flat_extras():
    definition = anthropic_tools_to_definitions(
        [{"name": "name with spaces", "input_schema": {}, "cache_control": {"type": "ephemeral"}}]
    )[0]
    assert definition.description == ""
    assert definition.input_schema == {}
    assert definition.extra_fields == {"cache_control": {"type": "ephemeral"}}
