"""Tests for proxy message conversion (Anthropic ↔ forge)."""

import json

from forge.core.messages import MessageRole, MessageType
from forge.core.workflow import ToolCall
from forge.proxy.convert_anthropic import (
    anthropic_to_messages,
    anthropic_tools_to_specs,
    anthropic_tool_choice_to_openai,
    anthropic_to_openai_passthrough,
    tool_calls_to_anthropic,
    text_response_to_anthropic,
    tool_calls_to_anthropic_sse,
    text_to_anthropic_sse,
)


# ── Inbound: Anthropic → forge ─────────────────────────────


class TestAnthropicToMessages:
    def test_system_string_becomes_system_message(self):
        msgs = anthropic_to_messages(
            [{"role": "user", "content": "hi"}],
            system="You are helpful.",
        )
        assert msgs[0].role == MessageRole.SYSTEM
        assert msgs[0].content == "You are helpful."
        assert msgs[0].metadata.type == MessageType.SYSTEM_PROMPT

    def test_system_block_list_concatenated(self):
        msgs = anthropic_to_messages(
            [{"role": "user", "content": "hi"}],
            system=[
                {"type": "text", "text": "Be terse."},
                {"type": "text", "text": "Use tools."},
            ],
        )
        assert msgs[0].role == MessageRole.SYSTEM
        assert "Be terse." in msgs[0].content
        assert "Use tools." in msgs[0].content

    def test_no_system_omits_system_message(self):
        msgs = anthropic_to_messages([{"role": "user", "content": "hi"}])
        assert msgs[0].role == MessageRole.USER

    def test_user_string_content(self):
        msgs = anthropic_to_messages([{"role": "user", "content": "Hello"}])
        assert msgs[0].role == MessageRole.USER
        assert msgs[0].content == "Hello"
        assert msgs[0].metadata.type == MessageType.USER_INPUT

    def test_assistant_text_block(self):
        msgs = anthropic_to_messages([{
            "role": "assistant",
            "content": [{"type": "text", "text": "Sure."}],
        }])
        assert msgs[0].role == MessageRole.ASSISTANT
        assert msgs[0].content == "Sure."
        assert msgs[0].metadata.type == MessageType.TEXT_RESPONSE

    def test_assistant_tool_use_block(self):
        msgs = anthropic_to_messages([{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "search",
                    "input": {"q": "weather"},
                },
            ],
        }])
        assert msgs[0].role == MessageRole.ASSISTANT
        assert msgs[0].metadata.type == MessageType.TOOL_CALL
        assert msgs[0].content == "Checking."
        assert len(msgs[0].tool_calls) == 1
        tc = msgs[0].tool_calls[0]
        assert tc.name == "search"
        assert tc.args == {"q": "weather"}
        assert tc.call_id == "toolu_abc"

    def test_user_tool_result_becomes_tool_message(self):
        msgs = anthropic_to_messages([{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_abc",
                "content": "sunny, 22C",
            }],
        }])
        assert msgs[0].role == MessageRole.TOOL
        assert msgs[0].metadata.type == MessageType.TOOL_RESULT
        assert msgs[0].content == "sunny, 22C"
        assert msgs[0].tool_call_id == "toolu_abc"

    def test_user_mixed_tool_result_plus_text(self):
        """Anthropic packs tool_result + follow-up user text in one user message;
        forge splits them into separate role=tool then role=user messages."""
        msgs = anthropic_to_messages([{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_x", "content": "ok"},
                {"type": "text", "text": "Now what?"},
            ],
        }])
        assert len(msgs) == 2
        assert msgs[0].role == MessageRole.TOOL
        assert msgs[1].role == MessageRole.USER
        assert msgs[1].content == "Now what?"

    def test_tool_result_content_as_block_list(self):
        msgs = anthropic_to_messages([{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_y",
                "content": [{"type": "text", "text": "block content"}],
            }],
        }])
        assert msgs[0].content == "block content"

    def test_thinking_image_document_dropped(self):
        """Anthropic-only block types with no forge analog are silently dropped."""
        msgs = anthropic_to_messages([{
            "role": "user",
            "content": [
                {"type": "thinking", "thinking": "internal monologue"},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
                {"type": "document", "source": {"type": "base64", "data": "..."}},
                {"type": "text", "text": "Here is the file."},
            ],
        }])
        assert len(msgs) == 1
        assert msgs[0].content == "Here is the file."


class TestAnthropicToolsToSpecs:
    def test_basic_tool(self):
        specs = anthropic_tools_to_specs([{
            "name": "get_weather",
            "description": "Get weather.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }])
        assert len(specs) == 1
        assert specs[0].name == "get_weather"
        assert specs[0].description == "Get weather."

    def test_empty_or_none(self):
        assert anthropic_tools_to_specs(None) == []
        assert anthropic_tools_to_specs([]) == []

    def test_skips_nameless_tool(self):
        specs = anthropic_tools_to_specs([{"description": "no name"}])
        assert specs == []


class TestAnthropicToolChoiceTranslation:
    def test_auto(self):
        assert anthropic_tool_choice_to_openai({"type": "auto"}) == "auto"

    def test_any_becomes_required(self):
        assert anthropic_tool_choice_to_openai({"type": "any"}) == "required"

    def test_none(self):
        assert anthropic_tool_choice_to_openai({"type": "none"}) == "none"

    def test_named_tool(self):
        result = anthropic_tool_choice_to_openai({"type": "tool", "name": "search"})
        assert result == {"type": "function", "function": {"name": "search"}}

    def test_passes_through_unknown_shape(self):
        # Non-dict input passes through (already in target shape).
        assert anthropic_tool_choice_to_openai("auto") == "auto"


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
        out = anthropic_to_openai_passthrough(body)
        assert out["model"] == "claude-3-5-sonnet"
        assert out["max_tokens"] == 1024
        assert out["temperature"] == 0.5
        assert out["top_p"] == 0.9
        assert out["top_k"] == 40
        assert out["stop"] == ["</done>"]
        assert out["tool_choice"] == "required"

    def test_excludes_forge_owned(self):
        body = {
            "messages": [],
            "tools": [],
            "system": "x",
            "stream": True,
        }
        out = anthropic_to_openai_passthrough(body)
        # None of the forge-owned fields appear in the passthrough.
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
        out = anthropic_to_openai_passthrough(body)
        assert out == {}


# ── Outbound: forge → Anthropic ────────────────────────────


class TestToolCallsToAnthropic:
    def test_shape(self):
        result = tool_calls_to_anthropic(
            [ToolCall(tool="get_weather", args={"city": "Paris"})],
            model="claude-3-5-sonnet",
        )
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-3-5-sonnet"
        assert result["stop_reason"] == "tool_use"
        assert result["id"].startswith("msg_")
        # One tool_use block
        tu_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tu_blocks) == 1
        assert tu_blocks[0]["name"] == "get_weather"
        assert tu_blocks[0]["input"] == {"city": "Paris"}
        assert tu_blocks[0]["id"].startswith("toolu_")

    def test_default_omits_reasoning_text_block(self):
        result = tool_calls_to_anthropic([
            ToolCall(tool="search", args={"q": "x"}, reasoning="Let me search."),
        ])
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        assert text_blocks == []

    def test_full_reasoning_replay_emits_reasoning_as_text_block(self):
        result = tool_calls_to_anthropic([
            ToolCall(tool="search", args={"q": "x"}, reasoning="Let me search."),
        ], reasoning_replay="full")
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        assert text_blocks and text_blocks[0]["text"] == "Let me search."

    def test_multiple_tool_calls(self):
        result = tool_calls_to_anthropic([
            ToolCall(tool="a", args={"x": 1}),
            ToolCall(tool="b", args={"y": 2}),
        ])
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


# ── SSE event sequence ─────────────────────────────────────


class TestToolCallsToAnthropicSSE:
    def test_event_sequence(self):
        events = tool_calls_to_anthropic_sse(
            [ToolCall(tool="get_weather", args={"city": "Paris"})],
        )
        types = [e["type"] for e in events]
        # Must start with message_start and end with message_stop
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"
        # Must contain a content_block triplet for the tool_use
        assert types.count("content_block_start") >= 1
        assert types.count("content_block_delta") >= 1
        assert types.count("content_block_stop") >= 1
        # Must include message_delta with stop_reason
        delta = next(e for e in events if e["type"] == "message_delta")
        assert delta["delta"]["stop_reason"] == "tool_use"

    def test_default_omits_reasoning_stream_block(self):
        events = tool_calls_to_anthropic_sse([
            ToolCall(tool="search", args={"q": "x"}, reasoning="Hmm."),
        ])
        starts = [e for e in events if e["type"] == "content_block_start"]
        assert starts[0]["content_block"]["type"] == "tool_use"

    def test_full_reasoning_replay_streams_text_block_before_tool_use(self):
        events = tool_calls_to_anthropic_sse([
            ToolCall(tool="search", args={"q": "x"}, reasoning="Hmm."),
        ], reasoning_replay="full")
        # First content_block_start should be type=text (reasoning), then tool_use
        starts = [e for e in events if e["type"] == "content_block_start"]
        assert starts[0]["content_block"]["type"] == "text"
        assert starts[1]["content_block"]["type"] == "tool_use"
        # The reasoning text delta has the text
        text_delta = next(
            e for e in events
            if e["type"] == "content_block_delta"
            and e["delta"]["type"] == "text_delta"
        )
        assert text_delta["delta"]["text"] == "Hmm."

    def test_input_json_delta_is_parseable(self):
        events = tool_calls_to_anthropic_sse(
            [ToolCall(tool="x", args={"a": 1, "b": "two"})],
        )
        json_delta = next(
            e for e in events
            if e["type"] == "content_block_delta"
            and e["delta"]["type"] == "input_json_delta"
        )
        assert json.loads(json_delta["delta"]["partial_json"]) == {"a": 1, "b": "two"}


class TestTextToAnthropicSSE:
    def test_event_sequence(self):
        events = text_to_anthropic_sse("Hi there.")
        types = [e["type"] for e in events]
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"
        delta = next(e for e in events if e["type"] == "message_delta")
        assert delta["delta"]["stop_reason"] == "end_turn"
        text_delta = next(
            e for e in events
            if e["type"] == "content_block_delta"
        )
        assert text_delta["delta"]["text"] == "Hi there."
