"""Tests for proxy message conversion (OpenAI ↔ forge)."""

import json

from forge.core.messages import MessageRole, MessageType
from forge.core.workflow import ToolCall
from forge.proxy.convert import (
    openai_to_messages,
    tool_calls_to_openai,
    text_response_to_openai,
    tool_calls_to_sse_events,
    text_to_sse_events,
)


# ── Inbound: OpenAI → forge ────────────────────────────────


class TestOpenaiToMessages:
    def test_system_message(self):
        msgs = openai_to_messages([{"role": "system", "content": "You are helpful."}])
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.SYSTEM
        assert msgs[0].content == "You are helpful."
        assert msgs[0].metadata.type == MessageType.SYSTEM_PROMPT

    def test_user_message(self):
        msgs = openai_to_messages([{"role": "user", "content": "Hello"}])
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.USER
        assert msgs[0].content == "Hello"
        assert msgs[0].metadata.type == MessageType.USER_INPUT

    def test_assistant_text(self):
        msgs = openai_to_messages([{"role": "assistant", "content": "Hi there"}])
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.ASSISTANT
        assert msgs[0].content == "Hi there"
        assert msgs[0].metadata.type == MessageType.TEXT_RESPONSE

    def test_assistant_tool_calls(self):
        msgs = openai_to_messages([{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_abc",
                "function": {"name": "search", "arguments": '{"q": "test"}'},
            }],
        }])
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.ASSISTANT
        assert msgs[0].metadata.type == MessageType.TOOL_CALL
        assert msgs[0].tool_calls is not None
        assert len(msgs[0].tool_calls) == 1
        assert msgs[0].tool_calls[0].name == "search"
        assert msgs[0].tool_calls[0].args == {"q": "test"}
        assert msgs[0].tool_calls[0].call_id == "call_abc"

    def test_tool_result(self):
        msgs = openai_to_messages([{
            "role": "tool",
            "content": "result data",
            "tool_call_id": "call_abc",
            "name": "search",
        }])
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.TOOL
        assert msgs[0].content == "result data"
        assert msgs[0].metadata.type == MessageType.TOOL_RESULT
        assert msgs[0].tool_name == "search"
        assert msgs[0].tool_call_id == "call_abc"

    def test_unknown_role_maps_to_user(self):
        msgs = openai_to_messages([{"role": "developer", "content": "stuff"}])
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.USER

    def test_list_content_blocks(self):
        msgs = openai_to_messages([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        }])
        assert msgs[0].content == "Hello\nWorld"

    def test_null_content_becomes_empty(self):
        msgs = openai_to_messages([{"role": "user", "content": None}])
        assert msgs[0].content == ""

    def test_multi_message_conversation(self):
        msgs = openai_to_messages([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ])
        assert len(msgs) == 4
        assert [m.role for m in msgs] == [
            MessageRole.SYSTEM, MessageRole.USER,
            MessageRole.ASSISTANT, MessageRole.USER,
        ]

    def test_assistant_tool_calls_with_dict_arguments(self):
        """Arguments as dict (not JSON string) should be handled."""
        msgs = openai_to_messages([{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "fetch", "arguments": {"url": "http://x"}},
            }],
        }])
        assert msgs[0].tool_calls[0].args == {"url": "http://x"}

    def test_empty_tool_calls_treated_as_text(self):
        """Empty tool_calls list should be treated as text response."""
        msgs = openai_to_messages([{
            "role": "assistant",
            "content": "thinking...",
            "tool_calls": [],
        }])
        assert msgs[0].metadata.type == MessageType.TEXT_RESPONSE


# ── Outbound: forge → OpenAI ───────────────────────────────


class TestToolCallsToOpenai:
    def test_single_tool_call(self):
        result = tool_calls_to_openai([ToolCall(tool="search", args={"q": "test"})])
        assert result["object"] == "chat.completion"
        choices = result["choices"]
        assert len(choices) == 1
        msg = choices[0]["message"]
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search"
        assert json.loads(tc["function"]["arguments"]) == {"q": "test"}
        assert choices[0]["finish_reason"] == "tool_calls"

    def test_multiple_tool_calls(self):
        result = tool_calls_to_openai([
            ToolCall(tool="search", args={"q": "a"}),
            ToolCall(tool="fetch", args={"url": "b"}),
        ])
        assert len(result["choices"][0]["message"]["tool_calls"]) == 2

    def test_reasoning_in_content(self):
        result = tool_calls_to_openai([
            ToolCall(tool="search", args={}, reasoning="Let me think..."),
        ])
        assert result["choices"][0]["message"]["content"] == "Let me think..."

    def test_no_reasoning_content_is_none(self):
        result = tool_calls_to_openai([ToolCall(tool="search", args={})])
        assert result["choices"][0]["message"]["content"] is None

    def test_model_name_propagated(self):
        result = tool_calls_to_openai(
            [ToolCall(tool="search", args={})], model="my-model",
        )
        assert result["model"] == "my-model"


class TestTextResponseToOpenai:
    def test_basic(self):
        result = text_response_to_openai("Hello!")
        assert result["object"] == "chat.completion"
        msg = result["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_empty_text(self):
        result = text_response_to_openai("")
        assert result["choices"][0]["message"]["content"] == ""

    def test_model_name(self):
        result = text_response_to_openai("hi", model="test-model")
        assert result["model"] == "test-model"


# ── SSE streaming ──────────────────────────────────────────


class TestToolCallsToSseEvents:
    def test_single_tool_call_structure(self):
        events = tool_calls_to_sse_events([ToolCall(tool="search", args={"q": "x"})])
        # Should have: tool call delta + final chunk
        assert len(events) == 2
        # First: tool call delta
        delta = events[0]["choices"][0]["delta"]
        assert "tool_calls" in delta
        assert delta["tool_calls"][0]["function"]["name"] == "search"
        # Last: finish_reason
        assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert events[-1]["choices"][0]["delta"] == {}

    def test_reasoning_prepended(self):
        events = tool_calls_to_sse_events([
            ToolCall(tool="search", args={}, reasoning="Thinking..."),
        ])
        # reasoning delta + tool call delta + final
        assert len(events) == 3
        assert events[0]["choices"][0]["delta"]["content"] == "Thinking..."

    def test_multiple_tool_calls(self):
        events = tool_calls_to_sse_events([
            ToolCall(tool="a", args={}),
            ToolCall(tool="b", args={}),
        ])
        # 2 tool call deltas + final
        assert len(events) == 3

    def test_consistent_completion_id(self):
        events = tool_calls_to_sse_events([ToolCall(tool="a", args={})])
        ids = {e["id"] for e in events}
        assert len(ids) == 1  # all events share the same completion ID


class TestTextToSseEvents:
    def test_single_chunk(self):
        events = text_to_sse_events("Hello!")
        # content chunk + final
        assert len(events) == 2
        assert events[0]["choices"][0]["delta"]["content"] == "Hello!"
        assert events[0]["choices"][0]["delta"]["role"] == "assistant"
        assert events[-1]["choices"][0]["finish_reason"] == "stop"

    def test_chunked_text(self):
        events = text_to_sse_events("Hello World!", chunk_size=5)
        # "Hello", " Worl", "d!" = 3 content chunks + final
        assert len(events) == 4
        texts = [e["choices"][0]["delta"]["content"] for e in events[:-1]]
        assert "".join(texts) == "Hello World!"

    def test_first_chunk_has_role(self):
        events = text_to_sse_events("Hi", chunk_size=1)
        assert "role" in events[0]["choices"][0]["delta"]
        # Second chunk should not have role
        assert "role" not in events[1]["choices"][0]["delta"]

    def test_empty_text(self):
        events = text_to_sse_events("")
        # Even empty: content chunk + final
        assert len(events) == 2
        assert events[0]["choices"][0]["delta"]["content"] == ""

    def test_consistent_completion_id(self):
        events = text_to_sse_events("test", chunk_size=1)
        ids = {e["id"] for e in events}
        assert len(ids) == 1
