from forge.core.messages import ReasoningPart, TextPart, ConversationState
from forge.clients.base import RequestMetadata
from forge.clients.requests import request_extras
from tests.canonical import assistant, call

"""Tests for proxy message conversion (OpenAI ↔ forge)."""
import json
from forge.proxy.convert import (
    openai_to_messages,
    reply_to_openai,
    text_response_to_openai,
    reply_to_sse_events,
    text_to_sse_events,
)


class TestToolCallsToOpenai:
    def test_single_tool_call(self):
        result = reply_to_openai(
            assistant(call(name="search", arguments={"q": "test"}))
        )
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
        result = reply_to_openai(
            assistant(
                call(name="search", arguments={"q": "a"}),
                call(name="fetch", arguments={"url": "b"}),
            )
        )
        assert len(result["choices"][0]["message"]["tool_calls"]) == 2

    def test_reasoning_omitted_by_default(self):
        result = reply_to_openai(
            assistant(
                ReasoningPart("Let me think..."), call(name="search", arguments={})
            )
        )
        msg = result["choices"][0]["message"]
        assert msg["content"] is None
        assert "reasoning_content" not in msg

    def test_keep_last_reasoning_replay_exposed_as_reasoning_content(self):
        result = reply_to_openai(
            assistant(
                ReasoningPart("Let me think..."), call(name="search", arguments={})
            ),
            reasoning_replay="keep-last",
        )
        msg = result["choices"][0]["message"]
        assert msg["content"] is None
        assert msg["reasoning_content"] == "Let me think..."

    def test_full_reasoning_replay_exposes_reasoning_in_content(self):
        result = reply_to_openai(
            assistant(
                ReasoningPart("Let me think..."), call(name="search", arguments={})
            ),
            reasoning_replay="full",
        )
        assert result["choices"][0]["message"]["content"] == "Let me think..."

    def test_none_reasoning_replay_omits_reasoning(self):
        result = reply_to_openai(
            assistant(
                ReasoningPart("Let me think..."), call(name="search", arguments={})
            ),
            reasoning_replay="none",
        )
        msg = result["choices"][0]["message"]
        assert msg["content"] is None
        assert "reasoning_content" not in msg

    def test_no_reasoning_content_is_none(self):
        result = reply_to_openai(assistant(call(name="search", arguments={})))
        assert result["choices"][0]["message"]["content"] is None

    def test_model_name_propagated(self):
        result = reply_to_openai(
            assistant(call(name="search", arguments={})), model="my-model"
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


class TestToolCallsToSseEvents:
    def test_single_tool_call_structure(self):
        events = reply_to_sse_events(
            assistant(call(name="search", arguments={"q": "x"}))
        )
        assert len(events) == 2
        delta = events[0]["choices"][0]["delta"]
        assert "tool_calls" in delta
        assert delta["tool_calls"][0]["function"]["name"] == "search"
        assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert events[-1]["choices"][0]["delta"] == {}

    def test_reasoning_omitted_from_stream_by_default(self):
        events = reply_to_sse_events(
            assistant(ReasoningPart("Thinking..."), call(name="search", arguments={}))
        )
        assert len(events) == 2
        assert "tool_calls" in events[0]["choices"][0]["delta"]

    def test_keep_last_reasoning_replay_streams_reasoning_content_delta(self):
        events = reply_to_sse_events(
            assistant(ReasoningPart("Thinking..."), call(name="search", arguments={})),
            reasoning_replay="keep-last",
        )
        assert len(events) == 3
        assert events[0]["choices"][0]["delta"]["reasoning_content"] == "Thinking..."

    def test_full_reasoning_replay_streams_content_delta(self):
        events = reply_to_sse_events(
            assistant(ReasoningPart("Thinking..."), call(name="search", arguments={})),
            reasoning_replay="full",
        )
        assert events[0]["choices"][0]["delta"]["content"] == "Thinking..."

    def test_none_reasoning_replay_omits_stream_reasoning_delta(self):
        events = reply_to_sse_events(
            assistant(ReasoningPart("Thinking..."), call(name="search", arguments={})),
            reasoning_replay="none",
        )
        assert len(events) == 2
        assert "tool_calls" in events[0]["choices"][0]["delta"]

    def test_multiple_tool_calls(self):
        events = reply_to_sse_events(
            assistant(call(name="a", arguments={}), call(name="b", arguments={}))
        )
        assert len(events) == 3

    def test_consistent_completion_id(self):
        events = reply_to_sse_events(assistant(call(name="a", arguments={})))
        ids = {e["id"] for e in events}
        assert len(ids) == 1


class TestTextToSseEvents:
    def test_single_chunk(self):
        events = text_to_sse_events("Hello!")
        assert len(events) == 2
        assert events[0]["choices"][0]["delta"]["content"] == "Hello!"
        assert events[0]["choices"][0]["delta"]["role"] == "assistant"
        assert events[-1]["choices"][0]["finish_reason"] == "stop"

    def test_chunked_text(self):
        events = text_to_sse_events("Hello World!", chunk_size=5)
        assert len(events) == 4
        texts = [e["choices"][0]["delta"]["content"] for e in events[:-1]]
        assert "".join(texts) == "Hello World!"

    def test_first_chunk_has_role(self):
        events = text_to_sse_events("Hi", chunk_size=1)
        assert "role" in events[0]["choices"][0]["delta"]
        assert "role" not in events[1]["choices"][0]["delta"]

    def test_empty_text(self):
        events = text_to_sse_events("")
        assert len(events) == 2
        assert events[0]["choices"][0]["delta"]["content"] == ""

    def test_consistent_completion_id(self):
        events = text_to_sse_events("test", chunk_size=1)
        ids = {e["id"] for e in events}
        assert len(ids) == 1
