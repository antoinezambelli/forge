"""Tests for OpenAI wire format <-> forge type conversion."""

import json

import pytest

from forge.core.messages import MessageRole, MessageType
from forge.core.workflow import TextResponse, ToolCall
from forge.proxy.convert import (
    forge_messages_to_openai,
    openai_messages_to_forge,
    parse_batch_response,
    parse_streamed_response,
    synthesize_batch_tool_calls,
    synthesize_sse_tool_calls,
)


# -- openai_messages_to_forge ----------------------------------


class TestOpenaiToForge:
    def test_system_message(self):
        msgs = openai_messages_to_forge([
            {"role": "system", "content": "You are helpful."},
        ])
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.SYSTEM
        assert msgs[0].metadata.type == MessageType.SYSTEM_PROMPT
        assert msgs[0].content == "You are helpful."

    def test_user_message(self):
        msgs = openai_messages_to_forge([
            {"role": "user", "content": "Hello"},
        ])
        assert msgs[0].role == MessageRole.USER
        assert msgs[0].metadata.type == MessageType.USER_INPUT

    def test_assistant_text(self):
        msgs = openai_messages_to_forge([
            {"role": "assistant", "content": "Hi there"},
        ])
        assert msgs[0].role == MessageRole.ASSISTANT
        assert msgs[0].metadata.type == MessageType.TEXT_RESPONSE
        assert msgs[0].content == "Hi there"

    def test_assistant_tool_calls(self):
        msgs = openai_messages_to_forge([
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_0", "type": "function",
                 "function": {"name": "search", "arguments": '{"q": "test"}'}},
            ]},
        ])
        assert msgs[0].role == MessageRole.ASSISTANT
        assert msgs[0].metadata.type == MessageType.TOOL_CALL
        assert len(msgs[0].tool_calls) == 1
        assert msgs[0].tool_calls[0].name == "search"
        assert msgs[0].tool_calls[0].args == {"q": "test"}

    def test_tool_result(self):
        msgs = openai_messages_to_forge([
            {"role": "tool", "content": "result data",
             "name": "search", "tool_call_id": "call_0"},
        ])
        assert msgs[0].role == MessageRole.TOOL
        assert msgs[0].metadata.type == MessageType.TOOL_RESULT
        assert msgs[0].tool_name == "search"
        assert msgs[0].tool_call_id == "call_0"

    def test_full_conversation(self):
        msgs = openai_messages_to_forge([
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
            ]},
            {"role": "tool", "content": "72F", "name": "get_weather", "tool_call_id": "c1"},
            {"role": "assistant", "content": "It's 72F in Paris."},
        ])
        types = [m.metadata.type for m in msgs]
        assert types == [
            MessageType.SYSTEM_PROMPT,
            MessageType.USER_INPUT,
            MessageType.TOOL_CALL,
            MessageType.TOOL_RESULT,
            MessageType.TEXT_RESPONSE,
        ]

    def test_null_content_handled(self):
        msgs = openai_messages_to_forge([
            {"role": "assistant", "content": None},
        ])
        assert msgs[0].content == ""

    def test_reasoning_content_field(self):
        """Assistant message with explicit reasoning_content field."""
        msgs = openai_messages_to_forge([
            {"role": "assistant", "reasoning_content": "Let me think...", "content": "Hello!"},
        ])
        assert len(msgs) == 2
        assert msgs[0].metadata.type == MessageType.REASONING
        assert msgs[0].content == "Let me think..."
        assert msgs[1].metadata.type == MessageType.TEXT_RESPONSE
        assert msgs[1].content == "Hello!"

    def test_think_tags_in_content(self):
        """Assistant message with [THINK] tags in content (batch format)."""
        msgs = openai_messages_to_forge([
            {"role": "assistant", "content": "[THINK]reasoning here[/THINK]\nHello!"},
        ])
        assert len(msgs) == 2
        assert msgs[0].metadata.type == MessageType.REASONING
        assert "[THINK]" in msgs[0].content
        assert msgs[1].metadata.type == MessageType.TEXT_RESPONSE
        assert msgs[1].content == "Hello!"

    def test_xml_think_tags(self):
        """Assistant message with <think> tags (Qwen/DeepSeek style)."""
        msgs = openai_messages_to_forge([
            {"role": "assistant", "content": "<think>hmm</think>\nResult"},
        ])
        assert len(msgs) == 2
        assert msgs[0].metadata.type == MessageType.REASONING
        assert msgs[1].content == "Result"

    def test_reasoning_only_no_content(self):
        """reasoning_content present but no text content."""
        msgs = openai_messages_to_forge([
            {"role": "assistant", "reasoning_content": "thinking...", "content": ""},
        ])
        assert len(msgs) == 1
        assert msgs[0].metadata.type == MessageType.REASONING

    def test_no_reasoning_plain_text(self):
        """Plain text without think tags stays TEXT_RESPONSE."""
        msgs = openai_messages_to_forge([
            {"role": "assistant", "content": "Just a regular response"},
        ])
        assert len(msgs) == 1
        assert msgs[0].metadata.type == MessageType.TEXT_RESPONSE


class TestStreamedReasoning:
    def test_captures_reasoning_content_deltas(self):
        chunks = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"Let me "}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"think..."}}]}',
            'data: {"choices":[{"delta":{"content":"Hello!"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        response, reasoning, usage = parse_streamed_response(chunks)
        assert isinstance(response, TextResponse)
        assert response.content == "Hello!"
        assert reasoning == "Let me think..."

    def test_no_reasoning_returns_none(self):
        chunks = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        response, reasoning, usage = parse_streamed_response(chunks)
        assert reasoning is None


class TestForgeToOpenai:
    def test_round_trips_system(self):
        original = [{"role": "system", "content": "Be helpful"}]
        msgs = openai_messages_to_forge(original)
        result = forge_messages_to_openai(msgs)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"

    def test_round_trips_tool_call(self):
        original = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "search", "arguments": '{"q": "a"}'}},
            ]},
        ]
        msgs = openai_messages_to_forge(original)
        result = forge_messages_to_openai(msgs)
        tc = result[0]["tool_calls"][0]
        assert tc["function"]["name"] == "search"
        assert tc["type"] == "function"
        assert tc["id"] == "c1"

    def test_round_trips_tool_result(self):
        original = [
            {"role": "tool", "content": "data", "name": "search", "tool_call_id": "c1"},
        ]
        msgs = openai_messages_to_forge(original)
        result = forge_messages_to_openai(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "data"
        assert result[0]["name"] == "search"
        assert result[0]["tool_call_id"] == "c1"


# -- synthesize_batch_tool_calls --------------------------------


class TestSynthesizeBatch:
    def test_synthesizes_batch_response(self):
        tool_calls = [ToolCall(tool="get_weather", args={"city": "Paris"})]
        raw = synthesize_batch_tool_calls(tool_calls, model="test")
        data = json.loads(raw)
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert data["choices"][0]["finish_reason"] == "tool_calls"


# -- parse_batch_response --------------------------------------


class TestParseBatchResponse:
    def test_parses_tool_calls(self):
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }],
                },
            }],
        }
        result = parse_batch_response(data)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].tool == "get_weather"
        assert result[0].args == {"city": "Paris"}

    def test_parses_multiple_tool_calls(self):
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "search", "arguments": '{"q": "a"}'}},
                        {"function": {"name": "lookup", "arguments": '{"id": 1}'}},
                    ],
                },
            }],
        }
        result = parse_batch_response(data)
        assert len(result) == 2
        assert result[0].tool == "search"
        assert result[1].tool == "lookup"

    def test_parses_text_response(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
            }],
        }
        result = parse_batch_response(data)
        assert isinstance(result, TextResponse)
        assert result.content == "Hello!"

    def test_empty_content_returns_text_response(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": None},
            }],
        }
        result = parse_batch_response(data)
        assert isinstance(result, TextResponse)
        assert result.content == ""


# -- parse_streamed_response -----------------------------------


class TestParseStreamedResponse:
    def test_parses_streamed_tool_calls(self):
        chunks = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\": \\"Paris\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}',
            "data: [DONE]",
        ]
        response, _reasoning, usage = parse_streamed_response(chunks)
        assert isinstance(response, list)
        assert len(response) == 1
        assert response[0].tool == "get_weather"
        assert response[0].args == {"city": "Paris"}
        assert usage["prompt_tokens"] == 10

    def test_parses_streamed_text(self):
        chunks = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"Hello "}}]}',
            'data: {"choices":[{"delta":{"content":"world"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        response, _reasoning, usage = parse_streamed_response(chunks)
        assert isinstance(response, TextResponse)
        assert response.content == "Hello world"

    def test_parses_multiple_streamed_tool_calls(self):
        chunks = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_0","function":{"name":"search","arguments":"{\\"q\\":\\"a\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_1","function":{"name":"lookup","arguments":"{\\"id\\":1}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
        response, _reasoning, usage = parse_streamed_response(chunks)
        assert isinstance(response, list)
        assert len(response) == 2
        assert response[0].tool == "search"
        assert response[1].tool == "lookup"

    def test_handles_malformed_json_gracefully(self):
        chunks = [
            "data: not json",
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ]
        response, _reasoning, usage = parse_streamed_response(chunks)
        assert isinstance(response, TextResponse)
        assert response.content == "ok"

    def test_empty_chunks(self):
        response, _reasoning, usage = parse_streamed_response([])
        assert isinstance(response, TextResponse)
        assert response.content == ""
        assert usage is None


# -- synthesize_sse_tool_calls ---------------------------------


class TestSynthesizeSse:
    def test_synthesizes_single_tool_call(self):
        tool_calls = [ToolCall(tool="get_weather", args={"city": "Paris"})]
        chunks = synthesize_sse_tool_calls(tool_calls, model="test-model")

        assert len(chunks) == 4  # role, tool_calls, finish, [DONE]
        assert chunks[-1] == b"data: [DONE]\n\n"

        # Parse the tool call chunk
        tc_chunk = json.loads(chunks[1].decode().removeprefix("data: ").strip())
        tc = tc_chunk["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}

    def test_synthesizes_multiple_tool_calls(self):
        tool_calls = [
            ToolCall(tool="search", args={"q": "a"}),
            ToolCall(tool="lookup", args={"id": 1}),
        ]
        chunks = synthesize_sse_tool_calls(tool_calls)

        tc_chunk = json.loads(chunks[1].decode().removeprefix("data: ").strip())
        tcs = tc_chunk["choices"][0]["delta"]["tool_calls"]
        assert len(tcs) == 2
        assert tcs[0]["function"]["name"] == "search"
        assert tcs[1]["function"]["name"] == "lookup"

    def test_finish_reason_is_tool_calls(self):
        tool_calls = [ToolCall(tool="x", args={})]
        chunks = synthesize_sse_tool_calls(tool_calls)

        finish_chunk = json.loads(chunks[2].decode().removeprefix("data: ").strip())
        assert finish_chunk["choices"][0]["finish_reason"] == "tool_calls"
