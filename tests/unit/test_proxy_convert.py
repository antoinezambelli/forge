"""Tests for OpenAI wire format <-> forge type conversion."""

import json

import pytest

from forge.core.workflow import TextResponse, ToolCall
from forge.proxy.convert import (
    parse_batch_response,
    parse_streamed_response,
    synthesize_sse_tool_calls,
)


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
        response, usage = parse_streamed_response(chunks)
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
        response, usage = parse_streamed_response(chunks)
        assert isinstance(response, TextResponse)
        assert response.content == "Hello world"

    def test_parses_multiple_streamed_tool_calls(self):
        chunks = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_0","function":{"name":"search","arguments":"{\\"q\\":\\"a\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_1","function":{"name":"lookup","arguments":"{\\"id\\":1}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
        response, usage = parse_streamed_response(chunks)
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
        response, usage = parse_streamed_response(chunks)
        assert isinstance(response, TextResponse)
        assert response.content == "ok"

    def test_empty_chunks(self):
        response, usage = parse_streamed_response([])
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
