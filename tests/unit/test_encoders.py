"""Edge case tests for encoder functions (outbound response conversion).

Tests for:
- tool_calls_to_openai(), text_response_to_openai() — forge → OpenAI
- openai_to_anthropic_response(), openai_to_anthropic_sse() — OpenAI → Anthropic
- anthropic_to_openai_response(), anthropic_to_openai_sse() — Anthropic → OpenAI

Focuses on edge cases and boundary conditions not covered in
test_proxy_convert.py and test_anthropic_convert.py.
"""

from __future__ import annotations

import json

import pytest

from forge.core.workflow import ToolCall
from forge.proxy.convert import (
    anthropic_to_openai_response,
    anthropic_to_openai_sse,
    openai_to_anthropic_response,
    openai_to_anthropic_sse,
    text_response_to_openai,
    text_to_sse_events,
    tool_calls_to_openai,
    tool_calls_to_sse_events,
)

# ── tool_calls_to_openai edge cases ───────────────────────────


class TestToolCallsToOpenaiEdgeCases:
    def test_single_tool_call(self) -> None:
        result = tool_calls_to_openai([ToolCall(tool="search", args={"q": "test"})])
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1

    def test_multiple_tool_calls(self) -> None:
        calls = [
            ToolCall(tool="search", args={"q": "x"}),
            ToolCall(tool="fetch", args={"url": "http://x"}),
        ]
        result = tool_calls_to_openai(calls)
        assert len(result["choices"][0]["message"]["tool_calls"]) == 2

    def test_tool_call_with_reasoning(self) -> None:
        calls = [ToolCall(tool="search", args={"q": "x"}, reasoning="Let me check.")]
        result = tool_calls_to_openai(calls)
        assert result["choices"][0]["message"]["content"] == "Let me check."

    def test_tool_call_with_empty_reasoning(self) -> None:
        calls = [ToolCall(tool="search", args={"q": "x"}, reasoning=None)]
        result = tool_calls_to_openai(calls)
        assert result["choices"][0]["message"]["content"] is None

    def test_tool_call_arguments_serialized_as_json(self) -> None:
        calls = [ToolCall(tool="f", args={"nested": {"key": "val"}})]
        result = tool_calls_to_openai(calls)
        args = json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        assert args == {"nested": {"key": "val"}}

    def test_usage_fields_present(self) -> None:
        result = tool_calls_to_openai([ToolCall(tool="f", args={})])
        assert "prompt_tokens" in result["usage"]
        assert "completion_tokens" in result["usage"]
        assert "total_tokens" in result["usage"]

    def test_model_propagated(self) -> None:
        result = tool_calls_to_openai([ToolCall(tool="f", args={})], model="my-model")
        assert result["model"] == "my-model"

    def test_id_generation(self) -> None:
        result1 = tool_calls_to_openai([ToolCall(tool="f", args={})])
        result2 = tool_calls_to_openai([ToolCall(tool="f", args={})])
        # IDs should be unique across calls
        assert result1["id"] != result2["id"]


# ── text_response_to_openai edge cases ────────────────────────


class TestTextResponseToOpenaiEdgeCases:
    def test_empty_text(self) -> None:
        result = text_response_to_openai("")
        assert result["choices"][0]["message"]["content"] == ""
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_long_text(self) -> None:
        text = "x" * 10000
        result = text_response_to_openai(text)
        assert result["choices"][0]["message"]["content"] == text

    def test_unicode_text(self) -> None:
        result = text_response_to_openai("Hello 世界 🌍")
        assert result["choices"][0]["message"]["content"] == "Hello 世界 🌍"

    def test_multiline_text(self) -> None:
        text = "line1\nline2\nline3"
        result = text_response_to_openai(text)
        assert result["choices"][0]["message"]["content"] == text

    def test_model_propagated(self) -> None:
        result = text_response_to_openai("hi", model="custom")
        assert result["model"] == "custom"


# ── tool_calls_to_sse_events edge cases ───────────────────────


class TestToolCallsToSseEventsEdgeCases:
    def test_single_tool_call_sse(self) -> None:
        events = tool_calls_to_sse_events([ToolCall(tool="f", args={})])
        assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"

    def test_multiple_tool_calls_sse(self) -> None:
        calls = [ToolCall(tool="a", args={}), ToolCall(tool="b", args={})]
        events = tool_calls_to_sse_events(calls)
        # Should have one delta per tool call
        tool_chunks = [e for e in events if e["choices"][0].get("delta", {}).get("tool_calls")]
        assert len(tool_chunks) == 2

    def test_reasoning_sent_first_in_sse(self) -> None:
        calls = [ToolCall(tool="f", args={}, reasoning="Thinking...")]
        events = tool_calls_to_sse_events(calls)
        first = events[0]
        assert first["choices"][0]["delta"]["content"] == "Thinking..."
        assert first["choices"][0]["delta"]["role"] == "assistant"

    def test_sse_model_propagated(self) -> None:
        events = tool_calls_to_sse_events([ToolCall(tool="f", args={})], model="m")
        for e in events:
            assert e["model"] == "m"


# ── text_to_sse_events edge cases ─────────────────────────────


class TestTextToSseEventsEdgeCases:
    def test_empty_text_sse(self) -> None:
        events = text_to_sse_events("")
        assert events[-1]["choices"][0]["finish_reason"] == "stop"

    def test_chunked_text(self) -> None:
        events = text_to_sse_events("abcdefgh", chunk_size=3)
        # Should have multiple chunks + final
        content_chunks = [e for e in events if e["choices"][0].get("delta", {}).get("content")]
        assert len(content_chunks) > 1

    def test_chunked_text_shorter_than_chunk_size(self) -> None:
        events = text_to_sse_events("hi", chunk_size=10)
        content_chunks = [e for e in events if e["choices"][0].get("delta", {}).get("content")]
        assert len(content_chunks) == 1

    def test_sse_first_chunk_has_role(self) -> None:
        events = text_to_sse_events("hello")
        first = events[0]
        assert first["choices"][0]["delta"]["role"] == "assistant"


# ── openai_to_anthropic_response edge cases ───────────────────


class TestOpenaiToAnthropicResponseEdgeCases:
    def test_content_filter_finish_reason(self) -> None:
        """content_filter → stop_sequence mapping."""
        openai_resp = {
            "choices": [{"finish_reason": "content_filter"}],
            "usage": {},
        }
        result = openai_to_anthropic_response(openai_resp, "model")
        assert result["stop_reason"] == "stop_sequence"

    def test_multiple_tool_calls(self) -> None:
        openai_resp = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "call_001",
                                "function": {"name": "f1", "arguments": '{"a":1}'},
                            },
                            {
                                "type": "function",
                                "id": "call_002",
                                "function": {"name": "f2", "arguments": '{"b":2}'},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = openai_to_anthropic_response(openai_resp, "model")
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == "f1"
        assert result["content"][1]["name"] == "f2"

    def test_reasoning_before_tool_use(self) -> None:
        openai_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me check the weather.",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "call_001",
                                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        result = openai_to_anthropic_response(openai_resp, "model")
        assert result["content"][0] == {"type": "text", "text": "Let me check the weather."}
        assert result["content"][1]["type"] == "tool_use"
        assert result["content"][1]["name"] == "get_weather"

    def test_no_choices_raises(self) -> None:
        """Empty choices list → IndexError in current code."""
        openai_resp = {"choices": [], "usage": {}}
        with pytest.raises(IndexError):
            openai_to_anthropic_response(openai_resp, "model")

    def test_missing_usage_fields(self) -> None:
        openai_resp = {"choices": [{"finish_reason": "stop"}]}
        result = openai_to_anthropic_response(openai_resp, "model")
        assert result["usage"]["input_tokens"] == 0
        assert result["usage"]["output_tokens"] == 0

    def test_id_propagated(self) -> None:
        openai_resp = {
            "id": "chatcmpl-custom",
            "choices": [{"finish_reason": "stop"}],
            "usage": {},
        }
        result = openai_to_anthropic_response(openai_resp, "model")
        assert result["id"] == "chatcmpl-custom"

    def test_arguments_as_dict_not_double_serialized(self) -> None:
        openai_resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "c1",
                                "function": {"name": "f", "arguments": {"key": "val"}},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        result = openai_to_anthropic_response(openai_resp, "model")
        assert result["content"][0]["input"] == {"key": "val"}


# ── openai_to_anthropic_sse edge cases ────────────────────────


class TestOpenaiToAnthropicSseEdgeCases:
    def test_empty_events(self) -> None:
        result = openai_to_anthropic_sse([], "model")
        assert len(result) == 2  # message_start + message_stop
        assert result[0]["type"] == "message_start"
        assert result[-1]["type"] == "message_stop"

    def test_text_stream_accumulates(self) -> None:
        events = [
            {"id": "c1", "choices": [{"delta": {"content": "A"}, "finish_reason": None}]},
            {"id": "c1", "choices": [{"delta": {"content": "B"}, "finish_reason": None}]},
            {"id": "c1", "choices": [{"delta": {"content": "C"}, "finish_reason": "stop"}]},
        ]
        result = openai_to_anthropic_sse(events, "model")
        text_deltas = [e for e in result if e.get("delta", {}).get("type") == "text_delta"]
        texts = [e["delta"]["text"] for e in text_deltas]
        assert texts == ["A", "B", "C"]

    def test_tool_call_accumulates_args(self) -> None:
        """Tool call args accumulate and emit as single input_json_delta at final chunk."""
        events = [
            {
                "id": "c1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "call_001", "type": "function", "function": {"name": "f"}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "c1",
                "choices": [
                    {
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"a":'}}]},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "c1",
                "choices": [
                    {
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]
        result = openai_to_anthropic_sse(events, "model")
        input_deltas = [e for e in result if e.get("delta", {}).get("type") == "input_json_delta"]
        # Args accumulate → single delta at final chunk with full accumulated args
        assert len(input_deltas) == 1
        assert input_deltas[0]["delta"]["partial_json"] == '{"a":1}'

    def test_content_filter_in_finish_reason(self) -> None:
        events = [
            {"id": "c1", "choices": [{"delta": {"content": "x"}, "finish_reason": "content_filter"}]},
        ]
        result = openai_to_anthropic_sse(events, "model")
        message_stops = [e for e in result if e.get("type") == "message_stop"]
        # The content_filter maps to stop_sequence
        assert any(e.get("stop_reason") == "stop_sequence" for e in message_stops)

    def test_multiple_tool_calls_in_stream(self) -> None:
        events = [
            {
                "id": "c1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "f1"}}]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "c1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 1, "id": "c2", "type": "function", "function": {"name": "f2"}}]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {"id": "c1", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        result = openai_to_anthropic_sse(events, "model")
        tool_starts = [e for e in result if e.get("content_block", {}).get("type") == "tool_use"]
        assert len(tool_starts) == 2

    def test_wrapped_with_id_and_model(self) -> None:
        events = [{"id": "my-id", "choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}]
        result = openai_to_anthropic_sse(events, "my-model")
        for ev in result:
            if ev.get("type") not in ("message_stop",):
                assert ev.get("id") == "my-id"
                assert ev.get("model") == "my-model"


# ── anthropic_to_openai_response edge cases ───────────────────


class TestAnthropicToOpenaiResponseEdgeCases:
    def test_stop_sequence_stop_reason(self) -> None:
        resp = {"content": [{"type": "text", "text": "x"}], "stop_reason": "stop_sequence", "usage": {}}
        result = anthropic_to_openai_response(resp, "model")
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_multiple_tool_use_blocks(self) -> None:
        resp = {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "f1", "input": {"a": 1}},
                {"type": "tool_use", "id": "t2", "name": "f2", "input": {"b": 2}},
            ],
            "stop_reason": "tool_use",
            "usage": {},
        }
        result = anthropic_to_openai_response(resp, "model")
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 2
        assert tc[0]["function"]["name"] == "f1"
        assert tc[1]["function"]["name"] == "f2"

    def test_text_and_tool_together(self) -> None:
        resp = {
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
            ],
            "stop_reason": "tool_use",
            "usage": {},
        }
        result = anthropic_to_openai_response(resp, "model")
        msg = result["choices"][0]["message"]
        assert msg["content"] == "Let me check."
        assert len(msg["tool_calls"]) == 1

    def test_thinking_wrapped_in_tags(self) -> None:
        resp = {
            "content": [{"type": "thinking", "text": "I think..."}],
            "stop_reason": "end_turn",
            "usage": {},
        }
        result = anthropic_to_openai_response(resp, "model")
        assert result["choices"][0]["message"]["content"] == "<think>I think...</think>"

    def test_total_tokens_computed(self) -> None:
        resp = {
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = anthropic_to_openai_response(resp, "model")
        assert result["usage"]["total_tokens"] == 15

    def test_missing_usage_defaults_to_zero(self) -> None:
        resp = {"content": [], "stop_reason": "end_turn"}
        result = anthropic_to_openai_response(resp, "model")
        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0
        assert result["usage"]["total_tokens"] == 0

    def test_id_propagated(self) -> None:
        resp = {"content": [], "stop_reason": "end_turn", "id": "msg-custom", "usage": {}}
        result = anthropic_to_openai_response(resp, "model")
        assert result["id"] == "msg-custom"

    def test_arguments_serialized_as_json_string(self) -> None:
        resp = {
            "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {"nested": {"k": "v"}}}],
            "stop_reason": "tool_use",
            "usage": {},
        }
        result = anthropic_to_openai_response(resp, "model")
        args = json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        assert args == {"nested": {"k": "v"}}


# ── anthropic_to_openai_sse edge cases ────────────────────────


class TestAnthropicToOpenaiSseEdgeCases:
    def test_empty_events(self) -> None:
        result = anthropic_to_openai_sse([], "model")
        assert result == []

    def test_text_stream_generates_chunks(self) -> None:
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop", "stop_reason": "end_turn"},
        ]
        result = anthropic_to_openai_sse(events, "model")
        content_chunks = [e for e in result if e["choices"][0].get("delta", {}).get("content")]
        assert len(content_chunks) == 1
        assert content_chunks[0]["choices"][0]["delta"]["content"] == "Hello"

    def test_tool_stream_generates_chunks(self) -> None:
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "f"}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"a":1}'},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop", "stop_reason": "tool_use"},
        ]
        result = anthropic_to_openai_sse(events, "model")
        tool_chunks = [e for e in result if "tool_calls" in e.get("choices", [{}])[0].get("delta", {})]
        assert len(tool_chunks) >= 1
        assert tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"a":1}'

    def test_stop_reason_max_tokens_maps_to_length(self) -> None:
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop", "stop_reason": "max_tokens"},
        ]
        result = anthropic_to_openai_sse(events, "model")
        assert result[-1]["choices"][0]["finish_reason"] == "length"

    def test_stop_reason_stop_sequence_maps_to_stop(self) -> None:
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop", "stop_reason": "stop_sequence"},
        ]
        result = anthropic_to_openai_sse(events, "model")
        assert result[-1]["choices"][0]["finish_reason"] == "stop"

    def test_fallback_single_chunk_for_text_only(self) -> None:
        """Events with text content but no proper delta events get a fallback."""
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
            {"type": "message_stop", "stop_reason": "end_turn"},
        ]
        result = anthropic_to_openai_sse(events, "model")
        # Should have content chunks + final chunk
        assert len(result) >= 2

    def test_model_propagated_in_chunks(self) -> None:
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop", "stop_reason": "end_turn"},
        ]
        result = anthropic_to_openai_sse(events, "my-model")
        for e in result:
            assert e["model"] == "my-model"

    def test_multiple_content_blocks(self) -> None:
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "A"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "f"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_stop", "stop_reason": "tool_use"},
        ]
        result = anthropic_to_openai_sse(events, "model")
        content_chunks = [e for e in result if e["choices"][0].get("delta", {}).get("content")]
        tool_chunks = [e for e in result if "tool_calls" in e.get("choices", [{}])[0].get("delta", {})]
        assert len(content_chunks) == 1
        assert len(tool_chunks) >= 1
