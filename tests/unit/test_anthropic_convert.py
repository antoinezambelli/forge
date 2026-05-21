"""Tests for forge.proxy.convert — Anthropic ↔ OpenAI conversion functions.

All tests exercise the 6 conversion functions directly.
No API calls or mocks needed.
"""

from __future__ import annotations

import json
from typing import Any

from forge.proxy.convert import (
    anthropic_to_openai_messages,
    anthropic_to_openai_response,
    anthropic_to_openai_sse,
    openai_to_anthropic_messages,
    openai_to_anthropic_response,
    openai_to_anthropic_sse,
)


# ── anthropic_to_openai_messages ──────────────────────────────────


class TestAnthropicToOpenAIMessages:
    """Convert Anthropic API request body → OpenAI messages list."""

    def test_simple_text(self) -> None:
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_openai_messages(body)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_system_string(self) -> None:
        body: dict[str, Any] = {
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_openai_messages(body)
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "Hi"}

    def test_assistant_tool_use(self) -> None:
        body: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_001", "name": "get_weather", "input": {"city": "Paris"}},
                    ],
                },
            ],
        }
        result = anthropic_to_openai_messages(body)
        assert len(result) == 2
        tc = result[1]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}
        assert tc["id"] == "toolu_001"

    def test_assistant_text_and_tool(self) -> None:
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "test"}},
                    ],
                },
            ],
        }
        result = anthropic_to_openai_messages(body)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me check."
        assert len(result[0]["tool_calls"]) == 1

    def test_tool_result_becomes_tool_role(self) -> None:
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "22C"},
                    ],
                },
            ],
        }
        result = anthropic_to_openai_messages(body)
        assert result[0] == {
            "role": "tool",
            "tool_call_id": "t1",
            "content": "22C",
            "name": "",
        }

    def test_thinking_block(self) -> None:
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "text": "I think..."}],
                },
            ],
        }
        result = anthropic_to_openai_messages(body)
        assert result[0]["content"] == "<think>I think...</think>"

    def test_full_conversation(self) -> None:
        body: dict[str, Any] = {
            "system": "You are a weather bot.",
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "22C sunny"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "It's 22C and sunny in Paris."}],
                },
            ],
        }
        result = anthropic_to_openai_messages(body)
        assert len(result) == 5
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert "tool_calls" in result[2]
        assert result[3]["role"] == "tool"
        assert result[4]["role"] == "assistant"

    def test_empty_messages(self) -> None:
        result = anthropic_to_openai_messages({})
        assert result == []

    def test_string_content(self) -> None:
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": "plain text"}],
        }
        result = anthropic_to_openai_messages(body)
        assert result == [{"role": "user", "content": "plain text"}]


# ── openai_to_anthropic_messages ──────────────────────────────────


class TestOpenAIToAnthropicMessages:
    """Convert OpenAI messages list → Anthropic body fields."""

    def test_simple_text(self) -> None:
        msgs = [{"role": "user", "content": "Hello"}]
        result = openai_to_anthropic_messages(msgs)
        assert result == {"messages": [{"role": "user", "content": "Hello"}]}

    def test_system_extracted(self) -> None:
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = openai_to_anthropic_messages(msgs)
        assert result["system"] == "Be helpful."
        assert len(result["messages"]) == 1

    def test_tool_call_conversion(self) -> None:
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {
                    "type": "function",
                    "id": "call_001",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ]},
        ]
        result = openai_to_anthropic_messages(msgs)
        block = result["messages"][0]["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "call_001"
        assert block["name"] == "get_weather"
        assert block["input"] == {"city": "Paris"}

    def test_tool_result_becomes_user_with_block(self) -> None:
        msgs = [
            {"role": "tool", "tool_call_id": "call_001", "content": "22C"},
        ]
        result = openai_to_anthropic_messages(msgs)
        assert result["messages"][0]["role"] == "user"
        block = result["messages"][0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call_001"
        assert block["content"] == "22C"

    def test_tool_result_with_error(self) -> None:
        msgs = [
            {"role": "tool", "tool_call_id": "call_001", "content": "error", "is_error": True},
        ]
        result = openai_to_anthropic_messages(msgs)
        block = result["messages"][0]["content"][0]
        assert block["is_error"] is True

    def test_assistant_text_with_tool(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {"type": "function", "id": "t1", "function": {"name": "search", "arguments": '{"q":"x"}'}},
                ],
            },
        ]
        result = openai_to_anthropic_messages(msgs)
        blocks = result["messages"][0]["content"]
        assert len(blocks) == 2
        assert blocks[0] == {"type": "text", "text": "Let me check."}
        assert blocks[1]["type"] == "tool_use"

    def test_arguments_as_dict(self) -> None:
        """Arguments already parsed as dict still works."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "id": "t1", "function": {"name": "f", "arguments": {"key": "val"}}},
                ],
            },
        ]
        result = openai_to_anthropic_messages(msgs)
        assert result["messages"][0]["content"][0]["input"] == {"key": "val"}

    def test_content_as_list_normalized(self) -> None:
        """User message with content as list of blocks gets normalized."""
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "thinking", "text": "Hmm"}]},
        ]
        result = openai_to_anthropic_messages(msgs)
        assert result["messages"][0]["content"] == [
            {"type": "text", "text": "Hello"},
            {"type": "thinking", "text": "Hmm"},
        ]

    def test_empty_content(self) -> None:
        msgs = [{"role": "assistant", "content": ""}]
        result = openai_to_anthropic_messages(msgs)
        assert result["messages"][0]["content"] == ""


# ── openai_to_anthropic_response ──────────────────────────────────


class TestOpenAItoAnthropicResponse:
    """Convert OpenAI non-streaming response → Anthropic response."""

    def test_text_response(self) -> None:
        openai_resp = {
            "id": "chatcmpl-abc",
            "choices": [{"message": {"content": "Hello!", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = openai_to_anthropic_response(openai_resp, "claude-sonnet-4")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-sonnet-4"
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert result["content"][0] == {"type": "text", "text": "Hello!"}
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_call_response(self) -> None:
        openai_resp = {
            "id": "chatcmpl-xyz",
            "choices": [{
                "message": {
                    "content": "Checking...",
                    "role": "assistant",
                    "tool_calls": [{
                        "type": "function",
                        "id": "call_001",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        result = openai_to_anthropic_response(openai_resp, "claude-haiku")
        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "Checking..."}
        tc = result["content"][1]
        assert tc["type"] == "tool_use"
        assert tc["name"] == "get_weather"
        assert tc["input"] == {"city": "Paris"}

    def test_finish_reason_mapping(self) -> None:
        """Verify all finish_reason → stop_reason mappings."""
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }
        for finish, expected_stop in mapping.items():
            openai_resp = {
                "choices": [{"finish_reason": finish}],
                "usage": {},
            }
            result = openai_to_anthropic_response(openai_resp, "model")
            assert result["stop_reason"] == expected_stop, f"Failed for finish_reason={finish}"

    def test_empty_content(self) -> None:
        openai_resp = {
            "choices": [{"message": {"content": "", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {},
        }
        result = openai_to_anthropic_response(openai_resp, "model")
        assert result["content"][0] == {"type": "text", "text": ""}


# ── openai_to_anthropic_sse ──────────────────────────────────────


class TestOpenAItoAnthropicSSE:
    """Convert OpenAI SSE chunks → Anthropic SSE events."""

    def test_text_stream(self) -> None:
        events = [
            {"id": "chatcmpl-1", "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"id": "chatcmpl-1", "choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
        ]
        result = openai_to_anthropic_sse(events, "claude-sonnet-4")
        # Should have message_start, content blocks, message_stop
        assert result[0]["type"] == "message_start"
        # Find message_stop
        stop_events = [e for e in result if e.get("type") == "message_stop"]
        assert len(stop_events) >= 1
        # Check text delta events exist
        text_deltas = [e for e in result if e.get("delta", {}).get("type") == "text_delta"]
        assert len(text_deltas) >= 2  # "Hello" and " world"

    def test_tool_call_stream(self) -> None:
        events = [
            {"id": "chatcmpl-2", "choices": [{
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call_001",
                    "type": "function",
                    "function": {"name": "get_weather"},
                }]},
                "finish_reason": None,
            }]},
            {"id": "chatcmpl-2", "choices": [{
                "delta": {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": '{"city": "'},
                }]},
                "finish_reason": None,
            }]},
            {"id": "chatcmpl-2", "choices": [{
                "delta": {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": 'Paris"}'},
                }]},
                "finish_reason": "tool_calls",
            }]},
        ]
        result = openai_to_anthropic_sse(events, "model")
        # Should have tool_use content blocks
        tool_starts = [e for e in result if e.get("content_block", {}).get("type") == "tool_use"]
        assert len(tool_starts) >= 1
        input_deltas = [e for e in result if e.get("delta", {}).get("type") == "input_json_delta"]
        assert len(input_deltas) >= 1

    def test_wrapped_with_id_and_model(self) -> None:
        events = [
            {"id": "chatcmpl-3", "choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]},
        ]
        result = openai_to_anthropic_sse(events, "claude-haiku")
        for ev in result:
            assert ev.get("id") == "chatcmpl-3"
            if ev.get("type") not in ("message_stop",):
                assert ev.get("model") == "claude-haiku"

    def test_message_stop_at_end(self) -> None:
        events = [
            {"id": "chatcmpl-4", "choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]},
        ]
        result = openai_to_anthropic_sse(events, "model")
        assert result[-1]["type"] == "message_stop"


# ── anthropic_to_openai_response ──────────────────────────────────


class TestAnthropicToOpenAIResponse:
    """Convert Anthropic non-streaming response → OpenAI response."""

    def test_text_response(self) -> None:
        anthropic_resp = {
            "id": "msg_abc",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = anthropic_to_openai_response(anthropic_resp, "claude-sonnet-4")
        assert result["object"] == "chat.completion"
        assert result["model"] == "claude-sonnet-4"
        assert result["choices"][0]["finish_reason"] == "stop"
        msg = result["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_tool_call_response(self) -> None:
        anthropic_resp = {
            "id": "msg_xyz",
            "content": [
                {"type": "tool_use", "id": "toolu_001", "name": "get_weather", "input": {"city": "Paris"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 15, "output_tokens": 8},
        }
        result = anthropic_to_openai_response(anthropic_resp, "model")
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        tc = result["choices"][0]["message"]["tool_calls"][0]
        assert tc["id"] == "toolu_001"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}

    def test_text_and_tool(self) -> None:
        anthropic_resp = {
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
            ],
            "stop_reason": "tool_use",
            "usage": {},
        }
        result = anthropic_to_openai_response(anthropic_resp, "model")
        msg = result["choices"][0]["message"]
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        assert msg["content"] == "Let me check."

    def test_thinking_block(self) -> None:
        anthropic_resp = {
            "content": [{"type": "thinking", "text": "I think..."}],
            "stop_reason": "end_turn",
            "usage": {},
        }
        result = anthropic_to_openai_response(anthropic_resp, "model")
        assert result["choices"][0]["message"]["content"] == "<think>I think...</think>"

    def test_stop_reason_mapping(self) -> None:
        for stop, expected_finish in [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_calls"),
        ]:
            resp = {"content": [], "stop_reason": stop, "usage": {}}
            result = anthropic_to_openai_response(resp, "model")
            assert result["choices"][0]["finish_reason"] == expected_finish, f"Failed for stop_reason={stop}"

    def test_empty_response(self) -> None:
        anthropic_resp = {
            "content": [],
            "stop_reason": "end_turn",
            "usage": {},
        }
        result = anthropic_to_openai_response(anthropic_resp, "model")
        assert result["choices"][0]["message"]["content"] == ""


# ── anthropic_to_openai_sse ──────────────────────────────────────


class TestAnthropicToOpenAISSE:
    """Convert Anthropic SSE events → OpenAI SSE chunks."""

    def test_text_stream(self) -> None:
        events = [
            {"type": "message_start", "model": "claude-sonnet-4"},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop", "stop_reason": "end_turn"},
        ]
        result = anthropic_to_openai_sse(events, "claude-sonnet-4")
        # Should have chunks with content deltas
        content_chunks = [e for e in result if e.get("choices", [{}])[0].get("delta", {}).get("content")]
        assert len(content_chunks) >= 2  # "Hello" and " world"
        # Final chunk should have finish_reason
        final = result[-1]
        assert final["choices"][0]["finish_reason"] == "stop"

    def test_tool_call_stream(self) -> None:
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_001", "name": "get_weather"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"city": "'}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": 'Paris"}'}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop", "stop_reason": "tool_use"},
        ]
        result = anthropic_to_openai_sse(events, "model")
        tool_chunks = [e for e in result if "tool_calls" in e.get("choices", [{}])[0].get("delta", {})]
        assert len(tool_chunks) >= 2

    def test_stop_reason_mapping(self) -> None:
        for stop, expected_finish in [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_calls"),
        ]:
            events = [
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_stop", "stop_reason": stop},
            ]
            result = anthropic_to_openai_sse(events, "model")
            final = result[-1]
            assert final["choices"][0]["finish_reason"] == expected_finish, f"Failed for stop_reason={stop}"

    def test_empty_stream(self) -> None:
        events: list[dict[str, Any]] = []
        result = anthropic_to_openai_sse(events, "model")
        assert result == []
