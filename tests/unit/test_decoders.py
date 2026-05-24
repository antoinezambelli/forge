"""Edge case tests for decoder functions (inbound message conversion).

Tests for:
- openai_to_messages() — OpenAI → forge Messages
- anthropic_to_openai_messages() — Anthropic body → OpenAI messages
- openai_to_anthropic_messages() — OpenAI messages → Anthropic body fields

Focuses on edge cases and boundary conditions not covered in the
main converter test files (test_proxy_convert.py, test_anthropic_convert.py).
"""

from __future__ import annotations

from typing import Any

from forge.core.messages import MessageRole, MessageType
from forge.proxy.convert import (
    anthropic_to_openai_messages,
    openai_to_anthropic_messages,
    openai_to_messages,
)

# ── openai_to_messages edge cases ──────────────────────────────


class TestOpenaiToMessagesEdgeCases:
    """Edge cases for openai_to_messages not covered in test_proxy_convert.py."""

    def test_empty_list(self) -> None:
        assert openai_to_messages([]) == []

    def test_none_content_becomes_empty_string(self) -> None:
        msgs = [{"role": "user", "content": None}]
        result = openai_to_messages(msgs)
        assert len(result) == 1
        assert result[0].content == ""

    def test_missing_content_key(self) -> None:
        msgs = [{"role": "user"}]
        result = openai_to_messages(msgs)
        assert len(result) == 1
        assert result[0].content == ""

    def test_unknown_role_maps_to_user(self) -> None:
        msgs = [{"role": "fake_role", "content": "data"}]
        result = openai_to_messages(msgs)
        assert result[0].role == MessageRole.USER
        assert result[0].metadata.type == MessageType.USER_INPUT

    def test_missing_role_defaults_to_user(self) -> None:
        msgs = [{"content": "no role"}]
        result = openai_to_messages(msgs)
        assert result[0].role == MessageRole.USER

    def test_list_content_with_mixed_types(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    "plain string",
                    {"type": "text", "text": "second"},
                ],
            }
        ]
        result = openai_to_messages(msgs)
        assert result[0].content == "first\nplain string\nsecond"

    def test_list_content_with_non_text_blocks_ignored(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "visible"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        result = openai_to_messages(msgs)
        # Only text blocks are extracted; image_url blocks contribute nothing
        assert result[0].content == "visible"

    def test_assistant_with_empty_tool_calls(self) -> None:
        """Assistant with tool_calls=[] should be treated as text response."""
        msgs = [{"role": "assistant", "content": "just text", "tool_calls": []}]
        result = openai_to_messages(msgs)
        assert len(result) == 1
        assert result[0].metadata.type == MessageType.TEXT_RESPONSE

    def test_assistant_with_tool_calls_and_reasoning(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "Let me think...",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_001",
                        "function": {"name": "search", "arguments": '{"q":"x"}'},
                    }
                ],
            }
        ]
        result = openai_to_messages(msgs)
        assert len(result) == 1
        assert result[0].metadata.type == MessageType.TOOL_CALL
        assert result[0].content == "Let me think..."
        assert len(result[0].tool_calls or []) == 1
        assert result[0].tool_calls[0].name == "search"

    def test_tool_message_without_tool_call_id(self) -> None:
        msgs = [{"role": "tool", "content": "result"}]
        result = openai_to_messages(msgs)
        assert result[0].role == MessageRole.TOOL
        assert result[0].metadata.type == MessageType.TOOL_RESULT
        assert result[0].tool_call_id == ""

    def test_multiple_system_messages(self) -> None:
        """Multiple system messages are all preserved."""
        msgs = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Hi"},
        ]
        result = openai_to_messages(msgs)
        assert len(result) == 3
        assert result[0].metadata.type == MessageType.SYSTEM_PROMPT
        assert result[1].metadata.type == MessageType.SYSTEM_PROMPT
        assert result[2].metadata.type == MessageType.USER_INPUT

    def test_tool_message_with_name(self) -> None:
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": "ok", "name": "search"}]
        result = openai_to_messages(msgs)
        assert result[0].tool_name == "search"
        assert result[0].tool_call_id == "c1"


# ── anthropic_to_openai_messages edge cases ───────────────────


class TestAnthropicToOpenAIMessagesEdgeCases:
    """Edge cases for anthropic_to_openai_messages."""

    def test_completely_empty_body(self) -> None:
        assert anthropic_to_openai_messages({}) == []

    def test_only_system_no_messages(self) -> None:
        body: dict[str, Any] = {"system": "You are helpful."}
        result = anthropic_to_openai_messages(body)
        assert len(result) == 1
        assert result[0] == {"role": "system", "content": "You are helpful."}

    def test_user_content_as_empty_list(self) -> None:
        """Empty content list produces no message (all block types filtered out)."""
        body: dict[str, Any] = {"messages": [{"role": "user", "content": []}]}
        result = anthropic_to_openai_messages(body)
        assert result == []

    def test_user_content_with_only_thinking_blocks(self) -> None:
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "thinking", "text": "Hmm..."}],
                }
            ],
        }
        result = anthropic_to_openai_messages(body)
        assert "<think>" in result[0]["content"]
        assert "Hmm..." in result[0]["content"]

    def test_assistant_with_thinking_only(self) -> None:
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "text": "I think..."}],
                }
            ],
        }
        result = anthropic_to_openai_messages(body)
        assert "thinking" not in str(result[0].get("content", "").lower())
        assert "<think>" in result[0]["content"]

    def test_tool_result_with_empty_content(self) -> None:
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": ""}],
                }
            ],
        }
        result = anthropic_to_openai_messages(body)
        assert result[0]["content"] == ""
        assert result[0]["tool_call_id"] == "t1"

    def test_assistant_text_with_empty_tool_use(self) -> None:
        """Assistant with text + empty tool_use block."""
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Done"},
                        {"type": "tool_use", "id": "t1", "name": "submit", "input": {}},
                    ],
                }
            ],
        }
        result = anthropic_to_openai_messages(body)
        assert result[0]["content"] == "Done"
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "submit"

    def test_missing_messages_key(self) -> None:
        body: dict[str, Any] = {"system": "hi"}
        result = anthropic_to_openai_messages(body)
        assert len(result) == 1
        assert result[0]["role"] == "system"

    def test_string_content_vs_list_content(self) -> None:
        """Same content as string vs list should produce same result."""
        string_body: dict[str, Any] = {"messages": [{"role": "user", "content": "hello"}]}
        list_body: dict[str, Any] = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
        string_result = anthropic_to_openai_messages(string_body)
        list_result = anthropic_to_openai_messages(list_body)
        assert string_result == list_result

    def test_tool_use_without_id_generates_one(self) -> None:
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "f", "input": {}}],
                }
            ],
        }
        result = anthropic_to_openai_messages(body)
        assert "id" in result[0]["tool_calls"][0]
        assert result[0]["tool_calls"][0]["id"].startswith("call_")


# ── openai_to_anthropic_messages edge cases ───────────────────


class TestOpenaiToAnthropicMessagesEdgeCases:
    """Edge cases for openai_to_anthropic_messages."""

    def test_empty_messages_list(self) -> None:
        result = openai_to_anthropic_messages([])
        assert result == {"messages": []}

    def test_system_only(self) -> None:
        msgs = [{"role": "system", "content": "Be helpful."}]
        result = openai_to_anthropic_messages(msgs)
        assert result["system"] == "Be helpful."
        assert result["messages"] == []

    def test_assistant_with_empty_tool_calls(self) -> None:
        """Assistant with empty tool_calls list: content becomes list of text blocks."""
        msgs = [{"role": "assistant", "content": "hello", "tool_calls": []}]
        result = openai_to_anthropic_messages(msgs)
        # content becomes a list of text blocks since tool_calls is falsy
        assert result["messages"][0]["content"] == [{"type": "text", "text": "hello"}]

    def test_tool_with_missing_tool_call_id(self) -> None:
        msgs = [{"role": "tool", "content": "result"}]
        result = openai_to_anthropic_messages(msgs)
        block = result["messages"][0]["content"][0]
        assert block["tool_use_id"] == ""

    def test_tool_with_is_error_false(self) -> None:
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": "ok", "is_error": False}]
        result = openai_to_anthropic_messages(msgs)
        block = result["messages"][0]["content"][0]
        assert "is_error" not in block

    def test_tool_with_is_error_true(self) -> None:
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": "fail", "is_error": True}]
        result = openai_to_anthropic_messages(msgs)
        block = result["messages"][0]["content"][0]
        assert block["is_error"] is True

    def test_assistant_with_text_and_tool_use(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "Checking...",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_001",
                        "function": {"name": "search", "arguments": '{"q":"x"}'},
                    }
                ],
            }
        ]
        result = openai_to_anthropic_messages(msgs)
        blocks = result["messages"][0]["content"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "tool_use"

    def test_user_content_as_list_with_thinking(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hi"}, {"type": "thinking", "text": "Hmm"}],
            }
        ]
        result = openai_to_anthropic_messages(msgs)
        blocks = result["messages"][0]["content"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "thinking"

    def test_user_content_list_with_unknown_block_type(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [{"type": "unknown_type", "data": "x"}],
            }
        ]
        result = openai_to_anthropic_messages(msgs)
        # Unknown block types should be passed through as-is
        block = result["messages"][0]["content"][0]
        assert block["type"] == "unknown_type"

    def test_tool_arguments_as_empty_string(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "c1",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            }
        ]
        result = openai_to_anthropic_messages(msgs)
        assert result["messages"][0]["content"][0]["input"] == {}

    def test_tool_arguments_missing(self) -> None:
        """Missing arguments defaults to empty dict (parsed from '{}')."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "c1",
                        "function": {"name": "f"},
                    }
                ],
            }
        ]
        result = openai_to_anthropic_messages(msgs)
        # defaults to "{}" string → json.loads → {}
        assert result["messages"][0]["content"][0]["input"] == {}

    def test_arguments_already_dict_not_double_serialized(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "c1",
                        "function": {"name": "f", "arguments": {"key": "val"}},
                    }
                ],
            }
        ]
        result = openai_to_anthropic_messages(msgs)
        assert result["messages"][0]["content"][0]["input"] == {"key": "val"}
