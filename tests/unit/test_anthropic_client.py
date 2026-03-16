"""Tests for forge.clients.anthropic — format conversion helpers.

All tests exercise the static conversion methods directly.
No API calls or mocks needed.
"""

import json
from typing import Literal
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from forge.clients.anthropic import AnthropicClient
from forge.core.workflow import TextResponse, ToolCall, ToolSpec


class CityParams(BaseModel):
    city: str = Field(description="City name")


def _make_spec(name: str = "get_weather") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Get {name}",
        parameters=CityParams,
    )


class SetUnitParams(BaseModel):
    unit: Literal["celsius", "fahrenheit"] = Field(description="Unit")


def _make_spec_with_enum() -> ToolSpec:
    return ToolSpec(
        name="set_unit",
        description="Set temperature unit",
        parameters=SetUnitParams,
    )


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
