"""Unit tests for forge.core.messages."""

import dataclasses

import pytest

from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo


class TestMessageRole:
    def test_values_and_string_behavior(self):
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.TOOL == "tool"
        for role in MessageRole:
            assert isinstance(role, str)


class TestMessageType:
    def test_values_and_string_behavior(self):
        expected = {
            "system_prompt",
            "user_input",
            "tool_call",
            "tool_result",
            "reasoning",
            "text_response",
            "step_nudge",
            "prerequisite_nudge",
            "retry_nudge",
            "context_warning",
            "summary",
        }
        actual = {mt.value for mt in MessageType}
        assert actual == expected
        for mt in MessageType:
            assert isinstance(mt, str)


class TestMessageMeta:
    def test_is_frozen(self):
        meta = MessageMeta(type=MessageType.TOOL_CALL, step_index=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.type = MessageType.SUMMARY  # type: ignore[misc]

    def test_defaults(self):
        meta = MessageMeta(type=MessageType.USER_INPUT)
        assert meta.step_index is None
        assert meta.original_type is None
        assert meta.token_estimate is None

    def test_all_fields(self):
        meta = MessageMeta(
            type=MessageType.SUMMARY,
            step_index=3,
            original_type=MessageType.TOOL_RESULT,
            token_estimate=120,
        )
        assert meta.type == MessageType.SUMMARY
        assert meta.step_index == 3
        assert meta.original_type == MessageType.TOOL_RESULT
        assert meta.token_estimate == 120


class TestMessage:
    def test_to_api_dict_returns_role_and_content(self):
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hello",
            metadata=MessageMeta(type=MessageType.REASONING),
        )
        api_dict = msg.to_api_dict()
        assert api_dict == {"role": "assistant", "content": "Hello"}

    def test_metadata_not_in_api_dict(self):
        msg = Message(
            role=MessageRole.USER,
            content="Do something",
            metadata=MessageMeta(
                type=MessageType.USER_INPUT, step_index=0, token_estimate=50
            ),
        )
        api_dict = msg.to_api_dict()
        assert "metadata" not in api_dict
        assert "_meta" not in api_dict
        assert set(api_dict.keys()) == {"role", "content"}

    def test_all_roles_round_trip(self):
        for role in MessageRole:
            msg = Message(
                role=role,
                content="test",
                metadata=MessageMeta(type=MessageType.USER_INPUT),
            )
            api_dict = msg.to_api_dict()
            assert api_dict["role"] == role.value

    def test_tool_call_emits_structured_tool_calls_ollama(self):
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="",
            metadata=MessageMeta(type=MessageType.TOOL_CALL),
            tool_calls=[ToolCallInfo(name="fetch", args={"key": "val"}, call_id="call_000000001")],
        )
        api_dict = msg.to_api_dict(format="ollama")
        assert api_dict["role"] == "assistant"
        assert api_dict["tool_calls"] == [{
            "function": {"name": "fetch", "arguments": {"key": "val"}},
        }]

    def test_tool_call_emits_structured_tool_calls_openai(self):
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="",
            metadata=MessageMeta(type=MessageType.TOOL_CALL),
            tool_calls=[ToolCallInfo(name="fetch", args={"key": "val"}, call_id="call_000000001")],
        )
        api_dict = msg.to_api_dict(format="openai")
        assert api_dict["role"] == "assistant"
        assert api_dict["tool_calls"] == [{
            "id": "call_000000001",
            "type": "function",
            "function": {"name": "fetch", "arguments": '{"key": "val"}'},
        }]

    def test_multi_tool_calls_ollama(self):
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="",
            metadata=MessageMeta(type=MessageType.TOOL_CALL),
            tool_calls=[
                ToolCallInfo(name="fetch", args={"key": "a"}, call_id="call_000000001"),
                ToolCallInfo(name="analyze", args={"key": "b"}, call_id="call_000000002"),
            ],
        )
        api_dict = msg.to_api_dict(format="ollama")
        assert len(api_dict["tool_calls"]) == 2
        assert api_dict["tool_calls"][0]["function"]["name"] == "fetch"
        assert api_dict["tool_calls"][1]["function"]["name"] == "analyze"

    def test_multi_tool_calls_openai(self):
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="",
            metadata=MessageMeta(type=MessageType.TOOL_CALL),
            tool_calls=[
                ToolCallInfo(name="fetch", args={"key": "a"}, call_id="call_000000001"),
                ToolCallInfo(name="analyze", args={"key": "b"}, call_id="call_000000002"),
            ],
        )
        api_dict = msg.to_api_dict(format="openai")
        assert len(api_dict["tool_calls"]) == 2
        assert api_dict["tool_calls"][0]["id"] == "call_000000001"
        assert api_dict["tool_calls"][1]["id"] == "call_000000002"

    def test_tool_result_emits_tool_name_ollama(self):
        msg = Message(
            role=MessageRole.TOOL,
            content="result data",
            metadata=MessageMeta(type=MessageType.TOOL_RESULT),
            tool_name="fetch",
        )
        api_dict = msg.to_api_dict(format="ollama")
        assert api_dict == {
            "role": "tool",
            "content": "result data",
            "tool_name": "fetch",
        }

    def test_tool_result_openai_uses_name_and_tool_call_id(self):
        msg = Message(
            role=MessageRole.TOOL,
            content="result data",
            metadata=MessageMeta(type=MessageType.TOOL_RESULT),
            tool_name="fetch",
            tool_call_id="call_000000001",
        )
        api_dict = msg.to_api_dict(format="openai")
        assert api_dict == {
            "role": "tool",
            "content": "result data",
            "name": "fetch",
            "tool_call_id": "call_000000001",
        }

    def test_tool_result_openai_without_id(self):
        msg = Message(
            role=MessageRole.TOOL,
            content="result data",
            metadata=MessageMeta(type=MessageType.TOOL_RESULT),
            tool_name="fetch",
        )
        api_dict = msg.to_api_dict(format="openai")
        assert api_dict == {
            "role": "tool",
            "content": "result data",
            "name": "fetch",
        }
