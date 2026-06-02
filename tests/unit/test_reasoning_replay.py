"""Tests for reasoning replay policy serialization."""

import pytest

from forge.core.inference import fold_and_serialize, prepare_backend_messages
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from forge.core.reasoning import filter_openai_reasoning_messages, validate_reasoning_replay

from tests.eval.metrics import count_wire_reasoning


def _reasoning(text: str) -> Message:
    return Message(MessageRole.ASSISTANT, text, MessageMeta(MessageType.REASONING))


def _tool_call(name: str) -> Message:
    return Message(
        MessageRole.ASSISTANT, "", MessageMeta(MessageType.TOOL_CALL),
        tool_calls=[ToolCallInfo(name=name, args={}, call_id=f"call_{name}")],
    )


def test_full_replays_every_reasoning_block():
    messages = [
        _reasoning("first"), _tool_call("a"),
        _reasoning("second"), _tool_call("b"),
    ]

    result = fold_and_serialize(messages, "openai", reasoning_replay="full")

    assert [m["content"] for m in result] == ["first", "second"]


def test_keep_last_replays_only_latest_reasoning_block():
    messages = [
        _reasoning("first"), _tool_call("a"),
        _reasoning("second"), _tool_call("b"),
    ]

    result = fold_and_serialize(messages, "openai", reasoning_replay="keep-last")

    assert [m["content"] for m in result] == ["", "second"]


def test_none_replays_no_reasoning_blocks():
    messages = [
        _reasoning("first"), _tool_call("a"),
        _reasoning("second"), _tool_call("b"),
    ]

    result = fold_and_serialize(messages, "openai", reasoning_replay="none")

    assert [m["content"] for m in result] == ["", ""]


def test_keep_last_orphan_reasoning_is_preserved_as_orphan():
    messages = [_reasoning("first"), _tool_call("a"), _reasoning("orphan")]

    result = fold_and_serialize(messages, "openai", reasoning_replay="keep-last")

    assert result[-1] == {"role": "assistant", "content": "orphan"}


def test_validate_reasoning_replay_rejects_unknown_policy():
    with pytest.raises(ValueError, match="reasoning_replay must be one of"):
        validate_reasoning_replay("latest")


def test_filter_openai_reasoning_messages_only_filters_assistant_messages():
    messages = [
        {
            "role": "user",
            "content": "keep this",
            "reasoning_content": "user metadata",
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "old assistant reasoning",
            "name": "a1",
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "latest assistant reasoning",
            "name": "a2",
        },
    ]

    result = filter_openai_reasoning_messages(messages, reasoning_replay="keep-last")

    assert result[0]["reasoning_content"] == "user metadata"
    assert result[1]["name"] == "a1"
    assert "reasoning_content" not in result[1]
    assert result[2]["name"] == "a2"
    assert result[2]["reasoning_content"] == "latest assistant reasoning"


def test_filter_openai_reasoning_messages_none_preserves_user_reasoning_fields():
    messages = [
        {"role": "user", "content": "keep", "reasoning": "user value"},
        {"role": "assistant", "content": None, "reasoning": "drop"},
    ]

    result = filter_openai_reasoning_messages(messages, reasoning_replay="none")

    assert result[0]["reasoning"] == "user value"
    assert "reasoning" not in result[1]


def test_prepare_backend_messages_filters_raw_openai_reasoning():
    raw_messages = [
        {"role": "assistant", "content": None, "reasoning_content": "old", "name": "a1"},
        {"role": "assistant", "content": None, "reasoning_content": "latest", "name": "a2"},
    ]

    result = prepare_backend_messages(
        [],
        "openai",
        raw_openai_messages=raw_messages,
        use_raw_messages=True,
        reasoning_replay="keep-last",
    )

    assert result[0]["name"] == "a1"
    assert "reasoning_content" not in result[0]
    assert result[1]["name"] == "a2"
    assert result[1]["reasoning_content"] == "latest"


def test_prepare_backend_messages_folds_forge_history_without_raw_messages():
    messages = [_reasoning("first"), _tool_call("a"), _reasoning("second"), _tool_call("b")]

    result = prepare_backend_messages(
        messages, "openai", reasoning_replay="keep-last",
    )

    assert [m["content"] for m in result] == ["", "second"]


# ── Eval-side on-wire reasoning counter (validates the knob end to end) ──

def _wire_transcript() -> list[Message]:
    return [
        _reasoning("first"), _tool_call("a"),
        _reasoning("second"), _tool_call("b"),
    ]


def test_count_wire_reasoning_full_keeps_all():
    survived, total = count_wire_reasoning(_wire_transcript(), "full")
    assert (survived, total) == (2, 2)


def test_count_wire_reasoning_keep_last_keeps_one():
    survived, total = count_wire_reasoning(_wire_transcript(), "keep-last")
    assert (survived, total) == (1, 2)


def test_count_wire_reasoning_none_strips_all():
    # The core claim: none puts zero reasoning on the wire.
    survived, total = count_wire_reasoning(_wire_transcript(), "none")
    assert (survived, total) == (0, 2)


def test_count_wire_reasoning_no_reasoning_is_zero_zero():
    survived, total = count_wire_reasoning([_tool_call("a"), _tool_call("b")], "full")
    assert (survived, total) == (0, 0)
