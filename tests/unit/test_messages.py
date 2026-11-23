"""Canonical state/ingress fidelity, replacing legacy flat serialization assertions."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
import pytest
from forge.core.messages import *
from forge.core.codecs import (
    openai_state,
    anthropic_state,
    render_openai,
    render_anthropic,
    decode_arguments,
    argument_value,
)


@pytest.mark.parametrize(
    "value", [{}, {"nested": [1, False]}, [], None, False, 0, 0.0, "", "broken{"]
)
def test_argument_variants_are_immutable_and_retain_received_values(value):
    wrapped = decode_arguments(json.dumps(value))
    assert isinstance(
        wrapped, ValidArguments if isinstance(value, dict) else InvalidArguments
    )
    part = ToolCallPart("f", wrapped)
    assert argument_value(part) == value
    with pytest.raises(FrozenInstanceError):
        wrapped.value = {}
    wire = render_openai([RichMessage(MessageRole.ASSISTANT, [part])])[0]
    assert json.loads(wire["tool_calls"][0]["function"]["arguments"]) == value
    assert (
        render_anthropic([RichMessage(MessageRole.ASSISTANT, [part])])[1][0]["content"][
            0
        ]["input"]
        == value
    )
    assert (
        render_openai([RichMessage(MessageRole.ASSISTANT, [part])], ollama=True)[0][
            "tool_calls"
        ][0]["function"]["arguments"]
        == value
    )


@pytest.mark.parametrize(
    "content",
    [
        "hello",
        "",
        None,
        [],
        [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "b", "x": 2},
        ],
        ["a", {"type": "text", "text": "b"}],
    ],
)
def test_openai_content_shapes_boundaries_and_fields(content):
    raw = [{"role": "user", "name": "n", "content": content, "x": {"nested": True}}]
    state = openai_state(raw)
    assert len(state.messages) == 1
    assert render_openai(state.messages) == raw


def test_openai_mixed_message_call_and_function_fields_win_extras():
    raw = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "visible", "x": 1}],
            "reasoning_content": "r1",
            "reasoning": "r2",
            "name": "a",
            "tool_calls": [
                {
                    "id": "imported",
                    "type": "function",
                    "x": "outer",
                    "function": {"name": "f", "arguments": '{"x":1}', "x": "inner"},
                }
            ],
        }
    ]
    state = openai_state(raw)
    before = deepcopy(state)
    wire = render_openai(state.messages)
    assert wire[0]["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
    raw[0]["tool_calls"][0]["function"]["arguments"] = '{"x": 1}'
    assert wire == raw
    assert state == before
    assert len(state.messages[0].parts) == 4
    part = state.messages[0].parts[-1]
    part.extra_fields.update(type="bad", id="bad")
    part.function_fields.update(name="bad", arguments="bad")
    rendered = render_openai(state.messages)[0]["tool_calls"][0]
    assert rendered["type"] == "function" and rendered["id"] == "imported"
    assert rendered["function"]["name"] == "f"


def test_anthropic_grouping_order_system_and_multipart_results_round_trip():
    blocks = [
        {"type": "text", "text": "before", "cache_control": {"type": "ephemeral"}},
        {
            "type": "tool_result",
            "tool_use_id": "",
            "is_error": True,
            "x": 4,
            "content": [
                {"type": "text", "text": "one", "x": 1},
                {"type": "text", "text": "two"},
            ],
        },
        {"type": "text", "text": "after"},
    ]
    raw = [
        {"role": "user", "content": blocks, "x": 2},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "r1", "signature": "s1"},
                {"type": "text", "text": "visible"},
                {"type": "tool_use", "id": "c", "name": "f", "input": {}, "x": 3},
                {"type": "thinking", "thinking": "r2", "signature": "s2"},
            ],
        },
    ]
    system = [
        {"type": "text", "text": "system", "cache_control": {"type": "ephemeral"}}
    ]
    state = anthropic_state(raw, system)
    assert len(state.messages) == 3
    assert len(state.messages[1].parts) == 3
    assert render_anthropic(state.messages) == (system, raw)
    result = state.messages[1].parts[1]
    assert result.outcome == "error" and result.call_id == ""
    assert not hasattr(result, "name")


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_unknown_structural_blocks_fail_clearly(protocol):
    parse = openai_state if protocol == "openai" else anthropic_state
    with pytest.raises(ValueError, match="Unsupported content block"):
        parse([{"role": "user", "content": [{"type": "image", "url": "x"}]}])


def test_forge_fields_are_direct_and_never_rendered():
    message = RichMessage(
        MessageRole.USER, [TextPart("x")], step_index=4, forge_purpose="retry"
    )
    assert render_openai([message]) == [{"role": "user", "content": "x"}]
    assert not hasattr(message, "metadata")


def test_missing_history_id_fallbacks_and_empty_result_reference():
    state = openai_state(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "result"},
        ]
    )
    assert state.messages[0].parts[-1].call_id.startswith("call_")
    assert state.messages[1].parts[0].call_id == ""
    assert render_openai(state.messages)[1]["tool_call_id"] == ""


def test_cross_protocol_native_result_error_flag_is_not_added_to_openai_wire():
    state = anthropic_state(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "c",
                        "is_error": True,
                        "content": "failed",
                    }
                ],
            }
        ]
    )
    assert state.messages[0].parts[0].outcome == "error"
    assert render_anthropic(state.messages)[1][0]["content"][0]["is_error"] is True
    assert "is_error" not in render_openai(state.messages)[0]


def test_imported_openai_result_does_not_acquire_tool_name_identity():
    raw = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "result", "tool_call_id": "c"},
    ]
    state = openai_state(raw)
    assert "name" not in render_openai(state.messages)[1]


def test_canonical_message_parts_win_over_structural_extras_even_when_absent():
    message = RichMessage(
        MessageRole.ASSISTANT,
        [TextPart("canonical")],
        extra_fields={
            "role": "user",
            "content": "wrong",
            "tool_calls": [{"wrong": True}],
            "reasoning_content": "wrong",
        },
    )
    assert render_openai([message]) == [{"role": "assistant", "content": "canonical"}]
