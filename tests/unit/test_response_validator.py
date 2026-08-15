"""ResponseValidator orchestration and handoff contracts."""

import pytest

from forge.core.workflow import TextResponse, ToolCall
from forge.guardrails import Nudge, ResponseValidator


def _validator(*, rescue_enabled: bool = True) -> ResponseValidator:
    return ResponseValidator(
        tool_names=["search", "answer"],
        rescue_enabled=rescue_enabled,
    )


def test_nudge_is_immutable() -> None:
    nudge = Nudge(role="user", content="try again", kind="retry")

    with pytest.raises(AttributeError):
        nudge.role = "system"


def test_text_response_rescue_hands_calls_to_validation_result() -> None:
    result = _validator().validate(
        TextResponse(content='{"tool": "search", "args": {"q": "hello"}}')
    )

    assert result.needs_retry is False
    assert result.nudge is None
    assert result.tool_calls == [ToolCall(tool="search", args={"q": "hello"})]


def test_unrescued_text_uses_retry_channel_and_content() -> None:
    cases = [
        ("plain text", _validator(), "I don't know"),
        (
            "rescue disabled",
            _validator(rescue_enabled=False),
            '{"tool": "search", "args": {"q": "hello"}}',
        ),
        (
            "unknown rescued tool",
            _validator(),
            '{"tool": "nonexistent", "args": {}}',
        ),
    ]

    for label, validator, content in cases:
        result = validator.validate(TextResponse(content=content))
        assert result.needs_retry is True, label
        assert result.tool_calls is None, label
        assert result.nudge is not None, label
        assert result.nudge.role == "user", label
        assert result.nudge.kind == "retry", label
        assert "tool call" in result.nudge.content.lower(), label


def test_valid_tool_call_batches_pass_through() -> None:
    batches = [
        [ToolCall(tool="search", args={"q": "hi"})],
        [
            ToolCall(tool="search", args={"q": "a"}),
            ToolCall(tool="answer", args={"text": "b"}),
        ],
        [],
    ]

    for calls in batches:
        result = _validator().validate(calls)
        assert result.needs_retry is False
        assert result.tool_calls == calls
        assert result.nudge is None


def test_unknown_tool_batches_use_tool_channel() -> None:
    batches = [
        [ToolCall(tool="nonexistent", args={})],
        [
            ToolCall(tool="search", args={"q": "hi"}),
            ToolCall(tool="bad_tool", args={}),
        ],
    ]

    for calls in batches:
        unknown = calls[-1].tool
        result = _validator().validate(calls)
        assert result.needs_retry is True
        assert result.tool_calls is None
        assert result.nudge is not None
        assert result.nudge.kind == "unknown_tool"
        assert result.nudge.role == "tool"
        assert unknown in result.nudge.content


def test_invalid_argument_shapes_use_tool_error_channel() -> None:
    cases = [
        ("string", [ToolCall(tool="search", args="")]),
        ("none", [ToolCall(tool="search", args=None)]),
        ("list", [ToolCall(tool="search", args=[1, 2])]),
        ("integer", [ToolCall(tool="search", args=42)]),
        (
            "mixed batch",
            [
                ToolCall(tool="search", args={"q": "hi"}),
                ToolCall(tool="answer", args=""),
            ],
        ),
    ]

    for label, calls in cases:
        result = _validator().validate(calls)
        bad_tool = calls[-1].tool
        assert result.needs_retry is True, label
        assert result.tool_calls is None, label
        assert result.nudge is not None, label
        assert result.nudge.kind == "tool_arg_validation", label
        assert result.nudge.role == "tool", label
        assert bad_tool in result.nudge.content, label
        assert "JSON object" in result.nudge.content, label


def test_unknown_tool_takes_precedence_over_argument_shape() -> None:
    result = _validator().validate([ToolCall(tool="hallucinated", args="")])

    assert result.needs_retry is True
    assert result.nudge is not None
    assert result.nudge.kind == "unknown_tool"
