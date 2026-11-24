"""ResponseValidator orchestration and handoff contracts."""

import pytest

from forge.core.messages import RichMessage, MessageRole
from forge.clients.replies import text_reply
from tests.canonical import assistant, call
from forge.guardrails import Nudge, ResponseValidator


def _validator(*, rescue_enabled: bool = True) -> ResponseValidator:
    return ResponseValidator(tool_names=["search", "answer"], rescue_enabled=rescue_enabled)


def test_nudge_is_immutable() -> None:
    nudge = Nudge(role="user", content="try again", kind="retry")

    with pytest.raises(AttributeError):
        nudge.role = "system"


def test_text_response_rescue_hands_calls_to_validation_result() -> None:
    result = _validator().validate(text_reply('{"tool": "search", "args": {"q": "hello"}}'))

    assert result.needs_retry is False
    assert result.nudge is None
    assert result.tool_calls == [call(name="search", arguments={"q": "hello"})]


def test_unrescued_text_uses_retry_channel_and_content() -> None:
    cases = [
        ("plain text", _validator(), "I don't know"),
        (
            "rescue disabled",
            _validator(rescue_enabled=False),
            '{"tool": "search", "args": {"q": "hello"}}',
        ),
        ("unknown rescued tool", _validator(), '{"tool": "nonexistent", "args": {}}'),
    ]

    for label, validator, content in cases:
        result = validator.validate(text_reply(content))
        assert result.needs_retry is True, label
        assert result.nudge is not None, label
        assert result.nudge.role == "user", label
        assert result.nudge.kind == "retry", label
        assert "tool call" in result.nudge.content.lower(), label


def test_valid_tool_call_batches_pass_through() -> None:
    batches = [
        [call(name="search", arguments={"q": "hi"})],
        [call(name="search", arguments={"q": "a"}), call(name="answer", arguments={"text": "b"})],
    ]

    for calls in batches:
        result = _validator().validate(RichMessage(MessageRole.ASSISTANT, calls))
        assert result.needs_retry is False
        assert result.tool_calls == calls
        assert result.nudge is None


def test_unknown_tool_batches_use_tool_channel() -> None:
    batches = [
        [call(name="nonexistent", arguments={})],
        [call(name="search", arguments={"q": "hi"}), call(name="bad_tool", arguments={})],
    ]

    for calls in batches:
        unknown = calls[-1].name
        result = _validator().validate(RichMessage(MessageRole.ASSISTANT, calls))
        assert result.needs_retry is True
        assert result.nudge is not None
        assert result.nudge.kind == "unknown_tool"
        assert result.nudge.role == "tool"
        assert unknown in result.nudge.content


def test_invalid_argument_shapes_use_tool_error_channel() -> None:
    cases = [
        ("string", [call(name="search", arguments="")]),
        ("none", [call(name="search", arguments=None)]),
        ("list", [call(name="search", arguments=[1, 2])]),
        ("integer", [call(name="search", arguments=42)]),
        (
            "mixed batch",
            [call(name="search", arguments={"q": "hi"}), call(name="answer", arguments="")],
        ),
    ]

    for label, calls in cases:
        result = _validator().validate(RichMessage(MessageRole.ASSISTANT, calls))
        bad_tool = calls[-1].name
        assert result.needs_retry is True, label
        assert result.nudge is not None, label
        assert result.nudge.kind == "tool_arg_validation", label
        assert result.nudge.role == "tool", label
        assert bad_tool in result.nudge.content, label
        assert "JSON object" in result.nudge.content, label


def test_unknown_tool_takes_precedence_over_argument_shape() -> None:
    result = _validator().validate(
        RichMessage(MessageRole.ASSISTANT, [call(name="hallucinated", arguments="")])
    )

    assert result.needs_retry is True
    assert result.nudge is not None
    assert result.nudge.kind == "unknown_tool"


@pytest.mark.parametrize("value", [None, [], False, 0, 0.0, ""])
def test_rescued_invalid_argument_values_never_become_executable(value):
    import json
    from forge.core.messages import InvalidArguments, TextPart

    reply = text_reply(json.dumps({"tool": "search", "args": value}))
    result = _validator().validate(reply)
    assert result.needs_retry and result.nudge.kind == "tool_arg_validation"
    assert isinstance(reply.parts[-1].arguments, InvalidArguments)
    assert reply.parts[-1].arguments.received == value
    assert not any(isinstance(part, TextPart) for part in reply.parts)
