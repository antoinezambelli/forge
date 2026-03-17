"""Tests for ResponseValidator and Nudge."""

import pytest

from forge.core.workflow import TextResponse, ToolCall
from forge.guardrails import Nudge, ResponseValidator, ValidationResult


class TestNudge:
    """Nudge dataclass basics."""

    def test_frozen(self):
        nudge = Nudge(role="user", content="try again", kind="retry")
        with pytest.raises(AttributeError):
            nudge.role = "system"

    def test_defaults(self):
        nudge = Nudge(role="user", content="x", kind="retry")
        assert nudge.tier == 0

    def test_tier_preserved(self):
        nudge = Nudge(role="user", content="x", kind="step", tier=2)
        assert nudge.tier == 2


class TestResponseValidatorTextResponse:
    """TextResponse handling — rescue and retry."""

    def setup_method(self):
        self.validator = ResponseValidator(
            tool_names=["search", "answer"], rescue_enabled=True
        )

    def test_plain_text_returns_retry_nudge(self):
        result = self.validator.validate(TextResponse(content="I don't know"))
        assert result.needs_retry is True
        assert result.tool_calls is None
        assert result.nudge is not None
        assert result.nudge.role == "user"
        assert result.nudge.kind == "retry"
        assert "tool call" in result.nudge.content.lower()

    def test_rescue_json_tool_call(self):
        text = '{"tool": "search", "args": {"q": "hello"}}'
        result = self.validator.validate(TextResponse(content=text))
        assert result.needs_retry is False
        assert result.nudge is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool == "search"
        assert result.tool_calls[0].args == {"q": "hello"}

    def test_rescue_code_fenced_json(self):
        text = '```json\n{"tool": "search", "args": {"q": "test"}}\n```'
        result = self.validator.validate(TextResponse(content=text))
        assert result.needs_retry is False
        assert result.tool_calls[0].tool == "search"

    def test_rescue_disabled(self):
        validator = ResponseValidator(
            tool_names=["search", "answer"], rescue_enabled=False
        )
        text = '{"tool": "search", "args": {"q": "hello"}}'
        result = validator.validate(TextResponse(content=text))
        assert result.needs_retry is True
        assert result.nudge.kind == "retry"

    def test_rescue_unknown_tool_not_rescued(self):
        text = '{"tool": "nonexistent", "args": {}}'
        result = self.validator.validate(TextResponse(content=text))
        assert result.needs_retry is True
        assert result.nudge.kind == "retry"


class TestResponseValidatorToolCalls:
    """list[ToolCall] handling — unknown tool detection."""

    def setup_method(self):
        self.validator = ResponseValidator(
            tool_names=["search", "answer"], rescue_enabled=True
        )

    def test_valid_tool_calls_pass(self):
        calls = [ToolCall(tool="search", args={"q": "hi"})]
        result = self.validator.validate(calls)
        assert result.needs_retry is False
        assert result.tool_calls == calls
        assert result.nudge is None

    def test_unknown_tool_returns_nudge(self):
        calls = [ToolCall(tool="nonexistent", args={})]
        result = self.validator.validate(calls)
        assert result.needs_retry is True
        assert result.tool_calls is None
        assert result.nudge.kind == "unknown_tool"
        assert "nonexistent" in result.nudge.content
        assert result.nudge.role == "user"

    def test_mixed_known_unknown_returns_nudge(self):
        calls = [
            ToolCall(tool="search", args={"q": "hi"}),
            ToolCall(tool="bad_tool", args={}),
        ]
        result = self.validator.validate(calls)
        assert result.needs_retry is True
        assert result.nudge.kind == "unknown_tool"
        assert "bad_tool" in result.nudge.content

    def test_multiple_valid_tools_pass(self):
        calls = [
            ToolCall(tool="search", args={"q": "a"}),
            ToolCall(tool="answer", args={"text": "b"}),
        ]
        result = self.validator.validate(calls)
        assert result.needs_retry is False
        assert result.tool_calls == calls

    def test_empty_tool_calls_pass(self):
        result = self.validator.validate([])
        assert result.needs_retry is False
        assert result.tool_calls == []
