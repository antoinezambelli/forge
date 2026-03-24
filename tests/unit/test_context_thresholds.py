"""Tests for context window self-awareness — threshold callbacks."""

import pytest

from forge.context.manager import ContextManager, default_context_warning
from forge.context.strategies import NoCompact
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType


def _msg(content: str, role: MessageRole = MessageRole.USER) -> Message:
    return Message(
        role=role,
        content=content,
        metadata=MessageMeta(type=MessageType.USER_INPUT),
    )


def _make_messages_at_pct(budget: int, pct: float) -> list[Message]:
    """Create messages whose estimated tokens are ~pct of budget."""
    # estimate_tokens uses len(content) // 4
    target_tokens = int(budget * pct)
    content = "x" * (target_tokens * 4)
    return [_msg("system"), _msg(content)]


# ── default_context_warning ──────────────────────────────────────


class TestDefaultContextWarning:
    def test_returns_string_at_50pct(self) -> None:
        result = default_context_warning(4000, 8000, 0.50)
        assert result is not None
        assert "50%" in result

    def test_returns_string_at_65pct(self) -> None:
        result = default_context_warning(5200, 8000, 0.65)
        assert result is not None
        assert "filling up" in result.lower()

    def test_returns_string_at_80pct(self) -> None:
        result = default_context_warning(6400, 8000, 0.80)
        assert result is not None
        assert "nearly full" in result.lower()

    def test_escalates_with_pct(self) -> None:
        msg_50 = default_context_warning(4000, 8000, 0.50)
        msg_80 = default_context_warning(6400, 8000, 0.80)
        assert msg_50 != msg_80
        # Higher urgency at 80%
        assert "nearly full" not in msg_50.lower()
        assert "nearly full" in msg_80.lower()


# ── ContextManager.check_thresholds ──────────────────────────────


class TestCheckThresholds:
    def test_no_callback_returns_none(self) -> None:
        ctx = ContextManager(
            strategy=NoCompact(),
            budget_tokens=8000,
            context_thresholds=[0.5],
            on_context_threshold=None,
        )
        msgs = _make_messages_at_pct(8000, 0.6)
        assert ctx.check_thresholds(msgs) is None

    def test_no_thresholds_returns_none(self) -> None:
        ctx = ContextManager(
            strategy=NoCompact(),
            budget_tokens=8000,
            context_thresholds=None,
            on_context_threshold=default_context_warning,
        )
        msgs = _make_messages_at_pct(8000, 0.6)
        assert ctx.check_thresholds(msgs) is None

    def test_fires_when_threshold_crossed(self) -> None:
        ctx = ContextManager(
            strategy=NoCompact(),
            budget_tokens=8000,
            context_thresholds=[0.5],
            on_context_threshold=default_context_warning,
        )
        msgs = _make_messages_at_pct(8000, 0.6)
        result = ctx.check_thresholds(msgs)
        assert result is not None
        assert "60%" in result

    def test_does_not_fire_below_threshold(self) -> None:
        ctx = ContextManager(
            strategy=NoCompact(),
            budget_tokens=8000,
            context_thresholds=[0.5],
            on_context_threshold=default_context_warning,
        )
        msgs = _make_messages_at_pct(8000, 0.4)
        assert ctx.check_thresholds(msgs) is None

    def test_fires_once_per_threshold(self) -> None:
        ctx = ContextManager(
            strategy=NoCompact(),
            budget_tokens=8000,
            context_thresholds=[0.5],
            on_context_threshold=default_context_warning,
        )
        msgs = _make_messages_at_pct(8000, 0.6)
        result1 = ctx.check_thresholds(msgs)
        result2 = ctx.check_thresholds(msgs)
        assert result1 is not None
        assert result2 is None  # already fired

    def test_resets_after_compaction_drops_below(self) -> None:
        ctx = ContextManager(
            strategy=NoCompact(),
            budget_tokens=8000,
            context_thresholds=[0.5],
            on_context_threshold=default_context_warning,
        )
        # Cross the threshold
        msgs_high = _make_messages_at_pct(8000, 0.6)
        result1 = ctx.check_thresholds(msgs_high)
        assert result1 is not None

        # Drop below (simulates post-compaction)
        msgs_low = _make_messages_at_pct(8000, 0.3)
        ctx.check_thresholds(msgs_low)

        # Cross again — should fire again
        result2 = ctx.check_thresholds(msgs_high)
        assert result2 is not None

    def test_multiple_thresholds_fires_highest(self) -> None:
        calls = []

        def track_callback(tokens, budget, pct):
            calls.append(pct)
            return f"warning at {pct:.0%}"

        ctx = ContextManager(
            strategy=NoCompact(),
            budget_tokens=8000,
            context_thresholds=[0.3, 0.5, 0.7],
            on_context_threshold=track_callback,
        )
        # Jump straight to 60% — crosses both 0.3 and 0.5
        msgs = _make_messages_at_pct(8000, 0.6)
        result = ctx.check_thresholds(msgs)
        assert result is not None
        # Should fire once with the highest crossed threshold
        assert len(calls) == 1

    def test_custom_callback_returning_none(self) -> None:
        ctx = ContextManager(
            strategy=NoCompact(),
            budget_tokens=8000,
            context_thresholds=[0.5],
            on_context_threshold=lambda t, b, p: None,
        )
        msgs = _make_messages_at_pct(8000, 0.6)
        result = ctx.check_thresholds(msgs)
        assert result is None

    def test_zero_budget_returns_none(self) -> None:
        ctx = ContextManager(
            strategy=NoCompact(),
            budget_tokens=0,
            context_thresholds=[0.5],
            on_context_threshold=default_context_warning,
        )
        assert ctx.check_thresholds([_msg("test")]) is None

    def test_importable_from_forge(self) -> None:
        from forge import default_context_warning
        assert callable(default_context_warning)


# ── Integration: injection into run_inference ─────────────────────


class TestInferenceInjection:
    """Verify the context warning is injected into api_messages sent to the LLM."""

    @pytest.mark.asyncio
    async def test_warning_injected_into_api_messages(self) -> None:
        from unittest.mock import AsyncMock
        from forge.core.inference import run_inference
        from forge.core.workflow import ToolCall, ToolSpec, TextResponse
        from forge.guardrails import ErrorTracker, ResponseValidator

        # Tiny budget, threshold at 30% so our messages will cross it
        ctx = ContextManager(
            strategy=NoCompact(),
            budget_tokens=100,
            context_thresholds=[0.3],
            on_context_threshold=default_context_warning,
        )

        # Messages that exceed 30% of 100 tokens (~30 tokens = ~120 chars)
        msgs = [
            _msg("You are a helpful assistant."),
            _msg("x" * 200),  # ~50 tokens, well over 30%
        ]

        # Mock client — capture what api_messages it receives
        captured_messages = []

        async def mock_send(api_messages, tools=None):
            captured_messages.extend(api_messages)
            return TextResponse(content="ok", intentional=True)

        mock_client = AsyncMock()
        mock_client.send = mock_send
        mock_client.api_format = "ollama"

        validator = ResponseValidator(tool_names=[], rescue_enabled=False)
        error_tracker = ErrorTracker(max_retries=1)

        result = await run_inference(
            messages=msgs,
            client=mock_client,
            context_manager=ctx,
            validator=validator,
            error_tracker=error_tracker,
            tool_specs=[],
            trust_text_intent=True,
        )

        # The injected warning should be the last message (user role)
        warning_msgs = [m for m in captured_messages if "Context usage" in m.get("content", "")]
        assert len(warning_msgs) > 0, \
            f"Expected context warning in api_messages, got: {[m['content'][:50] for m in captured_messages]}"
        assert warning_msgs[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_no_injection_without_threshold_config(self) -> None:
        from unittest.mock import AsyncMock
        from forge.core.inference import run_inference
        from forge.core.workflow import TextResponse
        from forge.guardrails import ErrorTracker, ResponseValidator

        # No thresholds configured
        ctx = ContextManager(
            strategy=NoCompact(),
            budget_tokens=100,
        )

        msgs = [_msg("system"), _msg("x" * 200)]

        captured_messages = []

        async def mock_send(api_messages, tools=None):
            captured_messages.extend(api_messages)
            return TextResponse(content="ok", intentional=True)

        mock_client = AsyncMock()
        mock_client.send = mock_send
        mock_client.api_format = "ollama"

        validator = ResponseValidator(tool_names=[], rescue_enabled=False)
        error_tracker = ErrorTracker(max_retries=1)

        await run_inference(
            messages=msgs,
            client=mock_client,
            context_manager=ctx,
            validator=validator,
            error_tracker=error_tracker,
            tool_specs=[],
            trust_text_intent=True,
        )

        # No context warning should be injected
        system_msgs = [m for m in captured_messages if m["role"] == "system"]
        assert not any("Context usage" in m.get("content", "") for m in system_msgs)
