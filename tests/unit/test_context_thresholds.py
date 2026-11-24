"""Tests for context window self-awareness — threshold callbacks."""

import pytest

from forge.context.manager import ContextManager, default_context_warning
from forge.context.strategies import NoCompact
from forge.core.messages import ConversationState, RichMessage, MessageRole, TextPart


def _msg(content: str, role: MessageRole = MessageRole.USER) -> RichMessage:
    return RichMessage(role, [TextPart(content)])


def _make_messages_at_pct(budget: int, pct: float) -> ConversationState:
    target_tokens = int(budget * pct)
    return ConversationState([_msg("system"), _msg("x" * (target_tokens * 4))])


# ── default_context_warning ──────────────────────────────────────


class TestDefaultContextWarning:
    def test_escalates_across_context_usage_tiers(self) -> None:
        cases = [
            ("50%", 4000, 0.50, "50%"),
            ("65%", 5200, 0.65, "filling up"),
            ("80%", 6400, 0.80, "nearly full"),
        ]
        results = []
        for label, tokens, pct, expected in cases:
            result = default_context_warning(tokens, 8000, pct)
            assert result is not None, label
            assert expected in result.lower(), label
            results.append(result)
        assert len(set(results)) == len(results)


# ── ContextManager.check_thresholds ──────────────────────────────


class TestCheckThresholds:
    def test_inactive_configurations_return_none(self) -> None:
        cases = [
            ("no callback", 8000, {"context_thresholds": [0.5], "on_context_threshold": None}),
            (
                "no thresholds",
                8000,
                {"context_thresholds": None, "on_context_threshold": default_context_warning},
            ),
            (
                "zero budget",
                0,
                {"context_thresholds": [0.5], "on_context_threshold": default_context_warning},
            ),
        ]
        for label, budget, kwargs in cases:
            ctx = ContextManager(strategy=NoCompact(), budget_tokens=budget, **kwargs)
            msgs = _make_messages_at_pct(8000, 0.6)
            assert ctx.check_thresholds(msgs) is None, label

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

    def test_importable_from_forge(self) -> None:
        from forge import default_context_warning

        assert callable(default_context_warning)

    def test_observed_usage_drives_threshold_callback(self) -> None:
        calls = []
        ctx = ContextManager(
            strategy=NoCompact(),
            budget_tokens=100,
            context_thresholds=[0.5],
            on_context_threshold=lambda tokens, budget, pct: calls.append((tokens, budget, pct))
            or "observed",
        )
        ctx.update_token_count(60)

        assert ctx.check_thresholds(ConversationState([_msg("tiny")])) == "observed"
        assert calls == [(60, 100, 0.6)]


# ── Integration: injection into run_inference ─────────────────────


class TestInferenceInjection:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("enabled", [True, False])
    async def test_warning_injected_transiently(self, enabled):
        from forge.core.inference import run_inference
        from forge.guardrails import ErrorTracker, ResponseValidator
        from tests.conftest import MockClient
        from tests.canonical import assistant, call

        ctx = ContextManager(
            NoCompact(),
            100,
            context_thresholds=[0.3] if enabled else None,
            on_context_threshold=default_context_warning if enabled else None,
        )
        state = ConversationState([_msg("system"), _msg("x" * 200)])
        client = MockClient([assistant(call("done"))])
        result = await run_inference(
            state,
            client,
            ctx,
            ResponseValidator(["done"], rescue_enabled=False),
            ErrorTracker(max_retries=1),
            [],
        )
        warnings = [
            m
            for m in client.send_calls[0].payload["messages"]
            if "Context usage" in m.get("content", "")
        ]
        assert len(warnings) == int(enabled)
        if enabled:
            assert warnings[0]["role"] == "user"
        assert not any(m.forge_purpose == "context_warning" for m in state.messages)
        assert sum(m.forge_purpose == "context_warning" for m in result.new_messages) == int(
            enabled
        )
