"""Tests for forge.context.manager — ContextManager and CompactEvent."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from forge.core.messages import Message, MessageMeta, MessageRole, MessageType
from forge.context.manager import CompactEvent, ContextManager
from forge.context.observations import ContextSession, ContextUsage
from forge.context.strategies import NoCompact, SlidingWindowCompact, TieredCompact


# ── Helpers ──────────────────────────────────────────────────────


def _msg(content: str, msg_type: MessageType = MessageType.TOOL_RESULT) -> Message:
    return Message(
        role=MessageRole.USER,
        content=content,
        metadata=MessageMeta(type=msg_type),
    )


def _build_messages(total_chars: int, count: int = 5) -> list[Message]:
    """Build messages with approximately total_chars of content.

    Messages after the header (index 0, 1) get sequential step_index values
    so that _find_eligible_end() can identify iteration boundaries.
    """
    per_msg = total_chars // count
    msgs: list[Message] = []
    step = 0
    for i in range(count):
        if i == 0:
            msgs.append(Message(
                role=MessageRole.SYSTEM,
                content="x" * per_msg,
                metadata=MessageMeta(type=MessageType.SYSTEM_PROMPT),
            ))
        elif i == 1:
            msgs.append(Message(
                role=MessageRole.USER,
                content="x" * per_msg,
                metadata=MessageMeta(type=MessageType.USER_INPUT),
            ))
        else:
            msgs.append(Message(
                role=MessageRole.ASSISTANT,
                content="x" * per_msg,
                metadata=MessageMeta(type=MessageType.TOOL_CALL, step_index=step),
            ))
            step += 1
    return msgs


class TestUnavailableBudget:
    def test_no_compact_without_budget_returns_original_messages(self) -> None:
        mgr = ContextManager(NoCompact(), budget_tokens=None)
        messages = [_msg("x" * 100_000)]

        assert mgr.check_thresholds(messages) is None
        assert mgr.maybe_compact(messages) is messages

    @pytest.mark.parametrize(
        "strategy",
        [SlidingWindowCompact(keep_recent=1), TieredCompact()],
    )
    def test_compacting_strategy_without_budget_is_rejected(self, strategy) -> None:
        with pytest.raises(ValueError, match="budget_tokens=None requires NoCompact"):
            ContextManager(strategy, budget_tokens=None)

    def test_thresholds_without_budget_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="no context thresholds"):
            ContextManager(
                NoCompact(),
                budget_tokens=None,
                context_thresholds=[0.5],
                on_context_threshold=lambda *_: None,
            )

    def test_threshold_callback_without_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no context thresholds"):
            ContextManager(
                NoCompact(),
                budget_tokens=None,
                on_context_threshold=lambda *_: None,
            )


# ── estimate_tokens ─────────────────────────────────────────────


class TestEstimateTokens:
    def test_basic(self) -> None:
        mgr = ContextManager(NoCompact(), budget_tokens=1000)
        msgs = [_msg("a" * 100), _msg("b" * 200)]
        assert mgr.estimate_tokens(msgs) == 300 // 4

    def test_empty(self) -> None:
        mgr = ContextManager(NoCompact(), budget_tokens=1000)
        assert mgr.estimate_tokens([]) == 0

    def test_char_div_4(self) -> None:
        mgr = ContextManager(NoCompact(), budget_tokens=1000)
        msgs = [_msg("a" * 41)]  # 41 / 4 = 10 (integer division)
        assert mgr.estimate_tokens(msgs) == 10

    def test_prefers_observed_usage_including_zero(self) -> None:
        mgr = ContextManager(NoCompact(), budget_tokens=1000)
        msgs = [_msg("a" * 400)]
        mgr.update_token_count(0)
        assert mgr.estimate_tokens(msgs) == 0

    def test_negative_compatibility_update_invalidates(self) -> None:
        mgr = ContextManager(NoCompact(), budget_tokens=1000)
        mgr.update_token_count(25)
        mgr.update_token_count(-1)
        assert mgr.usage is None
        assert mgr.estimate_tokens([_msg("a" * 40)]) == 10


class TestContextUsage:
    def test_record_read_invalidate_and_capacity_independence(self) -> None:
        mgr = ContextManager(NoCompact(), budget_tokens=123)
        usage = ContextUsage(
            current_usage_tokens=0,
            context_window_tokens=8192,
            model="opaque-model",
            context_window_source="opaque-source",
            observed_at=datetime.now(timezone.utc),
            session=ContextSession(id="opaque-id", source="opaque-session-source"),
        )

        mgr.record_usage(usage)

        assert mgr.usage is usage
        assert mgr.budget_tokens == 123
        mgr.invalidate_usage()
        assert mgr.usage is None

    def test_values_are_frozen_and_timestamp_must_be_utc_aware(self) -> None:
        usage = ContextUsage(
            current_usage_tokens=1,
            observed_at=datetime.now(timezone.utc),
        )
        with pytest.raises(FrozenInstanceError):
            usage.current_usage_tokens = 2  # type: ignore[misc]
        with pytest.raises(ValueError, match="UTC-aware"):
            ContextUsage(current_usage_tokens=1, observed_at=datetime.now())

    def test_scalar_adapter_records_aware_utc_time(self) -> None:
        mgr = ContextManager(NoCompact(), budget_tokens=1000)
        mgr.update_token_count(12)
        assert mgr.usage is not None
        assert mgr.usage.current_usage_tokens == 12
        assert mgr.usage.observed_at.utcoffset() is not None
        assert mgr.usage.observed_at.utcoffset().total_seconds() == 0

    def test_exports_are_additive(self) -> None:
        from forge import ContextUsage as TopLevelUsage
        from forge.context import ContextSession as ContextPackageSession

        assert TopLevelUsage is ContextUsage
        assert ContextPackageSession is ContextSession


class TestPublishedProxyUsage:
    def test_published_slot_is_distinct_from_native_policy_usage(self):
        manager = ContextManager(NoCompact(), budget_tokens=1000)
        native = ContextUsage(
            current_usage_tokens=10,
            observed_at=datetime.now(timezone.utc),
        )
        published = ContextUsage(
            current_usage_tokens=20,
            context_window_tokens=100,
            model="reported",
            context_window_source="backend_metadata",
            observed_at=datetime.now(timezone.utc),
        )

        manager.record_usage(native)
        manager.record_published_usage(published)
        manager.invalidate_usage()

        assert manager.usage is None
        assert manager.published_usage is published
        manager.clear_published_usage()
        assert manager.published_usage is None


# ── maybe_compact — under threshold ─────────────────────────────


class TestMaybeCompactUnderThreshold:
    def test_returns_messages_unchanged(self) -> None:
        mgr = ContextManager(NoCompact(), budget_tokens=10000)
        msgs = [_msg("short")]
        result = mgr.maybe_compact(msgs)
        assert result is msgs  # Same object — not compacted

    def test_on_compact_not_called(self) -> None:
        events: list[CompactEvent] = []
        mgr = ContextManager(NoCompact(), budget_tokens=10000, on_compact=events.append)
        msgs = [_msg("short")]
        mgr.maybe_compact(msgs)
        assert len(events) == 0


# ── maybe_compact — over threshold ──────────────────────────────


class TestMaybeCompactOverThreshold:
    def test_returns_compacted_messages(self) -> None:
        # Budget=100, threshold=0.75 -> trigger at 75 tokens = 300 chars
        # Build messages with ~400 chars total
        msgs = _build_messages(total_chars=400, count=6)
        mgr = ContextManager(
            SlidingWindowCompact(keep_recent=1, compact_threshold=0.75),
            budget_tokens=100,
        )
        result = mgr.maybe_compact(msgs)
        assert len(result) < len(msgs)

    def test_on_compact_called_with_event(self) -> None:
        events: list[CompactEvent] = []
        msgs = _build_messages(total_chars=400, count=6)
        mgr = ContextManager(
            SlidingWindowCompact(keep_recent=1, compact_threshold=0.75),
            budget_tokens=100,
            on_compact=events.append,
        )
        mgr.maybe_compact(msgs, step_index=3, step_hint="hint")
        assert len(events) == 1
        event = events[0]
        assert event.step_index == 3
        assert event.tokens_before > event.tokens_after
        assert event.budget_tokens == 100
        assert event.messages_before == 6
        assert event.messages_after < 6

    def test_on_compact_none_no_error(self) -> None:
        msgs = _build_messages(total_chars=400, count=6)
        mgr = ContextManager(
            SlidingWindowCompact(keep_recent=1, compact_threshold=0.75),
            budget_tokens=100,
            on_compact=None,
        )
        # Should not raise
        result = mgr.maybe_compact(msgs)
        assert len(result) < len(msgs)

    def test_observed_trigger_invalidates_after_real_rewrite(self) -> None:
        events: list[CompactEvent] = []
        msgs = _build_messages(total_chars=400, count=6)
        mgr = ContextManager(
            SlidingWindowCompact(keep_recent=1, compact_threshold=0.5),
            budget_tokens=1000,
            on_compact=events.append,
        )
        mgr.update_token_count(600)

        result = mgr.maybe_compact(msgs)

        assert result != msgs
        assert mgr.usage is None
        assert events[0].tokens_before == 600
        assert events[0].tokens_after == sum(len(m.content) for m in result) // 4

    def test_value_equal_phase_retains_usage(self) -> None:
        events: list[CompactEvent] = []
        msgs = _build_messages(total_chars=400, count=4)
        mgr = ContextManager(
            SlidingWindowCompact(keep_recent=3, compact_threshold=0.5),
            budget_tokens=1000,
            on_compact=events.append,
        )
        mgr.update_token_count(600)

        result = mgr.maybe_compact(msgs)

        assert result == msgs
        assert result is not msgs
        assert mgr.usage is not None
        assert events[0].tokens_before == 600
        assert events[0].tokens_after == 600


# ── CompactEvent fields ─────────────────────────────────────────


class TestCompactEvent:
    def test_fields_accurate(self) -> None:
        events: list[CompactEvent] = []
        msgs = _build_messages(total_chars=400, count=6)
        mgr = ContextManager(
            SlidingWindowCompact(keep_recent=1, compact_threshold=0.75),
            budget_tokens=100,
            on_compact=events.append,
        )
        mgr.maybe_compact(msgs, step_index=5)
        event = events[0]
        assert event.tokens_before == mgr.estimate_tokens(msgs)
        assert event.messages_before == 6
        assert event.budget_tokens == 100

    def test_phase_reached_from_tiered(self) -> None:
        events: list[CompactEvent] = []
        # Build enough content to trigger compaction
        msgs = _build_messages(total_chars=800, count=8)
        mgr = ContextManager(
            TieredCompact(keep_recent=2, compact_threshold=0.75),
            budget_tokens=100,
            on_compact=events.append,
        )
        mgr.maybe_compact(msgs, step_index=1, step_hint="[Steps completed: a]")
        assert len(events) == 1
        assert events[0].phase_reached >= 1

    def test_no_compact_when_under_threshold(self) -> None:
        events: list[CompactEvent] = []
        # Small content, high threshold — should not compact
        msgs = _build_messages(total_chars=100, count=4)
        mgr = ContextManager(
            TieredCompact(keep_recent=2, compact_threshold=0.99),
            budget_tokens=10000,
            on_compact=events.append,
        )
        result = mgr.maybe_compact(msgs)
        assert result is msgs  # Same object — not compacted
        assert len(events) == 0

    def test_per_phase_thresholds_through_manager(self) -> None:
        """Per-phase thresholds on TieredCompact flow through ContextManager."""
        events: list[CompactEvent] = []
        msgs = _build_messages(total_chars=800, count=8)
        tokens = sum(len(m.content) for m in msgs) // 4  # ~200
        # Set Phase 2 trigger above the token count so Phase 1 result
        # is always "under threshold" and escalation stops.
        # Phase 1 trigger: 0.0 * budget = 0 (fires)
        # Phase 2 trigger: very high (never escalates)
        budget = tokens * 10  # budget much larger than content
        mgr = ContextManager(
            TieredCompact(keep_recent=2, phase_thresholds=(0.0, 1.0, 1.0)),
            budget_tokens=budget,
            on_compact=events.append,
        )
        mgr.maybe_compact(msgs)
        assert len(events) == 1
        assert events[0].phase_reached == 1

    def test_all_phases_through_manager(self) -> None:
        events: list[CompactEvent] = []
        msgs = _build_messages(total_chars=800, count=8)
        mgr = ContextManager(
            TieredCompact(keep_recent=2, compact_threshold=0.0),
            budget_tokens=100,
            on_compact=events.append,
        )
        mgr.maybe_compact(msgs)
        assert len(events) == 1
        assert events[0].phase_reached == 3

    def test_frozen(self) -> None:
        event = CompactEvent(
            step_index=0,
            tokens_before=100,
            tokens_after=50,
            budget_tokens=200,
            messages_before=10,
            messages_after=6,
            phase_reached=1,
        )
        try:
            event.tokens_before = 999  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass
