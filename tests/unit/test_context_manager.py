"""Tests for forge.context.manager — ContextManager and CompactEvent."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from forge.core.messages import (
    ConversationState,
    RichMessage,
    MessageRole,
    TextPart,
    ToolResultPart,
)
from forge.core.messages import ReasoningPart, ToolCallPart, ValidArguments
from forge.context.accounting import estimate_content_tokens
from forge.context.manager import CompactEvent, ContextManager
from forge.context.observations import ContextSession, ContextUsage
from forge.context.strategies import NoCompact, SlidingWindowCompact, TieredCompact


# ── Helpers ──────────────────────────────────────────────────────


def _msg(content: str) -> RichMessage:
    return RichMessage(MessageRole.USER, [ToolResultPart("call", [TextPart(content)])])


def _build_messages(total_chars: int, count: int = 5) -> ConversationState:
    """Same content sizes and sequential post-header steps as the old fixture."""
    per_msg = total_chars // count
    return ConversationState(
        [
            RichMessage(
                MessageRole.SYSTEM
                if i == 0
                else MessageRole.USER
                if i == 1
                else MessageRole.ASSISTANT,
                [TextPart("x" * per_msg)],
                step_index=i - 2 if i >= 2 else None,
            )
            for i in range(count)
        ]
    )


class TestUnavailableBudget:
    def test_no_compact_without_budget_returns_original_messages(self) -> None:
        mgr = ContextManager(NoCompact(), budget_tokens=None)
        messages = ConversationState([_msg("x" * 100_000)])

        assert mgr.check_thresholds(messages) is None
        assert mgr.maybe_compact(messages) is messages

    def test_budget_dependent_features_without_budget_are_rejected(self) -> None:
        cases = [
            ("sliding strategy", SlidingWindowCompact(keep_recent=1), {}),
            ("tiered strategy", TieredCompact(), {}),
            (
                "thresholds",
                NoCompact(),
                {"context_thresholds": [0.5], "on_context_threshold": lambda *_: None},
            ),
            ("threshold callback", NoCompact(), {"on_context_threshold": lambda *_: None}),
        ]
        for label, strategy, kwargs in cases:
            with pytest.raises(ValueError) as exc_info:
                ContextManager(strategy, budget_tokens=None, **kwargs)
            assert "budget_tokens=None requires NoCompact" in str(exc_info.value), label


# ── estimate_tokens ─────────────────────────────────────────────


class TestEstimateTokens:
    def test_mixed_multiblock_accounting_counts_only_legacy_content(self):
        state = ConversationState(
            [
                RichMessage(
                    MessageRole.ASSISTANT,
                    [
                        TextPart("ab", {"ignored": "x" * 1000}),
                        ToolCallPart("read", ValidArguments({"huge": "x" * 10000}), "call"),
                        TextPart("cde"),
                        ReasoningPart("reason", field_name="reasoning_content"),
                        ToolResultPart(
                            "call",
                            [TextPart("fghi"), TextPart("jklmn")],
                            extra_fields={"ignored": "x" * 1000},
                        ),
                        ToolResultPart("other", [TextPart("op")]),
                    ],
                    extra_fields={"schema": "x" * 10000},
                )
            ]
        )
        before = deepcopy(state)
        manager = ContextManager(NoCompact(), 100)
        # Top-level text 2+1+3, reasoning 6, result 4+1+5, result 2.
        assert manager.estimate_tokens(state) == (6 + 6 + 10 + 2) // 4 == 6
        assert state == before
        manager.update_token_count(0)
        assert manager.estimate_tokens(state) == 0

    def test_uses_total_character_count_divided_by_four(self) -> None:
        cases = [
            ("multiple messages", [_msg("a" * 100), _msg("b" * 200)], 75),
            ("empty", [], 0),
            ("integer division", [_msg("a" * 41)], 10),
        ]
        for label, messages, expected in cases:
            mgr = ContextManager(NoCompact(), budget_tokens=1000)
            assert mgr.estimate_tokens(ConversationState(messages)) == expected, label

    def test_prefers_observed_usage_including_zero(self) -> None:
        mgr = ContextManager(NoCompact(), budget_tokens=1000)
        msgs = ConversationState([_msg("a" * 400)])
        mgr.update_token_count(0)
        assert mgr.estimate_tokens(msgs) == 0

    def test_negative_compatibility_update_invalidates(self) -> None:
        mgr = ContextManager(NoCompact(), budget_tokens=1000)
        mgr.update_token_count(25)
        mgr.update_token_count(-1)
        assert mgr.usage is None
        assert mgr.estimate_tokens(ConversationState([_msg("a" * 40)])) == 10


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
        usage = ContextUsage(current_usage_tokens=1, observed_at=datetime.now(timezone.utc))
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
        native = ContextUsage(current_usage_tokens=10, observed_at=datetime.now(timezone.utc))
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
    def test_returns_messages_unchanged_without_emitting_event(self) -> None:
        events: list[CompactEvent] = []
        mgr = ContextManager(NoCompact(), budget_tokens=10000, on_compact=events.append)
        msgs = ConversationState([_msg("short")])
        result = mgr.maybe_compact(msgs)
        assert result is msgs  # Same object — not compacted
        assert events == []


# ── maybe_compact — over threshold ──────────────────────────────


class TestMaybeCompactOverThreshold:
    def test_tiered_rewrite_invalidates_and_resets_warning_threshold(self):
        events = []
        manager = ContextManager(
            TieredCompact(1, phase_thresholds=(0.5, 0.7, 0.9)),
            1000,
            on_compact=events.append,
            context_thresholds=[0.5],
            on_context_threshold=lambda *_: "warning",
        )
        state = ConversationState(
            [
                RichMessage(MessageRole.SYSTEM, [TextPart("sys")]),
                RichMessage(MessageRole.USER, [TextPart("user")]),
                RichMessage(MessageRole.TOOL, [ToolResultPart("old", [TextPart("x" * 450)])], 0),
                RichMessage(MessageRole.ASSISTANT, [TextPart("recent")], 1),
            ]
        )
        manager.update_token_count(800)
        assert manager.check_thresholds(state) == "warning"
        compacted = manager.maybe_compact(state, step_index=7, step_hint="hint")
        assert state.messages[2].parts[0].content[0].text == "x" * 450
        assert manager.usage is None
        assert events[0].phase_reached == 1
        assert events[0].tokens_before == 800
        assert events[0].tokens_after == estimate_content_tokens(compacted)
        assert events[0].step_index == 7
        assert manager.check_thresholds(compacted) is None
        manager.update_token_count(800)
        assert manager.check_thresholds(compacted) == "warning"

    def test_returns_compacted_messages(self) -> None:
        # Budget=100, threshold=0.75 -> trigger at 75 tokens = 300 chars
        # Build messages with ~400 chars total
        msgs = _build_messages(total_chars=400, count=6)
        mgr = ContextManager(
            SlidingWindowCompact(keep_recent=1, compact_threshold=0.75), budget_tokens=100
        )
        result = mgr.maybe_compact(msgs)
        assert len(result.messages) < len(msgs.messages)

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
        assert events[0].tokens_after == estimate_content_tokens(result)

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
    def test_per_phase_thresholds_through_manager(self) -> None:
        """Per-phase thresholds on TieredCompact flow through ContextManager."""
        events: list[CompactEvent] = []
        msgs = _build_messages(total_chars=800, count=8)
        tokens = estimate_content_tokens(msgs)  # ~200
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
