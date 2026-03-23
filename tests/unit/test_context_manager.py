"""Tests for forge.context.manager — ContextManager and CompactEvent."""

from forge.core.messages import Message, MessageMeta, MessageRole, MessageType
from forge.context.manager import CompactEvent, ContextManager
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
