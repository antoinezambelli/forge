"""Tests for forge.context.strategies — CompactStrategy implementations."""

from forge.core.messages import Message, MessageMeta, MessageRole, MessageType
from forge.context.strategies import (
    CompactStrategy,
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
    _estimate_tokens,
)


# ── Helpers ──────────────────────────────────────────────────────


def _sys(content: str = "You are a helpful assistant.") -> Message:
    return Message(
        role=MessageRole.SYSTEM,
        content=content,
        metadata=MessageMeta(type=MessageType.SYSTEM_PROMPT),
    )


def _user_input(content: str = "Generate a quote for part X.") -> Message:
    return Message(
        role=MessageRole.USER,
        content=content,
        metadata=MessageMeta(type=MessageType.USER_INPUT),
    )


def _tool_call(content: str = "get_pricing({\"part\": \"X\"})", step: int = 0) -> Message:
    return Message(
        role=MessageRole.ASSISTANT,
        content=content,
        metadata=MessageMeta(type=MessageType.TOOL_CALL, step_index=step),
    )


def _tool_result(content: str = "Price: $10.69\nMOQ: 100\nLead time: 2 weeks", step: int = 0) -> Message:
    return Message(
        role=MessageRole.USER,
        content=content,
        metadata=MessageMeta(type=MessageType.TOOL_RESULT, step_index=step),
    )


def _reasoning(content: str = "The price is $10.69 at MOQ 100. Historical average was $9.50.", step: int = 0) -> Message:
    return Message(
        role=MessageRole.ASSISTANT,
        content=content,
        metadata=MessageMeta(type=MessageType.REASONING, step_index=step),
    )


def _step_nudge(content: str = "You must call get_pricing first.") -> Message:
    return Message(
        role=MessageRole.USER,
        content=content,
        metadata=MessageMeta(type=MessageType.STEP_NUDGE),
    )


def _retry_nudge(content: str = "Invalid format. Please try again.") -> Message:
    return Message(
        role=MessageRole.USER,
        content=content,
        metadata=MessageMeta(type=MessageType.RETRY_NUDGE),
    )


def _text_response(content: str = "Based on the data, the price looks reasonable.", step: int = 0) -> Message:
    return Message(
        role=MessageRole.ASSISTANT,
        content=content,
        metadata=MessageMeta(type=MessageType.TEXT_RESPONSE, step_index=step),
    )


def _build_history(
    num_pairs: int,
    *,
    include_nudges: bool = False,
    include_reasoning: bool = False,
    include_text_response: bool = False,
    long_results: bool = False,
    long_calls: bool = False,
) -> list[Message]:
    """Build a test message history with system prompt, user input, and N pairs."""
    msgs: list[Message] = [_sys(), _user_input()]
    for i in range(num_pairs):
        if include_reasoning:
            msgs.append(_reasoning(f"Analysis of result {i}: looks good.", step=i))
        if include_text_response:
            msgs.append(_text_response(f"Based on my analysis of step {i}, proceeding.", step=i))
        call_content = f"tool_{i}({{\"arg\": \"{i}\"}})"
        if long_calls:
            call_content += "\n" + f"Detailed explanation of tool_{i} call with arguments." * 5
        msgs.append(_tool_call(call_content, step=i))

        result_content = f"Result {i}"
        if long_results:
            result_content += "\n" + f"Detailed data line {i}\n" * 10
        msgs.append(_tool_result(result_content, step=i))

        if include_nudges and i % 2 == 0:
            msgs.append(_step_nudge(f"Nudge at step {i}"))
    return msgs


# Helper strategies for controlling phase escalation in tests.
# phase_thresholds=(0.0, 1.0, 1.0) → Phase 1 always fires, 2 and 3 never.
# phase_thresholds=(0.0, 0.0, 1.0) → Phase 1 and 2 fire, 3 never.
# compact_threshold=0.0             → All phases fire (uniform 0.0).

def _p1_only(keep_recent: int = 2) -> TieredCompact:
    return TieredCompact(keep_recent=keep_recent, phase_thresholds=(0.0, 1.0, 1.0))

def _p1_p2_only(keep_recent: int = 2) -> TieredCompact:
    return TieredCompact(keep_recent=keep_recent, phase_thresholds=(0.0, 0.0, 1.0))

def _all_phases(keep_recent: int = 2) -> TieredCompact:
    return TieredCompact(keep_recent=keep_recent, compact_threshold=0.0)

# Budget large enough that content is always "over threshold" when
# compact_threshold or phase_thresholds are 0.0.
BIG = 999999


class MinimalCustomStrategy(CompactStrategy):
    """A pre-existing custom strategy implementing only the public method."""

    def __init__(self) -> None:
        self.calls = 0

    def compact(
        self,
        messages: list[Message],
        budget_tokens: int,
        *,
        step_hint: str = "",
    ) -> tuple[list[Message], int]:
        self.calls += 1
        return list(messages), 0


# ── NoCompact ────────────────────────────────────────────────────


class TestNoCompact:
    def test_returns_an_equal_new_list_without_compaction(self) -> None:
        for label, msgs in [
            ("history", _build_history(3)),
            ("minimal", [_sys(), _user_input()]),
        ]:
            result, phase = NoCompact().compact(msgs, budget_tokens=BIG)
            assert result == msgs, label
            assert result is not msgs, label
            assert phase == 0, label

    def test_minimal_custom_strategy_uses_unchanged_public_contract(self) -> None:
        from forge.context.manager import ContextManager

        strategy = MinimalCustomStrategy()
        msgs = _build_history(1)

        assert ContextManager(strategy, budget_tokens=10).maybe_compact(msgs) is msgs
        assert strategy.calls == 1


# ── SlidingWindowCompact ─────────────────────────────────────────


class TestSlidingWindowCompact:
    def test_keeps_only_last_n_pairs(self) -> None:
        msgs = _build_history(6)  # 2 + 12 = 14 messages
        result, phase = SlidingWindowCompact(keep_recent=2, compact_threshold=0.0).compact(msgs, budget_tokens=BIG)
        # system + user + last 4 messages = 6
        assert len(result) == 6
        assert result[0].metadata.type == MessageType.SYSTEM_PROMPT
        assert result[1].metadata.type == MessageType.USER_INPUT
        # Last 4 should be the last 2 pairs (tool_call + tool_result each)
        assert result[2] is msgs[-4]
        assert result[-1] is msgs[-1]
        assert phase == 1

    def test_histories_within_window_return_everything(self) -> None:
        for pairs in [2, 3]:
            msgs = _build_history(pairs)
            result, phase = SlidingWindowCompact(
                keep_recent=3, compact_threshold=0.0
            ).compact(msgs, budget_tokens=BIG)
            assert len(result) == len(msgs), f"pairs={pairs}"
            assert phase == 1, f"pairs={pairs}"

    def test_returns_new_list(self) -> None:
        msgs = _build_history(4)
        result, _ = SlidingWindowCompact(keep_recent=2, compact_threshold=0.0).compact(msgs, budget_tokens=BIG)
        assert result is not msgs

    def test_no_compact_when_under_threshold(self) -> None:
        msgs = _build_history(3)
        # threshold=0.99 with huge budget — will never trigger
        result, phase = SlidingWindowCompact(keep_recent=1, compact_threshold=0.99).compact(msgs, budget_tokens=BIG)
        assert phase == 0
        assert len(result) == len(msgs)


# ── TieredCompact — Phase 1 ─────────────────────────────────────


class TestTieredPhase1:
    def test_applies_phase_one_rewrites_only_to_eligible_history(self) -> None:
        msgs = _build_history(
            6, include_nudges=True, include_text_response=True, long_results=True
        )
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, phase = _p1_only().compact(msgs, budget_tokens=BIG)
        nudge_types = {MessageType.STEP_NUDGE, MessageType.RETRY_NUDGE}
        for msg in result:
            if msg.metadata.type in nudge_types:
                assert msg in msgs[eligible_end:]
        for msg in result:
            if (
                msg.metadata.type == MessageType.TOOL_RESULT
                and msg not in msgs[eligible_end:]
            ):
                assert "[Truncated —" in msg.content
                before_marker = msg.content.split("\n[Truncated —")[0]
                assert len(before_marker) == TieredCompact.TRUNCATE_CHARS
        original_text = [m for m in msgs if m.metadata.type == MessageType.TEXT_RESPONSE]
        result_text = [m for m in result if m.metadata.type == MessageType.TEXT_RESPONSE]
        assert len(result_text) == len(original_text)
        assert phase == 1

    def test_preserves_header_and_recent_window(self) -> None:
        msgs = _build_history(6, long_results=True)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, _ = _p1_only().compact(msgs, budget_tokens=BIG)
        assert result[0].metadata.type == MessageType.SYSTEM_PROMPT
        assert result[1].metadata.type == MessageType.USER_INPUT
        assert result[-(len(msgs) - eligible_end):] == msgs[eligible_end:]

    def test_short_tool_result_not_truncated(self) -> None:
        """Tool results under TRUNCATE_CHARS are kept as-is."""
        msgs = [_sys(), _user_input(), _tool_call(step=0), _tool_result("Short result", step=0)]
        msgs += [_tool_call(step=1), _tool_result("Recent", step=1)]
        result, _ = _p1_only(keep_recent=1).compact(msgs, budget_tokens=BIG)
        for msg in result:
            if msg.content == "Short result":
                assert "[Truncated —" not in msg.content


# ── TieredCompact — Phase 2 ─────────────────────────────────────


class TestTieredPhase2:
    def test_drops_results_but_preserves_interpretive_history_and_calls(self) -> None:
        msgs = _build_history(
            6, long_results=True, include_reasoning=True, include_text_response=True
        )
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, phase = _p1_p2_only().compact(msgs, budget_tokens=BIG)
        protected_count = len(msgs) - eligible_end
        for msg in result[2:-protected_count]:
            assert msg.metadata.type != MessageType.TOOL_RESULT
        for msg_type in [
            MessageType.REASONING,
            MessageType.TEXT_RESPONSE,
            MessageType.TOOL_CALL,
        ]:
            original = [m.content for m in msgs if m.metadata.type == msg_type]
            compacted = [m.content for m in result if m.metadata.type == msg_type]
            assert compacted == original, msg_type
        assert phase == 2

    def test_recent_messages_untouched_in_phase2(self) -> None:
        msgs = _build_history(6, long_results=True)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, _ = _p1_p2_only().compact(msgs, budget_tokens=BIG)
        protected_count = len(msgs) - eligible_end
        assert result[-protected_count:] == msgs[eligible_end:]


# ── TieredCompact — Phase 3 ─────────────────────────────────────


class TestTieredPhase3:
    def test_drops_interpretive_history_but_keeps_tool_call_skeleton(self) -> None:
        msgs = _build_history(
            6, include_reasoning=True, include_text_response=True, long_results=True
        )
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, phase = _all_phases().compact(msgs, budget_tokens=BIG)
        protected_count = len(msgs) - eligible_end
        for msg in result[2:-protected_count]:
            assert msg.metadata.type not in {
                MessageType.REASONING,
                MessageType.TEXT_RESPONSE,
            }
        original_calls = [m for m in msgs if m.metadata.type == MessageType.TOOL_CALL]
        result_calls = [m for m in result if m.metadata.type == MessageType.TOOL_CALL]
        assert len(result_calls) == len(original_calls)
        assert phase == 3

    def test_preserves_header_and_recent_window(self) -> None:
        msgs = _build_history(6, long_results=True)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, _ = _all_phases().compact(msgs, budget_tokens=BIG)
        assert result[0] is msgs[0]
        assert result[1] is msgs[1]
        protected_count = len(msgs) - eligible_end
        assert result[-protected_count:] == msgs[eligible_end:]


# ── TieredCompact — Progressive Escalation ──────────────────────


class TestTieredEscalation:
    def test_no_eligible_messages_returns_unchanged(self) -> None:
        """When all messages are protected, nothing changes."""
        msgs = _build_history(2)  # 2 + 4 = 6 messages, keep_recent=3 protects 6
        result, _ = _all_phases(keep_recent=3).compact(msgs, budget_tokens=BIG)
        assert result[0].metadata.type == MessageType.SYSTEM_PROMPT
        assert result[1].metadata.type == MessageType.USER_INPUT
        assert len(result) == len(msgs)

    def test_does_not_mutate_input(self) -> None:
        """Compact must never mutate the input list."""
        msgs = _build_history(6, long_results=True)
        original = list(msgs)
        original_contents = [m.content for m in msgs]
        _all_phases().compact(msgs, budget_tokens=BIG)
        assert len(msgs) == len(original)
        for m, content in zip(msgs, original_contents):
            assert m.content == content

    def test_no_compaction_when_under_threshold(self) -> None:
        """Returns phase 0 when tokens are under the lowest threshold."""
        msgs = _build_history(3)  # Small content
        strategy = TieredCompact(keep_recent=2, compact_threshold=0.99)
        result, phase = strategy.compact(msgs, budget_tokens=BIG)
        assert phase == 0
        assert len(result) == len(msgs)

    def test_per_phase_thresholds(self) -> None:
        """Per-phase thresholds control which phases fire independently."""
        msgs = _build_history(6, long_results=True, include_reasoning=True)

        # Phase 1 only
        _, phase = _p1_only().compact(msgs, budget_tokens=BIG)
        assert phase == 1

        # Phase 1 + 2
        _, phase = _p1_p2_only().compact(msgs, budget_tokens=BIG)
        assert phase == 2

        # All phases
        _, phase = _all_phases().compact(msgs, budget_tokens=BIG)
        assert phase == 3

    def test_observed_usage_survives_phase1_noop_until_phase2_rewrite(self) -> None:
        msgs = _build_history(4)
        strategy = TieredCompact(
            keep_recent=1,
            phase_thresholds=(0.5, 0.7, 0.9),
        )

        result, phase = strategy._compact_with_initial_usage(
            msgs,
            budget_tokens=100,
            initial_tokens=80,
        )

        assert phase == 2
        assert result != msgs

    def test_real_phase1_rewrite_switches_to_heuristic(self) -> None:
        msgs = _build_history(4, long_results=True)
        strategy = TieredCompact(
            keep_recent=1,
            phase_thresholds=(0.5, 0.5, 0.9),
        )

        result, phase = strategy._compact_with_initial_usage(
            msgs,
            budget_tokens=1000,
            initial_tokens=600,
        )

        assert result != msgs
        assert _estimate_tokens(result) < 500
        assert phase == 1
