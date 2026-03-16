"""Tests for forge.context.strategies — CompactStrategy implementations."""

from forge.core.messages import Message, MessageMeta, MessageRole, MessageType
from forge.context.strategies import (
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


# ── NoCompact ────────────────────────────────────────────────────


class TestNoCompact:
    def test_returns_same_messages(self) -> None:
        msgs = _build_history(3)
        result, phase = NoCompact().compact(msgs, trigger_tokens=100)
        assert result == msgs
        assert phase == 0

    def test_returns_new_list(self) -> None:
        msgs = _build_history(3)
        result, _ = NoCompact().compact(msgs, trigger_tokens=100)
        assert result is not msgs

    def test_works_with_minimal_history(self) -> None:
        msgs = [_sys(), _user_input()]
        result, phase = NoCompact().compact(msgs, trigger_tokens=100)
        assert len(result) == 2
        assert phase == 0


# ── SlidingWindowCompact ─────────────────────────────────────────


class TestSlidingWindowCompact:
    def test_preserves_system_and_user(self) -> None:
        msgs = _build_history(6)
        result, phase = SlidingWindowCompact(keep_recent=2).compact(msgs, trigger_tokens=100)
        assert result[0].metadata.type == MessageType.SYSTEM_PROMPT
        assert result[1].metadata.type == MessageType.USER_INPUT
        assert phase == 1

    def test_keeps_only_last_n_pairs(self) -> None:
        msgs = _build_history(6)  # 2 + 12 = 14 messages
        result, _ = SlidingWindowCompact(keep_recent=2).compact(msgs, trigger_tokens=100)
        # system + user + last 4 messages = 6
        assert len(result) == 6
        # Last 4 should be the last 2 pairs (tool_call + tool_result each)
        assert result[2] is msgs[-4]
        assert result[-1] is msgs[-1]

    def test_short_history_returns_everything(self) -> None:
        msgs = _build_history(2)  # 2 + 4 = 6 messages
        result, phase = SlidingWindowCompact(keep_recent=3).compact(msgs, trigger_tokens=100)
        assert len(result) == len(msgs)
        assert phase == 1

    def test_returns_new_list(self) -> None:
        msgs = _build_history(4)
        result, _ = SlidingWindowCompact(keep_recent=2).compact(msgs, trigger_tokens=100)
        assert result is not msgs

    def test_exact_boundary(self) -> None:
        # Exactly keep_recent pairs + system + user — nothing to drop
        msgs = _build_history(3)  # 2 + 6 = 8 messages
        result, _ = SlidingWindowCompact(keep_recent=3).compact(msgs, trigger_tokens=100)
        assert len(result) == len(msgs)


# ── TieredCompact — Phase 1 ─────────────────────────────────────


class TestTieredPhase1:
    def test_drops_nudges_outside_keep_recent(self) -> None:
        msgs = _build_history(6, include_nudges=True)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        # Use a high trigger so Phase 1 alone is sufficient
        result, _ = TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=999999)
        nudge_types = {MessageType.STEP_NUDGE, MessageType.RETRY_NUDGE}
        # No nudges should remain outside keep_recent
        for msg in result:
            if msg.metadata.type in nudge_types:
                # If a nudge exists, it must be in the keep_recent window
                assert msg in msgs[eligible_end:]

    def test_truncates_tool_results_outside_keep_recent(self) -> None:
        # Build tool_results well over TRUNCATE_CHARS (200)
        msgs = [_sys(), _user_input()]
        big_result = "Header line\n" + "x" * 400  # 412 chars total
        for i in range(4):
            msgs.append(_tool_call(step=i))
            msgs.append(_tool_result(big_result, step=i))
        eligible_end = TieredCompact._find_eligible_end(msgs, 1)
        result, _ = TieredCompact(keep_recent=1).compact(msgs, trigger_tokens=999999)
        for msg in result:
            if msg.metadata.type == MessageType.TOOL_RESULT and msg not in msgs[eligible_end:]:
                assert "[Truncated —" in msg.content
                # Should keep first 200 chars + truncation marker
                before_marker = msg.content.split("\n[Truncated —")[0]
                assert len(before_marker) == TieredCompact.TRUNCATE_CHARS

    def test_keeps_text_response(self) -> None:
        """Phase 1 preserves TEXT_RESPONSE — it's the instruct model's reasoning."""
        msgs = _build_history(6, include_text_response=True, long_results=True)
        result, _ = TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=999999)
        original_text = [m for m in msgs if m.metadata.type == MessageType.TEXT_RESPONSE]
        result_text = [m for m in result if m.metadata.type == MessageType.TEXT_RESPONSE]
        assert len(result_text) == len(original_text)

    def test_keeps_recent_untouched(self) -> None:
        msgs = _build_history(6, long_results=True)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, _ = TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=999999)
        # Protected window should be identical to original
        assert result[-(len(msgs) - eligible_end):] == msgs[eligible_end:]

    def test_system_and_user_preserved(self) -> None:
        msgs = _build_history(6, include_nudges=True, long_results=True)
        result, _ = TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=999999)
        assert result[0].metadata.type == MessageType.SYSTEM_PROMPT
        assert result[1].metadata.type == MessageType.USER_INPUT

    def test_phase1_sufficient_stops_escalation(self) -> None:
        # Build history with nudges only (small content, Phase 1 drops them)
        msgs = _build_history(4, include_nudges=True)
        _, phase = TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=999999)
        assert phase == 1

    def test_short_tool_result_not_truncated(self) -> None:
        """Tool results under TRUNCATE_CHARS are kept as-is."""
        msgs = [_sys(), _user_input(), _tool_call(step=0), _tool_result("Short result", step=0)]
        msgs += [_tool_call(step=1), _tool_result("Recent", step=1)]
        result, _ = TieredCompact(keep_recent=1).compact(msgs, trigger_tokens=999999)
        for msg in result:
            if msg.content == "Short result":
                assert "[Truncated —" not in msg.content


# ── TieredCompact — Phase 2 ─────────────────────────────────────


class TestTieredPhase2:
    def test_drops_tool_results_outside_keep_recent(self) -> None:
        """Phase 2 drops tool_results entirely (not just truncates)."""
        msgs = _build_history(6, long_results=True)
        strategy = TieredCompact(keep_recent=2)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        tokens_after_p1 = _estimate_tokens(strategy._phase1(msgs, eligible_end))
        result, _ = strategy.compact(msgs, trigger_tokens=tokens_after_p1 - 1)
        # No tool_results should remain in the eligible zone
        protected_count = len(msgs) - eligible_end
        for msg in result[2:-protected_count]:
            assert msg.metadata.type != MessageType.TOOL_RESULT

    def test_preserves_reasoning_fully(self) -> None:
        msgs = _build_history(6, include_reasoning=True, long_results=True)
        strategy = TieredCompact(keep_recent=2)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        tokens_after_p1 = _estimate_tokens(strategy._phase1(msgs, eligible_end))
        result, _ = strategy.compact(msgs, trigger_tokens=tokens_after_p1 - 1)
        # All reasoning messages should be fully preserved
        original_reasoning = [m for m in msgs if m.metadata.type == MessageType.REASONING]
        result_reasoning = [m for m in result if m.metadata.type == MessageType.REASONING]
        assert len(result_reasoning) == len(original_reasoning)
        for orig, res in zip(original_reasoning, result_reasoning):
            assert res.content == orig.content

    def test_preserves_text_response(self) -> None:
        """Phase 2 preserves TEXT_RESPONSE — same as reasoning."""
        msgs = _build_history(6, include_text_response=True, long_results=True)
        strategy = TieredCompact(keep_recent=2)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        tokens_after_p1 = _estimate_tokens(strategy._phase1(msgs, eligible_end))
        result, _ = strategy.compact(msgs, trigger_tokens=tokens_after_p1 - 1)
        original_text = [m for m in msgs if m.metadata.type == MessageType.TEXT_RESPONSE]
        result_text = [m for m in result if m.metadata.type == MessageType.TEXT_RESPONSE]
        assert len(result_text) == len(original_text)

    def test_preserves_tool_calls(self) -> None:
        """Phase 2 keeps tool_call messages (skeleton of what was called)."""
        msgs = _build_history(6, long_results=True)
        strategy = TieredCompact(keep_recent=2)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        tokens_after_p1 = _estimate_tokens(strategy._phase1(msgs, eligible_end))
        result, _ = strategy.compact(msgs, trigger_tokens=tokens_after_p1 - 1)
        original_calls = [m for m in msgs if m.metadata.type == MessageType.TOOL_CALL]
        result_calls = [m for m in result if m.metadata.type == MessageType.TOOL_CALL]
        assert len(result_calls) == len(original_calls)

    def test_phase2_reached(self) -> None:
        msgs = _build_history(6, long_results=True)
        strategy = TieredCompact(keep_recent=2)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        tokens_after_p1 = _estimate_tokens(strategy._phase1(msgs, eligible_end))
        _, phase = strategy.compact(msgs, trigger_tokens=tokens_after_p1 - 1)
        assert phase == 2

    def test_recent_messages_untouched_in_phase2(self) -> None:
        msgs = _build_history(6, long_results=True)
        strategy = TieredCompact(keep_recent=2)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        tokens_after_p1 = _estimate_tokens(strategy._phase1(msgs, eligible_end))
        result, _ = strategy.compact(msgs, trigger_tokens=tokens_after_p1 - 1)
        protected_count = len(msgs) - eligible_end
        assert result[-protected_count:] == msgs[eligible_end:]


# ── TieredCompact — Phase 3 ─────────────────────────────────────


class TestTieredPhase3:
    def test_drops_reasoning_outside_keep_recent(self) -> None:
        """Phase 3 drops REASONING messages from the eligible zone."""
        msgs = _build_history(6, include_reasoning=True, long_results=True)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, _ = TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=0)
        protected_count = len(msgs) - eligible_end
        # No reasoning should remain in the eligible zone
        for msg in result[2:-protected_count]:
            assert msg.metadata.type != MessageType.REASONING

    def test_drops_text_response_outside_keep_recent(self) -> None:
        """Phase 3 drops TEXT_RESPONSE messages from the eligible zone."""
        msgs = _build_history(6, include_text_response=True, long_results=True)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, _ = TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=0)
        protected_count = len(msgs) - eligible_end
        for msg in result[2:-protected_count]:
            assert msg.metadata.type != MessageType.TEXT_RESPONSE

    def test_keeps_tool_call_skeleton(self) -> None:
        """Phase 3 preserves tool_call messages — the call history skeleton."""
        msgs = _build_history(6, include_reasoning=True, long_results=True)
        result, _ = TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=0)
        original_calls = [m for m in msgs if m.metadata.type == MessageType.TOOL_CALL]
        result_calls = [m for m in result if m.metadata.type == MessageType.TOOL_CALL]
        assert len(result_calls) == len(original_calls)

    def test_system_and_user_preserved(self) -> None:
        msgs = _build_history(6, long_results=True)
        result, _ = TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=0)
        assert result[0] is msgs[0]
        assert result[1] is msgs[1]

    def test_keep_recent_preserved(self) -> None:
        msgs = _build_history(6, long_results=True)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, _ = TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=0)
        protected_count = len(msgs) - eligible_end
        assert result[-protected_count:] == msgs[eligible_end:]

    def test_phase3_reached(self) -> None:
        msgs = _build_history(6, long_results=True, include_reasoning=True)
        _, phase = TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=0)
        assert phase == 3


# ── TieredCompact — Progressive Escalation ──────────────────────


class TestTieredEscalation:
    def test_phase1_sufficient(self) -> None:
        """When Phase 1 alone reduces tokens enough, Phase 2 doesn't fire."""
        # Build history with nudges (easy to drop)
        msgs = _build_history(6, include_nudges=True)
        # Use a very high trigger — Phase 1 will always be under it
        _, phase = TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=999999)
        assert phase == 1

    def test_phase1_insufficient_phase2_sufficient(self) -> None:
        """Phase 1 not enough, but Phase 2 (dropping tool_results) is enough."""
        msgs = _build_history(6, long_results=True, include_reasoning=True)
        strategy = TieredCompact(keep_recent=2)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)

        # Set trigger between Phase 1 and Phase 2 token counts
        tokens_after_p1 = _estimate_tokens(strategy._phase1(msgs, eligible_end))
        tokens_after_p2 = _estimate_tokens(strategy._phase2(msgs, eligible_end))

        # trigger must be > tokens_after_p2 (so Phase 2 succeeds)
        # and <= tokens_after_p1 (so Phase 1 fails)
        trigger = tokens_after_p2 + 1
        if trigger <= tokens_after_p1:
            result, phase = strategy.compact(msgs, trigger_tokens=trigger)
            assert phase == 2
            # Reasoning should be intact
            reasoning_msgs = [m for m in result if m.metadata.type == MessageType.REASONING]
            assert len(reasoning_msgs) > 0
            # Tool results should be gone from eligible zone
            protected_count = len(msgs) - eligible_end
            for msg in result[2:-protected_count]:
                assert msg.metadata.type != MessageType.TOOL_RESULT

    def test_only_phase3_saves_it(self) -> None:
        """Only Phase 3 (dropping reasoning + text_response) reduces tokens enough."""
        msgs = _build_history(8, long_results=True, include_reasoning=True)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        # trigger_tokens=0 forces through all phases
        result, phase = TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=0)
        assert phase == 3
        protected_count = len(msgs) - eligible_end
        # Tool_call skeleton should remain in eligible zone
        eligible_calls = [m for m in result[2:-protected_count] if m.metadata.type == MessageType.TOOL_CALL]
        assert len(eligible_calls) > 0
        # No reasoning in eligible zone
        for msg in result[2:-protected_count]:
            assert msg.metadata.type != MessageType.REASONING

    def test_no_eligible_messages_returns_unchanged(self) -> None:
        """When all messages are protected, nothing changes."""
        msgs = _build_history(2)  # 2 + 4 = 6 messages, keep_recent=3 protects 6
        result, _ = TieredCompact(keep_recent=3).compact(msgs, trigger_tokens=0)
        # Phase 3 fires but eligible_end == 2, so nothing to drop
        assert result[0].metadata.type == MessageType.SYSTEM_PROMPT
        assert result[1].metadata.type == MessageType.USER_INPUT
        assert len(result) == len(msgs)

    def test_does_not_mutate_input(self) -> None:
        """Compact must never mutate the input list."""
        msgs = _build_history(6, long_results=True)
        original = list(msgs)
        original_contents = [m.content for m in msgs]
        TieredCompact(keep_recent=2).compact(msgs, trigger_tokens=0)
        assert len(msgs) == len(original)
        for m, content in zip(msgs, original_contents):
            assert m.content == content
