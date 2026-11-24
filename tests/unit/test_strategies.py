"""Tests for forge.context.strategies — CompactStrategy implementations."""

from copy import deepcopy
import pytest
from forge.core.messages import (
    ConversationState,
    RichMessage,
    MessageRole,
    TextPart,
    ReasoningPart,
    ToolCallPart,
    ToolResultPart,
    ValidArguments,
)
from forge.context.accounting import estimate_content_tokens
from forge.context.strategies import CompactStrategy, NoCompact, SlidingWindowCompact, TieredCompact


def _sys(content="You are a helpful assistant."):
    return RichMessage(MessageRole.SYSTEM, [TextPart(content)])


def _user_input(content="Generate a quote for part X."):
    return RichMessage(MessageRole.USER, [TextPart(content)])


def _tool_call(content='get_pricing({"part": "X"})', step=0):
    return RichMessage(
        MessageRole.ASSISTANT,
        [
            TextPart(content),
            ToolCallPart("get_pricing", ValidArguments({"part": "X"}), f"call-{step}"),
        ],
        step_index=step,
    )


def _tool_result(content="Price: $10.69\nMOQ: 100\nLead time: 2 weeks", step=0):
    return RichMessage(
        MessageRole.USER, [ToolResultPart(f"call-{step}", [TextPart(content)])], step_index=step
    )


def _reasoning(content="The price is $10.69 at MOQ 100. Historical average was $9.50.", step=0):
    return RichMessage(MessageRole.ASSISTANT, [ReasoningPart(content)], step_index=step)


def _step_nudge(content="You must call get_pricing first."):
    return RichMessage(MessageRole.USER, [TextPart(content)], forge_purpose="step")


def _retry_nudge(content="Invalid format. Please try again."):
    return RichMessage(MessageRole.USER, [TextPart(content)], forge_purpose="retry")


def _text_response(content="Based on the data, the price looks reasonable.", step=0):
    return RichMessage(MessageRole.ASSISTANT, [TextPart(content)], step_index=step)


def _build_history(
    num_pairs,
    *,
    include_nudges=False,
    include_reasoning=False,
    include_text_response=False,
    long_results=False,
    long_calls=False,
):
    # Original fixture content/steps retained through the canonical bindings.
    msgs = [_sys(), _user_input()]
    for i in range(num_pairs):
        if include_reasoning:
            msgs.append(_reasoning(f"Analysis of result {i}: looks good.", step=i))
        if include_text_response:
            msgs.append(_text_response(f"Based on my analysis of step {i}, proceeding.", step=i))
        call_content = f'tool_{i}({{"arg": "{i}"}})'
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
        self, messages: ConversationState, budget_tokens: int, *, step_hint: str = ""
    ) -> tuple[ConversationState, int]:
        self.calls += 1
        return ConversationState(list(messages.messages)), 0


# ── NoCompact ────────────────────────────────────────────────────


class TestNoCompact:
    def test_returns_same_state_without_compaction(self) -> None:
        for label, msgs in [("history", _build_history(3)), ("minimal", [_sys(), _user_input()])]:
            state = ConversationState(msgs)
            before = deepcopy(state)
            result, phase = NoCompact().compact(state, budget_tokens=BIG)
            assert state == before
            assert result.messages == msgs, label
            assert result is state, label
            assert phase == 0, label

    def test_minimal_custom_strategy_uses_unchanged_public_contract(self) -> None:
        from forge.context.manager import ContextManager

        strategy = MinimalCustomStrategy()
        msgs = ConversationState(_build_history(1))

        assert ContextManager(strategy, budget_tokens=10).maybe_compact(msgs) is msgs
        assert strategy.calls == 1


# ── SlidingWindowCompact ─────────────────────────────────────────


class TestSlidingWindowCompact:
    def test_keeps_only_last_n_pairs(self) -> None:
        msgs = _build_history(6)  # 2 + 12 = 14 messages
        result, phase = SlidingWindowCompact(keep_recent=2, compact_threshold=0.0).compact(
            ConversationState(msgs), budget_tokens=BIG
        )
        result = result.messages
        # system + user + last 4 messages = 6
        assert len(result) == 6
        assert result[0] is msgs[0]
        assert result[1] is msgs[1]
        # Last 4 should be the last 2 pairs (tool_call + tool_result each)
        assert result[2] is msgs[-4]
        assert result[-1] is msgs[-1]
        assert phase == 1

    def test_histories_within_window_return_everything(self) -> None:
        for pairs in [2, 3]:
            msgs = _build_history(pairs)
            result, phase = SlidingWindowCompact(keep_recent=3, compact_threshold=0.0).compact(
                ConversationState(msgs), budget_tokens=BIG
            )
            assert result.messages == msgs, f"pairs={pairs}"
            assert phase == 1, f"pairs={pairs}"

    def test_returns_new_list(self) -> None:
        msgs = _build_history(4)
        result, _ = SlidingWindowCompact(keep_recent=2, compact_threshold=0.0).compact(
            ConversationState(msgs), budget_tokens=BIG
        )
        result = result.messages
        assert result is not msgs

    def test_no_compact_when_under_threshold(self) -> None:
        msgs = _build_history(3)
        # threshold=0.99 with huge budget — will never trigger
        result, phase = SlidingWindowCompact(keep_recent=1, compact_threshold=0.99).compact(
            ConversationState(msgs), budget_tokens=BIG
        )
        result = result.messages
        assert phase == 0
        assert len(result) == len(msgs)


# ── TieredCompact — Phase 1 ─────────────────────────────────────


class TestTieredPhase1:
    def test_applies_phase_one_rewrites_only_to_eligible_history(self) -> None:
        msgs = _build_history(6, include_nudges=True, include_text_response=True, long_results=True)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, phase = _p1_only().compact(ConversationState(msgs), budget_tokens=BIG)
        result = result.messages
        nudge_types = {"step", "retry"}
        for msg in result:
            if msg.forge_purpose in nudge_types:
                assert msg in msgs[eligible_end:]
        for msg in result:
            if isinstance(msg.parts[0], ToolResultPart) and msg not in msgs[eligible_end:]:
                assert "[Truncated —" in msg.parts[0].content[0].text
                before_marker = msg.parts[0].content[0].text.split("\n[Truncated —")[0]
                assert len(before_marker) == TieredCompact.TRUNCATE_CHARS
        original_text = [
            m
            for m in msgs
            if len(m.parts) == 1
            and isinstance(m.parts[0], TextPart)
            and m.role == MessageRole.ASSISTANT
        ]
        result_text = [
            m
            for m in result
            if len(m.parts) == 1
            and isinstance(m.parts[0], TextPart)
            and m.role == MessageRole.ASSISTANT
        ]
        assert len(result_text) == len(original_text)
        assert phase == 1

    def test_preserves_header_and_recent_window(self) -> None:
        msgs = _build_history(6, long_results=True)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, _ = _p1_only().compact(ConversationState(msgs), budget_tokens=BIG)
        result = result.messages
        assert result[0] is msgs[0]
        assert result[1] is msgs[1]
        assert result[-(len(msgs) - eligible_end) :] == msgs[eligible_end:]

    def test_short_tool_result_not_truncated(self) -> None:
        """Tool results under TRUNCATE_CHARS are kept as-is."""
        msgs = [_sys(), _user_input(), _tool_call(step=0), _tool_result("Short result", step=0)]
        msgs += [_tool_call(step=1), _tool_result("Recent", step=1)]
        result, _ = _p1_only(keep_recent=1).compact(ConversationState(msgs), budget_tokens=BIG)
        result = result.messages
        assert result[3] is msgs[3]
        assert result[3].parts[0].content[0].text == "Short result"


# ── TieredCompact — Phase 2 ─────────────────────────────────────


class TestTieredPhase2:
    def test_drops_results_but_preserves_interpretive_history_and_calls(self) -> None:
        msgs = _build_history(
            6, long_results=True, include_reasoning=True, include_text_response=True
        )
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, phase = _p1_p2_only().compact(ConversationState(msgs), budget_tokens=BIG)
        result = result.messages
        protected_count = len(msgs) - eligible_end
        for msg in result[2:-protected_count]:
            assert not any(isinstance(p, ToolResultPart) for p in msg.parts)
        for part_type in (ReasoningPart, TextPart, ToolCallPart):
            original = [p for m in msgs for p in m.parts if isinstance(p, part_type)]
            compacted = [p for m in result for p in m.parts if isinstance(p, part_type)]
            assert compacted == original, part_type
        assert phase == 2

    def test_recent_messages_untouched_in_phase2(self) -> None:
        msgs = _build_history(6, long_results=True)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, _ = _p1_p2_only().compact(ConversationState(msgs), budget_tokens=BIG)
        result = result.messages
        protected_count = len(msgs) - eligible_end
        assert result[-protected_count:] == msgs[eligible_end:]


# ── TieredCompact — Phase 3 ─────────────────────────────────────


class TestTieredPhase3:
    def test_drops_interpretive_history_but_keeps_tool_call_skeleton(self) -> None:
        msgs = _build_history(
            6, include_reasoning=True, include_text_response=True, long_results=True
        )
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, phase = _all_phases().compact(ConversationState(msgs), budget_tokens=BIG)
        result = result.messages
        protected_count = len(msgs) - eligible_end
        for msg in result[2:-protected_count]:
            assert all(isinstance(p, ToolCallPart) for p in msg.parts)
        original_calls = [p for m in msgs for p in m.parts if isinstance(p, ToolCallPart)]
        result_calls = [p for m in result for p in m.parts if isinstance(p, ToolCallPart)]
        assert result_calls == original_calls
        assert phase == 3

    def test_preserves_header_and_recent_window(self) -> None:
        msgs = _build_history(6, long_results=True)
        eligible_end = TieredCompact._find_eligible_end(msgs, 2)
        result, _ = _all_phases().compact(ConversationState(msgs), budget_tokens=BIG)
        result = result.messages
        assert result[0] is msgs[0]
        assert result[1] is msgs[1]
        protected_count = len(msgs) - eligible_end
        assert result[-protected_count:] == msgs[eligible_end:]


# ── TieredCompact — Progressive Escalation ──────────────────────


class TestTieredEscalation:
    def test_no_eligible_messages_returns_unchanged(self) -> None:
        """When all messages are protected, nothing changes."""
        msgs = _build_history(2)  # 2 + 4 = 6 messages, keep_recent=3 protects 6
        result, _ = _all_phases(keep_recent=3).compact(ConversationState(msgs), budget_tokens=BIG)
        result = result.messages
        assert result[0] is msgs[0]
        assert result[1] is msgs[1]
        assert len(result) == len(msgs)

    def test_does_not_mutate_input(self) -> None:
        """Compact must never mutate the input list."""
        msgs = _build_history(6, long_results=True)
        original = list(msgs)
        original_contents = deepcopy(msgs)
        _all_phases().compact(ConversationState(msgs), budget_tokens=BIG)
        assert len(msgs) == len(original)
        assert msgs == original_contents

    def test_no_compaction_when_under_threshold(self) -> None:
        """Returns phase 0 when tokens are under the lowest threshold."""
        msgs = _build_history(3)  # Small content
        strategy = TieredCompact(keep_recent=2, compact_threshold=0.99)
        result, phase = strategy.compact(ConversationState(msgs), budget_tokens=BIG)
        result = result.messages
        assert phase == 0
        assert len(result) == len(msgs)

    def test_per_phase_thresholds(self) -> None:
        """Per-phase thresholds control which phases fire independently."""
        msgs = _build_history(6, long_results=True, include_reasoning=True)

        # Phase 1 only
        _, phase = _p1_only().compact(ConversationState(msgs), budget_tokens=BIG)
        assert phase == 1

        # Phase 1 + 2
        _, phase = _p1_p2_only().compact(ConversationState(msgs), budget_tokens=BIG)
        assert phase == 2

        # All phases
        _, phase = _all_phases().compact(ConversationState(msgs), budget_tokens=BIG)
        assert phase == 3

    def test_observed_usage_survives_phase1_noop_until_phase2_rewrite(self) -> None:
        msgs = _build_history(4)
        strategy = TieredCompact(keep_recent=1, phase_thresholds=(0.5, 0.7, 0.9))

        result, phase = strategy._compact_with_initial_usage(
            ConversationState(msgs), budget_tokens=100, initial_tokens=80
        )
        result = result.messages

        assert phase == 2
        assert result != msgs

    def test_real_phase1_rewrite_switches_to_heuristic(self) -> None:
        msgs = _build_history(4, long_results=True)
        strategy = TieredCompact(keep_recent=1, phase_thresholds=(0.5, 0.5, 0.9))

        result, phase = strategy._compact_with_initial_usage(
            ConversationState(msgs), budget_tokens=1000, initial_tokens=600
        )
        result = result.messages

        assert result != msgs
        assert estimate_content_tokens(ConversationState(result)) < 500
        assert phase == 1


@pytest.mark.parametrize("lengths", [(150, 120), (200, 30, 40), (270,)])
def test_phase1_shared_block_budget_preserves_fields_and_correlation(lengths):
    from dataclasses import replace

    blocks = [TextPart(chr(65 + i) * n, {"block": i}, bare=True) for i, n in enumerate(lengths)]
    part = ToolResultPart(
        "correlated", blocks, "error", "string" if len(blocks) == 1 else "blocks", {"result": 1}
    )
    message = RichMessage(
        MessageRole.USER,
        [part],
        step_index=0,
        extra_fields={"name": "read"},
        content_shape="blocks",
    )
    state = ConversationState([_sys(), _user_input(), message, _tool_call(step=1)])
    before = deepcopy(state)
    result, phase = _p1_only(1).compact(state, BIG)
    truncated = result.messages[2].parts[0]
    removed = sum(lengths) - 200
    notice = f"\n[Truncated — {removed} chars removed]"
    assert phase == 1
    assert truncated == replace(part, content=truncated.content)
    assert result.messages[2] == replace(message, parts=[truncated])
    assert (
        "".join(b.text for b in truncated.content) == "".join(b.text for b in blocks)[:200] + notice
    )
    assert [(b.extra_fields, b.bare) for b in truncated.content] == [
        (b.extra_fields, b.bare) for b in blocks[: len(truncated.content)]
    ]
    if lengths == (150, 120):
        assert truncated.content[0] is blocks[0]
        assert truncated.content[1].text == "B" * 50 + notice
    assert state == before


def test_phase1_two_results_have_independent_budgets_and_keep_visible_text():
    results = [
        ToolResultPart(
            f"call-{i}", [TextPart("x" * size, {"i": i}, True)], outcome, "blocks", {"result": i}
        )
        for i, (size, outcome) in enumerate([(270, "success"), (350, "error")])
    ]
    text = TextPart("ordinary visible text", {"visible": True})
    message = RichMessage(
        MessageRole.USER,
        [results[0], text, results[1]],
        0,
        extra_fields={"message": True},
        content_shape="blocks",
    )
    state = ConversationState([_sys(), _user_input(), message, _tool_call(step=1)])
    result, _ = _p1_only(1).compact(state, BIG)
    first, kept_text, second = result.messages[2].parts
    assert kept_text is text
    for original, truncated, removed in zip(results, [first, second], [70, 150]):
        assert truncated.call_id == original.call_id
        assert truncated.outcome == original.outcome
        assert truncated.extra_fields == original.extra_fields
        assert truncated.content_shape == original.content_shape
        assert truncated.content[0].extra_fields == original.content[0].extra_fields
        assert truncated.content[0].bare is True
        assert truncated.content[0].text == "x" * 200 + f"\n[Truncated — {removed} chars removed]"


@pytest.mark.parametrize("phase", [1, 2, 3])
@pytest.mark.parametrize(
    "purpose", ["retry", "unknown_tool", "tool_arg_validation", "step", "prerequisite"]
)
def test_correction_parts_removed_without_losing_unrelated_parts(phase, purpose):
    reasoning = ReasoningPart("interpretation")
    call = ToolCallPart("read", ValidArguments({}), "call")
    correction = RichMessage(
        MessageRole.USER,
        [
            TextPart("correction"),
            ToolResultPart("bad", [TextPart("invalid")], "error"),
            reasoning,
            call,
        ],
        0,
        purpose,
        {"name": "retained"},
        "blocks",
    )
    ordinary = RichMessage(
        MessageRole.USER,
        [TextPart("visible"), ToolResultPart("execution-error", [TextPart("x" * 270)], "error")],
        0,
    )
    state = ConversationState([_sys(), _user_input(), correction, ordinary, _tool_call(step=1)])
    strategy = TieredCompact(
        keep_recent=1, phase_thresholds={1: (0, 1, 1), 2: (0, 0, 1), 3: (0, 0, 0)}[phase]
    )
    result, reached = strategy.compact(state, BIG)
    assert reached == phase
    assert result.messages[2].parts == [reasoning, call]
    assert result.messages[2].forge_purpose == purpose
    assert result.messages[2].extra_fields == correction.extra_fields
    assert result.messages[2].content_shape == "blocks"
    assert result.messages[3].parts[0] is ordinary.parts[0]
    if phase == 1:
        assert result.messages[3].parts[1].call_id == "execution-error"
        assert result.messages[3].parts[1].content[0].text.endswith("70 chars removed]")
    else:
        assert result.messages[3].parts == [ordinary.parts[0]]


@pytest.mark.parametrize("phase", [2, 3])
def test_mixed_assistant_preserves_calls_and_message_fields(phase):
    from dataclasses import replace

    call = ToolCallPart("read", ValidArguments({"x": 1}), "call", {"call": True})
    message = RichMessage(
        MessageRole.ASSISTANT,
        [
            TextPart("visible"),
            ReasoningPart("reasoning"),
            call,
            ToolResultPart("other", [TextPart("result")]),
        ],
        0,
        extra_fields={"name": "assistant"},
        content_shape="blocks",
        empty_tool_calls=True,
    )
    state = ConversationState([_sys(), _user_input(), message, _tool_call(step=1)])
    strategy = _p1_p2_only(1) if phase == 2 else _all_phases(1)
    result, _ = strategy.compact(state, BIG)
    retained = message.parts[:3] if phase == 2 else [call]
    assert result.messages[2] == replace(message, parts=retained)
    assert result.messages[2].parts[-1] is call


@pytest.mark.parametrize("strategy", [SlidingWindowCompact(1, 0), _all_phases(1)])
def test_variable_parallel_groups_and_retry_boundary(strategy):
    # Rejected replies/corrections stay at incoming step 1; acceptance/results
    # belong to completed step 2, exactly as Runner records them.
    old = [_tool_call(step=0)] + [_tool_result(step=0) for _ in range(3)]
    rejected = _text_response("rejected", step=1)
    correction = RichMessage(MessageRole.USER, [TextPart("retry")], 1, "retry")
    recent = [_tool_call(step=2)] + [_tool_result(step=2) for _ in range(2)]
    messages = [_sys(), _user_input(), *old, rejected, correction, *recent]
    assert TieredCompact._find_eligible_end(messages, 1) == len(messages) - 3
    result, _ = strategy.compact(ConversationState(messages), BIG)
    assert result.messages[:2] == messages[:2]
    assert result.messages[-3:] == recent
    assert all(a is b for a, b in zip(result.messages[-3:], recent))
    assert rejected not in result.messages and correction not in result.messages
    if isinstance(strategy, SlidingWindowCompact):
        assert result.messages == messages[:2] + recent
    else:
        assert [p for m in result.messages[2:-3] for p in m.parts] == [old[0].parts[1]]


def test_boundary_preserves_consecutive_transition_and_cutoff_comparison():
    messages = [_sys(), _user_input()] + [_tool_call(step=i) for i in [0, 2, 1, 2]]
    # Preserve the existing >= cutoff scan, not a regrouping of equal steps.
    assert TieredCompact._find_eligible_end(messages, 2) == 3
    assert TieredCompact._find_eligible_end(messages, 4) == 2


@pytest.mark.parametrize("strategy", [SlidingWindowCompact(1), TieredCompact(1)])
def test_exact_trigger_threshold_compacts(strategy):
    state = ConversationState(_build_history(4))
    _, below = strategy._compact_with_initial_usage(state, 100, initial_tokens=74)
    _, at = strategy._compact_with_initial_usage(state, 100, initial_tokens=75)
    assert below == 0 and at >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", [SlidingWindowCompact(1, 0), _all_phases(1)])
async def test_runner_retry_history_uses_incoming_and_completed_boundaries(strategy):
    from forge import ContextManager, WorkflowRunner, ToolDefinition, Workflow
    from tests.conftest import MockClient
    from tests.canonical import assistant, call

    snapshots = []

    class RecordingContext(ContextManager):
        def maybe_compact(self, state, **kwargs):
            snapshots.append(deepcopy(state))
            return super().maybe_compact(state, **kwargs)

    context = RecordingContext(NoCompact(), 10000)
    client = MockClient(
        [
            assistant(call("read"), call("read")),
            assistant(TextPart("rejected text")),
            assistant(call("read")),
            assistant(call("finish")),
        ]
    )
    workflow = Workflow(
        "retry-boundary",
        "test",
        tools={name: ToolDefinition(name) for name in ["read", "finish"]},
        handlers={"read": lambda: "result", "finish": lambda: "finished"},
        required_steps=[],
        terminal_tool="finish",
        system_prompt_template="Use tools.",
    )
    assert await WorkflowRunner(client, context).run(workflow, "Run") == "finished"
    state = snapshots[
        -1
    ]  # Before finish: retries consume iterations; acceptance/results use step 3.
    assert [m.step_index for m in state.messages] == [None, None, 1, 1, 1, 1, 1, 3, 3]
    rejected, correction = state.messages[5:7]
    assert rejected.parts[0].text == "rejected text"
    assert correction.forge_purpose == "retry"
    assert TieredCompact._find_eligible_end(state.messages, 1) == 7
    result, _ = strategy.compact(state, BIG)
    assert result.messages[:2] == state.messages[:2]
    assert result.messages[-2:] == state.messages[-2:]
    assert rejected not in result.messages and correction not in result.messages
    if isinstance(strategy, SlidingWindowCompact):
        assert result.messages == state.messages[:2] + state.messages[7:]
    else:
        assert [p for m in result.messages[2:-2] for p in m.parts] == state.messages[2].parts
