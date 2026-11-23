"""Compaction strategies for context window management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from forge.core.messages import (
    ConversationState, RichMessage, MessageRole, TextPart, ReasoningPart, ToolResultPart,
)
from forge.context.accounting import estimate_content_tokens


class CompactStrategy(ABC):
    """Interface for context compaction strategies.

    Policies operate directly on canonical parts, protecting the first two
    messages and recent iterations. Calls survive category-based removal.
    """

    @abstractmethod
    def compact(
        self,
        messages: ConversationState,
        budget_tokens: int,
        *,
        step_hint: str = "",
    ) -> tuple[ConversationState, int]:
        """Return a compacted copy of the message history and the phase reached.

        Returns a tuple of (compacted_messages, phase_reached). The phase int
        indicates how aggressively the strategy compacted: 0 means no
        compaction was applied, 1+ is implementation-defined. Strategies
        without internal phases should return 1.

        The strategy owns its own threshold logic. It receives the full
        budget_tokens and decides whether to compact and how aggressively.
        Return phase 0 if no compaction was needed.

        Must preserve (never cut):
        - The system prompt (messages[0])
        - The original user input (messages[1])
        """
        ...

    def _compact_with_initial_usage(
        self,
        messages: ConversationState,
        budget_tokens: int,
        *,
        initial_tokens: int,
        step_hint: str = "",
    ) -> tuple[ConversationState, int]:
        """Internal manager path; custom strategies retain the public contract."""
        return self.compact(messages, budget_tokens, step_hint=step_hint)


class NoCompact(CompactStrategy):
    """Passthrough strategy. Returns messages unchanged.

    Use when VRAM is abundant (32GB+) or workflows are short.
    """

    def compact(
        self,
        messages: ConversationState,
        budget_tokens: int,
        *,
        step_hint: str = "",
    ) -> tuple[ConversationState, int]:
        return messages, 0


class SlidingWindowCompact(CompactStrategy):
    """Keeps the system prompt, original user input, and the last N iterations.

    Simple and predictable. Good baseline for testing. Uses step_index to
    identify iteration boundaries (handles variable-size parallel tool batches).
    """

    def __init__(self, keep_recent: int, compact_threshold: float = 0.75) -> None:
        self.keep_recent = keep_recent
        self.compact_threshold = compact_threshold

    def compact(
        self,
        messages: ConversationState,
        budget_tokens: int,
        *,
        step_hint: str = "",
    ) -> tuple[ConversationState, int]:
        return self._compact_with_initial_usage(
            messages,
            budget_tokens,
            initial_tokens=estimate_content_tokens(messages),
            step_hint=step_hint,
        )

    def _compact_with_initial_usage(
        self,
        messages: ConversationState,
        budget_tokens: int,
        *,
        initial_tokens: int,
        step_hint: str = "",
    ) -> tuple[ConversationState, int]:
        trigger = int(budget_tokens * self.compact_threshold)
        if initial_tokens < trigger:
            return ConversationState(list(messages.messages)), 0
        eligible_end = TieredCompact._find_eligible_end(messages.messages, self.keep_recent)
        if eligible_end <= 2:
            return ConversationState(list(messages.messages)), 1
        return ConversationState(messages.messages[:2] + messages.messages[eligible_end:]), 1


class TieredCompact(CompactStrategy):
    """Three-phase compaction with explicit priority order.

    Each phase fires only if the previous phase didn't reduce tokens below
    the trigger threshold. keep_recent controls how many recent loop iterations
    (each iteration = one assistant message + N tool result messages) are
    fully preserved before older content is eligible for compaction.

    Phase priority (cut first -> preserve longest):
    1. Nudges/retries dropped, tool_results truncated to first ~200 chars
    2. Tool_results dropped entirely — reasoning and text_response preserved
    3. Reasoning and text_response dropped — only tool_call skeleton remains
    """

    TRUNCATE_CHARS = 200

    def __init__(
        self,
        keep_recent: int = 2,
        compact_threshold: float = 0.75,
        phase_thresholds: tuple[float, float, float] | None = None,
    ) -> None:
        """
        Args:
            keep_recent: Number of recent loop iterations to keep fully intact.
                Tune based on workflow depth — shallow workflows (3-5 steps)
                can use 2-3, deep workflows (8-10+) may need 4-6.
            compact_threshold: Fraction of budget that triggers compaction.
                Used as the threshold for all three phases when
                phase_thresholds is not set.
            phase_thresholds: Per-phase compaction thresholds as fractions
                of the context budget. A tuple of (phase1, phase2, phase3).
                Example: ``(0.60, 0.75, 0.90)`` means Phase 1 fires at 60%,
                Phase 2 at 75%, Phase 3 at 90%. Overrides compact_threshold.
        """
        self.keep_recent = keep_recent
        if phase_thresholds is not None:
            self._phase_triggers = phase_thresholds
        else:
            self._phase_triggers = (
                compact_threshold,
                compact_threshold,
                compact_threshold,
            )

    @staticmethod
    def _find_eligible_end(messages: list[RichMessage], keep_recent: int) -> int:
        """Find the boundary index: messages before this are eligible for compaction.

        Uses direct step_index to identify iteration boundaries.
        With parallel tool calls, one iteration may produce variable numbers of
        messages (1 TOOL_CALL + N TOOL_RESULTs), so counting by step_index is
        more accurate than a flat message count.
        """
        # Collect distinct step_index values from messages after the protected
        # header (messages[0] and [1] have step_index=None).
        seen_steps: list[int] = []
        for m in messages[2:]:
            si = m.step_index
            if si is not None and (not seen_steps or seen_steps[-1] != si):
                seen_steps.append(si)

        if len(seen_steps) <= keep_recent:
            # Not enough iterations to compact anything
            return 2

        # Protect the last keep_recent iterations
        cutoff_step = seen_steps[-keep_recent]
        # Find the first message index with step_index >= cutoff_step
        for i in range(2, len(messages)):
            si = messages[i].step_index
            if si is not None and si >= cutoff_step:
                return i
        return len(messages)

    def compact(
        self,
        messages: ConversationState,
        budget_tokens: int,
        *,
        step_hint: str = "",
    ) -> tuple[ConversationState, int]:
        """Apply tiered compaction: Phase 1 -> Phase 2 -> Phase 3.

        Each phase has its own threshold (fraction of budget_tokens).
        A phase runs when tokens reach its threshold.
        """
        return self._compact_with_initial_usage(
            messages,
            budget_tokens,
            initial_tokens=estimate_content_tokens(messages),
            step_hint=step_hint,
        )

    def _compact_with_initial_usage(
        self,
        messages: ConversationState,
        budget_tokens: int,
        *,
        initial_tokens: int,
        step_hint: str = "",
    ) -> tuple[ConversationState, int]:
        tokens = initial_tokens
        t1 = int(budget_tokens * self._phase_triggers[0])
        t2 = int(budget_tokens * self._phase_triggers[1])
        t3 = int(budget_tokens * self._phase_triggers[2])

        # Nothing to do if below the lowest threshold
        if tokens < t1:
            return ConversationState(list(messages.messages)), 0

        # Determine the boundary: everything before this index is eligible
        # messages[0] and messages[1] are always protected
        eligible_end = self._find_eligible_end(messages.messages, self.keep_recent)

        # Phase 1: Drop nudges/retries, truncate result text to 200 characters
        result = self._phase1(messages, eligible_end)
        if result != messages:
            tokens = estimate_content_tokens(result)
        if tokens < t2:
            return result, 1

        # Phase 2: Phase 1 + drop tool_results entirely
        phase2_result = self._phase2(messages, eligible_end)
        if phase2_result != result:
            tokens = estimate_content_tokens(phase2_result)
        if tokens < t3:
            return phase2_result, 2

        # Phase 3: Phase 2 + drop reasoning and text_response (tool_call skeleton only)
        return self._phase3(messages, eligible_end), 3

    def _phase1(self, state: ConversationState, eligible_end: int) -> ConversationState:
        return self._apply_phase(state, eligible_end, 1)

    def _phase2(self, state: ConversationState, eligible_end: int) -> ConversationState:
        return self._apply_phase(state, eligible_end, 2)

    def _phase3(self, state: ConversationState, eligible_end: int) -> ConversationState:
        return self._apply_phase(state, eligible_end, 3)

    def _apply_phase(
        self, state: ConversationState, eligible_end: int, phase: int,
    ) -> ConversationState:
        messages: list[RichMessage] = []
        for i, message in enumerate(state.messages):
            if not 2 <= i < eligible_end:
                messages.append(message)
                continue
            correction = message.forge_purpose in {
                "retry", "unknown_tool", "tool_arg_validation", "step", "prerequisite",
            }
            parts = []
            for part in message.parts:
                if correction and isinstance(part, (TextPart, ToolResultPart)):
                    continue
                if isinstance(part, ToolResultPart):
                    if phase >= 2:
                        continue
                    part = self._truncate_result(part)
                if (phase >= 3 and message.role == MessageRole.ASSISTANT
                        and isinstance(part, (ReasoningPart, TextPart))):
                    continue
                parts.append(part)
            if parts == message.parts:
                messages.append(message)
            elif parts:
                messages.append(replace(message, parts=parts))
        return ConversationState(messages)

    def _truncate_result(self, result: ToolResultPart) -> ToolResultPart:
        chars = sum(len(block.text) for block in result.content)
        if chars <= self.TRUNCATE_CHARS:
            return result
        remaining = self.TRUNCATE_CHARS
        blocks: list[TextPart] = []
        for block in result.content:
            if remaining == 0:
                break
            if len(block.text) <= remaining:
                blocks.append(block)
            else:
                blocks.append(replace(block, text=block.text[:remaining]))
            remaining -= len(blocks[-1].text)
        removed = chars - self.TRUNCATE_CHARS
        # Append to the last retained block: string results retain their exact
        # prefix/newline/notice output, without inserting a block separator.
        blocks[-1] = replace(
            blocks[-1], text=blocks[-1].text + f"\n[Truncated — {removed} chars removed]",
        )
        return replace(result, content=blocks)
