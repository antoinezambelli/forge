"""Compaction strategies for context window management."""

from __future__ import annotations

from abc import ABC, abstractmethod

from forge.core.messages import ConversationState


class CompactStrategy(ABC):
    """Interface for context compaction strategies.

    Recommended compaction priority (cut first -> preserve longest):
    1. step_nudge, retry_nudge — ephemeral corrections, no long-term value
    2. tool_result — truncate to first line; raw data is expendable once processed
    3. tool_call — collapse to one-liner (tool name + args)
    4. reasoning — preserve as long as possible; this is the model's interpretive context
    5. Recent iterations (within keep_recent window) — fully intact
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
