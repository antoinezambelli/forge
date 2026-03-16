"""Context manager for budget tracking and compaction triggering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from forge.context.strategies import CompactStrategy
from forge.core.messages import Message


@dataclass(frozen=True)
class CompactEvent:
    """Emitted by ContextManager when compaction fires."""

    step_index: int
    tokens_before: int
    tokens_after: int
    budget_tokens: int
    messages_before: int
    messages_after: int
    phase_reached: int


class ContextManager:
    """Manages context window budget and triggers compaction."""

    def __init__(
        self,
        strategy: CompactStrategy,
        budget_tokens: int,
        compact_threshold: float = 0.75,
        on_compact: Callable[[CompactEvent], None] | None = None,
    ) -> None:
        """
        Args:
            strategy: Compaction strategy to use when threshold is exceeded.
            budget_tokens: Maximum context budget in tokens.
            compact_threshold: Fraction of budget that triggers compaction.
            on_compact: Callback invoked when compaction fires. Receives a
                CompactEvent with before/after token counts, phase reached,
                and which messages were affected. Use for logging, debugging,
                or surfacing compaction to a UI.
        """
        self.strategy = strategy
        self.budget_tokens = budget_tokens
        self.compact_threshold = compact_threshold
        self.on_compact = on_compact

    def estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate token count via char/4 heuristic."""
        return sum(len(m.content) for m in messages) // 4

    def maybe_compact(
        self,
        messages: list[Message],
        step_index: int = 0,
        step_hint: str = "",
    ) -> list[Message]:
        """Compact if estimated tokens exceed budget * compact_threshold."""
        tokens_before = self.estimate_tokens(messages)
        trigger_tokens = int(self.budget_tokens * self.compact_threshold)

        if tokens_before < trigger_tokens:
            return messages

        result, phase = self.strategy.compact(
            messages, trigger_tokens, step_hint=step_hint
        )

        if self.on_compact is not None:
            event = CompactEvent(
                step_index=step_index,
                tokens_before=tokens_before,
                tokens_after=self.estimate_tokens(result),
                budget_tokens=self.budget_tokens,
                messages_before=len(messages),
                messages_after=len(result),
                phase_reached=phase,
            )
            self.on_compact(event)

        return result
