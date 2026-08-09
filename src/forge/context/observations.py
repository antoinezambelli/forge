"""Neutral immutable snapshots of current context usage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class ContextSession:
    """Optional opaque session identity attached to context usage."""

    id: str
    source: str | None = None


@dataclass(frozen=True)
class ContextUsage:
    """One completed snapshot of current prompt/input occupancy.

    Source values and identifiers are deliberately opaque to context policy.
    Callers that publish usage own their provenance vocabularies.
    """

    current_usage_tokens: int
    observed_at: datetime
    context_window_tokens: int | None = None
    model: str | None = None
    context_window_source: str | None = None
    session: ContextSession | None = None

    def __post_init__(self) -> None:
        if self.current_usage_tokens < 0:
            raise ValueError("current_usage_tokens must be nonnegative")
        if (
            self.observed_at.tzinfo is None
            or self.observed_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("observed_at must be UTC-aware")
