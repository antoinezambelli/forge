"""Reasoning replay policy shared by runner and proxy."""

from __future__ import annotations

from typing import Literal


ReasoningReplay = Literal["full", "keep-last", "none"]
REASONING_REPLAY_CHOICES: tuple[ReasoningReplay, ...] = ("full", "keep-last", "none")
DEFAULT_REASONING_REPLAY: ReasoningReplay = "none"


def validate_reasoning_replay(value: str) -> ReasoningReplay:
    """Validate and normalize a reasoning replay policy."""
    if value not in REASONING_REPLAY_CHOICES:
        choices = ", ".join(REASONING_REPLAY_CHOICES)
        raise ValueError(f"reasoning_replay must be one of: {choices}")
    return value  # type: ignore[return-value]


REASONING_MESSAGE_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")
