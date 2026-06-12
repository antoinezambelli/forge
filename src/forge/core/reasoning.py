"""Reasoning replay policy shared by runner and proxy."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal


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


def filter_openai_reasoning_messages(
    messages: list[dict[str, Any]],
    reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
) -> list[dict[str, Any]]:
    """Copy raw OpenAI messages and apply the reasoning replay policy.

    Non-reasoning fields are preserved verbatim so proxy passthrough keeps
    client-authored extensions, multimodal blocks, names, and other metadata.
    """
    reasoning_replay = validate_reasoning_replay(reasoning_replay)
    filtered = [deepcopy(msg) for msg in messages]
    if reasoning_replay == "full":
        return filtered

    last_reasoning_index: int | None = None
    if reasoning_replay == "keep-last":
        for i, msg in enumerate(filtered):
            if msg.get("role") == "assistant" and any(
                msg.get(field) for field in REASONING_MESSAGE_FIELDS
            ):
                last_reasoning_index = i

    for i, msg in enumerate(filtered):
        if msg.get("role") != "assistant":
            continue
        if reasoning_replay == "none" or i != last_reasoning_index:
            for field in REASONING_MESSAGE_FIELDS:
                msg.pop(field, None)
    return filtered
