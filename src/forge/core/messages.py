"""Canonical conversation content. Wire views belong to backend renderers."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ValidArguments:
    value: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.value, dict):
            raise ValueError("ValidArguments requires an object")


@dataclass(frozen=True)
class InvalidArguments:
    received: Any


def arguments_from_value(value: Any) -> ValidArguments | InvalidArguments:
    return ValidArguments(value) if isinstance(value, dict) else InvalidArguments(value)


@dataclass
class TextPart:
    text: str
    extra_fields: dict[str, Any] = field(default_factory=dict)
    bare: bool = False


@dataclass
class ReasoningPart:
    text: str
    extra_fields: dict[str, Any] = field(default_factory=dict)
    # Named OpenAI sibling field; None denotes an ordered thinking block.
    field_name: str | None = None


@dataclass
class ToolCallPart:
    name: str
    arguments: ValidArguments | InvalidArguments
    call_id: str = ""
    extra_fields: dict[str, Any] = field(default_factory=dict)
    function_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultPart:
    call_id: str
    content: list[TextPart]
    outcome: Literal["success", "error"] = "success"
    content_shape: Literal["string", "blocks", "null"] = "string"
    extra_fields: dict[str, Any] = field(default_factory=dict)


type ContentPart = TextPart | ReasoningPart | ToolCallPart | ToolResultPart


@dataclass
class RichMessage:
    role: MessageRole
    parts: list[ContentPart] = field(default_factory=list)
    step_index: int | None = None
    forge_purpose: str | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)
    content_shape: Literal["string", "blocks", "null"] = "string"
    # Preserve an explicit whole-message empty field; never denotes an invocation.
    empty_tool_calls: bool = False

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        """Return invocation parts in order; add or remove calls through parts."""
        return [part for part in self.parts if isinstance(part, ToolCallPart)]


@dataclass
class ConversationState:
    messages: list[RichMessage] = field(default_factory=list)
