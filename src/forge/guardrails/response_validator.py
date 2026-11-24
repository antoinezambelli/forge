"""Response validation over canonical assistant parts."""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from forge.core.messages import RichMessage, TextPart, ToolCallPart, InvalidArguments
from forge.guardrails.nudge import Nudge
from forge.prompts.nudges import (
    retry_nudge,
    tool_arg_validation_nudge,
    unknown_tool_nudge,
)
from forge.prompts.templates import rescue_tool_call


@dataclass
class ValidationResult:
    reply: RichMessage
    nudge: Nudge | None = None
    needs_retry: bool = False

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return self.reply.tool_calls


class ResponseValidator:
    def __init__(
        self,
        tool_names: list[str],
        rescue_enabled: bool = True,
        retry_nudge_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.tool_names = tool_names
        self.rescue_enabled = rescue_enabled
        self._retry_nudge_fn = retry_nudge_fn or retry_nudge

    def validate(self, response: RichMessage) -> ValidationResult:
        calls = response.tool_calls
        if not calls:
            text = "\n".join(p.text for p in response.parts if isinstance(p, TextPart))
            calls = (
                rescue_tool_call(text, self.tool_names) if self.rescue_enabled else []
            )
            if calls:
                response.parts = [
                    p for p in response.parts if not isinstance(p, TextPart)
                ] + calls
            else:
                return ValidationResult(
                    response,
                    Nudge(
                        role="user", content=self._retry_nudge_fn(text), kind="retry"
                    ),
                    True,
                )
        unknown = next((p for p in calls if p.name not in self.tool_names), None)
        if unknown is not None:
            return ValidationResult(
                response,
                Nudge(
                    role="tool",
                    content=unknown_tool_nudge(unknown.name, self.tool_names),
                    kind="unknown_tool",
                ),
                True,
            )
        bad = next(
            (p for p in calls if isinstance(p.arguments, InvalidArguments)), None
        )
        if bad is not None:
            return ValidationResult(
                response,
                Nudge(
                    role="tool",
                    content=tool_arg_validation_nudge(bad.name, bad.arguments.received),
                    kind="tool_arg_validation",
                ),
                True,
            )
        return ValidationResult(response)
