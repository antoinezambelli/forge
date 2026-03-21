"""Response validation — rescue, retry, and unknown-tool nudges."""

from __future__ import annotations

from dataclasses import dataclass

from forge.core.workflow import LLMResponse, TextResponse, ToolCall
from forge.guardrails.nudge import Nudge
from forge.prompts.nudges import retry_nudge, unknown_tool_nudge
from forge.prompts.templates import rescue_tool_call


@dataclass
class ValidationResult:
    """Result of validating an LLM response.

    Exactly one of ``tool_calls``, ``text_response``, or ``nudge`` is set:
    - If ``needs_retry`` is False and ``tool_calls`` is set: validated tool calls.
    - If ``needs_retry`` is False and ``text_response`` is set: intentional text.
    - If ``needs_retry`` is True: ``nudge`` contains the message to inject.
    """

    tool_calls: list[ToolCall] | None
    nudge: Nudge | None
    needs_retry: bool
    text_response: TextResponse | None = None


class ResponseValidator:
    """Validates LLM responses: rescues tool calls from text, checks tool names.

    Stateless — safe to reuse across turns and sessions.

    Args:
        tool_names: Valid tool names for this workflow.
        rescue_enabled: If True, attempt to parse tool calls from TextResponse
            before generating a retry nudge.
    """

    def __init__(self, tool_names: list[str], rescue_enabled: bool = True) -> None:
        self.tool_names = tool_names
        self.rescue_enabled = rescue_enabled

    def validate(
        self, response: LLMResponse, trust_text_intent: bool = False,
    ) -> ValidationResult:
        """Validate an LLM response.

        Args:
            response: Either a TextResponse or a list of ToolCall objects.
            trust_text_intent: If True, trust the backend's ``intentional``
                flag on TextResponse and skip retry. If False (default),
                ignore the flag and retry as usual.

        Returns:
            ValidationResult with tool_calls on success, or a Nudge on failure.
        """
        # TextResponse: intentional pass-through (if trusted), then rescue, then retry nudge
        if isinstance(response, TextResponse):
            if trust_text_intent and response.intentional:
                return ValidationResult(
                    tool_calls=None, nudge=None, needs_retry=False,
                    text_response=response,
                )
            if self.rescue_enabled:
                rescued = rescue_tool_call(response.content, self.tool_names)
                if rescued:
                    return ValidationResult(
                        tool_calls=rescued, nudge=None, needs_retry=False
                    )
            return ValidationResult(
                tool_calls=None,
                nudge=Nudge(
                    role="user",
                    content=retry_nudge(response.content),
                    kind="retry",
                ),
                needs_retry=True,
            )

        # list[ToolCall]: check for unknown tools
        tool_calls = response
        unknown = [tc for tc in tool_calls if tc.tool not in self.tool_names]
        if unknown:
            return ValidationResult(
                tool_calls=None,
                nudge=Nudge(
                    role="user",
                    content=unknown_tool_nudge(unknown[0].tool, self.tool_names),
                    kind="unknown_tool",
                ),
                needs_retry=True,
            )

        return ValidationResult(
            tool_calls=tool_calls, nudge=None, needs_retry=False
        )
