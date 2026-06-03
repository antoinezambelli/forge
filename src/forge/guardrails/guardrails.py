"""Guardrails -- bundled middleware for foreign orchestration loops.

Two-method API that wraps ResponseValidator, StepEnforcer, and ErrorTracker:

    result = guardrails.check(response)   # before execution
    done   = guardrails.record(["tool"])  # after execution

For granular control, use the individual components directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from forge.core.workflow import LLMResponse, ToolCall
from forge.guardrails.error_tracker import ErrorTracker
from forge.guardrails.nudge import TOOL_CHANNEL_KINDS, TOOL_ERROR_KINDS, Nudge
from forge.guardrails.response_validator import ResponseValidator
from forge.guardrails.step_enforcer import StepEnforcer


@dataclass(frozen=True)
class CheckResult:
    """Result of checking an LLM response against all guardrails.

    Attributes:
        action: What the caller should do next.
            "execute"      -- tool_calls are safe to run.
            "retry"        -- model produced unusable output (bad format / bare
                              text); inject nudge as a user message and re-prompt.
            "tool_error"   -- model called a tool incorrectly (unknown name or
                              malformed args); inject nudge as a tool result
                              (nudge.role == "tool") and re-prompt.
            "step_blocked" -- model tried to skip required steps; inject nudge.
            "fatal"        -- error budget exhausted; stop the workflow.
        tool_calls: Validated tool calls (only set when action == "execute").
        nudge: Corrective message to inject (set when action is "retry",
            "tool_error", or "step_blocked"). Emit it with its own role:
            ``messages.append({"role": nudge.role, "content": nudge.content})``.
        reason: Human-readable explanation (only set when action == "fatal").
    """

    action: Literal["execute", "retry", "tool_error", "step_blocked", "fatal"]
    tool_calls: list[ToolCall] | None = None
    nudge: Nudge | None = None
    reason: str | None = None


class Guardrails:
    """Bundled guardrail middleware for foreign orchestration loops.

    Wraps ResponseValidator, StepEnforcer, and ErrorTracker into a
    two-method API. Use ``check()`` after each LLM response and
    ``record()`` after executing tools.

    Args:
        tool_names: Valid tool names for this workflow.
        required_steps: Tools that must be called before the terminal tool.
            Defaults to no required steps.
        terminal_tool: The tool(s) that can end the workflow. Accepts a
            single name or a frozenset of names.
        max_retries: Consecutive bad responses before ``check()`` returns
            ``"fatal"``. Default 3.
        max_tool_errors: Consecutive tool execution failures before
            exhaustion. Default 2.
        rescue_enabled: Attempt to parse tool calls from plain text
            responses. Default True.
        max_premature_attempts: Premature terminal attempts before
            ``check()`` returns ``"fatal"``. Default 3.
        retry_nudge: Custom nudge for bare text responses. Pass a callable
            ``(raw_response) -> str`` for dynamic nudges. If None, uses
            the default.
    """

    def __init__(
        self,
        tool_names: list[str],
        terminal_tool: str | frozenset[str],
        required_steps: list[str] | None = None,
        max_retries: int = 3,
        max_tool_errors: int = 2,
        rescue_enabled: bool = True,
        max_premature_attempts: int = 3,
        retry_nudge: Callable[[str], str] | None = None,
    ) -> None:
        self._validator = ResponseValidator(
            tool_names=tool_names,
            rescue_enabled=rescue_enabled,
            retry_nudge_fn=retry_nudge,
        )
        if isinstance(terminal_tool, str):
            terminal_tools = frozenset([terminal_tool])
        else:
            terminal_tools = terminal_tool
        self._enforcer = StepEnforcer(
            required_steps=required_steps or [],
            terminal_tools=terminal_tools,
            max_premature_attempts=max_premature_attempts,
        )
        self._errors = ErrorTracker(
            max_retries=max_retries,
            max_tool_errors=max_tool_errors,
        )

    def check(
        self, response: LLMResponse,
    ) -> CheckResult:
        """Check an LLM response against all guardrails.

        Call this after each LLM response, before executing any tools.

        Args:
            response: The LLM response -- either a TextResponse or a
                list of ToolCall objects.

        Returns:
            CheckResult indicating what the caller should do next.
        """
        # Checkpoint 1: Is this response usable?
        validation = self._validator.validate(response)

        if validation.needs_retry:
            nudge = validation.nudge
            kind = nudge.kind if nudge is not None else ""
            # Budget: malformed args drain the tool-error budget; everything
            # else (bare text, unknown tool) drains the retry budget — matching
            # run_inference so all three integration modes account identically.
            if kind in TOOL_ERROR_KINDS:
                self._errors.record_result(success=False)
                if self._errors.tool_errors_exhausted:
                    return CheckResult(
                        action="fatal",
                        reason="too many consecutive tool-argument errors",
                    )
            else:
                self._errors.record_retry()
                if self._errors.retries_exhausted:
                    return CheckResult(
                        action="fatal",
                        reason="too many consecutive bad responses",
                    )
            # Channel: tool-call faults (unknown tool, malformed args) ride the
            # tool-result channel; bare-text failures stay a user-role retry.
            action = "tool_error" if kind in TOOL_CHANNEL_KINDS else "retry"
            return CheckResult(action=action, nudge=nudge)

        self._errors.reset_retries()

        # Checkpoint 2: Is the model skipping required steps?
        step_check = self._enforcer.check(validation.tool_calls)

        if step_check.needs_nudge:
            if self._enforcer.premature_exhausted:
                return CheckResult(
                    action="fatal",
                    reason="model repeatedly skipped required steps",
                )
            return CheckResult(action="step_blocked", nudge=step_check.nudge)

        return CheckResult(action="execute", tool_calls=validation.tool_calls)

    def record(self, executed: list[str | tuple[str, dict]]) -> bool:
        """Record which tools were successfully executed.

        Call this after executing tools to keep the middleware in sync.

        Args:
            executed: Names of tools that succeeded, or (name, args) tuples
                for prerequisite tracking.

        Returns:
            True if the terminal tool was reached and all required
            steps are satisfied (workflow is done).
        """
        for entry in executed:
            if isinstance(entry, tuple):
                name, args = entry
                self._enforcer.record(name, args)
            else:
                self._enforcer.record(entry)
        self._errors.reset_errors()
        self._enforcer.reset_premature()
        return self._enforcer.is_satisfied() and any(
            (entry if isinstance(entry, str) else entry[0]) in self._enforcer.terminal_tools
            for entry in executed
        )
