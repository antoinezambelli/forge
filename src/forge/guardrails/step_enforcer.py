"""Step enforcement — required step tracking and premature terminal nudges."""

from __future__ import annotations

from dataclasses import dataclass

from forge.core.steps import StepTracker
from forge.core.workflow import ToolCall
from forge.guardrails.nudge import Nudge
from forge.prompts.nudges import step_nudge


@dataclass
class StepCheck:
    """Result of checking tool calls against step requirements.

    If ``needs_nudge`` is True, ``nudge`` contains the message to inject.
    """

    nudge: Nudge | None
    needs_nudge: bool


class StepEnforcer:
    """Tracks required steps and enforces them with escalating nudges.

    Stateful — instantiate per session/task.

    Args:
        required_steps: Tool names that must be called before the terminal tool.
        terminal_tool: The tool that ends the workflow.
        max_premature_attempts: How many premature terminal attempts before
            the enforcer signals exhaustion (via StepCheck or raising).
    """

    def __init__(
        self,
        required_steps: list[str],
        terminal_tool: str,
        max_premature_attempts: int = 3,
    ) -> None:
        self._tracker = StepTracker(required_steps=required_steps)
        self.terminal_tool = terminal_tool
        self.max_premature_attempts = max_premature_attempts
        self._premature_attempts = 0

    def check(self, tool_calls: list[ToolCall]) -> StepCheck:
        """Check whether tool calls include a premature terminal call.

        If the terminal tool is in the batch and required steps aren't
        satisfied, returns a StepCheck with an escalating nudge. The
        escalation tier increments on each premature attempt (1=polite,
        2=direct, 3=aggressive).

        Args:
            tool_calls: The tool calls the model wants to execute.

        Returns:
            StepCheck with nudge if premature, or no nudge if clear to proceed.
        """
        has_terminal = any(tc.tool == self.terminal_tool for tc in tool_calls)

        if has_terminal and not self._tracker.is_satisfied():
            self._premature_attempts += 1
            tier = min(self._premature_attempts, 3)
            return StepCheck(
                nudge=Nudge(
                    role="user",
                    content=step_nudge(
                        self.terminal_tool,
                        self._tracker.pending(),
                        tier=tier,
                    ),
                    kind="step",
                    tier=tier,
                ),
                needs_nudge=True,
            )

        return StepCheck(nudge=None, needs_nudge=False)

    def record(self, tool_name: str) -> None:
        """Record a successful tool execution."""
        self._tracker.record(tool_name)

    def is_satisfied(self) -> bool:
        """True if all required steps have been completed."""
        return self._tracker.is_satisfied()

    def pending(self) -> list[str]:
        """Return required steps not yet completed."""
        return self._tracker.pending()

    def terminal_reached(self, tool_calls: list[ToolCall]) -> bool:
        """True if the terminal tool is in the batch and steps are satisfied."""
        has_terminal = any(tc.tool == self.terminal_tool for tc in tool_calls)
        return has_terminal and self._tracker.is_satisfied()

    @property
    def premature_attempts(self) -> int:
        """Number of premature terminal attempts so far."""
        return self._premature_attempts

    @property
    def premature_exhausted(self) -> bool:
        """True if premature attempts exceed the limit."""
        return self._premature_attempts > self.max_premature_attempts

    def summary_hint(self) -> str:
        """Human-readable hint for context compaction."""
        return self._tracker.summary_hint()
