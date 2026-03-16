"""Required-step tracking for workflow enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepTracker:
    """Tracks which required steps have been completed.

    Lives on the WorkflowRunner, outside the message history.
    Compaction cannot invalidate step completion. See P0-1.
    """

    required_steps: list[str]
    completed_steps: dict[str, None] = field(default_factory=dict)

    def record(self, tool_name: str) -> None:
        """Record a tool call as completed."""
        self.completed_steps[tool_name] = None

    def is_satisfied(self) -> bool:
        """True if all required steps have been called."""
        return all(s in self.completed_steps for s in self.required_steps)

    def pending(self) -> list[str]:
        """Return required steps not yet completed, preserving original order."""
        return [s for s in self.required_steps if s not in self.completed_steps]

    def summary_hint(self) -> str:
        """Human-readable hint for injection into compacted summaries."""
        if not self.completed_steps:
            return "[No steps completed yet]"
        return f"[Steps completed: {', '.join(self.completed_steps)}]"
