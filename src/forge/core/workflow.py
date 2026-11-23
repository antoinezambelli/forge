"""Tool and workflow definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolDefinition:
    """The complete model-visible definition of a function tool."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Tool name must be a nonempty string")
        if not isinstance(self.description, str):
            raise ValueError("Tool description must be a string")
        if not isinstance(self.input_schema, dict):
            raise ValueError("Tool input_schema must be a dictionary")
        self.extra_fields = {
            key: value
            for key, value in self.extra_fields.items()
            if key not in {"name", "description", "input_schema"}
        }


@dataclass
class Workflow:
    """Declarative workflow definition. Provided by downstream projects.

    Tools are ordered definitions keyed by name. Handlers and prerequisites
    are separately owned execution data, validated at construction time.
    It does NOT contain execution logic — that's the WorkflowRunner's job.
    """

    name: str
    description: str
    tools: dict[str, ToolDefinition]
    handlers: dict[str, Callable[..., Any]]
    required_steps: list[str]
    terminal_tool: str | list[str]
    system_prompt_template: str
    prerequisites: dict[str, list[str | dict[str, str]]] = field(default_factory=dict)
    terminal_tools: frozenset[str] = field(default_factory=frozenset, init=False)

    def __post_init__(self) -> None:
        # Normalize terminal_tool to frozenset for O(1) membership checks.
        if isinstance(self.terminal_tool, str):
            self.terminal_tools = frozenset([self.terminal_tool])
        else:
            self.terminal_tools = frozenset(self.terminal_tool)

        for key, tool_def in self.tools.items():
            if key != tool_def.name:
                raise ValueError(
                    f"Tool key '{key}' does not match ToolDefinition name '{tool_def.name}'"
                )
        tool_names = set(self.tools.keys())
        for name in self.handlers:
            if name not in tool_names:
                raise ValueError(f"Handler '{name}' not in tools: {tool_names}")
        for name in tool_names:
            if name not in self.handlers:
                raise ValueError(f"Missing handler for tool '{name}'")
        for name in self.prerequisites:
            if name not in tool_names:
                raise ValueError(
                    f"Prerequisite owner '{name}' not in tools: {tool_names}"
                )
        for step in self.required_steps:
            if step not in tool_names:
                raise ValueError(f"Required step '{step}' not in tools: {tool_names}")
        for tt in self.terminal_tools:
            if tt not in tool_names:
                raise ValueError(f"Terminal tool '{tt}' not in tools: {tool_names}")
            if tt in self.required_steps:
                raise ValueError(f"Terminal tool '{tt}' cannot also be a required step")
        for key, prerequisites in self.prerequisites.items():
            for prereq in prerequisites:
                prereq_name = prereq if isinstance(prereq, str) else prereq["tool"]
                if prereq_name not in tool_names:
                    raise ValueError(
                        f"Prerequisite '{prereq_name}' for tool '{key}' "
                        f"not in tools: {tool_names}"
                    )

    def build_system_prompt(self, **kwargs: str) -> str:
        """Render the system prompt with user-provided values."""
        return self.system_prompt_template.format(**kwargs)

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Return the ordered definitions for inference."""
        return list(self.tools.values())
