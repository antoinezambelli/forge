"""EvalScenario dataclass and shared helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


from forge.core.workflow import ToolDefinition, Workflow


__PLACEHOLDER_SCHEMA = {
    "properties": {
        "placeholder_field": {
            "description": "placeholder",
            "title": "Placeholder Field",
            "type": "string",
        }
    },
    "required": ["placeholder_field"],
    "title": "_PlaceholderParams",
    "type": "object",
}


@dataclass
class EvalScenario:
    """A single eval scenario with deterministic tools."""

    name: str
    description: str
    workflow: Workflow
    user_message: str
    budget_tokens: int = 8192
    max_iterations: int = 15
    max_retries_per_step: int = 5
    max_tool_errors: int = 2
    validate: Callable[[dict[str, Any]], bool] | None = None
    validate_state: Callable[[], bool] | None = None
    build_workflow: Callable[[], tuple[Workflow, Callable[[], bool] | None]] | None = None
    tags: list[str] = field(default_factory=list)
    ideal_iterations: int | None = None


def _placeholder_workflow(
    name: str,
    terminal_tool: str,
    required_steps: list[str] | None = None,
) -> Workflow:
    """Build a minimal valid Workflow placeholder for stateful scenarios.

    Stateful scenarios use ``build_workflow`` to create a fresh workflow
    per run.  The placeholder only needs to pass Workflow.__post_init__
    validation — it is never actually executed.
    """
    steps = required_steps or []
    tools = {
        tool_name: ToolDefinition(
            name=tool_name,
            description="placeholder",
            input_schema=__PLACEHOLDER_SCHEMA,
        )
        for tool_name in [*steps, terminal_tool]
    }
    return Workflow(
        name=name,
        description="placeholder",
        tools=tools,
        handlers={tool_name: lambda **kw: "" for tool_name in tools},
        required_steps=steps,
        terminal_tool=terminal_tool,
        system_prompt_template="",
    )


def _check(text: str | None, required: list[str]) -> bool:
    """Case-insensitive AND check: all required substrings must be present.

    Strips commas from text so '1,500' matches '1500'.
    """
    if not isinstance(text, str):
        return False
    lower = text.lower().replace(",", "")
    return all(term in lower for term in required)
