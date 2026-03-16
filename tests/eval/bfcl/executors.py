"""Stub executors and terminal tool for BFCL single-turn eval."""

from collections.abc import Callable

from pydantic import BaseModel, Field

from forge.core.workflow import ToolDef, ToolSpec


def make_stub_executor(spec: ToolSpec) -> Callable:
    """Create a stub executor that validates args via Pydantic.

    The stub:
    - Validates all args against the ToolSpec's Pydantic model (strict mode)
    - Returns a deterministic string on success

    Pydantic strict mode rejects type coercion (e.g. "5" for an int field),
    so the model must provide correctly-typed arguments. Validation errors
    are caught by forge's runner and fed back as nudges.
    """
    name = spec.name
    model_class = spec.parameters

    def executor(**kwargs):
        model_class.model_validate(kwargs, strict=True)
        return f"[stub] {name} executed"

    return executor


class DoneParams(BaseModel):
    message: str = Field(
        default="", description="Optional final message or summary."
    )


def make_done_tool() -> ToolDef:
    """Create the synthetic 'done' terminal tool."""
    return ToolDef(
        spec=ToolSpec(
            name="done",
            description=(
                "Call this tool when you have finished using the other tools "
                "and are ready to submit your answer."
            ),
            parameters=DoneParams,
        ),
        callable=lambda **kwargs: kwargs.get("message", ""),
    )
