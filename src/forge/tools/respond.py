"""Synthetic respond tool — structured alternative to bare text responses.

The model calls respond(message="...") instead of producing bare text.
This keeps the model in tool-calling mode where forge's full guardrail
stack applies. Small local models (~8B) cannot be trusted to choose
correctly between text and tool calls — guiding them to a tool is a must.

Usage:

    WorkflowRunner consumers:
        tools["respond"] = respond_tool()
        workflow = Workflow(..., tools=tools, terminal_tool="respond")

    Proxy:
        Injected automatically. respond() calls are converted to plain
        text responses before returning to the client.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from forge.core.workflow import ToolDef, ToolSpec


RESPOND_TOOL_NAME = "respond"

RESPOND_DESCRIPTION = (
    "Respond to the user with a message. Use this when the user is chatting, "
    "asking a question, when you need to ask a clarifying question before "
    "proceeding, or when no other tool action is needed. Also use this "
    "after completing the user's request to report the result."
)


class RespondParams(BaseModel):
    message: str = Field(description="The message to send to the user.")


def respond_spec() -> ToolSpec:
    """Return the ToolSpec for the respond tool (what the LLM sees)."""
    return ToolSpec(
        name=RESPOND_TOOL_NAME,
        description=RESPOND_DESCRIPTION,
        parameters=RespondParams,
    )


def respond_tool() -> ToolDef:
    """Return a complete ToolDef for the respond tool.

    The callable simply returns the message string. Use as a terminal
    tool in WorkflowRunner workflows.
    """

    def _respond(message: str) -> str:
        return message

    return ToolDef(
        spec=respond_spec(),
        callable=_respond,
    )
