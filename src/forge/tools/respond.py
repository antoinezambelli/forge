"""Synthetic function tool for structured user-facing responses."""

from forge.core.workflow import ToolDefinition

RESPOND_TOOL_NAME = "respond"
RESPOND_DESCRIPTION = (
    "Respond to the user with a message. Use this when the user is chatting, "
    "asking a question, when you need to ask a clarifying question before "
    "proceeding, or when no other tool action is needed. Also use this "
    "after completing the user's request to report the result."
)


def respond_definition() -> ToolDefinition:
    """Return the model-visible respond definition."""
    return ToolDefinition(
        name=RESPOND_TOOL_NAME,
        description=RESPOND_DESCRIPTION,
        input_schema={
            "type": "object",
            "title": "RespondParams",
            "properties": {
                "message": {
                    "type": "string", "title": "Message",
                    "description": "The message to send to the user.",
                },
            },
            "required": ["message"],
        },
    )


def respond(message: str) -> str:
    """Return the user-facing message; suitable as a Workflow handler."""
    return message
