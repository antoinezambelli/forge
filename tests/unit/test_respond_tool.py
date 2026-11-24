"""The synthetic definition and handler are independently authored."""

from forge import RESPOND_TOOL_NAME, ToolDefinition, respond, respond_definition


def test_respond_definition():
    definition = respond_definition()
    assert isinstance(definition, ToolDefinition)
    assert definition.name == RESPOND_TOOL_NAME == "respond"
    assert "message" in definition.description.lower()
    schema = definition.input_schema
    assert schema["properties"]["message"]["type"] == "string"
    assert schema["required"] == ["message"]


def test_respond_handler():
    for message in ["hello world", ""]:
        assert respond(message=message) == message
