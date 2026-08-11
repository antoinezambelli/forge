"""Tests for the synthetic respond tool."""

from forge.core.workflow import ToolDef, ToolSpec
from forge.tools.respond import respond_spec, respond_tool


class TestRespondSpec:
    def test_public_spec_contract(self) -> None:
        spec = respond_spec()
        assert isinstance(spec, ToolSpec)
        assert spec.name == "respond"
        assert "message" in spec.description.lower()

        schema = spec.get_json_schema()
        assert "message" in schema["properties"]
        assert schema["properties"]["message"]["type"] == "string"
        assert "message" in schema.get("required", [])


class TestRespondTool:
    def test_public_tool_contract(self) -> None:
        tool = respond_tool()
        spec = respond_spec()

        assert isinstance(tool, ToolDef)
        assert tool.name == "respond"
        assert tool.spec.name == spec.name
        assert tool.spec.description == spec.description
        for message in ["hello world", ""]:
            assert tool.callable(message=message) == message


class TestConstants:
    def test_importable_from_forge(self) -> None:
        from forge import RESPOND_TOOL_NAME, respond_spec, respond_tool

        assert RESPOND_TOOL_NAME == "respond"
        assert callable(respond_spec)
        assert callable(respond_tool)
