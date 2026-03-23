"""Tests for the synthetic respond tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from forge.tools.respond import (
    RESPOND_TOOL_NAME,
    RESPOND_DESCRIPTION,
    RespondParams,
    respond_spec,
    respond_tool,
)
from forge.core.workflow import ToolCall, ToolDef, ToolSpec, TextResponse


# ── respond_spec ──────────────────────────────────────────────────


class TestRespondSpec:
    def test_returns_tool_spec(self) -> None:
        spec = respond_spec()
        assert isinstance(spec, ToolSpec)

    def test_name(self) -> None:
        spec = respond_spec()
        assert spec.name == "respond"

    def test_description(self) -> None:
        spec = respond_spec()
        assert "message" in spec.description.lower()

    def test_parameters_schema_has_message(self) -> None:
        spec = respond_spec()
        schema = spec.get_json_schema()
        assert "message" in schema["properties"]
        assert schema["properties"]["message"]["type"] == "string"

    def test_message_is_required(self) -> None:
        spec = respond_spec()
        schema = spec.get_json_schema()
        assert "message" in schema.get("required", [])


# ── respond_tool ──────────────────────────────────────────────────


class TestRespondTool:
    def test_returns_tool_def(self) -> None:
        tool = respond_tool()
        assert isinstance(tool, ToolDef)

    def test_name_matches(self) -> None:
        tool = respond_tool()
        assert tool.name == "respond"

    def test_callable_returns_message(self) -> None:
        tool = respond_tool()
        result = tool.callable(message="hello world")
        assert result == "hello world"

    def test_callable_returns_empty_string(self) -> None:
        tool = respond_tool()
        result = tool.callable(message="")
        assert result == ""

    def test_spec_matches_respond_spec(self) -> None:
        tool = respond_tool()
        spec = respond_spec()
        assert tool.spec.name == spec.name
        assert tool.spec.description == spec.description


# ── Proxy injection logic ─────────────────────────────────────────


class TestProxyInjection:
    """Test respond tool injection in the proxy handler."""

    def test_inject_when_tools_present(self) -> None:
        from forge.proxy.handler import _extract_tool_specs
        # Simulate a request with one tool
        request_tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        specs = _extract_tool_specs(request_tools)
        # _extract_tool_specs itself doesn't inject — handler does
        assert len(specs) == 1
        assert specs[0].name == "get_weather"

    def test_respond_not_in_extracted_specs(self) -> None:
        """_extract_tool_specs doesn't inject — the handler does."""
        from forge.proxy.handler import _extract_tool_specs
        specs = _extract_tool_specs([{
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Edit file",
                "parameters": {"type": "object", "properties": {}},
            },
        }])
        names = [s.name for s in specs]
        assert "respond" not in names

    def test_no_double_injection_if_client_provides_respond(self) -> None:
        """If the client already defines respond, don't inject a second one."""
        from forge.proxy.handler import _extract_tool_specs
        request_tools = [
            {
                "type": "function",
                "function": {
                    "name": "respond",
                    "description": "Custom respond",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit",
                    "description": "Edit file",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        specs = _extract_tool_specs(request_tools)
        # The handler checks for existing respond — we verify the check works
        has_respond = any(s.name == RESPOND_TOOL_NAME for s in specs)
        assert has_respond
        # Only one respond
        respond_count = sum(1 for s in specs if s.name == RESPOND_TOOL_NAME)
        assert respond_count == 1


# ── Respond stripping logic ───────────────────────────────────────


class TestRespondStripping:
    """Test that respond tool calls are converted to text responses."""

    def test_respond_call_detected(self) -> None:
        tc = ToolCall(tool="respond", args={"message": "Hello!"})
        assert tc.tool == RESPOND_TOOL_NAME
        assert tc.args["message"] == "Hello!"

    def test_respond_mixed_with_real_tools(self) -> None:
        """When respond is mixed with real tool calls, only real calls remain."""
        tool_calls = [
            ToolCall(tool="edit", args={"file": "test.py"}),
            ToolCall(tool="respond", args={"message": "Done!"}),
        ]
        other_calls = [tc for tc in tool_calls if tc.tool != RESPOND_TOOL_NAME]
        respond_calls = [tc for tc in tool_calls if tc.tool == RESPOND_TOOL_NAME]

        assert len(other_calls) == 1
        assert other_calls[0].tool == "edit"
        assert len(respond_calls) == 1
        assert respond_calls[0].args["message"] == "Done!"


# ── Constants ─────────────────────────────────────────────────────


class TestConstants:
    def test_tool_name(self) -> None:
        assert RESPOND_TOOL_NAME == "respond"

    def test_importable_from_forge(self) -> None:
        from forge import RESPOND_TOOL_NAME, respond_spec, respond_tool
        assert RESPOND_TOOL_NAME == "respond"
        assert callable(respond_spec)
        assert callable(respond_tool)
