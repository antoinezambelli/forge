"""Unit tests for forge.core.workflow."""

import pytest
from pydantic import BaseModel, Field, ValidationError

from forge.core.workflow import (
    TextResponse,
    ToolCall,
    ToolDef,
    ToolSpec,
    Workflow,
)


class EmptyParams(BaseModel):
    pass


def _noop(**kwargs):
    """Dummy callable for ToolDef tests."""
    return kwargs


def _make_tool(name: str) -> ToolDef:
    """Create a minimal ToolDef for testing."""
    return ToolDef(
        spec=ToolSpec(
            name=name,
            description=f"Tool {name}",
            parameters=EmptyParams,
        ),
        callable=_noop,
    )


def _make_tools(*names: str) -> dict[str, ToolDef]:
    """Create a dict of minimal ToolDefs for testing."""
    return {name: _make_tool(name) for name in names}


def _make_workflow(**overrides) -> Workflow:
    """Create a valid Workflow with sensible defaults, overridable."""
    defaults = dict(
        name="test_workflow",
        description="A test workflow",
        tools=_make_tools("fetch_data", "submit_result"),
        required_steps=["fetch_data"],
        terminal_tool="submit_result",
        system_prompt_template="You are a {role}. Do {task}.",
    )
    defaults.update(overrides)
    return Workflow(**defaults)


class TestWorkflowValidation:
    def test_raises_on_unknown_required_step(self):
        with pytest.raises(ValueError, match="Required step 'nonexistent'"):
            _make_workflow(required_steps=["nonexistent"])

    def test_raises_on_unknown_terminal_tool(self):
        with pytest.raises(ValueError, match="Terminal tool 'nonexistent'"):
            _make_workflow(terminal_tool="nonexistent")

    def test_raises_on_key_name_mismatch(self):
        tool = _make_tool("actual_name")
        with pytest.raises(ValueError, match="does not match"):
            _make_workflow(tools={"wrong_key": tool, "submit_result": _make_tool("submit_result")})

    def test_raises_when_terminal_tool_in_required_steps(self):
        with pytest.raises(ValueError, match="cannot also be a required step"):
            _make_workflow(
                tools=_make_tools("fetch_data", "submit_result"),
                required_steps=["fetch_data", "submit_result"],
                terminal_tool="submit_result",
            )

    def test_valid_construction_succeeds(self):
        wf = _make_workflow()
        assert wf.name == "test_workflow"
        assert len(wf.tools) == 2

    def test_multiple_terminal_tools_accepted(self):
        tools = _make_tools("fetch_data", "approve", "reject")
        wf = _make_workflow(
            tools=tools,
            required_steps=["fetch_data"],
            terminal_tool=["approve", "reject"],
        )
        assert wf.terminal_tools == frozenset(["approve", "reject"])

    def test_single_terminal_tool_normalized_to_frozenset(self):
        wf = _make_workflow()
        assert isinstance(wf.terminal_tools, frozenset)
        assert wf.terminal_tools == frozenset(["submit_result"])

    def test_raises_on_unknown_terminal_tool_in_list(self):
        with pytest.raises(ValueError, match="Terminal tool 'nonexistent'"):
            _make_workflow(
                tools=_make_tools("fetch_data", "submit_result"),
                terminal_tool=["submit_result", "nonexistent"],
            )

    def test_raises_when_any_terminal_tool_in_required_steps(self):
        with pytest.raises(ValueError, match="cannot also be a required step"):
            _make_workflow(
                tools=_make_tools("fetch_data", "approve", "reject"),
                required_steps=["fetch_data", "approve"],
                terminal_tool=["approve", "reject"],
            )

    def test_raises_on_unknown_prerequisite_name_only(self):
        tools = _make_tools("fetch_data", "submit_result")
        tools["submit_result"].prerequisites = ["nonexistent"]
        with pytest.raises(ValueError, match="Prerequisite 'nonexistent'"):
            _make_workflow(tools=tools)

    def test_raises_on_unknown_prerequisite_arg_matched(self):
        tools = _make_tools("fetch_data", "submit_result")
        tools["submit_result"].prerequisites = [{"tool": "nonexistent", "match_arg": "path"}]
        with pytest.raises(ValueError, match="Prerequisite 'nonexistent'"):
            _make_workflow(tools=tools)

    def test_valid_prerequisite_succeeds(self):
        tools = _make_tools("fetch_data", "submit_result")
        tools["submit_result"].prerequisites = ["fetch_data"]
        wf = _make_workflow(tools=tools)
        assert wf.tools["submit_result"].prerequisites == ["fetch_data"]

    def test_valid_arg_matched_prerequisite_succeeds(self):
        tools = _make_tools("fetch_data", "submit_result")
        tools["submit_result"].prerequisites = [{"tool": "fetch_data", "match_arg": "id"}]
        wf = _make_workflow(tools=tools)
        assert len(wf.tools["submit_result"].prerequisites) == 1


class TestWorkflowMethods:
    def test_build_system_prompt_renders_template(self):
        wf = _make_workflow()
        prompt = wf.build_system_prompt(role="analyst", task="data analysis")
        assert prompt == "You are a analyst. Do data analysis."

    def test_get_tool_specs_returns_specs(self):
        wf = _make_workflow()
        specs = wf.get_tool_specs()
        assert len(specs) == 2
        assert all(isinstance(s, ToolSpec) for s in specs)
        names = {s.name for s in specs}
        assert names == {"fetch_data", "submit_result"}

    def test_get_callable_returns_correct_callable(self):
        def custom_fn(**kwargs):
            return "custom"

        tool = ToolDef(
            spec=ToolSpec(name="custom_tool", description="Custom", parameters=EmptyParams),
            callable=custom_fn,
        )
        wf = _make_workflow(
            tools={"custom_tool": tool, "submit_result": _make_tool("submit_result")},
            required_steps=["custom_tool"],
        )
        assert wf.get_callable("custom_tool") is custom_fn

    def test_get_callable_raises_keyerror_for_unknown(self):
        wf = _make_workflow()
        with pytest.raises(KeyError, match="nonexistent"):
            wf.get_callable("nonexistent")


class TestToolDef:
    def test_name_property_returns_spec_name(self):
        tool = _make_tool("my_tool")
        assert tool.name == "my_tool"


class TestToolCall:
    def test_is_pydantic_model(self):
        tc = ToolCall(tool="fetch", args={"key": "value"})
        assert tc.tool == "fetch"
        assert tc.args == {"key": "value"}

    def test_validates_on_construction(self):
        with pytest.raises(ValidationError):
            ToolCall(tool=123, args="not_a_dict")  # type: ignore[arg-type]

    def test_reasoning_defaults_to_none(self):
        tc = ToolCall(tool="fetch", args={})
        assert tc.reasoning is None

    def test_reasoning_captures_text(self):
        tc = ToolCall(tool="fetch", args={}, reasoning="I should fetch the data")
        assert tc.reasoning == "I should fetch the data"


class TestTextResponse:
    def test_is_pydantic_model(self):
        tr = TextResponse(content="I cannot do that.")
        assert tr.content == "I cannot do that."

    def test_validates_on_construction(self):
        with pytest.raises(ValidationError):
            TextResponse(content=12345)  # type: ignore[arg-type]


class TestFromJsonSchema:
    """Tests for ToolSpec.from_json_schema()."""

    def test_simple_string_params(self):
        """Basic string parameters."""
        spec = ToolSpec.from_json_schema("search", "Search", {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        })
        schema = spec.get_json_schema()
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_optional_params(self):
        """Optional params get default None."""
        spec = ToolSpec.from_json_schema("search", "Search", {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        })
        schema = spec.get_json_schema()
        assert "query" in schema["required"]
        # limit should not be required
        assert "limit" not in schema.get("required", [])

    def test_enum_params(self):
        """Enum values preserved."""
        spec = ToolSpec.from_json_schema("set_unit", "Set unit", {
            "type": "object",
            "properties": {
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["unit"],
        })
        schema = spec.get_json_schema()
        # Enum should appear somewhere in the schema output
        props = schema["properties"]["unit"]
        assert "enum" in props or "anyOf" in props or "$ref" in props

    def test_nested_object(self):
        """Nested object creates sub-model."""
        spec = ToolSpec.from_json_schema("create", "Create item", {
            "type": "object",
            "properties": {
                "item": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number"},
                    },
                    "required": ["name"],
                },
            },
            "required": ["item"],
        })
        schema = spec.get_json_schema()
        assert "item" in schema["properties"]

    def test_array_with_items(self):
        """Array with items type."""
        spec = ToolSpec.from_json_schema("batch", "Batch op", {
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["ids"],
        })
        schema = spec.get_json_schema()
        assert "ids" in schema["properties"]

    def test_default_values(self):
        """Default values preserved."""
        spec = ToolSpec.from_json_schema("config", "Configure", {
            "type": "object",
            "properties": {
                "timeout": {"type": "integer", "default": 30},
            },
        })
        schema = spec.get_json_schema()
        prop = schema["properties"]["timeout"]
        assert prop.get("default") == 30

    def test_empty_properties(self):
        """Schema with no properties produces valid empty model."""
        spec = ToolSpec.from_json_schema("noop", "No-op", {
            "type": "object",
            "properties": {},
        })
        schema = spec.get_json_schema()
        assert schema["properties"] == {} or "properties" in schema

    def test_round_trip(self):
        """from_json_schema -> get_json_schema produces valid schema."""
        input_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name"},
                "count": {"type": "integer", "default": 5},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name"],
        }
        spec = ToolSpec.from_json_schema("test", "Test", input_schema)
        output = spec.get_json_schema()
        # Output should have the same properties
        assert set(output["properties"].keys()) == {"name", "count", "tags"}
