"""Tests for forge.prompts.templates — build_tool_prompt, extract_tool_call, rescue_tool_call."""

from typing import Literal

from pydantic import BaseModel, Field

from forge.core.workflow import ToolCall, ToolSpec
from forge.prompts.templates import build_tool_prompt, extract_tool_call, rescue_tool_call


class GetPricingParams(BaseModel):
    part_number: str = Field(description="The part number")


def _make_spec(
    name: str = "get_pricing",
    description: str = "Get pricing for a part",
    params: type[BaseModel] | None = None,
) -> ToolSpec:
    if params is None:
        params = GetPricingParams
    return ToolSpec(name=name, description=description, parameters=params)


# ── build_tool_prompt ────────────────────────────────────────────


class TestBuildToolPrompt:
    def test_single_tool_contains_name_and_description(self) -> None:
        spec = _make_spec()
        result = build_tool_prompt([spec])
        assert "get_pricing" in result
        assert "Get pricing for a part" in result

    def test_single_tool_contains_parameter_info(self) -> None:
        spec = _make_spec()
        result = build_tool_prompt([spec])
        assert "part_number" in result
        assert "string" in result
        assert "The part number" in result

    def test_multiple_tools_all_included(self) -> None:
        specs = [
            _make_spec("get_pricing", "Get pricing"),
            _make_spec("get_history", "Get history"),
        ]
        result = build_tool_prompt(specs)
        assert "get_pricing" in result
        assert "get_history" in result

    def test_includes_json_format_instruction(self) -> None:
        spec = _make_spec()
        result = build_tool_prompt([spec])
        assert '"tool"' in result
        assert '"args"' in result

    def test_required_parameter_marked(self) -> None:
        spec = _make_spec()
        result = build_tool_prompt([spec])
        assert "required" in result.lower()

    def test_optional_parameter_marked(self) -> None:
        class OptionalQueryParams(BaseModel):
            query: str | None = Field(default=None, description="Search query")

        spec = _make_spec(params=OptionalQueryParams)
        result = build_tool_prompt([spec])
        assert "optional" in result.lower()

    def test_enum_values_shown(self) -> None:
        class SortParams(BaseModel):
            sort: Literal["asc", "desc"] = Field(description="Sort order")

        spec = _make_spec(params=SortParams)
        result = build_tool_prompt([spec])
        assert "asc" in result
        assert "desc" in result


# ── extract_tool_call ────────────────────────────────────────────


class TestExtractToolCall:
    def test_valid_json(self) -> None:
        text = '{"tool": "get_pricing", "args": {"part": "X123"}}'
        result = extract_tool_call(text, ["get_pricing"])
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].tool == "get_pricing"
        assert result[0].args == {"part": "X123"}

    def test_json_in_code_fences(self) -> None:
        text = '```json\n{"tool": "get_pricing", "args": {"part": "X"}}\n```'
        result = extract_tool_call(text, ["get_pricing"])
        assert len(result) == 1
        assert result[0].tool == "get_pricing"

    def test_json_in_bare_code_fences(self) -> None:
        text = '```\n{"tool": "get_pricing", "args": {}}\n```'
        result = extract_tool_call(text, ["get_pricing"])
        assert len(result) == 1
        assert result[0].tool == "get_pricing"

    def test_json_embedded_in_text(self) -> None:
        text = 'Sure, I will call the tool: {"tool": "get_pricing", "args": {"part": "X"}} Hope that helps!'
        result = extract_tool_call(text, ["get_pricing"])
        assert len(result) == 1
        assert result[0].tool == "get_pricing"

    def test_tool_not_in_available_tools(self) -> None:
        text = '{"tool": "delete_everything", "args": {}}'
        result = extract_tool_call(text, ["get_pricing", "get_history"])
        assert result == []

    def test_no_json(self) -> None:
        text = "I think we should look at the pricing data first."
        result = extract_tool_call(text, ["get_pricing"])
        assert result == []

    def test_malformed_json(self) -> None:
        text = '{"tool": "get_pricing", "args": {bad json}'
        result = extract_tool_call(text, ["get_pricing"])
        assert result == []

    def test_json_without_tool_key(self) -> None:
        text = '{"name": "get_pricing", "params": {}}'
        result = extract_tool_call(text, ["get_pricing"])
        assert result == []

    def test_missing_args_defaults_to_empty(self) -> None:
        text = '{"tool": "get_pricing"}'
        result = extract_tool_call(text, ["get_pricing"])
        assert len(result) == 1
        assert result[0].args == {}

    def test_nested_json_in_args(self) -> None:
        text = '{"tool": "search", "args": {"filter": {"type": "active"}}}'
        result = extract_tool_call(text, ["search"])
        assert len(result) == 1
        assert result[0].args == {"filter": {"type": "active"}}

    def test_multiple_tool_calls(self) -> None:
        text = '{"tool": "get_pricing", "args": {"part": "A"}} {"tool": "search", "args": {"q": "B"}}'
        result = extract_tool_call(text, ["get_pricing", "search"])
        assert len(result) == 2
        assert result[0].tool == "get_pricing"
        assert result[1].tool == "search"


# ── rescue_tool_call ────────────────────────────────────────────


class TestRescueToolCall:
    def test_json_tool_call_in_free_text(self) -> None:
        text = 'I will call the tool now: {"tool": "fetch", "args": {"key": "val"}}'
        result = rescue_tool_call(text, ["fetch", "submit"])
        assert len(result) == 1
        assert result[0].tool == "fetch"
        assert result[0].args == {"key": "val"}

    def test_rehearsal_syntax(self) -> None:
        text = 'fetch[ARGS]{"key": "value"}'
        result = rescue_tool_call(text, ["fetch", "submit"])
        assert len(result) == 1
        assert result[0].tool == "fetch"
        assert result[0].args == {"key": "value"}

    def test_rehearsal_inside_think_tags(self) -> None:
        text = '[THINK]reasoning here[/THINK] report[ARGS]{"findings": "data"}'
        result = rescue_tool_call(text, ["report", "submit"])
        assert len(result) == 1
        assert result[0].tool == "report"
        assert result[0].args == {"findings": "data"}

    def test_rehearsal_inside_xml_think_tags(self) -> None:
        text = '<think>reasoning here</think> fetch[ARGS]{"id": 42}'
        result = rescue_tool_call(text, ["fetch", "submit"])
        assert len(result) == 1
        assert result[0].tool == "fetch"
        assert result[0].args == {"id": 42}

    def test_unknown_tool_returns_empty(self) -> None:
        text = 'delete_everything[ARGS]{"force": true}'
        result = rescue_tool_call(text, ["fetch", "submit"])
        assert result == []

    def test_plain_text_returns_empty(self) -> None:
        text = "I think we should analyze the data first and then report."
        result = rescue_tool_call(text, ["fetch", "submit"])
        assert result == []

    def test_malformed_json_in_rehearsal_returns_empty(self) -> None:
        text = 'fetch[ARGS]{bad json here}'
        result = rescue_tool_call(text, ["fetch", "submit"])
        assert result == []

    def test_empty_string_returns_empty(self) -> None:
        result = rescue_tool_call("", ["fetch", "submit"])
        assert result == []

    def test_only_think_tags_returns_empty(self) -> None:
        text = "[THINK]just thinking, no tool call[/THINK]"
        result = rescue_tool_call(text, ["fetch", "submit"])
        assert result == []

    def test_json_in_code_fences(self) -> None:
        text = '```json\n{"tool": "fetch", "args": {"q": "test"}}\n```'
        result = rescue_tool_call(text, ["fetch", "submit"])
        assert len(result) == 1
        assert result[0].tool == "fetch"

    def test_json_preferred_over_rehearsal(self) -> None:
        """If both JSON and rehearsal syntax are present, JSON wins (tried first)."""
        text = '{"tool": "fetch", "args": {"a": 1}} submit[ARGS]{"b": 2}'
        result = rescue_tool_call(text, ["fetch", "submit"])
        # JSON extraction finds both tool calls
        assert len(result) >= 1
        assert result[0].tool == "fetch"
        assert result[0].args == {"a": 1}
