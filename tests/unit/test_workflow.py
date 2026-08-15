"""Behavioral tests for :mod:`forge.core.workflow`."""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from forge.core.workflow import ToolDef, ToolSpec, Workflow


class EmptyParams(BaseModel):
    pass


def _noop(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def _make_tool(name: str, fn=_noop) -> ToolDef:
    return ToolDef(
        spec=ToolSpec(
            name=name,
            description=f"Tool {name}",
            parameters=EmptyParams,
        ),
        callable=fn,
    )


def _make_tools(*names: str) -> dict[str, ToolDef]:
    return {name: _make_tool(name) for name in names}


def _make_workflow(**overrides: Any) -> Workflow:
    values = {
        "name": "test_workflow",
        "description": "A test workflow",
        "tools": _make_tools("fetch_data", "submit_result"),
        "required_steps": ["fetch_data"],
        "terminal_tool": "submit_result",
        "system_prompt_template": "You are a {role}. Do {task}.",
    }
    values.update(overrides)
    return Workflow(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"required_steps": ["nonexistent"]}, "Required step 'nonexistent'"),
        ({"terminal_tool": "nonexistent"}, "Terminal tool 'nonexistent'"),
        (
            {"terminal_tool": ["submit_result", "nonexistent"]},
            "Terminal tool 'nonexistent'",
        ),
        (
            {
                "tools": {
                    "wrong_key": _make_tool("actual_name"),
                    "submit_result": _make_tool("submit_result"),
                }
            },
            "does not match",
        ),
        (
            {"required_steps": ["fetch_data", "submit_result"]},
            "cannot also be a required step",
        ),
        (
            {
                "tools": _make_tools("fetch_data", "approve", "reject"),
                "required_steps": ["fetch_data", "approve"],
                "terminal_tool": ["approve", "reject"],
            },
            "cannot also be a required step",
        ),
    ],
)
def test_invalid_workflow_construction(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _make_workflow(**overrides)


@pytest.mark.parametrize(
    "prerequisites",
    [["nonexistent"], [{"tool": "nonexistent", "match_arg": "path"}]],
    ids=["name", "argument-matched"],
)
def test_rejects_unknown_prerequisite(prerequisites: list[Any]) -> None:
    tools = _make_tools("fetch_data", "submit_result")
    tools["submit_result"].prerequisites = prerequisites

    with pytest.raises(ValueError, match="Prerequisite 'nonexistent'"):
        _make_workflow(tools=tools)


@pytest.mark.parametrize(
    "prerequisites",
    [["fetch_data"], [{"tool": "fetch_data", "match_arg": "id"}]],
    ids=["name", "argument-matched"],
)
def test_accepts_supported_prerequisite(prerequisites: list[Any]) -> None:
    tools = _make_tools("fetch_data", "submit_result")
    tools["submit_result"].prerequisites = prerequisites

    workflow = _make_workflow(tools=tools)

    assert workflow.tools["submit_result"].prerequisites == prerequisites


@pytest.mark.parametrize(
    ("terminal_tool", "expected"),
    [
        ("submit_result", {"submit_result"}),
        (["approve", "reject"], {"approve", "reject"}),
    ],
)
def test_terminal_tools_are_normalized(
    terminal_tool: str | list[str], expected: set[str]
) -> None:
    tools = _make_tools("fetch_data", "submit_result", "approve", "reject")
    workflow = _make_workflow(tools=tools, terminal_tool=terminal_tool)

    assert workflow.terminal_tools == frozenset(expected)


def test_workflow_public_methods_render_and_dispatch() -> None:
    def custom_fn(**kwargs: Any) -> str:
        return f"handled {kwargs['value']}"

    tools = {
        "custom_tool": _make_tool("custom_tool", custom_fn),
        "submit_result": _make_tool("submit_result"),
    }
    workflow = _make_workflow(
        tools=tools,
        required_steps=["custom_tool"],
    )

    assert workflow.build_system_prompt(role="analyst", task="analysis") == (
        "You are a analyst. Do analysis."
    )
    assert {spec.name for spec in workflow.get_tool_specs()} == set(tools)
    assert workflow.get_callable("custom_tool")(value="input") == "handled input"
    with pytest.raises(KeyError, match="nonexistent"):
        workflow.get_callable("nonexistent")


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Name"},
        "count": {"type": "integer", "default": 5},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "item": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "price": {"type": "number"},
            },
            "required": ["sku"],
        },
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "unit", "item", "tags"],
}


@pytest.mark.parametrize(
    ("payload", "valid"),
    [
        (
            {
                "name": "sample",
                "unit": "celsius",
                "item": {"sku": "A-1", "price": 2.5},
                "tags": ["new"],
            },
            True,
        ),
        (
            {
                "name": "sample",
                "unit": "kelvin",
                "item": {"sku": "A-1"},
                "tags": [],
            },
            False,
        ),
        (
            {
                "name": "sample",
                "unit": "celsius",
                "item": {"price": 2.5},
                "tags": [],
            },
            False,
        ),
        (
            {
                "unit": "celsius",
                "item": {"sku": "A-1"},
                "tags": [],
            },
            False,
        ),
    ],
    ids=["valid", "invalid-enum", "invalid-nested", "missing-required"],
)
def test_json_schema_model_validation(payload: dict[str, Any], valid: bool) -> None:
    parameters = ToolSpec.from_json_schema("create", "Create", JSON_SCHEMA).parameters

    if not valid:
        with pytest.raises(ValidationError):
            parameters.model_validate(payload)
        return

    parsed = parameters.model_validate(payload)
    assert parsed.count == 5
    assert parsed.item.sku == "A-1"
    assert parsed.tags == ["new"]


def test_json_schema_round_trip_preserves_public_contract() -> None:
    output = ToolSpec.from_json_schema(
        "create", "Create", JSON_SCHEMA
    ).get_json_schema()

    assert set(output["properties"]) == {"name", "count", "unit", "item", "tags"}
    assert set(output["required"]) == {"name", "unit", "item", "tags"}
    assert output["properties"]["count"]["default"] == 5
