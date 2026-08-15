"""Prompt rendering and text tool-call parser contracts."""

from typing import Literal

from pydantic import BaseModel, Field

from forge.core.workflow import ToolCall, ToolSpec
from forge.prompts.templates import (
    build_tool_prompt,
    extract_tool_call,
    rescue_tool_call,
)


class GetPricingParams(BaseModel):
    part_number: str = Field(description="The part number")


def _make_spec(
    name: str = "get_pricing",
    description: str = "Get pricing for a part",
    params: type[BaseModel] = GetPricingParams,
) -> ToolSpec:
    return ToolSpec(name=name, description=description, parameters=params)


def _signatures(calls: list[ToolCall]) -> list[tuple[str, object]]:
    return [(call.tool, call.args) for call in calls]


def test_build_tool_prompt_describes_tools_and_call_format() -> None:
    prompt = build_tool_prompt(
        [
            _make_spec(),
            _make_spec("get_history", "Get history"),
        ]
    )

    for expected in (
        "get_pricing",
        "Get pricing for a part",
        "get_history",
        "part_number",
        "string",
        "The part number",
        '"tool"',
        '"args"',
    ):
        assert expected in prompt


def test_build_tool_prompt_marks_parameter_styles() -> None:
    class QueryParams(BaseModel):
        query: str = Field(description="Search query")
        limit: int | None = Field(default=None, description="Result limit")
        sort: Literal["asc", "desc"] = Field(description="Sort order")

    prompt = build_tool_prompt([_make_spec(params=QueryParams)])

    assert "required" in prompt.lower()
    assert "optional" in prompt.lower()
    assert "asc" in prompt
    assert "desc" in prompt


def test_extract_tool_call_positive_matrix() -> None:
    cases = [
        (
            "forge JSON",
            '{"tool": "get_pricing", "args": {"part": "X123"}}',
            ["get_pricing"],
            [("get_pricing", {"part": "X123"})],
        ),
        (
            "JSON fence",
            '```json\n{"tool": "get_pricing", "args": {"part": "X"}}\n```',
            ["get_pricing"],
            [("get_pricing", {"part": "X"})],
        ),
        (
            "bare fence",
            '```\n{"tool": "get_pricing", "args": {}}\n```',
            ["get_pricing"],
            [("get_pricing", {})],
        ),
        (
            "embedded nested JSON",
            'Calling: {"tool": "search", "args": {"filter": {"type": "active"}}}.',
            ["search"],
            [("search", {"filter": {"type": "active"}})],
        ),
        (
            "OpenAI keys",
            '{"name": "get_pricing", "arguments": {"part": "X123"}}',
            ["get_pricing"],
            [("get_pricing", {"part": "X123"})],
        ),
        (
            "Granite wrapper",
            '<tool_call>{"name": "get_pricing", "arguments": {"part": "X"}}</tool_call>',
            ["get_pricing"],
            [("get_pricing", {"part": "X"})],
        ),
        (
            "missing args",
            '{"tool": "get_pricing"}',
            ["get_pricing"],
            [("get_pricing", {})],
        ),
        (
            "multiple calls",
            '{"tool": "get_pricing", "args": {"part": "A"}} '
            '{"tool": "search", "args": {"q": "B"}}',
            ["get_pricing", "search"],
            [
                ("get_pricing", {"part": "A"}),
                ("search", {"q": "B"}),
            ],
        ),
    ]

    for label, text, available, expected in cases:
        assert _signatures(extract_tool_call(text, available)) == expected, label


def test_extract_tool_call_negative_matrix() -> None:
    cases = [
        (
            "unknown tool",
            '{"tool": "delete_everything", "args": {}}',
            ["get_pricing"],
        ),
        ("plain text", "We should inspect pricing first.", ["get_pricing"]),
        (
            "malformed JSON",
            '{"tool": "get_pricing", "args": {bad json}',
            ["get_pricing"],
        ),
        (
            "unsupported keys",
            '{"function": "get_pricing", "params": {}}',
            ["get_pricing"],
        ),
    ]

    for label, text, available in cases:
        assert extract_tool_call(text, available) == [], label


def test_rescue_tool_call_style_matrix() -> None:
    multiline_command = 'find . -name "*.py" -exec grep -l "percentage" {} \\;'
    cases = [
        (
            "JSON in prose",
            'I will call: {"tool": "fetch", "args": {"key": "val"}}',
            ["fetch", "submit"],
            [("fetch", {"key": "val"})],
        ),
        (
            "rehearsal",
            'fetch[ARGS]{"key": "value"}',
            ["fetch", "submit"],
            [("fetch", {"key": "value"})],
        ),
        (
            "rehearsal after bracket thinking",
            '[THINK]reasoning[/THINK] report[ARGS]{"findings": "data"}',
            ["report", "submit"],
            [("report", {"findings": "data"})],
        ),
        (
            "rehearsal after XML thinking",
            '<think>reasoning</think> fetch[ARGS]{"id": 42}',
            ["fetch", "submit"],
            [("fetch", {"id": 42})],
        ),
        (
            "Qwen parameters",
            "<function=fetch>\n<parameter=query>hello</parameter>\n"
            "<parameter=limit>10</parameter>\n</function>",
            ["fetch", "submit"],
            [("fetch", {"query": "hello", "limit": "10"})],
        ),
        (
            "Qwen multiline parameter",
            "<function=bash>\n<parameter=command>\n"
            f"{multiline_command}\n</parameter>\n</function>",
            ["bash", "edit"],
            [("bash", {"command": multiline_command})],
        ),
        (
            "multiple Qwen calls",
            "<function=fetch><parameter=q>one</parameter></function>\n"
            "<function=submit><parameter=data>two</parameter></function>",
            ["fetch", "submit"],
            [("fetch", {"q": "one"}), ("submit", {"data": "two"})],
        ),
        (
            "Qwen after thinking",
            "<think>reasoning</think>"
            "<function=fetch><parameter=query>weather</parameter></function>",
            ["fetch", "submit"],
            [("fetch", {"query": "weather"})],
        ),
        (
            "Mistral no separator",
            '[TOOL_CALLS]read{"file_path": "transformer.py"}',
            ["read", "edit"],
            [("read", {"file_path": "transformer.py"})],
        ),
        (
            "Mistral space separator",
            '[TOOL_CALLS]read {"file_path": "transformer.py"}',
            ["read", "edit"],
            [("read", {"file_path": "transformer.py"})],
        ),
        (
            "Mistral newline separator",
            '[TOOL_CALLS]read\n{"file_path": "transformer.py"}',
            ["read", "edit"],
            [("read", {"file_path": "transformer.py"})],
        ),
        (
            "Mistral nested braces",
            '[TOOL_CALLS]edit{"file_path": "x.py", '
            '"old_string": "if x: { print(1) }", '
            '"new_string": "if x: { print(2) }"}',
            ["read", "edit"],
            [
                (
                    "edit",
                    {
                        "file_path": "x.py",
                        "old_string": "if x: { print(1) }",
                        "new_string": "if x: { print(2) }",
                    },
                )
            ],
        ),
        (
            "Mistral escaped quote",
            '[TOOL_CALLS]edit{"file_path": "x.py", "old_string": "say \\"hi\\""}',
            ["read", "edit"],
            [("edit", {"file_path": "x.py", "old_string": 'say "hi"'})],
        ),
        (
            "multiple Mistral calls",
            '[TOOL_CALLS]read{"file_path": "a.py"}\n'
            '[TOOL_CALLS]read{"file_path": "b.py"}',
            ["read", "edit"],
            [
                ("read", {"file_path": "a.py"}),
                ("read", {"file_path": "b.py"}),
            ],
        ),
        (
            "Mistral after apology and thinking",
            "I apologize.\n<think>read first</think>\n"
            '[TOOL_CALLS]read{"file_path": "x.py"}',
            ["read", "edit"],
            [("read", {"file_path": "x.py"})],
        ),
    ]

    for label, text, available, expected in cases:
        assert _signatures(rescue_tool_call(text, available)) == expected, label


def test_rescue_tool_call_negative_matrix() -> None:
    cases = [
        ("unknown rehearsal", 'unknown[ARGS]{"force": true}'),
        ("malformed rehearsal", "fetch[ARGS]{bad json}"),
        ("unknown Qwen", "<function=unknown><parameter=x>1</parameter></function>"),
        ("unclosed Qwen", "<function=fetch><parameter=q>never closed"),
        ("unknown Mistral", '[TOOL_CALLS]unknown{"x": 1}'),
        ("unclosed Mistral", '[TOOL_CALLS]fetch{"x": "never closed"'),
        ("plain text", "Analyze the data first."),
        ("empty", ""),
        ("thinking only", "[THINK]just thinking[/THINK]"),
    ]

    for label, text in cases:
        assert rescue_tool_call(text, ["fetch", "submit"]) == [], label


def test_rescue_tool_call_precedence_matrix() -> None:
    cases = [
        (
            "rehearsal",
            '{"tool": "fetch", "args": {"source": "json"}} '
            'submit[ARGS]{"source": "rehearsal"}',
        ),
        (
            "Qwen",
            '{"tool": "fetch", "args": {"source": "json"}} '
            "<function=submit><parameter=source>qwen</parameter></function>",
        ),
        (
            "Mistral",
            '{"tool": "fetch", "args": {"source": "json"}} '
            '[TOOL_CALLS]submit{"source": "mistral"}',
        ),
    ]

    for label, text in cases:
        assert _signatures(rescue_tool_call(text, ["fetch", "submit"])) == [
            ("fetch", {"source": "json"})
        ], label
