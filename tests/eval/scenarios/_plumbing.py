"""Plumbing scenarios — basic FC, sequential steps, error recovery."""

from __future__ import annotations

from typing import Any


from forge.core.workflow import ToolDefinition, Workflow

from ._base import EvalScenario, _check

# ── Explicit input schemas ──────────────────────────────────


COUNTRY_SCHEMA = {
    "properties": {
        "country": {"description": "Country name", "title": "Country", "type": "string"}
    },
    "required": ["country"],
    "title": "CountryParams",
    "type": "object",
}


CONTENT_SCHEMA = {
    "properties": {
        "content": {
            "description": "The content to summarize",
            "title": "Content",
            "type": "string",
        }
    },
    "required": ["content"],
    "title": "ContentParams",
    "type": "object",
}


FETCH_SALES_SCHEMA = {
    "properties": {
        "quarter": {
            "description": "Quarter number (1-4)",
            "title": "Quarter",
            "type": "integer",
        },
        "year": {"description": "Four-digit year", "title": "Year", "type": "integer"},
    },
    "required": ["quarter", "year"],
    "title": "FetchSalesParams",
    "type": "object",
}


EMPTY_SCHEMA = {"properties": {}, "title": "EmptyParams", "type": "object"}


FINDINGS_SCHEMA = {
    "properties": {
        "findings": {
            "description": "The findings to include in the report",
            "title": "Findings",
            "type": "string",
        }
    },
    "required": ["findings"],
    "title": "FindingsParams",
    "type": "object",
}


COUNT_SCHEMA = {
    "properties": {
        "count": {
            "description": "Number of records to fetch (must be a numeric string)",
            "title": "Count",
            "type": "string",
        }
    },
    "required": ["count"],
    "title": "CountParams",
    "type": "object",
}


# ── Scenario 1: basic_2step ──────────────────────────────────────

_basic_2step_tools: dict[str, ToolDefinition] = {
    "get_country_info": ToolDefinition(
            name="get_country_info",
            description="Look up facts about a country.",
            input_schema=COUNTRY_SCHEMA,
        ),
    "summarize": ToolDefinition(
            name="summarize",
            description="Summarize content and provide the final answer.",
            input_schema=CONTENT_SCHEMA,
        ),
}
_basic_2step_tools_handlers = {
    "get_country_info": lambda **kwargs: "The capital of France is Paris. Population: 2.1 million.",
    "summarize": lambda **kwargs: kwargs.get("content", ""),
}

basic_2step = EvalScenario(
    name="basic_2step",
    description="Baseline FC check — does the model do function calling at all?",
    workflow=Workflow(
        name="basic_2step",
        description="Simple 2-step information retrieval and summary",
        tools=_basic_2step_tools,
        handlers=_basic_2step_tools_handlers,
        required_steps=["get_country_info"],
        terminal_tool="summarize",
        system_prompt_template=(
            "You are a helpful assistant. Use the available tools to answer "
            "the user's question. First use get_country_info to retrieve "
            "information, then use summarize to provide the final answer."
        ),
    ),
    user_message="What is the capital of France?",
    validate=lambda args: _check(args.get("content", ""), ["paris", "capital"]),
    tags=["plumbing"],
)


# ── Scenario 2: sequential_3step ────────────────────────────────

_sequential_3step_tools: dict[str, ToolDefinition] = {
    "fetch_sales_data": ToolDefinition(
            name="fetch_sales_data",
            description="Fetch sales data for a given quarter and year.",
            input_schema=FETCH_SALES_SCHEMA,
        ),
    "analyze_sales": ToolDefinition(
            name="analyze_sales",
            description="Analyze the loaded sales data and produce findings.",
            input_schema=EMPTY_SCHEMA,
        ),
    "report": ToolDefinition(
            name="report",
            description="Produce a final report from findings.",
            input_schema=FINDINGS_SCHEMA,
        ),
}
_sequential_3step_tools_handlers = {
    "fetch_sales_data": lambda **kwargs: "Dataset: 150 records, 12 columns, covering Q1–Q4 2024 sales data.",
    "analyze_sales": lambda **kwargs: "Analysis: Revenue grew 23% YoY. Top product: Widget Pro. Weakest region: APAC.",
    "report": lambda **kwargs: kwargs.get("findings", ""),
}

sequential_3step = EvalScenario(
    name="sequential_3step",
    description="Required step enforcement — 3-step sequential workflow.",
    workflow=Workflow(
        name="sequential_3step",
        description="Fetch data, analyze, then report",
        tools=_sequential_3step_tools,
        handlers=_sequential_3step_tools_handlers,
        required_steps=["fetch_sales_data", "analyze_sales"],
        terminal_tool="report",
        system_prompt_template=(
            "You are a data analyst assistant. Fetch the sales data first, "
            "then analyze it, then produce a report using the report tool."
        ),
    ),
    user_message="Generate a sales report from the Q4 2024 dataset.",
    validate=lambda args: _check(
        args.get("findings", ""), ["23", "widget pro", "apac"]
    ),
    tags=["plumbing"],
)


# ── Scenario 3: error_recovery ───────────────────────────────────


def _fetch_with_validation(**kwargs: Any) -> str:
    count = kwargs.get("count", "")
    if not (isinstance(count, str) and len(count) == 4 and count.isdigit()):
        raise TypeError(
            f"count must be a zero-padded 4-digit string, got '{count}'"
        )
    return f"Fetched {int(count)} records."


_error_recovery_tools: dict[str, ToolDefinition] = {
    "fetch": ToolDefinition(
            name="fetch",
            description="Fetch records. The count parameter must be a numeric string.",
            input_schema=COUNT_SCHEMA,
        ),
    "summarize": ToolDefinition(
            name="summarize",
            description="Summarize the fetched content.",
            input_schema=CONTENT_SCHEMA,
        ),
}
_error_recovery_tools_handlers = {
    "fetch": _fetch_with_validation,
    "summarize": lambda **kwargs: kwargs.get("content", ""),
}

error_recovery = EvalScenario(
    name="error_recovery",
    description="Tool error self-correction — model must recover from TypeError.",
    workflow=Workflow(
        name="error_recovery",
        description="Fetch with validation, then summarize",
        tools=_error_recovery_tools,
        handlers=_error_recovery_tools_handlers,
        required_steps=["fetch"],
        terminal_tool="summarize",
        system_prompt_template=(
            "You are a helpful assistant. Fetch the requested records, "
            "then summarize them."
        ),
    ),
    user_message="Fetch 10 records and summarize them.",
    validate=lambda args: _check(args.get("content", ""), ["10", "record"]),
    tags=["plumbing"],
)
