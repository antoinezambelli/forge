"""Plumbing scenarios — basic FC, sequential steps, compaction, error recovery."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from forge.core.workflow import ToolDef, ToolSpec, Workflow

from ._base import EvalScenario, _check

# ── Pydantic parameter models ──────────────────────────────────


class CountryParams(BaseModel):
    country: str = Field(description="Country name")


class ContentParams(BaseModel):
    content: str = Field(description="The content to summarize")


class FetchSalesParams(BaseModel):
    quarter: int = Field(description="Quarter number (1-4)")
    year: int = Field(description="Four-digit year")


class EmptyParams(BaseModel):
    pass


class FindingsParams(BaseModel):
    findings: str = Field(description="The findings to include in the report")


class CountParams(BaseModel):
    count: str = Field(description="Number of records to fetch (must be a numeric string)")


# ── Scenario 1: basic_2step ──────────────────────────────────────

_basic_2step_tools: dict[str, ToolDef] = {
    "get_country_info": ToolDef(
        spec=ToolSpec(
            name="get_country_info",
            description="Look up facts about a country.",
            parameters=CountryParams,
        ),
        callable=lambda **kwargs: "The capital of France is Paris. Population: 2.1 million.",
    ),
    "summarize": ToolDef(
        spec=ToolSpec(
            name="summarize",
            description="Summarize content and provide the final answer.",
            parameters=ContentParams,
        ),
        callable=lambda **kwargs: kwargs.get("content", ""),
    ),
}

basic_2step = EvalScenario(
    name="basic_2step",
    description="Baseline FC check — does the model do function calling at all?",
    workflow=Workflow(
        name="basic_2step",
        description="Simple 2-step information retrieval and summary",
        tools=_basic_2step_tools,
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

_sequential_3step_tools: dict[str, ToolDef] = {
    "fetch_sales_data": ToolDef(
        spec=ToolSpec(
            name="fetch_sales_data",
            description="Fetch sales data for a given quarter and year.",
            parameters=FetchSalesParams,
        ),
        callable=lambda **kwargs: "Dataset: 150 records, 12 columns, covering Q1–Q4 2024 sales data.",
    ),
    "analyze_sales": ToolDef(
        spec=ToolSpec(
            name="analyze_sales",
            description="Analyze the loaded sales data and produce findings.",
            parameters=EmptyParams,
        ),
        callable=lambda **kwargs: "Analysis: Revenue grew 23% YoY. Top product: Widget Pro. Weakest region: APAC.",
    ),
    "report": ToolDef(
        spec=ToolSpec(
            name="report",
            description="Produce a final report from findings.",
            parameters=FindingsParams,
        ),
        callable=lambda **kwargs: kwargs.get("findings", ""),
    ),
}

sequential_3step = EvalScenario(
    name="sequential_3step",
    description="Required step enforcement — 3-step sequential workflow.",
    workflow=Workflow(
        name="sequential_3step",
        description="Fetch data, analyze, then report",
        tools=_sequential_3step_tools,
        required_steps=["fetch_sales_data", "analyze_sales"],
        terminal_tool="report",
        system_prompt_template=(
            "You are a data analyst assistant. Fetch the sales data first, "
            "then analyze it, then produce a report using the report tool."
        ),
    ),
    user_message="Generate a sales report from the Q4 2024 dataset.",
    validate=lambda args: _check(args.get("findings", ""), ["23", "widget pro", "apac"]),
    tags=["plumbing"],
)


# ── Scenario 3: compaction_stress ────────────────────────────────

_COMPACTION_FETCH_RESULT = (
    "Q4 2024 Sales Summary (North America)\n"
    "Generated: 2025-01-15 | Source: SAP ERP export | Currency: USD\n"
    "\n"
    "═══ Regional Breakdown ═══\n"
    "Region: Northeast | Revenue: $2,847,300 | Units: 14,200 | Avg Price: $200.51\n"
    "  Top SKU: Widget Pro (6,200 units, $1,364,000) | Returns: 2.1% | Net margin: 34.2%\n"
    "  Channel mix: Direct 58%, Distributor 31%, Online 11%\n"
    "  Notable: Boston metro +31% QoQ, NYC flat, Philadelphia -8% (warehouse issue Dec)\n"
    "Region: Southeast | Revenue: $1,932,100 | Units: 11,400 | Avg Price: $169.48\n"
    "  Top SKU: Widget Standard (4,800 units, $672,000) | Returns: 3.4% | Net margin: 28.1%\n"
    "  Channel mix: Direct 42%, Distributor 44%, Online 14%\n"
    "  Notable: Atlanta +22% QoQ, Miami +15%, Charlotte new territory launched Oct\n"
    "Region: Midwest   | Revenue: $2,105,600 | Units: 12,800 | Avg Price: $164.50\n"
    "  Top SKU: Widget Standard (5,100 units, $714,000) | Returns: 1.8% | Net margin: 31.5%\n"
    "  Channel mix: Direct 35%, Distributor 52%, Online 13%\n"
    "  Notable: Chicago stable, Detroit +18% (auto sector recovery), Minneapolis -3%\n"
    "Region: Southwest | Revenue: $1,478,200 | Units:  8,900 | Avg Price: $166.09\n"
    "  Top SKU: Widget Lite (3,200 units, $288,000) | Returns: 4.1% | Net margin: 22.7%\n"
    "  Channel mix: Direct 29%, Distributor 48%, Online 23%\n"
    "  Notable: Phoenix +9%, Dallas flat, high return rate traced to packaging defect lot #4471\n"
    "Region: West      | Revenue: $3,201,400 | Units: 15,600 | Avg Price: $205.22\n"
    "  Top SKU: Widget Pro (7,800 units, $1,716,000) | Returns: 1.5% | Net margin: 38.6%\n"
    "  Channel mix: Direct 61%, Distributor 22%, Online 17%\n"
    "  Notable: SF Bay +28% QoQ, Seattle +19%, LA +12%, Portland -5% (rep vacancy)\n"
    "\n"
    "═══ Product Summary ═══\n"
    "1. Widget Pro      — 18,400 units ($3,812,000) | ASP: $207.17 | Margin: 36.4%\n"
    "   Q3→Q4 trend: +14% units, +16% revenue | Inventory: 4,200 units (3.1 weeks)\n"
    "2. Widget Standard — 12,200 units ($1,708,000) | ASP: $140.00 | Margin: 29.8%\n"
    "   Q3→Q4 trend: +8% units, +7% revenue | Inventory: 6,800 units (7.6 weeks)\n"
    "3. Widget Lite     —  9,800 units ($882,000)   | ASP: $90.00  | Margin: 21.3%\n"
    "   Q3→Q4 trend: -2% units, -3% revenue | Inventory: 8,100 units (11.3 weeks)\n"
    "4. Accessory Pack  — 22,500 units ($1,125,000) | ASP: $50.00  | Margin: 52.1%\n"
    "   Q3→Q4 trend: +21% units, +21% revenue | Inventory: 12,400 units (7.5 weeks)\n"
    "\n"
    "═══ Monthly Trend (Q4) ═══\n"
    "Oct 2024: Revenue $3,412,800 | Units 18,900 | ASP $180.57 | Returns 2.8%\n"
    "  Strongest: West ($982K), Weakest: Southwest ($398K, lot #4471 returns peak)\n"
    "Nov 2024: Revenue $3,891,200 | Units 21,200 | ASP $183.55 | Returns 2.1%\n"
    "  Strongest: Northeast ($1,012K, Black Friday), Weakest: Southwest ($442K)\n"
    "Dec 2024: Revenue $4,260,600 | Units 22,800 | ASP $186.87 | Returns 2.2%\n"
    "  Strongest: West ($1,180K, year-end deals), Weakest: Midwest ($612K, weather)\n"
    "\n"
    "═══ Totals ═══\n"
    "Total Revenue: $11,564,600 | Total Units: 62,900 | Blended ASP: $183.86\n"
    "YoY Growth: +23.1% revenue, +18.4% units\n"
    "Gross Margin: $3,842,100 (33.2%) | Returns: $418,200 (2.4% of gross)\n"
    "Outstanding AR: $2,891,150 (DSO: 45 days, up from 38 in Q3)"
)

_COMPACTION_ANALYZE_RESULT = (
    "Analysis of Q4 2024 Sales Data:\n"
    "\n"
    "REVENUE CONCENTRATION:\n"
    "- West region leads revenue ($3.2M, 27.7%) driven by Widget Pro premium pricing\n"
    "- Northeast second ($2.8M, 24.6%) with strongest direct channel presence (58%)\n"
    "- Top 2 regions account for 52.3% of total revenue — geographic concentration risk\n"
    "- Southwest weakest at $1.5M (12.8%) — lowest unit volume and below-average pricing\n"
    "\n"
    "PRODUCT MIX:\n"
    "- Widget Pro: 29.3% of units but 33.0% of revenue — premium mix driving ASP up\n"
    "- Widget Standard: stable workhorse, 19.4% of units, predictable margins (29.8%)\n"
    "- Widget Lite: declining (-2% QoQ), excess inventory (11.3 weeks vs 6-week target)\n"
    "  Action needed: mark down or bundle with Pro to clear inventory before Q1 refresh\n"
    "- Accessory Pack: highest margin (52.1%) and fastest growth (+21% QoQ)\n"
    "  Attach rate varies: 68% with Pro, 41% with Standard, 22% with Lite\n"
    "\n"
    "PRICING & MARGIN:\n"
    "- YoY revenue growth (+23.1%) outpaces unit growth (+18.4%) — price increases\n"
    "  contributing ~4.7pp of revenue growth. Sustainability risk if demand is price elastic\n"
    "- Southwest net margin (22.7%) drags blended margin down — packaging defect lot #4471\n"
    "  drove 4.1% return rate. Estimated defect cost: $61,200 in returns + $18,400 reshipping\n"
    "- West achieves best margin (38.6%) via direct channel dominance and Pro concentration\n"
    "\n"
    "CHANNEL TRENDS:\n"
    "- Direct: 45% of revenue, highest margin, but requires most sales headcount\n"
    "- Distributor: 39% of revenue, growing in Midwest (52%) — margin compression risk\n"
    "  as distributors push for volume discounts in Q1 negotiations\n"
    "- Online: 16% of revenue, fastest growing channel (+34% QoQ), lowest cost to serve\n"
    "  Opportunity: shift Widget Lite sales online to recover margin\n"
    "\n"
    "OPERATIONAL FLAGS:\n"
    "- DSO increased from 38 to 45 days — AR aging in Southeast (distributor terms)\n"
    "- Widget Lite inventory at 11.3 weeks — well above 6-week target, write-down risk\n"
    "- Portland rep vacancy since Oct — $180K revenue gap, territory reassigned to Seattle\n"
    "- Packaging defect lot #4471 resolved Nov 28, returns normalizing in Dec data"
)

_compaction_stress_tools: dict[str, ToolDef] = {
    "fetch_sales_data": ToolDef(
        spec=ToolSpec(
            name="fetch_sales_data",
            description="Fetch sales data for a given quarter and year.",
            parameters=FetchSalesParams,
        ),
        callable=lambda **kwargs: _COMPACTION_FETCH_RESULT,
    ),
    "analyze_sales": ToolDef(
        spec=ToolSpec(
            name="analyze_sales",
            description="Analyze the loaded sales data and produce findings.",
            parameters=EmptyParams,
        ),
        callable=lambda **kwargs: _COMPACTION_ANALYZE_RESULT,
    ),
    "report": ToolDef(
        spec=ToolSpec(
            name="report",
            description="Produce a final report.",
            parameters=FindingsParams,
        ),
        callable=lambda **kwargs: kwargs.get("findings", ""),
    ),
}

compaction_stress = EvalScenario(
    name="compaction_stress",
    description="Large tool results under tight budget — forces compaction.",
    workflow=Workflow(
        name="compaction_stress",
        description="Fetch, analyze, report under tight token budget",
        tools=_compaction_stress_tools,
        required_steps=["fetch_sales_data", "analyze_sales"],
        terminal_tool="report",
        system_prompt_template=(
            "You are a data analyst assistant. Fetch the sales data first, "
            "then analyze it, then produce a report using the report tool."
        ),
    ),
    user_message="Generate a sales report from the Q4 2024 dataset.",
    budget_tokens=2048,
    validate=lambda args: _check(args.get("findings", ""), ["23", "widget pro", "west"]),
    tags=["plumbing", "compaction"],
)


# ── Scenario 4: error_recovery ───────────────────────────────────


def _fetch_with_validation(**kwargs: Any) -> str:
    count = kwargs.get("count", "")
    if not (isinstance(count, str) and len(count) == 4 and count.isdigit()):
        raise TypeError(
            f"count must be a zero-padded 4-digit string, got '{count}'"
        )
    return f"Fetched {int(count)} records."


_error_recovery_tools: dict[str, ToolDef] = {
    "fetch": ToolDef(
        spec=ToolSpec(
            name="fetch",
            description="Fetch records. The count parameter must be a numeric string.",
            parameters=CountParams,
        ),
        callable=_fetch_with_validation,
    ),
    "summarize": ToolDef(
        spec=ToolSpec(
            name="summarize",
            description="Summarize the fetched content.",
            parameters=ContentParams,
        ),
        callable=lambda **kwargs: kwargs.get("content", ""),
    ),
}

error_recovery = EvalScenario(
    name="error_recovery",
    description="Tool error self-correction — model must recover from TypeError.",
    workflow=Workflow(
        name="error_recovery",
        description="Fetch with validation, then summarize",
        tools=_error_recovery_tools,
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
