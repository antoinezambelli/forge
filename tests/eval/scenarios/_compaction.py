"""Compaction scenarios — phase2 compaction stress and relevance detection."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from forge.core.workflow import ToolDef, ToolSpec, Workflow

from ._base import EvalScenario, _check


class SupplierParams(BaseModel):
    supplier: str = Field(description="The supplier name to check")


class RecommendParams(BaseModel):
    primary_supplier: str = Field(description="The recommended primary supplier name")
    reasoning: str = Field(description="Justification for the recommendation, including key metrics")


class CityParams(BaseModel):
    city: str = Field(description="The city name")


class FlightParams(BaseModel):
    origin: str = Field(description="Departure city")
    destination: str = Field(description="Arrival city")


class HotelParams(BaseModel):
    city: str = Field(description="The city to check")
    checkin: str = Field(description="Check-in date")


class CurrencyParams(BaseModel):
    amount: str = Field(description="The amount to convert")
    from_currency: str = Field(description="Source currency code")
    to_currency: str = Field(description="Target currency code")


class ReasonParams(BaseModel):
    reason: str = Field(description="Brief explanation of why no tool is appropriate")


# ── Scenario 10: phase2_compaction ────────────────────────────────

# Five suppliers, each returning ~450 chars of structured data.
# budget_tokens=950, trigger at 712 (75%) with eval keep_recent=2.
# Tuned via bisection: 800 too tight (MaxIterErrors), 1200 too loose (always P3).
# P2 (drop tool_results) removes ~100 tok/iteration — target landing phase.
# Reasoning models preserve key facts in <think> traces (survive P2).
# Instruct models lose the data entirely from compacted iterations.

_SUPPLIER_DATA: dict[str, str] = {
    "alphatech": (
        "Supplier Assessment — AlphaTech Components (Shenzhen)\n"
        "Component: MCU-4200 microcontroller\n"
        "Unit Price: $4.82 (MOQ 5,000) | $4.55 (MOQ 10,000)\n"
        "Lead Time: 6 weeks (standard) | 4 weeks (expedited, +15%)\n"
        "Quality: Defect rate 0.8% | ISO 9001 certified\n"
        "Capacity: 50,000 units/month | Current utilization: 72%\n"
        "Payment Terms: Net 45 | 2% discount for Net 15\n"
        "Shipping: FOB Shenzhen | DDP available (+$0.35/unit)\n"
        "Risk Flags: Single-source for ceramic substrate\n"
        "Last Audit: 2024-09-15 — passed, minor finding on traceability"
    ),
    "meridian": (
        "Supplier Assessment — Meridian Electronics (Taipei)\n"
        "Component: MCU-4200 microcontroller\n"
        "Unit Price: $5.10 (MOQ 5,000) | $4.78 (MOQ 10,000)\n"
        "Lead Time: 8 weeks (standard) | 5 weeks (expedited, +20%)\n"
        "Quality: Defect rate 0.3% | ISO 9001 + IATF 16949 certified\n"
        "Capacity: 35,000 units/month | Current utilization: 88%\n"
        "Payment Terms: Net 30 | No early payment discount\n"
        "Shipping: FOB Taipei | DDP available (+$0.42/unit)\n"
        "Risk Flags: High utilization, limited surge capacity\n"
        "Last Audit: 2024-11-20 — passed, no findings"
    ),
    "novacore": (
        "Supplier Assessment — NovaCore Industries (Guadalajara)\n"
        "Component: MCU-4200 microcontroller\n"
        "Unit Price: $5.45 (MOQ 3,000) | $5.12 (MOQ 8,000)\n"
        "Lead Time: 4 weeks (standard) | 2 weeks (expedited, +25%)\n"
        "Quality: Defect rate 1.2% | ISO 9001 certified\n"
        "Capacity: 25,000 units/month | Current utilization: 61%\n"
        "Payment Terms: Net 30 | 3% discount for Net 10\n"
        "Shipping: FCA Guadalajara | DDP to US included in price\n"
        "Risk Flags: Newer facility, limited track record (est. 2022)\n"
        "Last Audit: 2024-08-03 — passed, finding on ESD controls"
    ),
    "precisionworks": (
        "Supplier Assessment — PrecisionWorks Ltd (Penang)\n"
        "Component: MCU-4200 microcontroller\n"
        "Unit Price: $4.65 (MOQ 5,000) | $4.30 (MOQ 15,000)\n"
        "Lead Time: 7 weeks (standard) | 5 weeks (expedited, +18%)\n"
        "Quality: Defect rate 0.5% | ISO 9001 + AS9100 certified\n"
        "Capacity: 60,000 units/month | Current utilization: 55%\n"
        "Payment Terms: Net 60 | 1.5% discount for Net 30\n"
        "Shipping: FOB Penang | DDP available (+$0.38/unit)\n"
        "Risk Flags: None identified\n"
        "Last Audit: 2024-10-28 — passed, no findings"
    ),
    "circuitedge": (
        "Supplier Assessment — CircuitEdge Inc (Juárez)\n"
        "Component: MCU-4200 microcontroller\n"
        "Unit Price: $5.25 (MOQ 2,000) | $4.95 (MOQ 7,000)\n"
        "Lead Time: 3 weeks (standard) | 1 week (expedited, +30%)\n"
        "Quality: Defect rate 1.8% | ISO 9001 pending (expected Q2 2025)\n"
        "Capacity: 20,000 units/month | Current utilization: 45%\n"
        "Payment Terms: Net 30 | No early payment discount\n"
        "Shipping: FCA Juárez | DDP to US included in price\n"
        "Risk Flags: ISO 9001 not yet certified, high defect rate\n"
        "Last Audit: 2024-06-12 — conditional pass, corrective actions pending"
    ),
}

# Ordered list so the model gets a deterministic sequence
_SUPPLIER_ORDER = ["alphatech", "meridian", "novacore", "precisionworks", "circuitedge"]


def _check_supplier(**kwargs: Any) -> str:
    name = kwargs.get("supplier", "").strip().lower()
    # Fuzzy match: accept partial names
    for key in _SUPPLIER_DATA:
        if key in name or name in key:
            return _SUPPLIER_DATA[key]
    return (
        f"Unknown supplier: '{kwargs.get('supplier', '')}'. "
        f"Available suppliers: AlphaTech, Meridian, NovaCore, PrecisionWorks, CircuitEdge."
    )


_phase2_compaction_tools: dict[str, ToolDef] = {
    "check_supplier": ToolDef(
        spec=ToolSpec(
            name="check_supplier",
            description="Check a supplier's assessment data for the MCU-4200 component.",
            parameters=SupplierParams,
        ),
        callable=_check_supplier,
    ),
    "recommend": ToolDef(
        spec=ToolSpec(
            name="recommend",
            description="Submit a procurement recommendation based on supplier assessments.",
            parameters=RecommendParams,
        ),
        callable=lambda **kwargs: f"Recommendation recorded: {kwargs.get('primary_supplier', '')}",
    ),
}


def _validate_phase2_compaction(args: dict[str, Any]) -> bool:
    text = f"{args.get('primary_supplier', '')} {args.get('reasoning', '')}".lower()
    # PrecisionWorks is objectively the best: lowest price ($4.30 at MOQ),
    # low defect rate (0.5%), highest capacity (60K), lowest utilization (55%),
    # no risk flags, AS9100 certified
    has_supplier = "precisionworks" in text or "precision" in text
    has_metric = any(t in text for t in [
        "4.30", "4.65", "0.5%", "defect", "60,000", "60000", "capacity",
        "as9100", "no risk", "no finding",
    ])
    return has_supplier and has_metric


phase2_compaction = EvalScenario(
    name="phase2_compaction",
    description="5-supplier audit under tight budget — forces TieredCompact phase 2.",
    workflow=Workflow(
        name="phase2_compaction",
        description="Check 5 suppliers for MCU-4200 component, then recommend the best one",
        tools=_phase2_compaction_tools,
        required_steps=["check_supplier"],
        terminal_tool="recommend",
        system_prompt_template=(
            "You are a procurement analyst. Evaluate all five suppliers for "
            "the MCU-4200 microcontroller by checking each one, then submit a "
            "recommendation for the primary supplier. Check suppliers in this "
            "order: AlphaTech, Meridian, NovaCore, PrecisionWorks, CircuitEdge. "
            "Your recommendation must name the best supplier and cite specific "
            "metrics (price, quality, capacity) that justify the choice."
        ),
    ),
    user_message=(
        "We need to source 10,000 MCU-4200 microcontrollers. Evaluate all "
        "five suppliers and recommend the best primary supplier."
    ),
    budget_tokens=925,
    validate=_validate_phase2_compaction,
    tags=["compaction", "reasoning"],
    ideal_iterations=6,
)


# ── Scenario 11: relevance_detection ─────────────────────────────

_relevance_detection_tools: dict[str, ToolDef] = {
    "get_forecast": ToolDef(
        spec=ToolSpec(name="get_forecast", description="Get weather forecast for a city.",
                      parameters=CityParams),
        callable=lambda **kwargs: f"Forecast for {kwargs.get('city', '???')}: 22°C, partly cloudy.",
    ),
    "book_flight": ToolDef(
        spec=ToolSpec(name="book_flight", description="Book a flight between two cities.",
                      parameters=FlightParams),
        callable=lambda **kwargs: f"Flight booked: {kwargs.get('origin', '')} → {kwargs.get('destination', '')}.",
    ),
    "check_hotel": ToolDef(
        spec=ToolSpec(name="check_hotel", description="Check hotel availability in a city.",
                      parameters=HotelParams),
        callable=lambda **kwargs: f"3 hotels available in {kwargs.get('city', '')}.",
    ),
    "convert_currency": ToolDef(
        spec=ToolSpec(name="convert_currency", description="Convert an amount between currencies.",
                      parameters=CurrencyParams),
        callable=lambda **kwargs: f"{kwargs.get('amount', '0')} {kwargs.get('from_currency', '')} = 0.00 {kwargs.get('to_currency', '')}.",
    ),
    "decline": ToolDef(
        spec=ToolSpec(name="decline", description="Call this when none of the available tools are relevant to the user's request.",
                      parameters=ReasonParams),
        callable=lambda **kwargs: kwargs.get("reason", ""),
    ),
}

relevance_detection = EvalScenario(
    name="relevance_detection",
    description="Hallucination resistance — model should refuse to call irrelevant tools.",
    workflow=Workflow(
        name="relevance_detection",
        description="Travel tools that are irrelevant to the user's question",
        tools=_relevance_detection_tools,
        required_steps=[],
        terminal_tool="decline",
        system_prompt_template=(
            "You are a helpful assistant. You have access to travel-related "
            "tools. If the user's request cannot be answered using the "
            "available tools, call the decline tool to explain why. "
            "Do NOT call a tool unless it is directly relevant."
        ),
    ),
    user_message="What is the square root of 144?",
    max_iterations=5,
    validate=lambda args: bool(args.get("reason", "").strip()),
    tags=["model_quality"],
    ideal_iterations=1,
)
