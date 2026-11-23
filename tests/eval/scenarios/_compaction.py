"""Compaction-related input schemas and relevance detection scenario."""

from __future__ import annotations


from forge.core.workflow import ToolDefinition, Workflow

from ._base import EvalScenario


CITY_SCHEMA = {
    "properties": {
        "city": {"description": "The city name", "title": "City", "type": "string"}
    },
    "required": ["city"],
    "title": "CityParams",
    "type": "object",
}


FLIGHT_SCHEMA = {
    "properties": {
        "origin": {
            "description": "Departure city",
            "title": "Origin",
            "type": "string",
        },
        "destination": {
            "description": "Arrival city",
            "title": "Destination",
            "type": "string",
        },
    },
    "required": ["origin", "destination"],
    "title": "FlightParams",
    "type": "object",
}


HOTEL_SCHEMA = {
    "properties": {
        "city": {"description": "The city to check", "title": "City", "type": "string"},
        "checkin": {
            "description": "Check-in date",
            "title": "Checkin",
            "type": "string",
        },
    },
    "required": ["city", "checkin"],
    "title": "HotelParams",
    "type": "object",
}


CURRENCY_SCHEMA = {
    "properties": {
        "amount": {
            "description": "The amount to convert",
            "title": "Amount",
            "type": "string",
        },
        "from_currency": {
            "description": "Source currency code",
            "title": "From Currency",
            "type": "string",
        },
        "to_currency": {
            "description": "Target currency code",
            "title": "To Currency",
            "type": "string",
        },
    },
    "required": ["amount", "from_currency", "to_currency"],
    "title": "CurrencyParams",
    "type": "object",
}


REASON_SCHEMA = {
    "properties": {
        "reason": {
            "description": "Brief explanation of why no tool is appropriate",
            "title": "Reason",
            "type": "string",
        }
    },
    "required": ["reason"],
    "title": "ReasonParams",
    "type": "object",
}


# ── Scenario: relevance_detection ─────────────────────────────

_relevance_detection_tools: dict[str, ToolDefinition] = {
    "get_forecast": ToolDefinition(name="get_forecast", description="Get weather forecast for a city.",
                      input_schema=CITY_SCHEMA),
    "book_flight": ToolDefinition(name="book_flight", description="Book a flight between two cities.",
                      input_schema=FLIGHT_SCHEMA),
    "check_hotel": ToolDefinition(name="check_hotel", description="Check hotel availability in a city.",
                      input_schema=HOTEL_SCHEMA),
    "convert_currency": ToolDefinition(name="convert_currency", description="Convert an amount between currencies.",
                      input_schema=CURRENCY_SCHEMA),
    "decline": ToolDefinition(name="decline", description="Call this when none of the available tools are relevant to the user's request.",
                      input_schema=REASON_SCHEMA),
}
_relevance_detection_tools_handlers = {
    "get_forecast": lambda **kwargs: f"Forecast for {kwargs.get('city', '???')}: 22°C, partly cloudy.",
    "book_flight": lambda **kwargs: f"Flight booked: {kwargs.get('origin', '')} → {kwargs.get('destination', '')}.",
    "check_hotel": lambda **kwargs: f"3 hotels available in {kwargs.get('city', '')}.",
    "convert_currency": lambda **kwargs: f"{kwargs.get('amount', '0')} {kwargs.get('from_currency', '')} = 0.00 {kwargs.get('to_currency', '')}.",
    "decline": lambda **kwargs: kwargs.get("reason", ""),
}

relevance_detection = EvalScenario(
    name="relevance_detection",
    description="Hallucination resistance — model should refuse to call irrelevant tools.",
    workflow=Workflow(
        name="relevance_detection",
        description="Travel tools that are irrelevant to the user's question",
        tools=_relevance_detection_tools,
        handlers=_relevance_detection_tools_handlers,
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
