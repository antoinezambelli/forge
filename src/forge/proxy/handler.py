"""Request handler — the bridge between HTTP and run_inference."""

from __future__ import annotations

import json
import logging
from typing import Any

from forge.clients.base import LLMClient
from forge.context.manager import ContextManager
from forge.core.inference import run_inference
from forge.core.workflow import ToolCall, ToolSpec, TextResponse
from forge.guardrails import ErrorTracker, ResponseValidator
from forge.proxy.convert import (
    openai_to_messages,
    tool_calls_to_openai,
    tool_calls_to_sse_events,
    text_response_to_openai,
    text_to_sse_events,
)

logger = logging.getLogger("forge.proxy")


def _extract_tool_specs(request_tools: list[dict[str, Any]] | None) -> list[ToolSpec]:
    """Extract ToolSpec objects from the OpenAI tools array in the request."""
    if not request_tools:
        return []
    specs = []
    for tool in request_tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {})
        specs.append(ToolSpec.from_json_schema(
            name=name,
            description=description,
            schema=parameters,
        ))
    return specs


def _extract_tool_names(tool_specs: list[ToolSpec]) -> list[str]:
    """Get tool names from specs."""
    return [s.name for s in tool_specs]


async def handle_chat_completions(
    body: dict[str, Any],
    client: LLMClient,
    context_manager: ContextManager,
    max_retries: int = 3,
    rescue_enabled: bool = True,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Handle a /v1/chat/completions request.

    Converts inbound OpenAI messages to forge Messages, runs inference
    with guardrails, and converts the result back to OpenAI format.

    Args:
        body: Parsed JSON request body.
        client: The forge LLM client for the backend.
        context_manager: For context compaction.
        max_retries: Max consecutive retries for bad responses.
        rescue_enabled: Whether to attempt rescue parsing.

    Returns:
        If stream=false: a single OpenAI response dict.
        If stream=true: a list of SSE chunk dicts.
    """
    openai_messages = body.get("messages", [])
    request_tools = body.get("tools")
    is_stream = body.get("stream", False)
    model_name = body.get("model", "forge")

    # Convert inbound
    messages = openai_to_messages(openai_messages)
    tool_specs = _extract_tool_specs(request_tools)
    tool_names = _extract_tool_names(tool_specs)

    # Set up guardrails
    validator = ResponseValidator(tool_names, rescue_enabled=rescue_enabled)
    error_tracker = ErrorTracker(max_retries=max_retries)

    # Run inference (compact → fold → serialize → send → validate → retry)
    result = await run_inference(
        messages=messages,
        client=client,
        context_manager=context_manager,
        validator=validator,
        error_tracker=error_tracker,
        tool_specs=tool_specs,
    )

    # run_inference returns None when max_attempts exhausted (shouldn't happen
    # without max_attempts set, but handle defensively)
    if result is None:
        if is_stream:
            return text_to_sse_events("Error: inference failed", model=model_name)
        return text_response_to_openai("Error: inference failed", model=model_name)

    tool_calls = result.response

    # Convert outbound
    if is_stream:
        return tool_calls_to_sse_events(tool_calls, model=model_name)
    return tool_calls_to_openai(tool_calls, model=model_name)
