"""Request handler — the bridge between HTTP and run_inference."""

from __future__ import annotations

import json
import logging
from typing import Any

from forge.clients.base import LLMClient
from forge.context.manager import ContextManager
from forge.core.inference import fold_and_serialize, run_inference
from forge.core.workflow import ToolCall, ToolSpec, TextResponse
from forge.errors import ToolCallError
from forge.guardrails import ErrorTracker, ResponseValidator
from forge.proxy.convert import (
    openai_to_messages,
    tool_calls_to_openai,
    tool_calls_to_sse_events,
    text_response_to_openai,
    text_to_sse_events,
)
from forge.tools.respond import RESPOND_TOOL_NAME, respond_spec

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

    # Inject respond tool when tools are present.  The model calls
    # respond(message="...") instead of producing bare text, keeping it
    # in tool-calling mode where guardrails apply.  The respond call is
    # stripped from the outbound response — the client never sees it.
    if tool_specs and not any(s.name == RESPOND_TOOL_NAME for s in tool_specs):
        tool_specs.append(respond_spec())

    tool_names = _extract_tool_names(tool_specs)

    # No tools → plain chat completion, no guardrails needed.
    # Forward to backend and return the response directly.
    if not tool_specs:
        logger.info("No tools in request, passing through to backend")
        api_format = getattr(client, "api_format", "ollama")
        api_messages = fold_and_serialize(messages, api_format)
        response = await client.send(api_messages, tools=None)
        text = response.content if isinstance(response, TextResponse) else ""
        if is_stream:
            return text_to_sse_events(text, model=model_name)
        return text_response_to_openai(text, model=model_name)

    # Set up guardrails
    validator = ResponseValidator(tool_names, rescue_enabled=rescue_enabled)
    error_tracker = ErrorTracker(max_retries=max_retries)

    # Run inference (compact → fold → serialize → send → validate → retry)
    try:
        result = await run_inference(
            messages=messages,
            client=client,
            context_manager=context_manager,
            validator=validator,
            error_tracker=error_tracker,
            tool_specs=tool_specs,
            trust_text_intent=True,
        )
    except ToolCallError as exc:
        # Retries exhausted — the model kept returning text instead of tool
        # calls. Return the last text response to the client rather than an
        # error. The client's own agentic loop can decide what to do.
        raw = exc.raw_response or ""
        logger.warning("Retries exhausted, passing through text: %.120s", raw)
        if is_stream:
            return text_to_sse_events(raw, model=model_name)
        return text_response_to_openai(raw, model=model_name)

    # run_inference returns None when max_attempts exhausted
    if result is None:
        if is_stream:
            return text_to_sse_events("", model=model_name)
        return text_response_to_openai("", model=model_name)

    # Intentional text response — pass through to client
    if isinstance(result.response, TextResponse):
        logger.info("Intentional text response, passing through")
        if is_stream:
            return text_to_sse_events(result.response.content, model=model_name)
        return text_response_to_openai(result.response.content, model=model_name)

    tool_calls = result.response

    # Strip respond() calls — convert to plain text for the client.
    # If the model called respond(message="..."), the client sees a
    # normal text response (finish_reason="stop"), not a tool call.
    respond_calls = [tc for tc in tool_calls if tc.tool == RESPOND_TOOL_NAME]
    other_calls = [tc for tc in tool_calls if tc.tool != RESPOND_TOOL_NAME]

    if respond_calls and not other_calls:
        # Pure respond — convert to text
        text = respond_calls[0].args.get("message", "")
        logger.info("Stripping respond() call, returning as text")
        if is_stream:
            return text_to_sse_events(text, model=model_name)
        return text_response_to_openai(text, model=model_name)

    if other_calls:
        # Real tool calls (possibly mixed with respond) — return the
        # real tool calls only, drop respond.
        if is_stream:
            return tool_calls_to_sse_events(other_calls, model=model_name)
        return tool_calls_to_openai(other_calls, model=model_name)

    # Shouldn't happen, but handle empty tool_calls gracefully
    if is_stream:
        return text_to_sse_events("", model=model_name)
    return text_response_to_openai("", model=model_name)
