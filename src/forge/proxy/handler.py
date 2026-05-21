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
    anthropic_to_openai_messages,
    openai_to_anthropic_response,
    openai_to_anthropic_sse,
    anthropic_to_openai_response,
    anthropic_to_openai_sse,
)
from forge.tools.respond import RESPOND_TOOL_NAME, respond_spec

logger = logging.getLogger("forge.proxy")


# OpenAI-compatible top-level body fields plumbed from inbound body to
# client. llama-server / Ollama support the sampling fields below as
# top-level body / options fields. Anthropic ignores them.
# ``chat_template_kwargs`` is a nested dict of Jinja template variables
# (e.g. {"reasoning_effort": "high"}) — passed through to the LlamafileClient
# as part of the ``sampling`` kwarg; OllamaClient drops it (no analog field).


def _format_response(
    response: list[ToolCall] | TextResponse | str | None,
    is_stream: bool,
    model: str,
    anthropic_backend: bool,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Format an inference result into the appropriate wire format."""
    if response is None or isinstance(response, str):
        text = response or ""
        return _format_text(text, is_stream, model, anthropic_backend)
    elif isinstance(response, TextResponse):
        return _format_text(response.content, is_stream, model, anthropic_backend)
    else:
        # list[ToolCall]
        return _format_tool_calls(response, is_stream, model, anthropic_backend)


def _format_text(
    text: str,
    is_stream: bool,
    model: str,
    anthropic_backend: bool,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Format a text response."""
    if anthropic_backend:
        if is_stream:
            return openai_to_anthropic_sse(text_to_sse_events(text, model), model)
        return openai_to_anthropic_response(text_response_to_openai(text, model), model)
    else:
        if is_stream:
            return text_to_sse_events(text, model)
        return text_response_to_openai(text, model)


def _format_tool_calls(
    tool_calls: list[ToolCall],
    is_stream: bool,
    model: str,
    anthropic_backend: bool,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Format a tool call response."""
    if anthropic_backend:
        if is_stream:
            return openai_to_anthropic_sse(tool_calls_to_sse_events(tool_calls, model), model)
        return openai_to_anthropic_response(tool_calls_to_openai(tool_calls, model), model)
    else:
        if is_stream:
            return tool_calls_to_sse_events(tool_calls, model)
        return tool_calls_to_openai(tool_calls, model)
_SAMPLING_FIELDS = (
    "temperature", "top_p", "top_k", "min_p",
    "repeat_penalty", "presence_penalty", "seed",
    "chat_template_kwargs", "model",
)


def _extract_sampling(body: dict[str, Any]) -> dict[str, Any] | None:
    """Pull recognized sampling fields out of the inbound request body.

    Returns None if the body carries no sampling fields, matching the
    "no overrides; use client instance state" path in the clients.
    """
    extracted = {f: body[f] for f in _SAMPLING_FIELDS if f in body}
    return extracted or None


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
    anthropic_backend: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Handle a /v1/chat/completions request.

    Converts inbound OpenAI messages to forge Messages, runs inference
    with guardrails, and converts the result back to the appropriate
    format (OpenAI or Anthropic).

    Args:
        body: Parsed JSON request body.
        client: The forge LLM client for the backend.
        context_manager: For context compaction.
        max_retries: Max consecutive retries for bad responses.
        rescue_enabled: Whether to attempt rescue parsing.
        anthropic_backend: If True, return Anthropic-format responses.

    Returns:
        If stream=false: a response dict (OpenAI or Anthropic format).
        If stream=true: a list of SSE chunk dicts.
    """
    openai_messages = body.get("messages", [])
    request_tools = body.get("tools")
    is_stream = body.get("stream", False)
    model_name = body.get("model", "forge")
    sampling = _extract_sampling(body)

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
        response = await client.send(api_messages, tools=None, sampling=sampling)
        text = response.content if isinstance(response, TextResponse) else ""
        return _format_response(text, is_stream, model_name, anthropic_backend)

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
            sampling=sampling,
        )
    except ToolCallError as exc:
        # Retries exhausted — the model kept returning text instead of tool
        # calls. Return the last text response to the client rather than an
        # error. The client's own agentic loop can decide what to do.
        raw = exc.raw_response or ""
        logger.warning("Retries exhausted, passing through text: %.120s", raw)
        return _format_response(raw, is_stream, model_name, anthropic_backend)

    # run_inference returns None when max_attempts exhausted
    if result is None:
        return _format_response("", is_stream, model_name, anthropic_backend)

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
        return _format_response(text, is_stream, model_name, anthropic_backend)

    if other_calls:
        # Real tool calls (possibly mixed with respond) — return the
        # real tool calls only, drop respond.
        return _format_response(other_calls, is_stream, model_name, anthropic_backend)

    # Shouldn't happen, but handle empty tool_calls gracefully
    return _format_response("", is_stream, model_name, anthropic_backend)


async def handle_messages(
    body: dict[str, Any],
    client: LLMClient,
    context_manager: ContextManager,
    max_retries: int = 3,
    rescue_enabled: bool = True,
    anthropic_backend: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Handle a /v1/messages request (Anthropic incoming format).

    Converts inbound Anthropic messages to OpenAI format, then delegates
    to handle_chat_completions for the inference pipeline.

    Args:
        body: Parsed JSON request body in Anthropic format.
        client: The forge LLM client for the backend.
        context_manager: For context compaction.
        max_retries: Max consecutive retries for bad responses.
        rescue_enabled: Whether to attempt rescue parsing.
        anthropic_backend: If True, return Anthropic-format responses.

    Returns:
        If stream=false: a response dict.
        If stream=true: a list of SSE chunk dicts.
    """
    # Convert Anthropic body to OpenAI messages
    openai_messages = anthropic_to_openai_messages(body)

    # Build an OpenAI-format body for handle_chat_completions
    openai_body: dict[str, Any] = {
        "model": body.get("model", "forge"),
        "messages": openai_messages,
        "stream": body.get("stream", False),
    }

    # Convert Anthropic tools to OpenAI tools
    anthropic_tools = body.get("tools")
    if anthropic_tools:
        openai_body["tools"] = []
        for tool in anthropic_tools:
            openai_body["tools"].append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })

    # Map Anthropic sampling params to OpenAI
    if "temperature" in body:
        openai_body["temperature"] = body["temperature"]
    if "top_p" in body:
        openai_body["top_p"] = body["top_p"]
    if "top_k" in body:
        openai_body["top_k"] = body["top_k"]

    return await handle_chat_completions(
        body=openai_body,
        client=client,
        context_manager=context_manager,
        max_retries=max_retries,
        rescue_enabled=rescue_enabled,
        anthropic_backend=anthropic_backend,
    )
