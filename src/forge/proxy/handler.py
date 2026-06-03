"""Request handler — the bridge between HTTP and run_inference."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Literal

from forge.clients.base import LLMClient, format_tool
from forge.context.manager import ContextManager
from forge.core.inference import _get_usage, fold_and_serialize, run_inference
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
from forge.proxy.convert_anthropic import (
    anthropic_to_messages,
    anthropic_to_openai_passthrough,
    anthropic_tools_to_specs,
    tool_calls_to_anthropic,
    tool_calls_to_anthropic_sse,
    text_response_to_anthropic,
    text_to_anthropic_sse,
)
from forge.tools.respond import RESPOND_TOOL_NAME, respond_spec

logger = logging.getLogger("forge.proxy")


# OpenAI-compatible sampling fields plumbed from inbound to the client.
# llama-server / Ollama accept these as top-level body / options fields.
# Anthropic ignores them. ``chat_template_kwargs`` is a nested dict of
# Jinja template variables (e.g. {"reasoning_effort": "high"}) — passed
# through as part of ``sampling``; clients that don't understand it drop it.
# Everything else in the inbound body rides through via ``passthrough`` —
# the client merges it into its outbound body verbatim.
_SAMPLING_FIELDS = (
    "temperature", "top_p", "top_k", "min_p",
    "repeat_penalty", "presence_penalty", "seed",
    "chat_template_kwargs",
)

# Body fields forge owns and reasons about — never go into passthrough.
_FORGE_OWNED = frozenset({"messages", "tools", "stream", "stream_options", "system"})


def _extract_sampling(body: dict[str, Any]) -> dict[str, Any] | None:
    """Pull recognized sampling fields out of the inbound request body.

    Returns None if the body carries no sampling fields, matching the
    "no overrides; use client instance state" path in the clients.
    """
    extracted = {f: body[f] for f in _SAMPLING_FIELDS if f in body}
    return extracted or None


def _extract_passthrough(body: dict[str, Any]) -> dict[str, Any] | None:
    """Pull non-forge-owned, non-sampling fields for the passthrough channel.

    Everything that isn't ``messages``/``tools``/``stream``/``system`` and
    isn't already in the sampling extraction flows to the outbound body
    unchanged. Lets the proxy honor user-set ``max_tokens``, ``stop``,
    ``tool_choice``, ``model``, etc. without forge needing to enumerate
    every supported field.
    """
    extras = {
        k: v for k, v in body.items()
        if k not in _FORGE_OWNED and k not in _SAMPLING_FIELDS
    }
    return extras or None


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


def _raw_openai_tools(request_tools: Any) -> list[dict[str, Any]] | None:
    """Return a detached deep copy of the inbound OpenAI tools array."""
    if not isinstance(request_tools, list) or not request_tools:
        return None
    return [deepcopy(tool) for tool in request_tools if isinstance(tool, dict)]


def _raw_openai_messages(request_messages: Any) -> list[dict[str, Any]] | None:
    """Return a detached deep copy of the inbound OpenAI messages array."""
    if not isinstance(request_messages, list) or not request_messages:
        return None
    return [deepcopy(msg) for msg in request_messages if isinstance(msg, dict)]


async def handle_chat_completions(
    body: dict[str, Any],
    client: LLMClient,
    context_manager: ContextManager,
    max_retries: int = 3,
    max_tool_errors: int = 2,
    rescue_enabled: bool = True,
    native_passthrough: bool = True,
    inject_respond_tool: bool = False,
    protocol: Literal["openai", "anthropic"] = "openai",
) -> dict[str, Any] | list[dict[str, Any]]:
    """Handle an inbound completions request.

    Converts inbound messages to forge Messages, runs inference with
    guardrails, and converts the result back to the inbound protocol's
    shape.

    Args:
        body: Parsed JSON request body.
        client: The forge LLM client for the backend.
        context_manager: For context compaction.
        max_retries: Max consecutive retries for bad responses.
        max_tool_errors: Max consecutive tool-call errors (malformed args).
        rescue_enabled: Whether to attempt rescue parsing.
        native_passthrough: When True (default, native capability), forward the
            client's verbatim OpenAI tools/messages to the backend on the clean
            first attempt (transparent passthrough). When False (prompt
            capability), suppress the raw passthrough so the request folds
            normally and the client's prompt path injects the tool prompt and
            downgrades tool history — the raw passthrough is meaningless when
            tools are serialized into the prompt text.
        inject_respond_tool: When True and the client request supplies tools,
            inject forge's synthetic respond() tool so the model stays in
            tool-calling mode (the call is stripped from the outbound
            response). Default False — the proxy forwards the client's tools
            untouched unless explicitly opted in.
        protocol: Inbound wire format. ``openai`` for
            ``/v1/chat/completions``; ``anthropic`` for ``/v1/messages``.

    Returns:
        If stream=false: a single response dict (protocol-shaped).
        If stream=true: a list of SSE event dicts (protocol-shaped).
    """
    is_stream = body.get("stream", False)
    model_name = body.get("model", "forge")

    # Inbound parse + sampling/passthrough extraction (protocol-specific)
    if protocol == "anthropic":
        messages = anthropic_to_messages(
            body.get("messages", []),
            body.get("system"),
        )
        tool_specs = anthropic_tools_to_specs(body.get("tools"))
        # Anthropic inbound's sampling fields don't overlap with the OpenAI
        # sampling set; the translated passthrough carries them in OpenAI
        # shape for the (OpenAI-shape) backend client.
        sampling = None
        passthrough = anthropic_to_openai_passthrough(body) or None
        # Path-1 cache_control opt-in. The Anthropic client uses this when
        # the runner hasn't mutated messages (clean first-attempt call);
        # OpenAI-shape clients (LlamafileClient) accept and ignore. See
        # ADR-015.
        inbound_anthropic_body = body
    else:
        request_messages = body.get("messages", [])
        request_tools = body.get("tools")
        messages = openai_to_messages(request_messages)
        tool_specs = _extract_tool_specs(request_tools)
        sampling = _extract_sampling(body)
        passthrough = _extract_passthrough(body)
        inbound_anthropic_body = None
        # Detached verbatim copies of the client's OpenAI tools/messages.
        # Forwarded to the native backend on the clean first attempt so it
        # sees the exact schema/transcript the client authored, bypassing the
        # lossy ToolSpec round-trip. tool_specs stays as forge's validation
        # sidecar. (Anthropic protocol converts shapes itself → None.)
        #
        # In prompt capability (native_passthrough=False) we suppress the raw
        # passthrough: the request folds normally and the client's prompt path
        # (LlamafileClient._send_prompt) strips the tools into the prompt and
        # downgrades tool history. A verbatim native transcript is meaningless
        # once tools are injected as prompt text.
        if native_passthrough:
            raw_tools_for_backend = _raw_openai_tools(request_tools)
            raw_messages_for_backend = _raw_openai_messages(request_messages)
        else:
            raw_tools_for_backend = None
            raw_messages_for_backend = None

    if protocol == "anthropic":
        raw_tools_for_backend = None
        raw_messages_for_backend = None

    # Optionally inject the respond tool (default off). When on, the model
    # calls respond(message="...") instead of producing bare text, keeping it
    # in tool-calling mode where guardrails apply. The respond call is
    # stripped from the outbound response — the client never sees it.
    if (
        inject_respond_tool
        and tool_specs
        and not any(s.name == RESPOND_TOOL_NAME for s in tool_specs)
    ):
        respond = respond_spec()
        tool_specs.append(respond)
        if raw_tools_for_backend is not None:
            raw_tools_for_backend.append(format_tool(respond))

    tool_names = _extract_tool_names(tool_specs)

    # No tools → plain chat completion, no guardrails needed.
    # Forward to backend and return the response directly.
    if not tool_specs:
        logger.info("No tools in request, passing through to backend")
        api_format = getattr(client, "api_format", "ollama")
        api_messages = raw_messages_for_backend or fold_and_serialize(messages, api_format)
        response = await client.send(
            api_messages, tools=None, sampling=sampling, passthrough=passthrough,
            inbound_anthropic_body=inbound_anthropic_body,
        )
        usage = _get_usage(client)
        text = response.content if isinstance(response, TextResponse) else ""
        return _emit_text(text, model_name, protocol, is_stream, usage=usage)

    # Set up guardrails
    validator = ResponseValidator(tool_names, rescue_enabled=rescue_enabled)
    error_tracker = ErrorTracker(max_retries=max_retries, max_tool_errors=max_tool_errors)

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
            passthrough=passthrough,
            inbound_anthropic_body=inbound_anthropic_body,
            raw_openai_messages=raw_messages_for_backend,
            raw_openai_tools=raw_tools_for_backend,
        )
    except ToolCallError as exc:
        # Retries exhausted — the model kept returning text instead of tool
        # calls. Return the last text response to the client rather than an
        # error. The client's own agentic loop can decide what to do.
        raw = exc.raw_response or ""
        logger.warning("Retries exhausted, passing through text: %.120s", raw)
        usage = _get_usage(client)
        return _emit_text(raw, model_name, protocol, is_stream, usage=usage)

    # run_inference returns None when max_attempts exhausted
    if result is None:
        return _emit_text("", model_name, protocol, is_stream)

    tool_calls = result.response
    usage = result.usage

    # Strip respond() calls — convert to plain text for the client.
    # If the model called respond(message="..."), the client sees a
    # normal text response (stop_reason/finish_reason indicates "stop"),
    # not a tool call.
    respond_calls = [tc for tc in tool_calls if tc.tool == RESPOND_TOOL_NAME]
    other_calls = [tc for tc in tool_calls if tc.tool != RESPOND_TOOL_NAME]

    if respond_calls and not other_calls:
        # Pure respond — convert to text
        text = respond_calls[0].args.get("message", "")
        logger.info("Stripping respond() call, returning as text")
        return _emit_text(text, model_name, protocol, is_stream, usage=usage)

    if other_calls:
        # Real tool calls (possibly mixed with respond) — return the
        # real tool calls only, drop respond.
        return _emit_tool_calls(other_calls, model_name, protocol, is_stream, usage=usage)

    # Shouldn't happen, but handle empty tool_calls gracefully
    return _emit_text("", model_name, protocol, is_stream, usage=usage)


def _emit_text(
    text: str,
    model: str,
    protocol: str,
    is_stream: bool,
    usage: Any | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Protocol-aware text response emitter."""
    if protocol == "anthropic":
        if is_stream:
            return text_to_anthropic_sse(text, model=model, usage=usage)
        return text_response_to_anthropic(text, model=model, usage=usage)
    if is_stream:
        return text_to_sse_events(text, model=model, usage=usage)
    return text_response_to_openai(text, model=model, usage=usage)


def _emit_tool_calls(
    tool_calls: list[ToolCall],
    model: str,
    protocol: str,
    is_stream: bool,
    usage: Any | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Protocol-aware tool-call response emitter."""
    if protocol == "anthropic":
        if is_stream:
            return tool_calls_to_anthropic_sse(tool_calls, model=model, usage=usage)
        return tool_calls_to_anthropic(tool_calls, model=model, usage=usage)
    if is_stream:
        return tool_calls_to_sse_events(tool_calls, model=model, usage=usage)
    return tool_calls_to_openai(tool_calls, model=model, usage=usage)
