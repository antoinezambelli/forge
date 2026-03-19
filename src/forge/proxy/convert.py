"""Conversion between OpenAI wire format and forge types.

Parses OpenAI chat completion responses (both streaming and batch)
into forge ToolCall/TextResponse objects, and synthesizes OpenAI SSE
chunks from forge ToolCall objects (for rescued responses).
"""

from __future__ import annotations

import json
from typing import Any

from forge.core.workflow import LLMResponse, TextResponse, ToolCall


def parse_batch_response(data: dict[str, Any]) -> LLMResponse:
    """Parse a non-streaming OpenAI chat completion response.

    Args:
        data: Parsed JSON response body.

    Returns:
        list[ToolCall] if the model produced tool calls,
        TextResponse if the model produced text.
    """
    choice = data["choices"][0]
    message = choice["message"]

    if message.get("tool_calls"):
        return [
            ToolCall(
                tool=tc["function"]["name"],
                args=json.loads(tc["function"]["arguments"]),
            )
            for tc in message["tool_calls"]
        ]

    return TextResponse(content=message.get("content") or "")


def parse_streamed_response(chunks: list[str]) -> tuple[LLMResponse, dict[str, Any] | None]:
    """Parse buffered SSE chunks into a forge response.

    Assembles tool calls or text content from streaming deltas.
    Also extracts usage data from the final chunk if present.

    Args:
        chunks: SSE data lines (with "data: " prefix).

    Returns:
        Tuple of (LLMResponse, usage_dict_or_None).
    """
    text_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}  # index -> {id, name, arguments}
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None

    for chunk_line in chunks:
        if chunk_line == "data: [DONE]":
            continue

        payload = chunk_line.removeprefix("data: ").strip()
        if not payload:
            continue

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if "usage" in data and data["usage"]:
            usage = data["usage"]

        choices = data.get("choices", [])
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta", {})

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        # Text content
        if "content" in delta and delta["content"]:
            text_parts.append(delta["content"])

        # Tool call deltas
        if "tool_calls" in delta:
            for tc_delta in delta["tool_calls"]:
                idx = tc_delta.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": "", "name": "", "arguments": ""}

                tc = tool_calls[idx]
                if "id" in tc_delta:
                    tc["id"] = tc_delta["id"]
                fn = tc_delta.get("function", {})
                if "name" in fn:
                    tc["name"] = fn["name"]
                if "arguments" in fn:
                    tc["arguments"] += fn["arguments"]

    # Build response
    if tool_calls:
        result: list[ToolCall] = []
        for idx in sorted(tool_calls.keys()):
            tc = tool_calls[idx]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            result.append(ToolCall(tool=tc["name"], args=args))
        return result, usage

    return TextResponse(content="".join(text_parts)), usage


def synthesize_sse_tool_calls(
    tool_calls: list[ToolCall],
    model: str = "forge-proxy",
) -> list[bytes]:
    """Synthesize OpenAI SSE chunks from forge ToolCall objects.

    Used when the proxy rescues tool calls from a text response --
    the client expects SSE tool_calls format, not text.

    Args:
        tool_calls: Validated tool calls to serialize.
        model: Model name for the response metadata.

    Returns:
        List of SSE chunk bytes ready to send to the client.
    """
    chunks: list[bytes] = []
    resp_id = "chatcmpl-forge-rescued"

    # Role chunk
    chunks.append(_sse_chunk(resp_id, model, {"role": "assistant"}))

    # Tool call chunks
    tc_list = []
    for i, tc in enumerate(tool_calls):
        tc_list.append({
            "id": f"call_rescued_{i}",
            "type": "function",
            "function": {
                "name": tc.tool,
                "arguments": json.dumps(tc.args),
            },
        })

    # Emit tool calls in one delta
    chunks.append(_sse_chunk(resp_id, model, {"tool_calls": tc_list}))

    # Finish chunk
    finish = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    chunks.append(f"data: {json.dumps(finish)}\n\n".encode())

    # Done
    chunks.append(b"data: [DONE]\n\n")

    return chunks


def _sse_chunk(resp_id: str, model: str, delta: dict[str, Any]) -> bytes:
    """Build a single SSE chunk."""
    data = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta}],
    }
    return f"data: {json.dumps(data)}\n\n".encode()
