"""Conversion between OpenAI wire format and forge types.

Inbound:  OpenAI message dicts -> forge Message objects
Outbound: forge ToolCall objects -> OpenAI SSE chunks (for rescued responses)
Parsing:  OpenAI response (batch or streamed) -> forge ToolCall/TextResponse
"""

from __future__ import annotations

import json
from typing import Any

import re

from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from forge.core.workflow import LLMResponse, TextResponse, ToolCall

# Patterns for detecting think tags in content (Mistral and Qwen/DeepSeek styles).
_THINK_PATTERN = re.compile(
    r"^\s*(\[THINK\].*?\[/THINK\]|<think>.*?</think>)\s*",
    re.DOTALL | re.IGNORECASE,
)


# -- Inbound: OpenAI dicts -> forge Messages -------------------


def openai_messages_to_forge(messages: list[dict[str, Any]]) -> list[Message]:
    """Convert an OpenAI messages array to forge Message objects.

    Infers MessageType from role and structure:
      - system                         -> SYSTEM_PROMPT
      - user (first)                   -> USER_INPUT
      - user (subsequent)              -> USER_INPUT
      - assistant with tool_calls      -> TOOL_CALL
      - assistant with text content    -> TEXT_RESPONSE
      - tool                           -> TOOL_RESULT

    Args:
        messages: OpenAI-format message dicts.

    Returns:
        List of forge Message objects with inferred metadata.
    """
    result: list[Message] = []
    seen_user = False

    for msg in messages:
        role_str = msg.get("role", "user")
        content = msg.get("content") or ""

        if role_str == "system":
            result.append(Message(
                MessageRole.SYSTEM,
                content,
                MessageMeta(MessageType.SYSTEM_PROMPT),
            ))

        elif role_str == "user":
            msg_type = MessageType.USER_INPUT
            seen_user = True
            result.append(Message(
                MessageRole.USER,
                content,
                MessageMeta(msg_type),
            ))

        elif role_str == "assistant":
            if msg.get("tool_calls"):
                # Assistant message with tool calls
                tc_infos = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except json.JSONDecodeError:
                        args = {}
                    tc_infos.append(ToolCallInfo(
                        name=fn.get("name", ""),
                        args=args,
                        call_id=tc.get("id", ""),
                    ))
                result.append(Message(
                    MessageRole.ASSISTANT,
                    content,
                    MessageMeta(MessageType.TOOL_CALL),
                    tool_calls=tc_infos,
                ))
            else:
                # Check for reasoning content.
                # 1. Explicit reasoning_content field (OpenAI/llama-server streaming format)
                # 2. Think tags in content (batch format: [THINK]...[/THINK] or <think>...</think>)
                reasoning = msg.get("reasoning_content") or ""
                if reasoning:
                    result.append(Message(
                        MessageRole.ASSISTANT,
                        reasoning,
                        MessageMeta(MessageType.REASONING),
                    ))
                    if content:
                        result.append(Message(
                            MessageRole.ASSISTANT,
                            content,
                            MessageMeta(MessageType.TEXT_RESPONSE),
                        ))
                elif content and _THINK_PATTERN.match(content):
                    # Split think tags from content
                    match = _THINK_PATTERN.match(content)
                    think_text = match.group(1)
                    remaining = content[match.end():].strip()
                    result.append(Message(
                        MessageRole.ASSISTANT,
                        think_text,
                        MessageMeta(MessageType.REASONING),
                    ))
                    if remaining:
                        result.append(Message(
                            MessageRole.ASSISTANT,
                            remaining,
                            MessageMeta(MessageType.TEXT_RESPONSE),
                        ))
                else:
                    # Plain text assistant message
                    result.append(Message(
                        MessageRole.ASSISTANT,
                        content,
                        MessageMeta(MessageType.TEXT_RESPONSE),
                    ))

        elif role_str == "tool":
            result.append(Message(
                MessageRole.TOOL,
                content,
                MessageMeta(MessageType.TOOL_RESULT),
                tool_name=msg.get("name"),
                tool_call_id=msg.get("tool_call_id"),
            ))

    return result


def forge_messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert forge Messages back to OpenAI wire format.

    Uses Message.to_api_dict(format="openai").
    """
    return [m.to_api_dict(format="openai") for m in messages]


# -- Response parsing: backend response -> forge types ---------


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


def parse_streamed_response(chunks: list[str]) -> tuple[LLMResponse, str | None, dict[str, Any] | None]:
    """Parse buffered SSE chunks into a forge response.

    Assembles tool calls or text content from streaming deltas.
    Also extracts reasoning_content and usage data.

    Args:
        chunks: SSE data lines (with "data: " prefix).

    Returns:
        Tuple of (LLMResponse, reasoning_or_None, usage_dict_or_None).
    """
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}  # index -> {id, name, arguments}
    usage: dict[str, Any] | None = None

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

        # Reasoning content (separate field from llama-server streaming)
        if "reasoning_content" in delta and delta["reasoning_content"]:
            reasoning_parts.append(delta["reasoning_content"])

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
    reasoning = "".join(reasoning_parts) if reasoning_parts else None

    if tool_calls:
        result: list[ToolCall] = []
        for idx in sorted(tool_calls.keys()):
            tc = tool_calls[idx]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            result.append(ToolCall(tool=tc["name"], args=args))
        return result, reasoning, usage

    return TextResponse(content="".join(text_parts)), reasoning, usage


# -- Outbound: forge ToolCall -> OpenAI SSE chunks -------------


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


def synthesize_batch_tool_calls(
    tool_calls: list[ToolCall],
    model: str = "forge-proxy",
) -> bytes:
    """Synthesize an OpenAI batch response from forge ToolCall objects."""
    return json.dumps({
        "id": "chatcmpl-forge-rescued",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_rescued_{i}",
                        "type": "function",
                        "function": {
                            "name": tc.tool,
                            "arguments": json.dumps(tc.args),
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ],
            },
            "finish_reason": "tool_calls",
        }],
    }).encode()


def _sse_chunk(resp_id: str, model: str, delta: dict[str, Any]) -> bytes:
    """Build a single SSE chunk."""
    data = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta}],
    }
    return f"data: {json.dumps(data)}\n\n".encode()
