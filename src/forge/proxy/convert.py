"""Convert between OpenAI chat completions format and forge Messages."""

from __future__ import annotations

import json
import uuid
from typing import Any

from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from forge.core.workflow import ToolCall

# ── Inbound: OpenAI request → forge Messages ─────────────────────


def openai_to_messages(openai_messages: list[dict[str, Any]]) -> list[Message]:
    """Convert OpenAI chat completions messages to forge Message objects.

    Handles system, user, assistant (with optional tool_calls), and tool
    role messages. Unknown roles are mapped to USER.
    """
    messages: list[Message] = []

    for msg in openai_messages:
        role_str = msg.get("role", "user")
        content = msg.get("content", "") or ""
        # Normalize list-style content blocks to a plain string.
        # OpenAI format allows content as [{"type": "text", "text": "..."}].
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            content = "\n".join(parts)

        if role_str == "system":
            messages.append(
                Message(
                    MessageRole.SYSTEM,
                    content,
                    MessageMeta(MessageType.SYSTEM_PROMPT),
                )
            )

        elif role_str == "assistant":
            if msg.get("tool_calls"):
                tc_infos = []
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        args = json.loads(args)
                    tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                    tc_infos.append(
                        ToolCallInfo(
                            name=func.get("name", ""),
                            args=args,
                            call_id=tc_id,
                        )
                    )
                messages.append(
                    Message(
                        MessageRole.ASSISTANT,
                        content,
                        MessageMeta(MessageType.TOOL_CALL),
                        tool_calls=tc_infos,
                    )
                )
            else:
                messages.append(
                    Message(
                        MessageRole.ASSISTANT,
                        content,
                        MessageMeta(MessageType.TEXT_RESPONSE),
                    )
                )

        elif role_str == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tool_name = msg.get("name", "")
            messages.append(
                Message(
                    MessageRole.TOOL,
                    content,
                    MessageMeta(MessageType.TOOL_RESULT),
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )

        else:
            # "user" or anything else
            messages.append(
                Message(
                    MessageRole.USER,
                    content,
                    MessageMeta(MessageType.USER_INPUT),
                )
            )

    return messages


# ── Outbound: forge response → OpenAI format ─────────────────────


def tool_calls_to_openai(
    tool_calls: list[ToolCall],
    model: str = "forge",
) -> dict[str, Any]:
    """Convert forge ToolCalls to an OpenAI chat completions response object."""
    tc_list = []
    for _i, tc in enumerate(tool_calls):
        tc_list.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tc.tool,
                    "arguments": json.dumps(tc.args),
                },
            }
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": tool_calls[0].reasoning or None,
                    "tool_calls": tc_list,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def text_response_to_openai(
    text: str,
    model: str = "forge",
) -> dict[str, Any]:
    """Convert a text response to an OpenAI chat completions response object."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ── SSE streaming helpers ────────────────────────────────────────


def tool_calls_to_sse_events(
    tool_calls: list[ToolCall],
    model: str = "forge",
) -> list[dict[str, Any]]:
    """Convert forge ToolCalls to a sequence of SSE chunk objects.

    Returns the complete list of chunk dicts ready to be formatted as
    SSE data lines. The caller handles the actual SSE wire format.
    """
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    events: list[dict[str, Any]] = []

    # If there's reasoning, send it as a content delta first
    if tool_calls[0].reasoning:
        events.append(
            {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": tool_calls[0].reasoning},
                        "finish_reason": None,
                    }
                ],
            }
        )

    # Tool call deltas
    for i, tc in enumerate(tool_calls):
        tc_id = f"call_{uuid.uuid4().hex[:8]}"
        # First chunk for this tool: name + start of args
        events.append(
            {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": i,
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.tool,
                                        "arguments": json.dumps(tc.args),
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )

    # Final chunk with finish_reason
    events.append(
        {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )

    return events


def text_to_sse_events(
    text: str,
    model: str = "forge",
    chunk_size: int = 0,
) -> list[dict[str, Any]]:
    """Convert a text response to SSE chunk objects.

    If chunk_size > 0, splits the text into chunks of that size for
    more realistic streaming. Otherwise sends the full text in one chunk.
    """
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    events: list[dict[str, Any]] = []

    if chunk_size > 0 and len(text) > chunk_size:
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    else:
        chunks = [text]

    for i, chunk in enumerate(chunks):
        delta: dict[str, Any] = {"content": chunk}
        if i == 0:
            delta["role"] = "assistant"
        events.append(
            {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }
        )

    # Final chunk
    events.append(
        {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
    )

    return events


# ── Anthropic ↔ OpenAI conversion ──────────────────────────────────

# Mapping Anthropic stop_reason → OpenAI finish_reason
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
}

# Reverse: OpenAI finish_reason → Anthropic stop_reason
# Note: both end_turn and stop_sequence map to "stop", so we prefer end_turn
_STOP_REASON_MAP_REV = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "stop_sequence",
}


def anthropic_to_openai_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an Anthropic API request body to OpenAI-format messages list.

    Handles:
    - Top-level ``system`` string → ``role: "system"`` message
    - ``messages`` array with Anthropic format (``role``, ``content`` blocks)
    - ``tool_use`` content blocks → ``assistant`` with ``tool_calls``
    - ``tool_result`` content blocks → ``tool`` role messages
    - Plain text content → string content
    """
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": system})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content_blocks = msg.get("content", "")

        if role == "assistant":
            if isinstance(content_blocks, list):
                tool_calls = []
                text_parts = []
                for block in content_blocks:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_calls.append(
                                {
                                    "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name", ""),
                                        "arguments": json.dumps(block.get("input", {})),
                                    },
                                }
                            )
                        elif block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "thinking":
                            text_parts.append(f"<think>{block.get('text', '')}</think>")
                ai_msg: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    ai_msg["content"] = "\n".join(text_parts)
                if tool_calls:
                    ai_msg["tool_calls"] = tool_calls
                messages.append(ai_msg)
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content_blocks or "",
                    }
                )

        elif role == "user":
            if isinstance(content_blocks, list):
                # Check if there are tool_result blocks
                tool_results = [b for b in content_blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
                text_blocks = [
                    b for b in content_blocks if isinstance(b, dict) and b.get("type") in ("text", "thinking")
                ]

                for tr in tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id", ""),
                            "content": tr.get("content", ""),
                            "name": tr.get("name", ""),
                        }
                    )

                if text_blocks:
                    text_content = ""
                    for tb in text_blocks:
                        if tb.get("type") == "thinking":
                            text_content += f"<think>{tb.get('text', '')}</think>"
                        else:
                            text_content += tb.get("text", "")
                    messages.append(
                        {
                            "role": "user",
                            "content": text_content,
                        }
                    )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": content_blocks or "",
                    }
                )

    return messages


def openai_to_anthropic_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert OpenAI-format messages list to Anthropic API body fields.

    Extracts system message and converts messages to Anthropic format.
    Returns a dict with ``system`` and ``messages`` keys suitable for
    the Anthropic API body (not the full body — caller adds model,
    max_tokens, tools, etc.).
    """
    system: str | None = None
    anthropic_msgs: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""

        if role == "system":
            system = content
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        args = json.loads(args)
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                            "name": func.get("name", ""),
                            "input": args,
                        }
                    )
            anthropic_msgs.append({"role": "assistant", "content": blocks if blocks else ""})

        elif role == "tool":
            # Tool results go inside a user message
            tc_id = msg.get("tool_call_id", "")
            is_error = msg.get("is_error", False)
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tc_id,
                "content": content,
            }
            if is_error:
                block["is_error"] = True
            anthropic_msgs.append({"role": "user", "content": [block]})

        else:
            # "user" or anything else
            if isinstance(content, list):
                # Already in content-block format; normalize
                blocks = []
                for b in content:
                    if isinstance(b, dict):
                        if b.get("type") == "text":
                            blocks.append({"type": "text", "text": b.get("text", "")})
                        elif b.get("type") == "thinking":
                            blocks.append({"type": "thinking", "text": b.get("text", "")})
                        else:
                            blocks.append(b)
                    else:
                        blocks.append({"type": "text", "text": str(b)})
                anthropic_msgs.append({"role": "user", "content": blocks})
            else:
                anthropic_msgs.append({"role": "user", "content": content})

    result: dict[str, Any] = {"messages": anthropic_msgs}
    if system:
        result["system"] = system
    return result


def openai_to_anthropic_response(openai_resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert an OpenAI chat.completion response to Anthropic format.

    Handles both text and tool-call responses.
    """
    choice = openai_resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = _STOP_REASON_MAP_REV.get(finish_reason, "end_turn")

    # Build content blocks
    content_blocks: list[dict[str, Any]] = []
    reasoning = message.get("content", "") or ""

    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                    "name": func.get("name", ""),
                    "input": args,
                }
            )
        # Reasoning/thinking goes as text block before tool_use
        if reasoning:
            content_blocks.insert(0, {"type": "text", "text": reasoning})
    else:
        content_blocks.append({"type": "text", "text": reasoning})

    usage = openai_resp.get("usage", {})
    anthropic_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }

    return {
        "id": openai_resp.get("id", f"msg_{uuid.uuid4().hex[:12]}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage,
    }


def openai_to_anthropic_sse(openai_events: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """Convert OpenAI SSE chunks to Anthropic SSE events.

    Handles text deltas, tool call deltas, and finish_reason.
    """
    events: list[dict[str, Any]] = []
    # Track tool_use blocks by index for streaming
    tool_indices: dict[int, dict[str, Any]] = {}
    cmpl_id = (
        openai_events[0].get("id", f"msg_{uuid.uuid4().hex[:12]}") if openai_events else f"msg_{uuid.uuid4().hex[:12]}"
    )
    accumulated_text = ""
    next_index = 0  # Track next available content block index
    text_started = False  # Whether content_block_start for text has been emitted

    for event in openai_events:
        choices = event.get("choices", [])
        for choice in choices:
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # Text delta
            if "content" in delta and delta["content"] is not None:
                accumulated_text += delta["content"]
                # Only emit content_block_start on the first text delta
                if not text_started:
                    events.append(
                        {
                            "type": "content_block_start",
                            "index": next_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                    )
                    text_started = True
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": next_index,
                        "delta": {"type": "text_delta", "text": delta["content"]},
                    }
                )

            # Tool call deltas
            tool_calls = delta.get("tool_calls", [])
            for tc in tool_calls:
                idx = tc.get("index", 0)
                if "id" in tc and idx not in tool_indices:
                    tool_indices[idx] = {"name": "", "args": "", "id": tc["id"]}
                if idx in tool_indices:
                    if "function" in tc and "name" in tc.get("function", {}):
                        tool_indices[idx]["name"] = tc["function"]["name"]
                    if "function" in tc and "arguments" in tc.get("function", {}):
                        tool_indices[idx]["args"] += tc["function"]["arguments"]

            # Final chunk — emit content_block_stop for text and all tool_use blocks
            if finish_reason:
                # Close text block if it was started
                if text_started:
                    events.append(
                        {
                            "type": "content_block_stop",
                            "index": next_index,
                        }
                    )

                # Emit tool_use blocks
                for _tidx, tdata in sorted(tool_indices.items()):
                    events.append(
                        {
                            "type": "content_block_start",
                            "index": next_index,
                            "content_block": {
                                "type": "tool_use",
                                "name": tdata["name"],
                                "id": tdata["id"],
                                "input": json.loads(tdata["args"])
                                if isinstance(tdata["args"], str) and tdata["args"]
                                else tdata["args"],
                            },
                        }
                    )
                    events.append(
                        {
                            "type": "content_block_delta",
                            "index": next_index,
                            "delta": {"type": "input_json_delta", "partial_json": tdata["args"]},
                        }
                    )
                    events.append(
                        {
                            "type": "content_block_stop",
                            "index": next_index,
                        }
                    )
                    next_index += 1

                events.append(
                    {
                        "type": "message_stop",
                        "stop_reason": _STOP_REASON_MAP_REV.get(finish_reason, "end_turn"),
                    }
                )

    # Wrap in Anthropic SSE message format
    wrapped: list[dict[str, Any]] = []
    for ev in events:
        wrapped.append({**ev, "id": cmpl_id, "model": model})

    # Add message_start
    wrapped.insert(
        0,
        {
            "id": cmpl_id,
            "type": "message_start",
            "model": model,
            "message": {
                "id": cmpl_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    # Add message_stop if not already emitted (e.g. empty events)
    has_stop = any(ev.get("type") == "message_stop" for ev in wrapped)
    if not has_stop:
        wrapped.append({"id": cmpl_id, "type": "message_stop"})

    return wrapped


def anthropic_to_openai_response(anthropic_resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert an Anthropic API response to OpenAI chat.completion format.

    Handles both text and tool-call responses.
    """
    stop_reason = anthropic_resp.get("stop_reason", "end_turn")
    finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop")

    content_blocks = anthropic_resp.get("content", [])
    tool_calls = []
    text_parts = []

    for block in content_blocks:
        if isinstance(block, dict):
            if block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
            elif block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "thinking":
                text_parts.append(f"<think>{block.get('text', '')}</think>")

    usage = anthropic_resp.get("usage", {})
    openai_usage = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    }

    message: dict[str, Any] = {"role": "assistant"}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if text_parts:
        message["content"] = "\n".join(text_parts)
    else:
        message["content"] = ""

    return {
        "id": anthropic_resp.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": openai_usage,
    }


def anthropic_to_openai_sse(anthropic_events: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """Convert Anthropic SSE events to OpenAI SSE chunks.

    Handles content_block_delta (text and input_json), message_stop.
    """
    events: list[dict[str, Any]] = []
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    accumulated_text = ""
    tool_blocks: dict[int, dict[str, Any]] = {}
    current_tool_idx = 0

    for event in anthropic_events:
        ev_type = event.get("type", "")

        if ev_type == "content_block_start":
            idx = event.get("index", 0)
            block = event.get("content_block", {})
            if block.get("type") == "text":
                tool_blocks[idx] = {"type": "text", "chunks": []}
            elif block.get("type") == "tool_use":
                tool_blocks[idx] = {
                    "type": "tool_use",
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "name": block.get("name", ""),
                    "args": "",
                }

        elif ev_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type", "")
            idx = event.get("index", 0)

            if delta_type == "text_delta":
                accumulated_text += delta.get("text", "")
                events.append(
                    {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta.get("text", "")},
                                "finish_reason": None,
                            }
                        ],
                    }
                )

            elif delta_type == "input_json_delta" and idx in tool_blocks:
                partial = delta.get("partial_json", "")
                tool_blocks[idx]["args"] += partial
                events.append(
                    {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": current_tool_idx,
                                            "id": tool_blocks[idx].get("id", ""),
                                            "type": "function",
                                            "function": {"arguments": partial},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                current_tool_idx += 1

        elif ev_type == "message_stop":
            stop_reason = event.get("stop_reason", "end_turn")
            finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop")
            events.append(
                {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            )

    # If we have accumulated text but no SSE events were generated,
    # send the full text as a single chunk
    if not events and accumulated_text:
        events.append(
            {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": accumulated_text},
                        "finish_reason": None,
                    }
                ],
            }
        )
        events.append(
            {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    return events
