"""Convert between OpenAI chat completions format and forge Messages."""

from __future__ import annotations

import json
import uuid
from typing import Any

from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from forge.core.workflow import ToolCall, TextResponse


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
            messages.append(Message(
                MessageRole.SYSTEM,
                content,
                MessageMeta(MessageType.SYSTEM_PROMPT),
            ))

        elif role_str == "assistant":
            if "tool_calls" in msg and msg["tool_calls"]:
                tc_infos = []
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        args = json.loads(args)
                    tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                    tc_infos.append(ToolCallInfo(
                        name=func.get("name", ""),
                        args=args,
                        call_id=tc_id,
                    ))
                messages.append(Message(
                    MessageRole.ASSISTANT,
                    content,
                    MessageMeta(MessageType.TOOL_CALL),
                    tool_calls=tc_infos,
                ))
            else:
                messages.append(Message(
                    MessageRole.ASSISTANT,
                    content,
                    MessageMeta(MessageType.TEXT_RESPONSE),
                ))

        elif role_str == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tool_name = msg.get("name", "")
            messages.append(Message(
                MessageRole.TOOL,
                content,
                MessageMeta(MessageType.TOOL_RESULT),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            ))

        else:
            # "user" or anything else
            messages.append(Message(
                MessageRole.USER,
                content,
                MessageMeta(MessageType.USER_INPUT),
            ))

    return messages


# ── Outbound: forge response → OpenAI format ─────────────────────

def tool_calls_to_openai(
    tool_calls: list[ToolCall],
    model: str = "forge",
) -> dict[str, Any]:
    """Convert forge ToolCalls to an OpenAI chat completions response object."""
    tc_list = []
    for i, tc in enumerate(tool_calls):
        tc_list.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": tc.tool,
                "arguments": json.dumps(tc.args),
            },
        })

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": tool_calls[0].reasoning or None,
                "tool_calls": tc_list,
            },
            "finish_reason": "tool_calls",
        }],
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
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "finish_reason": "stop",
        }],
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
        events.append({
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": tool_calls[0].reasoning},
                "finish_reason": None,
            }],
        })

    # Tool call deltas
    for i, tc in enumerate(tool_calls):
        tc_id = f"call_{uuid.uuid4().hex[:8]}"
        # First chunk for this tool: name + start of args
        events.append({
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": i,
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool,
                            "arguments": json.dumps(tc.args),
                        },
                    }],
                },
                "finish_reason": None,
            }],
        })

    # Final chunk with finish_reason
    events.append({
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls",
        }],
    })

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
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    else:
        chunks = [text]

    for i, chunk in enumerate(chunks):
        delta: dict[str, Any] = {"content": chunk}
        if i == 0:
            delta["role"] = "assistant"
        events.append({
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": None,
            }],
        })

    # Final chunk
    events.append({
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    })

    return events
