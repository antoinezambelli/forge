"""Convert between OpenAI chat completions format and forge Messages."""

from __future__ import annotations

import json
import uuid
from typing import Any

from forge.clients.base import flatten_content_to_text
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from forge.core.reasoning import DEFAULT_REASONING_REPLAY, ReasoningReplay, validate_reasoning_replay
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
        # Normalize list-style content blocks to a plain string.
        # OpenAI format allows content as [{"type": "text", "text": "..."}].
        content = flatten_content_to_text(msg.get("content", "") or "")

        if role_str == "system":
            messages.append(Message(
                MessageRole.SYSTEM,
                content,
                MessageMeta(MessageType.SYSTEM_PROMPT),
            ))

        elif role_str == "assistant":
            reasoning = (
                msg.get("reasoning_content")
                or msg.get("reasoning")
                or msg.get("reasoning_text")
            )
            if reasoning:
                messages.append(Message(
                    MessageRole.ASSISTANT,
                    str(reasoning),
                    MessageMeta(MessageType.REASONING),
                ))
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
            elif content:
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
    usage: Any | None = None,
    reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
) -> dict[str, Any]:
    """Convert forge ToolCalls to an OpenAI chat completions response object."""
    reasoning_replay = validate_reasoning_replay(reasoning_replay)
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

    reasoning = tool_calls[0].reasoning if tool_calls else None
    message: dict[str, Any] = {
        "role": "assistant",
        "content": reasoning if reasoning_replay == "full" else None,
        "tool_calls": tc_list,
    }
    if reasoning and reasoning_replay == "keep-last":
        message["reasoning_content"] = reasoning

    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    if usage:
        response["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    return response


def text_response_to_openai(
    text: str,
    model: str = "forge",
    usage: Any | None = None,
) -> dict[str, Any]:
    """Convert a text response to an OpenAI chat completions response object."""
    response = {
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

    if usage:
        response["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    return response


# ── SSE streaming helpers ────────────────────────────────────────

def tool_calls_to_sse_events(
    tool_calls: list[ToolCall],
    model: str = "forge",
    usage: Any | None = None,
    reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
) -> list[dict[str, Any]]:
    """Convert forge ToolCalls to a sequence of SSE chunk objects.

    Returns the complete list of chunk dicts ready to be formatted as
    SSE data lines. The caller handles the actual SSE wire format.
    """
    reasoning_replay = validate_reasoning_replay(reasoning_replay)
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    events: list[dict[str, Any]] = []

    reasoning = tool_calls[0].reasoning if tool_calls else None
    if reasoning and reasoning_replay != "none":
        delta: dict[str, Any] = {"role": "assistant"}
        if reasoning_replay == "full":
            delta["content"] = reasoning
        else:
            delta["reasoning_content"] = reasoning
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
    final_event = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls",
        }],
    }

    if usage:
        final_event["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    events.append(final_event)
    return events


def text_to_sse_events(
    text: str,
    model: str = "forge",
    chunk_size: int = 0,
    usage: Any | None = None,
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
    final_event = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }

    if usage:
        final_event["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    events.append(final_event)
    return events
