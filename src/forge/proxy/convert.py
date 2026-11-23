"""OpenAI ingress and caller-facing reply rendering."""

from __future__ import annotations
import json
import uuid
from typing import Any
from forge.core.messages import (
    ConversationState,
    RichMessage,
    TextPart,
    ReasoningPart,
    ToolCallPart,
)
from forge.core.codecs import openai_state, argument_value
from forge.core.reasoning import (
    DEFAULT_REASONING_REPLAY,
    ReasoningReplay,
    validate_reasoning_replay,
)


def openai_to_messages(messages: list[dict[str, Any]]) -> ConversationState:
    return openai_state(messages)


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

    if usage:
        response["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    return response


# ── SSE streaming helpers ────────────────────────────────────────


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
    final_event = {
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

    if usage:
        final_event["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    events.append(final_event)
    return events


def reply_to_openai(
    reply: RichMessage,
    model="forge",
    usage=None,
    reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
) -> dict:
    mode = validate_reasoning_replay(reasoning_replay)
    text = "\n".join(p.text for p in reply.parts if isinstance(p, TextPart))
    reasoning = "\n".join(p.text for p in reply.parts if isinstance(p, ReasoningPart))
    response = text_response_to_openai(text, model, usage)
    message = response["choices"][0]["message"]
    calls = [p for p in reply.parts if isinstance(p, ToolCallPart)]
    if calls:
        message["content"] = text or None
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": p.name,
                    "arguments": json.dumps(argument_value(p)),
                },
            }
            for p in calls
        ]
        response["choices"][0]["finish_reason"] = "tool_calls"
        if reasoning and mode == "full":
            message["content"] = "\n".join(filter(None, [reasoning, text]))
        elif reasoning and mode == "keep-last":
            message["reasoning_content"] = reasoning
    return response


def reply_to_sse_events(
    reply: RichMessage,
    model="forge",
    usage=None,
    reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
) -> list[dict]:
    response = reply_to_openai(reply, model, usage, reasoning_replay)
    choice = response["choices"][0]
    message = choice["message"]
    events = []
    base = {k: response[k] for k in ("id", "model")}
    base["object"] = "chat.completion.chunk"
    if message.get("content") or message.get("reasoning_content"):
        delta = {
            k: v
            for k, v in message.items()
            if k in {"role", "content", "reasoning_content"}
        }
        events.append(
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        )
    for i, call in enumerate(message.get("tool_calls", [])):
        events.append(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": i, **call}]},
                        "finish_reason": None,
                    }
                ],
            }
        )
    final = {
        **base,
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}
        ],
    }
    if usage:
        final["usage"] = response["usage"]
    events.append(final)
    return events
