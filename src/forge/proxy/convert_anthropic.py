"""Anthropic ingress and caller-facing reply rendering."""

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
from forge.core.codecs import anthropic_state, argument_value
from forge.core.workflow import ToolDefinition
from forge.core.reasoning import (
    DEFAULT_REASONING_REPLAY,
    ReasoningReplay,
    validate_reasoning_replay,
)


def anthropic_to_messages(
    messages: list[dict[str, Any]], system=None
) -> ConversationState:
    return anthropic_state(messages, system)


def anthropic_tools_to_definitions(
    tools: list[dict[str, Any]] | None,
) -> list[ToolDefinition]:
    """Parse flat Anthropic function tools into canonical definitions."""
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ValueError("Anthropic tools must be a list")
    definitions = []
    names = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "custom") != "custom":
            raise ValueError("Expected an Anthropic function tool")
        if "input_schema" not in tool:
            raise ValueError("Anthropic tool requires input_schema")
        definition = ToolDefinition(
            name=tool.get("name"),
            description=tool.get("description", ""),
            input_schema=tool["input_schema"],
            extra_fields={
                key: value
                for key, value in tool.items()
                if key not in {"name", "description", "input_schema"}
            },
        )
        if definition.name in names:
            raise ValueError(f"Duplicate tool name '{definition.name}'")
        names.add(definition.name)
        definitions.append(definition)
    return definitions


def _anthropic_usage(usage: Any | None) -> dict[str, int]:
    """Map a forge TokenUsage to Anthropic's input/output token shape."""
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0),
        "output_tokens": getattr(usage, "completion_tokens", 0),
    }


def text_response_to_anthropic(
    text: str,
    model: str = "forge",
    usage: Any | None = None,
) -> dict[str, Any]:
    """Convert a text response to an Anthropic Messages API response object."""
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": _anthropic_usage(usage),
    }


# ── SSE event builders ───────────────────────────────────────────


def text_to_anthropic_sse(
    text: str,
    model: str = "forge",
    usage: Any | None = None,
) -> list[dict[str, Any]]:
    """Build the Anthropic SSE event sequence for a text response."""
    au = _anthropic_usage(usage)
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    return [
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": au["input_tokens"], "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": au["output_tokens"]},
        },
        {"type": "message_stop"},
    ]


def reply_to_anthropic(
    reply: RichMessage,
    model="forge",
    usage=None,
    reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
) -> dict:
    mode = validate_reasoning_replay(reasoning_replay)
    response = text_response_to_anthropic("", model, usage)
    blocks = []
    for part in reply.parts:
        if isinstance(part, TextPart) and part.text:
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ReasoningPart) and mode == "full" and part.text:
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ToolCallPart):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": part.name,
                    "input": argument_value(part),
                }
            )
    response["content"] = blocks
    if any(isinstance(p, ToolCallPart) for p in reply.parts):
        response["stop_reason"] = "tool_use"
    return response


def reply_to_anthropic_sse(
    reply: RichMessage,
    model="forge",
    usage=None,
    reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
) -> list[dict]:
    response = reply_to_anthropic(reply, model, usage, reasoning_replay)
    events = [
        {
            "type": "message_start",
            "message": {
                **response,
                "content": [],
                "stop_reason": None,
                "usage": {
                    "input_tokens": response["usage"]["input_tokens"],
                    "output_tokens": 1,
                },
            },
        }
    ]
    for i, block in enumerate(response["content"]):
        if block["type"] == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            start = {**block, "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"]),
            }
        events.extend(
            [
                {"type": "content_block_start", "index": i, "content_block": start},
                {"type": "content_block_delta", "index": i, "delta": delta},
                {"type": "content_block_stop", "index": i},
            ]
        )
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": response["stop_reason"],
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": response["usage"]["output_tokens"]},
            },
            {"type": "message_stop"},
        ]
    )
    return events
