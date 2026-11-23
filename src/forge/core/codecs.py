"""Supported protocol ingress and outgoing content views."""

from __future__ import annotations
import json
import uuid
from copy import deepcopy
from typing import Any
from forge.core.messages import (
    ConversationState,
    RichMessage,
    MessageRole,
    TextPart,
    ReasoningPart,
    ToolCallPart,
    ToolResultPart,
    ValidArguments,
    arguments_from_value,
)
from forge.core.reasoning import REASONING_MESSAGE_FIELDS, validate_reasoning_replay


def extras(value: dict, owned: set[str]) -> dict:
    return {k: deepcopy(v) for k, v in value.items() if k not in owned}


def decode_arguments(raw: Any, *, json_encoded: bool = True):
    if json_encoded and isinstance(raw, str):
        if raw == "":
            return ValidArguments({})
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            pass
    return arguments_from_value(raw)


def argument_value(part: ToolCallPart) -> Any:
    return (
        part.arguments.value
        if isinstance(part.arguments, ValidArguments)
        else part.arguments.received
    )


def text_parts(content: Any) -> tuple[list[TextPart], str]:
    if content is None:
        return [], "null"
    if isinstance(content, str):
        return [TextPart(content)], "string"
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(TextPart(block, bare=True))
        elif block.get("type") == "text":
            parts.append(
                TextPart(block.get("text", ""), extras(block, {"type", "text"}))
            )
        else:
            raise ValueError(f"Unsupported content block kind: {block.get('type')!r}")
    return parts, "blocks"


def openai_message(
    value: dict, *, imported: bool = True, native_arguments: bool = False
) -> RichMessage:
    role = (
        MessageRole(value.get("role", "user"))
        if value.get("role", "user") in MessageRole
        else MessageRole.USER
    )
    parts, shape = text_parts(value.get("content", ""))
    fields = extras(
        value,
        {
            "role",
            "content",
            "tool_calls",
            "tool_call_id",
            *(REASONING_MESSAGE_FIELDS if role == MessageRole.ASSISTANT else ()),
        },
    )
    if role == MessageRole.TOOL:
        result = ToolResultPart(
            value.get("tool_call_id", ""), parts, content_shape=shape
        )
        return RichMessage(role, [result], extra_fields=fields, content_shape=shape)
    reasoning = []
    for key in REASONING_MESSAGE_FIELDS:
        if role == MessageRole.ASSISTANT and key in value:
            text = value[key]
            if not isinstance(text, str):
                raise ValueError(f"reasoning field {key!r} must be a string")
            reasoning.append(ReasoningPart(text, field_name=key))
    parts = reasoning + parts
    for call in value.get("tool_calls") or []:
        if call.get("type", "function") != "function":
            raise ValueError("Unsupported tool call kind")
        fn = call.get("function", {})
        parts.append(
            ToolCallPart(
                fn.get("name", ""),
                decode_arguments(
                    fn.get("arguments", {}), json_encoded=not native_arguments
                ),
                call.get("id", f"call_{uuid.uuid4().hex[:8]}") if imported else "",
                extras(call, {"type", "id", "function"}),
                extras(fn, {"name", "arguments"}),
            )
        )
    return RichMessage(
        role,
        parts,
        extra_fields=fields,
        content_shape=shape,
        empty_tool_calls="tool_calls" in value and value["tool_calls"] == [],
    )


def openai_state(messages: list[dict]) -> ConversationState:
    return ConversationState([openai_message(m) for m in messages])


def anthropic_message(value: dict, *, imported: bool = True) -> RichMessage:
    content = value.get("content", "")
    if isinstance(content, str):
        parts, shape = text_parts(content)
    else:
        parts, shape = [], "blocks"
        for block in content:
            kind = block.get("type")
            if kind == "text":
                parts.append(
                    TextPart(block.get("text", ""), extras(block, {"type", "text"}))
                )
            elif kind == "thinking":
                parts.append(
                    ReasoningPart(
                        block.get("thinking", ""), extras(block, {"type", "thinking"})
                    )
                )
            elif kind == "tool_use":
                parts.append(
                    ToolCallPart(
                        block.get("name", ""),
                        arguments_from_value(block.get("input", {})),
                        block.get("id", f"toolu_{uuid.uuid4().hex[:12]}")
                        if imported
                        else "",
                        extras(block, {"type", "id", "name", "input"}),
                    )
                )
            elif kind == "tool_result":
                texts, result_shape = text_parts(block.get("content", ""))
                parts.append(
                    ToolResultPart(
                        block.get("tool_use_id", ""),
                        texts,
                        "error" if block.get("is_error") else "success",
                        result_shape,
                        extras(block, {"type", "tool_use_id", "content"}),
                    )
                )
            else:
                raise ValueError(f"Unsupported content block kind: {kind!r}")
    return RichMessage(
        MessageRole(value.get("role", "user")),
        parts,
        extra_fields=extras(value, {"role", "content"}),
        content_shape=shape,
    )


def anthropic_state(messages: list[dict], system=None) -> ConversationState:
    state = ConversationState()
    if system is not None:
        state.messages.append(anthropic_message({"role": "system", "content": system}))
    state.messages.extend(anthropic_message(m) for m in messages)
    return state


def replay_messages(state: ConversationState, mode: str) -> list[RichMessage]:
    mode = validate_reasoning_replay(mode)
    last = next(
        (
            i
            for i in range(len(state.messages) - 1, -1, -1)
            if state.messages[i].role == MessageRole.ASSISTANT
            and any(isinstance(p, ReasoningPart) for p in state.messages[i].parts)
        ),
        None,
    )
    view = deepcopy(state.messages)
    for i, message in enumerate(view):
        if mode == "none" or (mode == "keep-last" and i != last):
            message.parts = [
                p for p in message.parts if not isinstance(p, ReasoningPart)
            ]
    return view


def render_text(parts: list[TextPart], shape: str):
    if shape == "null":
        return None
    if shape == "blocks":
        return [
            p.text if p.bare else {**p.extra_fields, "type": "text", "text": p.text}
            for p in parts
        ]
    return "\n".join(p.text for p in parts)


def render_openai(messages: list[RichMessage], *, ollama: bool = False) -> list[dict]:
    result = []
    names = {
        p.call_id: p.name
        for m in messages
        for p in m.parts
        if isinstance(p, ToolCallPart)
    }
    for message in messages:
        text = [p for p in message.parts if isinstance(p, TextPart)]
        content = render_text(text, message.content_shape)
        if ollama and isinstance(content, list):
            content = "\n".join(p.text for p in text)
        item = {
            **extras(
                message.extra_fields,
                {
                    "role",
                    "content",
                    "tool_calls",
                    *(
                        REASONING_MESSAGE_FIELDS
                        if message.role == MessageRole.ASSISTANT
                        else ()
                    ),
                },
            ),
            "role": message.role.value,
            "content": content,
        }
        calls = []
        results = []
        for part in message.parts:
            if isinstance(part, ReasoningPart):
                key = "thinking" if ollama else (part.field_name or "reasoning_content")
                if key in item:
                    item[key] += "\n" + part.text
                else:
                    item[key] = part.text
            elif isinstance(part, ToolCallPart):
                value = argument_value(part)
                call = {
                    **deepcopy(part.extra_fields),
                    "function": {
                        **deepcopy(part.function_fields),
                        "name": part.name,
                        "arguments": value if ollama else json.dumps(value),
                    },
                }
                if not ollama:
                    call.update(type="function", id=part.call_id)
                calls.append(call)
            elif isinstance(part, ToolResultPart):
                result_item = {
                    **deepcopy(message.extra_fields),
                    **{
                        k: deepcopy(v)
                        for k, v in part.extra_fields.items()
                        if k != "is_error"
                    },
                    "role": "tool",
                    "content": render_text(part.content, part.content_shape),
                }
                if ollama:
                    result_item["tool_name"] = message.extra_fields.get(
                        "name", names.get(part.call_id, "")
                    )
                    result_item.pop("tool_call_id", None)
                    result_item["content"] = "\n".join(p.text for p in part.content)
                else:
                    result_item["tool_call_id"] = part.call_id
                    if message.forge_purpose is not None:
                        result_item.setdefault("name", names.get(part.call_id, ""))
                results.append(result_item)
        if calls or message.empty_tool_calls:
            item["tool_calls"] = calls
        if results:
            result.extend(results)
            if text or calls:
                result.append(item)
        else:
            result.append(item)
    return result


def render_anthropic(
    messages: list[RichMessage], *, merge_consecutive: bool = False
) -> tuple[Any, list[dict]]:
    system = None
    result = []
    pending = []
    for message in messages:
        blocks = []
        for part in message.parts:
            if isinstance(part, TextPart):
                if (
                    not part.text
                    and message.content_shape != "blocks"
                    and len(message.parts) > 1
                ):
                    continue
                blocks.append(
                    {**deepcopy(part.extra_fields), "type": "text", "text": part.text}
                )
            elif isinstance(part, ReasoningPart):
                if part.field_name is not None:
                    # OpenAI reasoning has no signed native thinking-block form.
                    continue
                blocks.append(
                    {
                        **deepcopy(part.extra_fields),
                        "type": "thinking",
                        "thinking": part.text,
                    }
                )
            elif isinstance(part, ToolCallPart):
                blocks.append(
                    {
                        **deepcopy(part.extra_fields),
                        "type": "tool_use",
                        "id": part.call_id,
                        "name": part.name,
                        "input": deepcopy(argument_value(part)),
                    }
                )
                pending.append(part.call_id)
            elif isinstance(part, ToolResultPart):
                blocks.append(
                    {
                        **deepcopy(part.extra_fields),
                        "type": "tool_result",
                        "tool_use_id": part.call_id,
                        "content": render_text(part.content, part.content_shape),
                    }
                )
                if part.call_id in pending:
                    pending.remove(part.call_id)
        if not blocks and any(
            isinstance(part, ReasoningPart) and part.field_name is not None
            for part in message.parts
        ):
            continue
        role = "user" if message.role == MessageRole.TOOL else message.role.value
        if (
            role == "user"
            and pending
            and not any(isinstance(p, ToolResultPart) for p in message.parts)
        ):
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": "Not executed.",
                    "is_error": True,
                }
                for call_id in pending
            ] + blocks
            pending.clear()
        plain = all(isinstance(p, TextPart) for p in message.parts) and all(
            b["type"] == "text" for b in blocks
        )
        content = (
            render_text(message.parts, message.content_shape)
            if plain and message.content_shape == "string"
            else blocks
        )
        if role == "system":
            system = content
        else:
            fields = deepcopy(message.extra_fields)
            if message.role == MessageRole.TOOL:
                fields.pop("name", None)
                fields.pop("tool_call_id", None)
            result.append({**fields, "role": role, "content": content})
    if merge_consecutive:
        merged = []
        for item in result:
            if merged and merged[-1]["role"] == item["role"]:
                left, right = merged[-1]["content"], item["content"]
                left = (
                    [{"type": "text", "text": left}] if isinstance(left, str) else left
                )
                right = (
                    [{"type": "text", "text": right}]
                    if isinstance(right, str)
                    else right
                )
                merged[-1] = {**merged[-1], "content": left + right}
            else:
                merged.append(item)
        result = merged
    return system, result
