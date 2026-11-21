"""Convert between Anthropic Messages API format and forge Messages.

Mirrors convert.py's shape: inbound parser + outbound emitter + SSE helpers.
Used by the proxy's /v1/messages route.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from forge.core.reasoning import DEFAULT_REASONING_REPLAY, ReasoningReplay, validate_reasoning_replay
from forge.core.workflow import ToolCall, ToolSpec


# ── Inbound: Anthropic request → forge Messages ──────────────────

def anthropic_to_messages(
    anthropic_messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
) -> list[Message]:
    """Convert Anthropic messages + system to forge Message objects.

    The Anthropic protocol carries ``system`` as a top-level field (string
    or list of content blocks); forge represents it as a SYSTEM-role message
    prepended to the conversation.

    Content blocks: ``text`` → string; ``tool_use`` → ToolCallInfo on the
    assistant message; ``tool_result`` → role=tool message; ``thinking`` /
    ``document`` / ``image`` → dropped (no forge analog today).
    """
    messages: list[Message] = []

    if system:
        sys_text = _flatten_text_blocks(system) if isinstance(system, list) else system
        if sys_text:
            messages.append(Message(
                MessageRole.SYSTEM,
                sys_text,
                MessageMeta(MessageType.SYSTEM_PROMPT),
            ))

    for msg in anthropic_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            if role == "assistant":
                messages.append(Message(
                    MessageRole.ASSISTANT,
                    content,
                    MessageMeta(MessageType.TEXT_RESPONSE),
                ))
            else:
                messages.append(Message(
                    MessageRole.USER,
                    content,
                    MessageMeta(MessageType.USER_INPUT),
                ))
            continue

        # content is a list of blocks
        text_parts: list[str] = []
        tool_calls: list[ToolCallInfo] = []
        tool_results: list[tuple[str, str, str]] = []  # (tool_use_id, name, content)

        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tc_args = block.get("input", {})
                tool_calls.append(ToolCallInfo(
                    name=block.get("name", ""),
                    args=tc_args if isinstance(tc_args, dict) else {},
                    call_id=block.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                ))
            elif btype == "tool_result":
                tr_content = block.get("content", "")
                if isinstance(tr_content, list):
                    tr_content = _flatten_text_blocks(tr_content)
                tool_results.append((
                    block.get("tool_use_id", ""),
                    "",  # Anthropic tool_result doesn't carry the tool name
                    str(tr_content),
                ))
            # thinking, image, document, etc. → dropped

        text = "\n".join(p for p in text_parts if p)

        if role == "assistant" and tool_calls:
            messages.append(Message(
                MessageRole.ASSISTANT,
                text,
                MessageMeta(MessageType.TOOL_CALL),
                tool_calls=tool_calls,
            ))
        elif role == "assistant":
            messages.append(Message(
                MessageRole.ASSISTANT,
                text,
                MessageMeta(MessageType.TEXT_RESPONSE),
            ))
        elif role == "user" and tool_results:
            # Anthropic packs tool_results inside user messages; forge splits
            # them into separate role=tool messages.
            for tc_id, tname, tcontent in tool_results:
                messages.append(Message(
                    MessageRole.TOOL,
                    tcontent,
                    MessageMeta(MessageType.TOOL_RESULT),
                    tool_name=tname,
                    tool_call_id=tc_id,
                ))
            if text:
                messages.append(Message(
                    MessageRole.USER,
                    text,
                    MessageMeta(MessageType.USER_INPUT),
                ))
        else:
            messages.append(Message(
                MessageRole.USER,
                text,
                MessageMeta(MessageType.USER_INPUT),
            ))

    return messages


def anthropic_tools_to_specs(tools: list[dict[str, Any]] | None) -> list[ToolSpec]:
    """Anthropic tools array → forge ToolSpec list.

    Anthropic shape: ``{"name", "description", "input_schema"}``.
    """
    if not tools:
        return []
    specs = []
    for tool in tools:
        name = tool.get("name", "")
        if not name:
            continue
        specs.append(ToolSpec.from_json_schema(
            name=name,
            description=tool.get("description", ""),
            schema=tool.get("input_schema", {}),
        ))
    return specs


def anthropic_tool_choice_to_openai(tc: Any) -> Any:
    """Translate Anthropic tool_choice to OpenAI tool_choice shape.

    Anthropic: ``{type:"auto"|"any"|"none"|"tool", name?:"X"}``.
    OpenAI: ``"auto"|"required"|"none"|{type:"function", function:{name:"X"}}``.
    """
    if not isinstance(tc, dict):
        return tc
    t = tc.get("type")
    if t == "auto":
        return "auto"
    if t == "any":
        return "required"
    if t == "none":
        return "none"
    if t == "tool":
        return {"type": "function", "function": {"name": tc.get("name", "")}}
    return tc


def openai_tool_choice_to_anthropic(tc: Any) -> Any:
    """Translate OpenAI tool_choice to Anthropic tool_choice shape.

    OpenAI: ``"auto"|"required"|"none"|{type:"function", function:{name:"X"}}``.
    Anthropic: ``{type:"auto"|"any"|"none"|"tool", name?:"X"}``. A dict already
    in one of those shapes passes through unchanged; an unrecognized value is
    forwarded unchanged so the backend rejects it loudly.
    """
    if isinstance(tc, str):
        if tc == "auto":
            return {"type": "auto"}
        if tc == "required":
            return {"type": "any"}
        if tc == "none":
            return {"type": "none"}
        return tc
    if isinstance(tc, dict) and tc.get("type") == "function":
        return {"type": "tool", "name": tc.get("function", {}).get("name", "")}
    return tc


def anthropic_to_openai_passthrough(body: dict[str, Any]) -> dict[str, Any]:
    """Build the OpenAI-shape passthrough body from an Anthropic inbound body.

    Output is what the handler passes via ``passthrough`` to the OpenAI-
    shape client (LlamafileClient) on path 2. Translates field names +
    tool_choice/stop_sequences shapes. Excludes fields forge owns
    (``messages``, ``tools``, ``system``, ``stream``) — those flow through
    forge's normal parsing. Excludes Anthropic-only fields without an
    OpenAI analog (``cache_control`` lives on blocks; ``thinking``,
    ``metadata``, ``service_tier`` have no analog) — they drop here for
    path 2 and are preserved via ``inbound_anthropic_body`` on path 1
    (see ADR-015).
    """
    out: dict[str, Any] = {}

    if "model" in body:
        out["model"] = body["model"]
    if "litellm_session_id" in body:
        out["litellm_session_id"] = body["litellm_session_id"]
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        out["temperature"] = body["temperature"]
    if "top_p" in body:
        out["top_p"] = body["top_p"]
    if "top_k" in body:
        out["top_k"] = body["top_k"]

    if "stop_sequences" in body:
        out["stop"] = body["stop_sequences"]

    if "tool_choice" in body:
        out["tool_choice"] = anthropic_tool_choice_to_openai(body["tool_choice"])

    return out


# Mirrors handler._FORGE_OWNED plus "model"; keeping the literal
# avoids importing the handler (which imports this module).
_NATIVE_OWNED = frozenset({
    "messages", "tools", "stream", "stream_options", "system", "model",
})


def anthropic_backend_passthrough(
    body: dict[str, Any],
    *,
    from_anthropic: bool,
) -> dict[str, Any]:
    """Build the extras for an Anthropic backend from either inbound protocol.

    ``from_anthropic=True`` keeps every inbound field the verbatim first
    attempt already forwards — only forge-owned fields (``messages``,
    ``system``, ``tools``, ``stream``, ``stream_options``) and ``model``
    are removed — so a rebuilt retry sends the same request shape instead
    of OpenAI spellings. ``thinking`` is the exception: a rebuilt transcript
    cannot carry thinking blocks, so the config is excluded (see the
    comment in that branch).

    ``from_anthropic=False`` starts from the handler's ``passthrough``
    extras (already forge-owned and sampling filtered) and translates only
    the OpenAI spellings the proxy knows Anthropic spells differently:
    ``stop`` → ``stop_sequences`` (a bare string is wrapped in a list),
    ``max_completion_tokens`` → ``max_tokens``, ``user`` →
    ``metadata.user_id``, and ``parallel_tool_calls: false`` →
    ``disable_parallel_tool_use: true`` on an auto/any/tool
    ``tool_choice`` (``none`` needs no flag). An explicit value wins in
    each case and the replaced OpenAI spelling is removed. An explicit
    ``stop_sequences`` wins over ``stop``; when ``tool_choice`` is absent
    or null — the OpenAI ``auto`` default — the flag rides an explicit
    ``{"type": "auto"}`` so the choice is not silently left to Anthropic's
    parallel-enabled default. ``metadata`` narrows to the one key Anthropic
    defines (``user_id``); any other value is removed. Every other key rides
    through unchanged, including keys with no Anthropic analog, matching
    ``main``'s pass-through. ``litellm_session_id`` is forwarded by key
    presence, matching ``anthropic_to_openai_passthrough``. ``thinking`` is
    not forwarded because a rebuilt transcript cannot carry signed thinking
    blocks; ``tool_choice`` is therefore translated as given. Sampling
    fields ride the per-call ``sampling`` channel untouched;
    AnthropicClient ignores them by design.
    """
    if from_anthropic:
        extras = {k: v for k, v in body.items() if k not in _NATIVE_OWNED}
        # A rebuilt transcript carries no signed thinking blocks: the inbound
        # parser drops them and _convert_messages rebuilds assistant turns
        # without them (ADR-017 keeps thinking replay out of scope). The
        # Messages API requires thinking blocks to be returned complete and
        # unmodified within a tool-use turn, so forwarding the config would
        # send a body the backend rejects. A rebuilt attempt runs with
        # thinking off, as it did on main.
        extras.pop("thinking", None)
        return extras

    # The caller's extras already exclude forge-owned and sampling fields.
    # Copy before rewriting so the handler's passthrough dict is untouched.
    out = dict(body)
    # ``thinking`` is omitted like the native branch: a rebuilt transcript
    # cannot carry signed thinking blocks, so forwarding it would be rejected.
    out.pop("thinking", None)

    if "stop" in out:
        stop = out.pop("stop")
        if "stop_sequences" not in out:
            out["stop_sequences"] = [stop] if isinstance(stop, str) else stop

    if "max_completion_tokens" in out:
        max_completion_tokens = out.pop("max_completion_tokens")
        out.setdefault("max_tokens", max_completion_tokens)

    if "tool_choice" in out:
        tool_choice = openai_tool_choice_to_anthropic(out["tool_choice"])
        if tool_choice is None:
            del out["tool_choice"]
        else:
            out["tool_choice"] = tool_choice

    parallel_tool_calls = out.pop("parallel_tool_calls", None)
    if parallel_tool_calls is False:
        choice = out.get("tool_choice")
        if choice is None:
            # OpenAI's absent/null tool_choice means "auto"; Anthropic's
            # default is also auto, but only an explicit choice can carry
            # the flag.
            out["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        elif isinstance(choice, dict) and choice.get("type") in ("auto", "any", "tool"):
            out["tool_choice"] = {**choice, "disable_parallel_tool_use": True}

    metadata = out.get("metadata")
    user_id = metadata.get("user_id") if isinstance(metadata, dict) else None
    if not isinstance(user_id, str) and isinstance(out.get("user"), str):
        user_id = out["user"]
    out.pop("user", None)
    if isinstance(user_id, str):
        out["metadata"] = {"user_id": user_id}
    else:
        out.pop("metadata", None)

    return out


def _flatten_text_blocks(blocks: list[dict[str, Any]] | str) -> str:
    """Concatenate text from a list of content blocks. Non-text blocks dropped."""
    if isinstance(blocks, str):
        return blocks
    parts = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif isinstance(b, str):
            parts.append(b)
    return "\n".join(p for p in parts if p)


# ── Outbound: forge response → Anthropic response ────────────────

def _anthropic_usage(usage: Any | None) -> dict[str, int]:
    """Map a forge TokenUsage to Anthropic's input/output token shape."""
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0),
        "output_tokens": getattr(usage, "completion_tokens", 0),
    }


def tool_calls_to_anthropic(
    tool_calls: list[ToolCall],
    model: str = "forge",
    usage: Any | None = None,
    reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
) -> dict[str, Any]:
    """Convert forge ToolCalls to an Anthropic Messages API response object."""
    reasoning_replay = validate_reasoning_replay(reasoning_replay)
    blocks: list[dict[str, Any]] = []

    if tool_calls and tool_calls[0].reasoning and reasoning_replay == "full":
        blocks.append({"type": "text", "text": tool_calls[0].reasoning})

    for tc in tool_calls:
        blocks.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": tc.tool,
            "input": tc.args,
        })

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": _anthropic_usage(usage),
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

def tool_calls_to_anthropic_sse(
    tool_calls: list[ToolCall],
    model: str = "forge",
    usage: Any | None = None,
    reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
) -> list[dict[str, Any]]:
    """Build the Anthropic SSE event sequence for a tool-use response.

    Each returned dict carries a top-level ``type`` field; the SSE wire
    formatter reads that to emit ``event: <type>`` lines. Spec:
    https://platform.claude.com/docs/en/build-with-claude/streaming
    """
    reasoning_replay = validate_reasoning_replay(reasoning_replay)
    au = _anthropic_usage(usage)
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    events: list[dict[str, Any]] = []

    events.append({
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
    })

    block_idx = 0

    # Reasoning text first, if present.
    reasoning = tool_calls[0].reasoning if tool_calls else None
    if reasoning and reasoning_replay == "full":
        events.append({
            "type": "content_block_start",
            "index": block_idx,
            "content_block": {"type": "text", "text": ""},
        })
        events.append({
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {"type": "text_delta", "text": reasoning},
        })
        events.append({"type": "content_block_stop", "index": block_idx})
        block_idx += 1

    # One tool_use block per call.
    for tc in tool_calls:
        tc_id = f"toolu_{uuid.uuid4().hex[:24]}"
        events.append({
            "type": "content_block_start",
            "index": block_idx,
            "content_block": {
                "type": "tool_use",
                "id": tc_id,
                "name": tc.tool,
                "input": {},
            },
        })
        events.append({
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps(tc.args),
            },
        })
        events.append({"type": "content_block_stop", "index": block_idx})
        block_idx += 1

    events.append({
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": au["output_tokens"]},
    })
    events.append({"type": "message_stop"})

    return events


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
