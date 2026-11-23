"""Semantic reply parsers, shared by buffered and streamed backend replies."""

from forge.core.codecs import openai_message, anthropic_message
from forge.core.messages import RichMessage, TextPart, ReasoningPart, MessageRole
from forge.prompts.think_tags import extract_think_tags


def openai_reply(
    message: dict, *, capture_reasoning: bool = True, ollama: bool = False
) -> RichMessage:
    value = dict(message)
    value["role"] = "assistant"
    if ollama and "thinking" in value:
        value["reasoning_content"] = value.pop("thinking")
    from forge.core.reasoning import REASONING_MESSAGE_FIELDS
    from forge.errors import BackendError

    for key in REASONING_MESSAGE_FIELDS:
        if value.get(key) and not isinstance(value[key], str):
            raise BackendError(500, f"reasoning field {key!r} is not a string")
    value = {
        k: v
        for k, v in value.items()
        if k not in REASONING_MESSAGE_FIELDS or v not in (None, "")
    }
    reply = openai_message(value, imported=False, native_arguments=ollama)
    for part in list(reply.parts):
        if isinstance(part, TextPart):
            thinking, text = extract_think_tags(part.text)
            part.text = text
            if (
                thinking
                and capture_reasoning
                and not any(
                    isinstance(p, ReasoningPart) and p.text for p in reply.parts
                )
            ):
                reply.parts.insert(
                    0, ReasoningPart(thinking, field_name="reasoning_content")
                )
    if not capture_reasoning:
        reply.parts = [p for p in reply.parts if not isinstance(p, ReasoningPart)]
    return reply


def anthropic_reply(message: dict) -> RichMessage:
    return anthropic_message(
        {"role": "assistant", "content": message["content"]}, imported=False
    )


def text_reply(content: str) -> RichMessage:
    return RichMessage(MessageRole.ASSISTANT, [TextPart(content)])


async def openai_stream(
    response, record_usage, *, capture_reasoning=True, emit_tool_deltas=True
):
    """Accumulate supported deltas and parse FINAL with the buffered codec."""
    import json
    from forge.clients.base import ChunkType, StreamChunk
    from forge.core.reasoning import REASONING_MESSAGE_FIELDS

    content = ""
    reasoning = {}
    calls = {}
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        record_usage(chunk)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        if delta.get("content"):
            content += delta["content"]
            yield StreamChunk(ChunkType.TEXT_DELTA, content=delta["content"])
        for key in REASONING_MESSAGE_FIELDS:
            if delta.get(key):
                if not isinstance(delta[key], str):
                    from forge.errors import BackendError

                    raise BackendError(
                        500, f"streamed reasoning field {key!r} is not a string"
                    )
                reasoning[key] = reasoning.get(key, "") + delta[key]
        for call in delta.get("tool_calls") or []:
            slot = calls.setdefault(
                call.get("index", 0), {"function": {"name": "", "arguments": ""}}
            )
            fn = call.get("function", {})
            slot["function"]["name"] += fn.get("name") or ""
            if fn.get("arguments") is not None:
                slot["function"]["arguments"] += (
                    fn["arguments"]
                    if isinstance(fn["arguments"], str)
                    else json.dumps(fn["arguments"])
                )
                if emit_tool_deltas:
                    yield StreamChunk(
                        ChunkType.TOOL_CALL_DELTA,
                        content=(
                            fn["arguments"]
                            if isinstance(fn["arguments"], str)
                            else json.dumps(fn["arguments"])
                        ),
                    )
    reply = openai_reply(
        {
            "content": content,
            **reasoning,
            "tool_calls": [calls[k] for k in sorted(calls)],
        },
        capture_reasoning=capture_reasoning,
    )
    yield StreamChunk(ChunkType.FINAL, response=reply)
