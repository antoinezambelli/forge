"""Rich fixtures for the migrated canonical inference/client tests."""

from forge.clients.base import RequestMetadata
from forge.core.codecs import openai_state
from forge.core.messages import (
    RichMessage,
    MessageRole,
    ToolCallPart,
    TextPart,
    ReasoningPart,
    arguments_from_value,
)


def call(name="search", arguments={}):
    return ToolCallPart(name, arguments_from_value(arguments))


def assistant(*parts):
    return RichMessage(MessageRole.ASSISTANT, list(parts))


def calls(reply):
    return [p for p in reply.parts if isinstance(p, ToolCallPart)]


def visible(reply):
    return "\n".join(p.text for p in reply.parts if isinstance(p, TextPart))


def reasoning(reply):
    return "\n".join(p.text for p in reply.parts if isinstance(p, ReasoningPart))


def render(
    client,
    messages,
    tools=None,
    sampling=None,
    passthrough=None,
    extra_headers=None,
    *,
    stream=False,
):
    return client.render_request(
        openai_state(messages),
        tools,
        RequestMetadata(
            sampling=sampling,
            extra_fields=passthrough or {},
            extra_headers=extra_headers,
        ),
        stream=stream,
    )


async def send(
    client, messages, tools=None, sampling=None, passthrough=None, extra_headers=None
):
    return await client.send(
        render(client, messages, tools, sampling, passthrough, extra_headers)
    )


def send_stream(
    client, messages, tools=None, sampling=None, passthrough=None, extra_headers=None
):
    return client.send_stream(
        render(
            client, messages, tools, sampling, passthrough, extra_headers, stream=True
        )
    )


def render_native(client, body, tools=None, extra_headers=None, *, stream=False):
    from forge.core.codecs import anthropic_state
    from forge.proxy.convert_anthropic import anthropic_tools_to_definitions

    return client.render_request(
        anthropic_state(body.get("messages", []), body.get("system")),
        tools
        if tools is not None
        else anthropic_tools_to_definitions(body.get("tools")) or None,
        RequestMetadata(
            model=body.get("model"),
            source_protocol="anthropic",
            extra_headers=extra_headers,
            extra_fields={
                k: v
                for k, v in body.items()
                if k not in {"messages", "system", "tools", "stream", "model"}
            },
        ),
        stream=stream,
    )
