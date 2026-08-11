"""Tests for run_inference threading the per-call credential to the client.

The proxy hands run_inference a relocated inbound auth header via
``extra_headers``; it must reach the client's send/send_stream. When unset, the
kwarg must be omitted entirely (splat-only-when-set) so clients and test
doubles that don't declare it keep their original signature.
"""

from unittest.mock import AsyncMock

import pytest

from forge.clients.base import ChunkType, StreamChunk
from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.inference import run_inference
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType
from forge.core.workflow import ToolCall, ToolSpec
from forge.guardrails import ErrorTracker, ResponseValidator


def _ctx():
    return ContextManager(strategy=NoCompact(), budget_tokens=8192)


def _search_spec():
    return ToolSpec.from_json_schema(
        name="search", description="", schema={"type": "object", "properties": {}},
    )


def _messages():
    return [Message(MessageRole.USER, "hi", MessageMeta(MessageType.USER_INPUT))]


def _send_client(*responses):
    client = AsyncMock()
    client.api_format = "ollama"
    client.send = AsyncMock(side_effect=list(responses))
    client.last_usage = {}
    client._slot_id = 0
    return client


async def _run(client, **kw):
    return await run_inference(
        messages=_messages(),
        client=client,
        context_manager=_ctx(),
        validator=ResponseValidator(["search"], rescue_enabled=True),
        error_tracker=ErrorTracker(max_retries=1),
        tool_specs=[_search_spec()],
        **kw,
    )


@pytest.mark.asyncio
async def test_extra_headers_forwarded_or_omitted_from_send():
    for case, extra_headers in [
        ("forwarded", {"Authorization": "Bearer X"}),
        ("omitted", None),
    ]:
        client = _send_client([ToolCall(tool="search", args={})])
        kwargs = {"extra_headers": extra_headers} if extra_headers is not None else {}
        await _run(client, **kwargs)
        sent = client.send.call_args.kwargs
        if extra_headers is None:
            # Splat-only-when-set keeps clients without this parameter compatible.
            assert "extra_headers" not in sent, case
        else:
            assert sent["extra_headers"] == extra_headers, case


@pytest.mark.asyncio
async def test_extra_headers_forwarded_or_omitted_from_send_stream():
    for case, extra_headers in [
        ("forwarded", {"x-api-key": "K"}),
        ("omitted", None),
    ]:
        captured: dict = {}

        async def fake_stream(*args, **kwargs):
            captured.update(kwargs)
            yield StreamChunk(
                type=ChunkType.FINAL, response=[ToolCall(tool="search", args={})]
            )

        client = AsyncMock()
        client.api_format = "ollama"
        client.send_stream = fake_stream
        client.last_usage = {}
        client._slot_id = 0

        kwargs = {"extra_headers": extra_headers} if extra_headers is not None else {}
        await _run(client, stream=True, **kwargs)
        if extra_headers is None:
            assert "extra_headers" not in captured, case
        else:
            assert captured["extra_headers"] == extra_headers, case
