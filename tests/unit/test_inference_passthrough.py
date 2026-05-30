"""Tests for run_inference's raw-OpenAI passthrough first-attempt gate.

The proxy hands run_inference the client's verbatim OpenAI transcript/tools.
They must be forwarded ONLY on the clean first attempt; any forge mutation
(retry here) falls back to fold_and_serialize + the parsed tool_specs.
"""

from unittest.mock import AsyncMock

import pytest

from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.inference import run_inference
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType
from forge.core.workflow import TextResponse, ToolCall, ToolSpec
from forge.guardrails import ErrorTracker, ResponseValidator


def _client(*responses):
    client = AsyncMock()
    client.api_format = "ollama"
    client.send = AsyncMock(side_effect=list(responses))
    client.last_usage = {}
    client._slot_id = 0
    return client


def _ctx():
    return ContextManager(strategy=NoCompact(), budget_tokens=8192)


def _search_spec():
    return ToolSpec.from_json_schema(
        name="search", description="", schema={"type": "object", "properties": {}},
    )


@pytest.mark.asyncio
async def test_raw_used_on_first_attempt_folded_on_retry():
    # Attempt 0: text (invalid → retry). Attempt 1: valid tool call.
    client = _client(
        TextResponse(content="just narrating, no tool"),
        [ToolCall(tool="search", args={})],
    )
    messages = [Message(
        MessageRole.USER, "folded-form",
        MessageMeta(MessageType.USER_INPUT),
    )]
    raw_messages = [{"role": "user", "content": "VERBATIM", "name": "u1"}]
    raw_tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    result = await run_inference(
        messages=messages,
        client=client,
        context_manager=_ctx(),
        validator=ResponseValidator(["search"], rescue_enabled=True),
        error_tracker=ErrorTracker(max_retries=2),
        tool_specs=[_search_spec()],
        raw_openai_messages=raw_messages,
        raw_openai_tools=raw_tools,
    )

    assert result is not None
    assert client.send.await_count == 2

    # Attempt 0 (clean): forwarded the verbatim raw messages + raw tools.
    first = client.send.call_args_list[0]
    assert first.args[0] == raw_messages
    assert first.kwargs["raw_openai_tools"] == raw_tools

    # Attempt 1 (post-retry mutation): folded messages, no raw tools kwarg.
    second = client.send.call_args_list[1]
    assert second.args[0] != raw_messages
    assert second.args[0][0]["content"] == "folded-form"
    assert "raw_openai_tools" not in second.kwargs


@pytest.mark.asyncio
async def test_no_raw_falls_back_to_fold():
    """Without raw_openai_* (the non-proxy runner path), folding is used and
    no raw_openai_tools kwarg is passed to the client."""
    client = _client([ToolCall(tool="search", args={})])
    messages = [Message(
        MessageRole.USER, "hello",
        MessageMeta(MessageType.USER_INPUT),
    )]

    await run_inference(
        messages=messages,
        client=client,
        context_manager=_ctx(),
        validator=ResponseValidator(["search"], rescue_enabled=True),
        error_tracker=ErrorTracker(max_retries=1),
        tool_specs=[_search_spec()],
    )

    call = client.send.call_args
    assert call.args[0][0]["content"] == "hello"
    assert "raw_openai_tools" not in call.kwargs
