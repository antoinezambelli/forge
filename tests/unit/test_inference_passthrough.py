"""Tests for run_inference's raw-OpenAI passthrough first-attempt gate.

The proxy hands run_inference the client's verbatim OpenAI transcript/tools.
They must be forwarded ONLY on the clean first attempt; any forge mutation
(retry here) falls back to fold_and_serialize + the parsed tool_specs.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from forge.clients.base import TokenUsage, _capture_usage
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


def test_usage_capture_restores_nested_outer_sink():
    with _capture_usage() as outer:
        outer_usage = TokenUsage(1, 2, 3)
        with _capture_usage() as inner:
            inner_usage = TokenUsage(4, 5, 9)
        assert inner.usage is inner_usage
        TokenUsage(6, 7, 13)

    assert outer.usage != outer_usage
    assert outer.usage == TokenUsage(6, 7, 13)


@pytest.mark.asyncio
async def test_usage_capture_is_task_local_for_overlapping_attempts():
    both_started = asyncio.Event()
    arrived = 0

    async def capture(value):
        nonlocal arrived
        with _capture_usage() as captured:
            arrived += 1
            if arrived == 2:
                both_started.set()
            await both_started.wait()
            usage = TokenUsage(value, 0, value)
            await asyncio.sleep(0)
        return captured.usage, usage

    first, second = await asyncio.gather(capture(10), capture(20))
    assert first[0] is first[1]
    assert second[0] is second[1]


@pytest.mark.asyncio
async def test_raw_used_on_first_attempt_folded_on_retry():
    # Attempt 0: text (invalid → retry). Attempt 1: valid tool call.
    client = _client(
        TextResponse(content="just narrating, no tool"),
        [ToolCall(tool="search", args={})],
    )
    messages = [
        Message(
            MessageRole.USER, "folded-form",
            MessageMeta(MessageType.USER_INPUT),
        ),
        Message(
            MessageRole.ASSISTANT, "prior-answer",
            MessageMeta(MessageType.TEXT_RESPONSE),
        ),
        Message(
            MessageRole.USER, "complete-tool-output",
            MessageMeta(MessageType.TOOL_RESULT),
        ),
    ]
    raw_messages = [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "old",
            "tool_calls": [],
            "name": "a1",
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "latest",
            "tool_calls": [],
            "name": "a2",
        },
    ]
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
        reasoning_replay="full",
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
    assert [message["content"] for message in second.args[0][:3]] == [
        "folded-form",
        "prior-answer",
        "complete-tool-output",
    ]
    # Retry preserves the complete prefix, then appends the rejected response
    # and its corrective nudge.
    assert len(second.args[0]) == 5
    assert "raw_openai_tools" not in second.kwargs


@pytest.mark.asyncio
async def test_threshold_sized_proxy_request_stays_on_clean_raw_path():
    client = _client([ToolCall(tool="search", args={})])
    large = "caller-history-" * 10_000
    messages = [Message(
        MessageRole.USER,
        large,
        MessageMeta(MessageType.USER_INPUT),
    )]
    raw_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": large},
        {"role": "assistant", "content": "prior answer"},
    ]
    raw_tools = [
        {"type": "function", "function": {"name": "search", "parameters": {}}},
    ]
    context_manager = ContextManager(NoCompact(), budget_tokens=8)

    await run_inference(
        messages=messages,
        client=client,
        context_manager=context_manager,
        validator=ResponseValidator(["search"], rescue_enabled=True),
        error_tracker=ErrorTracker(max_retries=1),
        tool_specs=[_search_spec()],
        raw_openai_messages=raw_messages,
        raw_openai_tools=raw_tools,
    )

    call = client.send.await_args
    assert call.args[0] == raw_messages
    assert call.kwargs["raw_openai_tools"] == raw_tools


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


@pytest.mark.asyncio
async def test_non_full_reasoning_replay_filters_raw_reasoning_but_keeps_raw_shape():
    client = _client([ToolCall(tool="search", args={})])
    messages = [Message(
        MessageRole.USER, "folded-form",
        MessageMeta(MessageType.USER_INPUT),
    )]
    raw_messages = [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "old",
            "tool_calls": [],
            "name": "a1",
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "latest",
            "tool_calls": [],
            "name": "a2",
        },
    ]
    raw_tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    await run_inference(
        messages=messages,
        client=client,
        context_manager=_ctx(),
        validator=ResponseValidator(["search"], rescue_enabled=True),
        error_tracker=ErrorTracker(max_retries=1),
        tool_specs=[_search_spec()],
        raw_openai_messages=raw_messages,
        raw_openai_tools=raw_tools,
        reasoning_replay="keep-last",
    )

    call = client.send.call_args
    assert call.args[0][0]["name"] == "a1"
    assert "reasoning_content" not in call.args[0][0]
    assert call.args[0][1]["name"] == "a2"
    assert call.args[0][1]["reasoning_content"] == "latest"
    assert call.kwargs["raw_openai_tools"] == raw_tools


async def _run_with_context(client, context_manager):
    return await run_inference(
        messages=[Message(
            MessageRole.USER, "x" * 40,
            MessageMeta(MessageType.USER_INPUT),
        )],
        client=client,
        context_manager=context_manager,
        validator=ResponseValidator(["search"], rescue_enabled=True),
        error_tracker=ErrorTracker(max_retries=2),
        tool_specs=[_search_spec()],
    )


@pytest.mark.asyncio
async def test_fresh_usage_records_prompt_not_total_occupancy():
    ctx = _ctx()
    client = _client([ToolCall(tool="search", args={})])

    async def send(*args, **kwargs):
        client.last_usage[0] = TokenUsage(25, 75, 100)
        return [ToolCall(tool="search", args={})]

    client.send = AsyncMock(side_effect=send)
    result = await _run_with_context(client, ctx)

    assert result is not None
    assert result.usage == TokenUsage(25, 75, 100)
    assert ctx.usage is not None
    assert ctx.usage.current_usage_tokens == 25


@pytest.mark.asyncio
async def test_cached_input_contributes_to_prompt_occupancy():
    ctx = _ctx()
    client = _client([ToolCall(tool="search", args={})])

    async def send(*args, **kwargs):
        client.last_usage[0] = TokenUsage(
            5,
            7,
            12,
            cache_creation_input_tokens=100,
            cache_read_input_tokens=200,
        )
        return [ToolCall(tool="search", args={})]

    client.send = AsyncMock(side_effect=send)
    result = await _run_with_context(client, ctx)

    assert result is not None
    assert result.usage is client.last_usage[0]
    assert ctx.usage is not None
    assert ctx.usage.current_usage_tokens == 305


@pytest.mark.asyncio
async def test_stale_mirror_usage_is_not_returned_or_observed():
    ctx = _ctx()
    ctx.update_token_count(999)
    stale = TokenUsage(30, 10, 40)
    client = _client([ToolCall(tool="search", args={})])
    client.last_usage[0] = stale

    result = await _run_with_context(client, ctx)

    assert result is not None
    assert result.usage is None
    assert client.last_usage[0] is stale
    assert ctx.usage is None
    assert ctx.estimate_tokens([Message(
        MessageRole.USER, "x" * 40,
        MessageMeta(MessageType.USER_INPUT),
    )]) == 10


@pytest.mark.asyncio
async def test_equal_valued_replacement_is_fresh_attempt_usage():
    ctx = _ctx()
    old = TokenUsage(30, 10, 40)
    client = _client([ToolCall(tool="search", args={})])
    client.last_usage[0] = old

    async def send(*args, **kwargs):
        client.last_usage[0] = TokenUsage(30, 10, 40)
        return [ToolCall(tool="search", args={})]

    client.send = AsyncMock(side_effect=send)
    result = await _run_with_context(client, ctx)

    assert result is not None
    assert result.usage == old
    assert result.usage is not old
    assert ctx.usage is not None
    assert ctx.usage.current_usage_tokens == 30


@pytest.mark.asyncio
async def test_retry_invalidates_then_final_attempt_replaces_usage():
    ctx = _ctx()
    client = _client()
    usage_before_send = []

    async def send(*args, **kwargs):
        usage_before_send.append(ctx.usage)
        if len(usage_before_send) == 1:
            client.last_usage[0] = TokenUsage(10, 2, 12)
            return TextResponse(content="retry me")
        client.last_usage[0] = TokenUsage(20, 3, 23)
        return [ToolCall(tool="search", args={})]

    client.send = AsyncMock(side_effect=send)
    result = await _run_with_context(client, ctx)

    assert result is not None
    assert usage_before_send == [None, None]
    assert result.usage == TokenUsage(20, 3, 23)
    assert ctx.usage is not None
    assert ctx.usage.current_usage_tokens == 20
