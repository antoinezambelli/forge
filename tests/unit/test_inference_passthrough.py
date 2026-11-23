"""Inference lifecycle and per-attempt canonical rendering regressions."""

from copy import deepcopy
import pytest
from forge.clients.base import RequestMetadata, TokenUsage, ChunkType
from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.codecs import openai_state, anthropic_state, argument_value
from forge.core.inference import run_inference
from forge.core.messages import (
    TextPart,
    ReasoningPart,
    ToolCallPart,
    ToolResultPart,
    InvalidArguments,
)
from forge.core.workflow import ToolDefinition
from forge.errors import ToolCallError, StreamError
from forge.guardrails import ResponseValidator, ErrorTracker
from tests.conftest import MockClient
from tests.canonical import assistant, call, visible


def ctx(budget=100):
    return ContextManager(NoCompact(), budget)


async def infer(state, client, *, context=None, errors=None, **kwargs):
    return await run_inference(
        state,
        client,
        context or ctx(),
        ResponseValidator(["search"]),
        errors or ErrorTracker(),
        [
            ToolDefinition(
                "search",
                input_schema={
                    "$defs": {"X": {"type": "string"}},
                    "additionalProperties": False,
                },
            )
        ],
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_initial_and_retry_share_renderer_keep_fields_and_record_each_reply(
    stream,
):
    state = openai_state(
        [
            {
                "role": "user",
                "name": "caller",
                "content": [
                    {
                        "type": "text",
                        "text": "one",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "two", "x": 2},
                ],
            }
        ]
    )
    before = deepcopy(state.messages[0])
    rejected = assistant(TextPart("no call"), ReasoningPart("private"))
    accepted = assistant(TextPart("normal text"), ReasoningPart("reason"), call())
    client = MockClient([rejected, accepted])
    chunks = []

    async def on_chunk(chunk):
        chunks.append(chunk)

    result = await infer(state, client, stream=stream, on_chunk=on_chunk)
    requests = client.send_stream_calls if stream else client.send_calls
    assert len(requests) == 2
    assert requests[0].payload["messages"][0] == requests[1].payload["messages"][0]
    assert requests[0].payload["tools"] == requests[1].payload["tools"]
    assert (
        requests[0].payload["tools"][0]["function"]["parameters"][
            "additionalProperties"
        ]
        is False
    )
    assert state.messages == [before, rejected, state.messages[2], accepted]
    assert (
        visible(state.messages[2]).startswith("Your response")
        or "tool call" in visible(state.messages[2]).lower()
    )
    assert result.response is accepted and result.new_messages[-1] is accepted
    assert sum(m is accepted for m in state.messages) == 1
    assert sum(m is rejected for m in state.messages) == 1
    assert result.attempts == 2 and result.tool_call_counter == 1
    assert accepted.parts[-1].call_id == "call_000000000"
    if stream:
        assert chunks[-1].type == ChunkType.FINAL and chunks[-1].response is accepted


@pytest.mark.asyncio
async def test_rescue_only_visible_parts_joined_by_newline_and_recorded_as_calls():
    reply = assistant(
        ReasoningPart('{"tool":"search","args":{"wrong":true}}'),
        TextPart('{"tool":"search",'),
        TextPart('"args":{"q":"right"}}'),
    )
    state = openai_state([])
    client = MockClient([reply])
    result = await infer(state, client)
    assert result.attempts == 1
    assert len(client.send_calls) == 1 and state.messages == [reply]
    assert not any(isinstance(p, TextPart) for p in reply.parts)
    assert reply.parts[-1].arguments.value == {"q": "right"}
    assert isinstance(reply.parts[0], ReasoningPart)


@pytest.mark.asyncio
async def test_structured_calls_take_precedence_over_visible_rescue_encoding():
    reply = assistant(
        TextPart('{"tool":"search","args":{"wrong":true}}'),
        call(arguments={"q": "structured"}),
    )
    state = openai_state([])
    await infer(state, MockClient([reply]))
    assert len(reply.parts) == 2 and reply.parts[-1].arguments.value == {
        "q": "structured"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [[], None, False, 0, 0.0, ""])
async def test_invalid_calls_remain_correlated_on_retry_without_coercion(value):
    failed_call = ToolCallPart("search", InvalidArguments(value))
    failed = assistant(TextPart("ordinary"), failed_call)
    state = openai_state([])
    result = await infer(state, MockClient([failed, assistant(call())]))
    assert state.messages[0] is failed and argument_value(failed_call) == value
    result_part = state.messages[1].parts[0]
    assert isinstance(result_part, ToolResultPart) and result_part.outcome == "error"
    assert result_part.call_id == failed_call.call_id == "call_000000000"
    assert result_part.content[0].text.startswith("[ToolArgValidationError]")
    assert result.tool_call_counter == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["retry", "unknown", "arguments"])
async def test_error_budget_exhaustion_retains_final_reply_once_and_no_unused_correction(
    kind,
):
    def response():
        if kind == "retry":
            return assistant(TextPart("last text"))
        return assistant(
            ToolCallPart(
                "ghost" if kind == "unknown" else "search", InvalidArguments(False)
            )
        )

    first, last = response(), response()
    state = openai_state([])
    client = MockClient([first, last])
    with pytest.raises(
        ToolCallError,
        match="max_tool_errors=1" if kind == "arguments" else "max_retries=1",
    ) as error:
        await infer(
            state, client, errors=ErrorTracker(max_retries=1, max_tool_errors=1)
        )
    assert len(client.send_calls) == 2
    assert state.messages[-1] is last
    assert sum(m is last for m in state.messages) == 1 and len(state.messages) == 3
    assert (
        error.value.raw_response == "last text"
        if kind == "retry"
        else "False" in error.value.raw_response
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["retry", "arguments"])
async def test_max_attempts_returns_none_final_recorded_no_correction(kind):
    reply = (
        assistant(TextPart("last"))
        if kind == "retry"
        else assistant(ToolCallPart("search", InvalidArguments(0)))
    )
    state = openai_state([])
    client = MockClient([reply])
    assert await infer(state, client, max_attempts=1) is None
    assert state.messages == [reply] and len(client.send_calls) == 1


@pytest.mark.asyncio
async def test_zero_attempt_budget_never_sends():
    client = MockClient([])
    state = openai_state([])
    assert await infer(state, client, max_attempts=0) is None
    assert state.messages == [] and client.send_calls == []


@pytest.mark.asyncio
async def test_request_owned_usage_priority_invalidation_and_legacy_content_estimate():
    class UsageClient(MockClient):
        async def send(self, request):
            TokenUsage(7, 3, 10, 11, 13)
            return await super().send(request)

    state = openai_state([{"role": "user", "content": "x" * 40}])
    context = ctx()
    assert context.estimate_tokens(state) == 10
    result = await infer(
        state,
        UsageClient([assistant(call(arguments={"large": "x" * 10000}))]),
        context=context,
    )
    assert context.estimate_tokens(state) == 31
    assert result.usage.total_tokens == 10
    context.invalidate_usage()
    assert context.estimate_tokens(state) == 10


@pytest.mark.asyncio
async def test_threshold_warning_is_transient_and_rich_accounting_is_exercised_before_nocompact():
    state = openai_state([{"role": "user", "content": "x" * 40}])
    context = ContextManager(
        NoCompact(),
        10,
        context_thresholds=[0.5],
        on_context_threshold=lambda *_: "warning",
    )
    client = MockClient([assistant(call())])
    result = await infer(state, client, context=context)
    assert client.send_calls[0].payload["messages"][-1]["content"] == "warning"
    assert result.new_messages[0].forge_purpose == "context_warning"
    assert len(state.messages) == 2


@pytest.mark.asyncio
async def test_missing_stream_final_preserves_streamerror_and_does_not_retry():
    from forge.clients.base import StreamChunk

    class IncompleteStream(MockClient):
        async def send_stream(self, request):
            self.send_stream_calls.append(request)
            yield StreamChunk(ChunkType.TEXT_DELTA, content="partial")

    client = IncompleteStream([])
    state = openai_state([])
    with pytest.raises(StreamError, match="Stream ended without FINAL"):
        await infer(state, client, stream=True)
    assert len(client.send_stream_calls) == 1 and state.messages == []
