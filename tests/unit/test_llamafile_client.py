"""Tests for forge.clients.llamafile — LlamafileClient with mocked HTTP."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

import httpx

from forge.clients.llamafile import LlamafileClient, _extract_think_tags, _merge_consecutive
from forge.core.workflow import TextResponse, ToolCall, ToolSpec
from pydantic import BaseModel, Field
from forge.clients.base import ChunkType


class PartParams(BaseModel):
    part: str = Field(description="Part number")


def _make_spec(name: str = "get_pricing") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Get {name}",
        parameters=PartParams,
    )


def _make_client(mode: str = "native", think: bool | None = None) -> LlamafileClient:
    """Create a LlamafileClient with a mocked HTTP client."""
    client = LlamafileClient(
        base_url="http://test:8080/v1", model="test-model", mode=mode, think=think
    )
    mock_http = AsyncMock()
    # stream() is a sync method returning an async context manager, not a coroutine
    mock_http.stream = MagicMock()
    client._http = mock_http
    return client


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx Response."""
    resp = MagicMock()
    resp.json.return_value = data
    resp.status_code = status_code
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _openai_tool_call_response(
    tool_name: str = "get_pricing", args: str = '{"part": "X"}'
) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": args,
                            },
                        }
                    ],
                }
            }
        ]
    }


def _openai_text_response(content: str = "Hello") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ]
    }


# ── send — native mode ──────────────────────────────────────────


class TestLlamafileNativeSend:
    @pytest.mark.asyncio
    async def test_returns_tool_call(self) -> None:
        client = _make_client("native")
        client._http.post.return_value = _mock_response(
            _openai_tool_call_response("get_pricing", '{"part": "X123"}')
        )
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].tool == "get_pricing"
        assert result[0].args == {"part": "X123"}

    @pytest.mark.asyncio
    async def test_returns_text_response(self) -> None:
        client = _make_client("native")
        client._http.post.return_value = _mock_response(
            _openai_text_response("I need more info")
        )
        result = await client.send([{"role": "user", "content": "test"}])
        assert isinstance(result, TextResponse)
        assert result.content == "I need more info"

    @pytest.mark.asyncio
    async def test_arguments_parsed_from_string(self) -> None:
        """OpenAI format sends arguments as JSON string, not dict."""
        client = _make_client("native")
        client._http.post.return_value = _mock_response(
            _openai_tool_call_response("get_pricing", '{"part": "X", "qty": 100}')
        )
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].args == {"part": "X", "qty": 100}

    @pytest.mark.asyncio
    async def test_captures_reasoning_with_tool_call(self) -> None:
        """When content accompanies tool_calls, it is captured as reasoning."""
        client = _make_client("native")
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I should check the pricing.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_pricing", "arguments": '{"part": "X"}'},
                    }],
                }
            }]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning == "I should check the pricing."

    @pytest.mark.asyncio
    async def test_reasoning_content_preferred_over_content(self) -> None:
        """reasoning_content field (reasoning model) is preferred over content."""
        client = _make_client("native")
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Final answer",
                    "reasoning_content": "I need to think about this carefully...",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_pricing", "arguments": '{"part": "X"}'},
                    }],
                }
            }]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning == "I need to think about this carefully..."

    @pytest.mark.asyncio
    async def test_content_used_when_no_reasoning_content(self) -> None:
        """Without reasoning_content, content is used as reasoning (instruct model)."""
        client = _make_client("native")
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_pricing", "arguments": '{"part": "X"}'},
                    }],
                }
            }]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning == "Let me check."

    @pytest.mark.asyncio
    async def test_null_content_gives_no_reasoning(self) -> None:
        """None/null content alongside tool_calls → reasoning is None."""
        client = _make_client("native")
        # _openai_tool_call_response uses content: None
        client._http.post.return_value = _mock_response(
            _openai_tool_call_response()
        )
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning is None

    @pytest.mark.asyncio
    async def test_empty_content_gives_no_reasoning(self) -> None:
        """Empty string content alongside tool_calls → reasoning is None."""
        client = _make_client("native")
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_pricing", "arguments": '{"part": "X"}'},
                    }],
                }
            }]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning is None

    @pytest.mark.asyncio
    async def test_native_mode_preserves_tool_role(self) -> None:
        """Native mode passes role='tool' and structured tool_calls through
        unchanged — llama-server with --jinja supports them natively."""
        client = _make_client("native")
        client._http.post.return_value = _mock_response(
            _openai_text_response("ok")
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "fetch", "arguments": {}}}
            ]},
            {"role": "tool", "content": "fetch_result"},
        ]
        await client.send(messages)

        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        sent_messages = body["messages"]
        # tool role preserved
        assert sent_messages[3]["role"] == "tool"
        assert sent_messages[3]["content"] == "fetch_result"
        # Structured tool_calls preserved
        assert "tool_calls" in sent_messages[2]
        assert sent_messages[2]["tool_calls"][0]["function"]["name"] == "fetch"


# ── send — prompt mode ──────────────────────────────────────────


class TestLlamafilePromptSend:
    @pytest.mark.asyncio
    async def test_extracts_tool_call_from_text(self) -> None:
        client = _make_client("prompt")
        # Model responds with JSON in its text output
        client._http.post.return_value = _mock_response(
            _openai_text_response('{"tool": "get_pricing", "args": {"part": "X"}}')
        )
        result = await client.send(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "test"}],
            tools=[_make_spec()],
        )
        assert isinstance(result, list)
        assert result[0].tool == "get_pricing"

    @pytest.mark.asyncio
    async def test_returns_text_when_no_json(self) -> None:
        client = _make_client("prompt")
        client._http.post.return_value = _mock_response(
            _openai_text_response("I don't know which tool to call")
        )
        result = await client.send(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "test"}],
            tools=[_make_spec()],
        )
        assert isinstance(result, TextResponse)

    @pytest.mark.asyncio
    async def test_injects_tool_prompt_into_system_message(self) -> None:
        client = _make_client("prompt")
        client._http.post.return_value = _mock_response(
            _openai_text_response("no tool call")
        )
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "test"},
        ]
        await client.send(messages, tools=[_make_spec()])

        # Check that the system message was modified
        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        sent_system = body["messages"][0]["content"]
        assert "get_pricing" in sent_system
        assert "You are helpful." in sent_system
        # Original should not be modified
        assert messages[0]["content"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_no_tools_parameter_in_request(self) -> None:
        client = _make_client("prompt")
        client._http.post.return_value = _mock_response(
            _openai_text_response("ok")
        )
        await client.send(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "test"}],
            tools=[_make_spec()],
        )
        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "tools" not in body

    @pytest.mark.asyncio
    async def test_tool_role_downgraded_in_prompt_mode(self) -> None:
        """Prompt mode also downgrades role='tool' to role='user'."""
        client = _make_client("prompt")
        client._http.post.return_value = _mock_response(
            _openai_text_response("no tool call")
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "fetch", "arguments": {}}}
            ]},
            {"role": "tool", "content": "fetch_result"},
        ]
        await client.send(messages, tools=[_make_spec()])

        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        sent_messages = body["messages"]
        # tool → user downgrade
        assert sent_messages[3]["role"] == "user"
        assert sent_messages[3]["content"] == "fetch_result"


# ── auto mode ────────────────────────────────────────────────────


class TestLlamafileAutoMode:
    @pytest.mark.asyncio
    async def test_auto_resolves_native_on_tool_call(self) -> None:
        client = _make_client("auto")
        assert client.resolved_mode is None
        client._http.post.return_value = _mock_response(
            _openai_tool_call_response()
        )
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert client.resolved_mode == "native"

    @pytest.mark.asyncio
    async def test_auto_stays_native_on_text_response(self) -> None:
        """TextResponse with tools provided means FC is supported but the
        model chose not to call a tool. Should resolve to native, not
        fall back to prompt mode."""
        client = _make_client("auto")

        client._http.post.return_value = _mock_response(
            _openai_text_response("Let me think about this...")
        )

        result = await client.send(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "test"}],
            tools=[_make_spec()],
        )
        assert client.resolved_mode == "native"
        assert isinstance(result, TextResponse)
        assert result.content == "Let me think about this..."

    @pytest.mark.asyncio
    async def test_auto_falls_back_on_http_error(self) -> None:
        client = _make_client("auto")

        # First call (native attempt) raises HTTP error
        error_resp = _mock_response({}, status_code=400)
        prompt_resp = _mock_response(
            _openai_text_response('{"tool": "get_pricing", "args": {}}')
        )
        client._http.post.side_effect = [error_resp, prompt_resp]

        result = await client.send(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "test"}],
            tools=[_make_spec()],
        )
        assert client.resolved_mode == "prompt"
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_auto_without_tools_defaults_native(self) -> None:
        client = _make_client("auto")
        client._http.post.return_value = _mock_response(
            _openai_text_response("hello")
        )
        result = await client.send([{"role": "user", "content": "hi"}])
        assert isinstance(result, TextResponse)
        assert client.resolved_mode == "native"

    @pytest.mark.asyncio
    async def test_send_stream_auto_resolves_native(self) -> None:
        """send_stream() probes mode before streaming when resolved_mode is None."""
        client = _make_client("auto")
        assert client.resolved_mode is None

        # Probe call resolves to native
        client._http.post.return_value = _mock_response(
            _openai_tool_call_response()
        )
        # Streaming call returns a tool call
        sse_lines = [
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_pricing", "arguments": ""}}]}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"part\": \"X\"}"}}]}}]})}',
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "test"}],
            tools=[_make_spec()],
        ):
            chunks.append(chunk)

        assert client.resolved_mode == "native"
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, list)

    @pytest.mark.asyncio
    async def test_send_stream_auto_falls_back_to_prompt(self) -> None:
        """send_stream() falls back to prompt mode when native probe fails."""
        client = _make_client("auto")
        assert client.resolved_mode is None

        # Probe: native fails with HTTP error, prompt fallback succeeds
        error_resp = _mock_response({}, status_code=400)
        prompt_resp = _mock_response(
            _openai_text_response('{"tool": "get_pricing", "args": {"part": "X"}}')
        )
        client._http.post.side_effect = [error_resp, prompt_resp]

        # Streaming call (now in prompt mode) returns extracted tool call
        sse_lines = [
            f'data: {json.dumps({"choices": [{"delta": {"content": "{\"tool\": \"get_pricing\", \"args\": {\"part\": \"Y\"}}"}, "finish_reason": "stop"}]})}',
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "test"}],
            tools=[_make_spec()],
        ):
            chunks.append(chunk)

        assert client.resolved_mode == "prompt"
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, list)
        assert finals[0].response[0].tool == "get_pricing"


# ── get_context_length ───────────────────────────────────────────


class TestLlamafileGetContextLength:
    @pytest.mark.asyncio
    async def test_parses_n_ctx(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({
            "default_generation_settings": {"n_ctx": 8192}
        })
        result = await client.get_context_length()
        assert result == 8192

    @pytest.mark.asyncio
    async def test_strips_v1_from_url(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({
            "default_generation_settings": {"n_ctx": 4096}
        })
        await client.get_context_length()
        call_args = client._http.get.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert "/v1" not in url
        assert url.endswith("/props")

    @pytest.mark.asyncio
    async def test_returns_none_on_missing_data(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({})
        result = await client.get_context_length()
        assert result is None

    @pytest.mark.asyncio
    async def test_propagates_error(self) -> None:
        client = _make_client()
        client._http.get.side_effect = Exception("connection error")
        with pytest.raises(Exception, match="connection error"):
            await client.get_context_length()


# ── send_stream ──────────────────────────────────────────────────


class _MockSSEStreamResponse:
    """Mock for httpx streaming response with SSE format."""

    status_code = 200

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class TestLlamafileSendStream:
    @pytest.mark.asyncio
    async def test_yields_text_deltas_and_final(self) -> None:
        client = _make_client("native")
        sse_lines = [
            f'data: {json.dumps({"choices": [{"delta": {"content": "Hello"}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"content": " world"}}]})}',
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "hi"}]
        ):
            chunks.append(chunk)

        text_deltas = [c for c in chunks if c.type == ChunkType.TEXT_DELTA]
        assert len(text_deltas) == 2
        assert text_deltas[0].content == "Hello"

        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, TextResponse)
        assert finals[0].response.content == "Hello world"

    @pytest.mark.asyncio
    async def test_yields_tool_call_from_stream(self) -> None:
        client = _make_client("native")
        sse_lines = [
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_pricing", "arguments": ""}}]}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"part\":"}}]}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": " \"X\"}"}}]}}]})}',
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        tc_deltas = [c for c in chunks if c.type == ChunkType.TOOL_CALL_DELTA]
        assert len(tc_deltas) >= 1

        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, list)
        assert finals[0].response[0].tool == "get_pricing"

    @pytest.mark.asyncio
    async def test_streaming_captures_reasoning_with_tool_call(self) -> None:
        """Content streamed before tool_calls is captured as reasoning."""
        client = _make_client("native")
        sse_lines = [
            f'data: {json.dumps({"choices": [{"delta": {"content": "Let me "}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"content": "check..."}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_pricing", "arguments": ""}}]}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"part\": \"X\"}"}}]}}]})}',
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, list)
        assert final.response[0].reasoning == "Let me check..."

    @pytest.mark.asyncio
    async def test_streaming_reasoning_content_preferred(self) -> None:
        """Streamed reasoning_content is preferred over content for reasoning."""
        client = _make_client("native")
        sse_lines = [
            f'data: {json.dumps({"choices": [{"delta": {"reasoning_content": "Let me "}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"reasoning_content": "reason..."}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"content": "Final."}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_pricing", "arguments": ""}}]}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"part\": \"X\"}"}}]}}]})}',
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, list)
        assert final.response[0].reasoning == "Let me reason..."

    @pytest.mark.asyncio
    async def test_streaming_no_reasoning_when_no_content(self) -> None:
        """Tool call stream with no content deltas → reasoning is None."""
        client = _make_client("native")
        sse_lines = [
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_pricing", "arguments": ""}}]}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"part\": \"X\"}"}}]}}]})}',
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, list)
        assert final.response[0].reasoning is None


# ── resolved_mode ────────────────────────────────────────────────


class TestResolvedMode:
    def test_native_mode_set_immediately(self) -> None:
        client = LlamafileClient(model="test", mode="native")
        assert client.resolved_mode == "native"

    def test_prompt_mode_set_immediately(self) -> None:
        client = LlamafileClient(model="test", mode="prompt")
        assert client.resolved_mode == "prompt"

    def test_auto_mode_unset(self) -> None:
        client = LlamafileClient(model="test", mode="auto")
        assert client.resolved_mode is None


# ── _apply_sampling ──────────────────────────────────────────────


class TestApplySampling:
    """Tests that sampling kwargs land in llama-server request bodies."""

    def test_sampling_absent_by_default(self) -> None:
        """Unset sampling params don't appear in the body."""
        client = LlamafileClient(model="test", mode="native")
        body: dict = {}
        client._apply_sampling(body)
        assert body == {}

    def test_sampling_params_populate_body(self) -> None:
        """All sampling kwargs land as top-level body fields when set."""
        client = LlamafileClient(
            model="test",
            mode="native",
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            repeat_penalty=1.05,
            presence_penalty=1.5,
        )
        body: dict = {}
        client._apply_sampling(body)
        assert body == {
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.05,
            "presence_penalty": 1.5,
        }

    @pytest.mark.asyncio
    async def test_native_send_includes_sampling(self) -> None:
        """_send_native request body includes sampling params."""
        client = _make_client(mode="native")
        client.top_p = 0.95
        client.top_k = 20
        client.min_p = 0.0
        client.repeat_penalty = 1.05
        client._http.post = AsyncMock(return_value=_mock_response({
            "choices": [{"message": {"content": "hi"}}],
        }))
        await client._send_native([{"role": "user", "content": "hi"}], None)
        sent_body = client._http.post.call_args.kwargs["json"]
        assert sent_body["top_p"] == 0.95
        assert sent_body["top_k"] == 20
        assert sent_body["min_p"] == 0.0
        assert sent_body["repeat_penalty"] == 1.05

    @pytest.mark.asyncio
    async def test_prompt_send_includes_sampling(self) -> None:
        """_send_prompt request body includes sampling params."""
        client = _make_client(mode="prompt")
        client.top_p = 0.95
        client._http.post = AsyncMock(return_value=_mock_response({
            "choices": [{"message": {"content": "hi"}}],
        }))
        await client._send_prompt([{"role": "user", "content": "hi"}], None)
        sent_body = client._http.post.call_args.kwargs["json"]
        assert sent_body["top_p"] == 0.95


# ── _merge_consecutive ────────────────────────────────────────────


class TestMergeConsecutive:
    def test_merges_consecutive_user_messages(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "retry nudge"},
        ]
        result = _merge_consecutive(messages)
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "hello\n\nretry nudge"

    def test_merges_consecutive_assistant_messages(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "I should check pricing."},
            {"role": "assistant", "content": "[tool_call] get_pricing({})"},
        ]
        result = _merge_consecutive(messages)
        assert len(result) == 3
        assert result[2]["role"] == "assistant"
        assert "I should check pricing." in result[2]["content"]
        assert "[tool_call]" in result[2]["content"]

    def test_does_not_merge_across_tool_calls(self) -> None:
        """Assistant messages with tool_calls are never merged."""
        messages = [
            {"role": "assistant", "content": "thinking..."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "fetch", "arguments": {}}}
            ]},
        ]
        result = _merge_consecutive(messages)
        assert len(result) == 2
        assert "tool_calls" in result[1]

    def test_does_not_merge_tool_role(self) -> None:
        """role='tool' messages are never merged."""
        messages = [
            {"role": "tool", "content": "result1"},
            {"role": "tool", "content": "result2"},
        ]
        result = _merge_consecutive(messages)
        assert len(result) == 2

    def test_preserves_correct_alternation(self) -> None:
        """Already-alternating messages pass through unchanged."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
        ]
        result = _merge_consecutive(messages)
        assert len(result) == 4

    def test_empty_messages(self) -> None:
        assert _merge_consecutive([]) == []

    def test_does_not_merge_into_tool_calls(self) -> None:
        """A plain assistant message before a tool_calls message stays separate."""
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "a", "arguments": {}}}
            ]},
            {"role": "assistant", "content": "reasoning"},
        ]
        result = _merge_consecutive(messages)
        assert len(result) == 2

    def test_merges_users_separated_by_invisible_messages(self) -> None:
        """User messages separated by assistant(tc)+tool are merged.

        The Jinja parity checker ignores assistant(tool_calls) and tool
        messages, so two user messages with only invisible messages between
        them are consecutive from the checker's perspective.
        """
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "fetch", "arguments": {}}}
            ]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "report", "arguments": {}}}
            ]},
            {"role": "user", "content": "step nudge"},
        ]
        result = _merge_consecutive(messages)
        # user messages merged, invisible messages preserved between
        user_msgs = [m for m in result if m["role"] == "user" and "tool_calls" not in m]
        assert len(user_msgs) == 1
        assert "go" in user_msgs[0]["content"]
        assert "step nudge" in user_msgs[0]["content"]


# ── _extract_think_tags ──────────────────────────────────────────


class TestExtractThinkTags:
    def test_extracts_single_block(self) -> None:
        text = "[THINK]I need to check pricing.[/THINK]Let me call the tool."
        reasoning, remaining = _extract_think_tags(text)
        assert reasoning == "I need to check pricing."
        assert remaining == "Let me call the tool."

    def test_extracts_multiple_blocks(self) -> None:
        text = "[THINK]First thought.[/THINK] middle [THINK]Second thought.[/THINK] end"
        reasoning, remaining = _extract_think_tags(text)
        assert reasoning == "First thought.\n\nSecond thought."
        assert remaining == "middle  end"

    def test_no_tags_returns_original(self) -> None:
        text = "Just plain content with no tags."
        reasoning, remaining = _extract_think_tags(text)
        assert reasoning == ""
        assert remaining == text

    def test_multiline_think_block(self) -> None:
        text = "[THINK]Line 1\nLine 2\nLine 3[/THINK]Result"
        reasoning, remaining = _extract_think_tags(text)
        assert "Line 1" in reasoning
        assert "Line 3" in reasoning
        assert remaining == "Result"

    def test_empty_think_block(self) -> None:
        text = "[THINK][/THINK]Content"
        reasoning, remaining = _extract_think_tags(text)
        assert reasoning == ""
        assert remaining == "Content"

    def test_empty_string(self) -> None:
        reasoning, remaining = _extract_think_tags("")
        assert reasoning == ""
        assert remaining == ""

    # ── <think> tag format (Qwen/DeepSeek) ──

    def test_extracts_xml_think_block(self) -> None:
        text = "<think>I should analyze the data.</think>Let me call the tool."
        reasoning, remaining = _extract_think_tags(text)
        assert reasoning == "I should analyze the data."
        assert remaining == "Let me call the tool."

    def test_extracts_multiple_xml_think_blocks(self) -> None:
        text = "<think>First.</think> middle <think>Second.</think> end"
        reasoning, remaining = _extract_think_tags(text)
        assert reasoning == "First.\n\nSecond."
        assert remaining == "middle  end"

    def test_multiline_xml_think_block(self) -> None:
        text = "<think>Line 1\nLine 2\nLine 3</think>Result"
        reasoning, remaining = _extract_think_tags(text)
        assert "Line 1" in reasoning
        assert "Line 3" in reasoning
        assert remaining == "Result"

    def test_empty_xml_think_block(self) -> None:
        text = "<think></think>Content"
        reasoning, remaining = _extract_think_tags(text)
        assert reasoning == ""
        assert remaining == "Content"

    def test_mixed_tag_formats(self) -> None:
        """Both [THINK] and <think> in same text (unlikely but should work)."""
        text = "[THINK]Mistral thought.[/THINK] <think>Qwen thought.</think> end"
        reasoning, remaining = _extract_think_tags(text)
        assert "Mistral thought." in reasoning
        assert "Qwen thought." in reasoning
        assert remaining == "end"


# ── think flag behavior — sync ────────────────────────────────────


class TestThinkFlagSync:
    @pytest.mark.asyncio
    async def test_think_true_captures_think_tags_as_reasoning(self) -> None:
        """think=True: [THINK] content parsed into ToolCall.reasoning."""
        client = _make_client("native", think=True)
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "[THINK]I should check pricing.[/THINK]",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_pricing", "arguments": '{"part": "X"}'},
                    }],
                }
            }]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning == "I should check pricing."

    @pytest.mark.asyncio
    async def test_think_false_discards_reasoning_content(self) -> None:
        """think=False: reasoning_content field discarded."""
        client = _make_client("native", think=False)
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Deep reasoning here...",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_pricing", "arguments": '{"part": "X"}'},
                    }],
                }
            }]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning is None

    @pytest.mark.asyncio
    async def test_think_false_discards_think_tags(self) -> None:
        """think=False: [THINK] tags in content discarded."""
        client = _make_client("native", think=False)
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "[THINK]Reasoning here[/THINK]",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_pricing", "arguments": '{"part": "X"}'},
                    }],
                }
            }]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning is None

    @pytest.mark.asyncio
    async def test_think_true_reasoning_content_preferred(self) -> None:
        """think=True: reasoning_content field preferred over [THINK] in content."""
        client = _make_client("native", think=True)
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "[THINK]Content reasoning[/THINK]",
                    "reasoning_content": "Server-parsed reasoning",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_pricing", "arguments": '{"part": "X"}'},
                    }],
                }
            }]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning == "Server-parsed reasoning"

    @pytest.mark.asyncio
    async def test_think_default_is_true(self) -> None:
        """think=None (auto) defaults to capturing reasoning."""
        client = _make_client("native")  # think=None → _think=True
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "[THINK]Auto-captured reasoning[/THINK]",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_pricing", "arguments": '{"part": "X"}'},
                    }],
                }
            }]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning == "Auto-captured reasoning"

    @pytest.mark.asyncio
    async def test_think_tags_stripped_from_text_response(self) -> None:
        """[THINK] tags are always stripped from TextResponse content."""
        client = _make_client("native", think=True)
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "[THINK]Some reasoning[/THINK]I don't know which tool to call.",
                }
            }]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await client.send([{"role": "user", "content": "test"}])
        assert isinstance(result, TextResponse)
        assert result.content == "I don't know which tool to call."
        assert "[THINK]" not in result.content

    @pytest.mark.asyncio
    async def test_content_fallback_when_no_think_tags(self) -> None:
        """think=True, no tags, no reasoning_content: content used as reasoning."""
        client = _make_client("native", think=True)
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Let me check the pricing.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_pricing", "arguments": '{"part": "X"}'},
                    }],
                }
            }]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning == "Let me check the pricing."


# ── think flag behavior — stream ──────────────────────────────────


class TestThinkFlagStream:
    @pytest.mark.asyncio
    async def test_stream_think_true_captures_think_tags(self) -> None:
        """Streaming: [THINK] tags in accumulated content → reasoning."""
        client = _make_client("native", think=True)
        sse_lines = [
            f'data: {json.dumps({"choices": [{"delta": {"content": "[THINK]I need to "}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"content": "check.[/THINK]"}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_pricing", "arguments": ""}}]}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"part\": \"X\"}"}}]}}]})}',
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, list)
        assert final.response[0].reasoning == "I need to check."

    @pytest.mark.asyncio
    async def test_stream_think_false_discards_reasoning(self) -> None:
        """Streaming: think=False discards reasoning_content."""
        client = _make_client("native", think=False)
        sse_lines = [
            f'data: {json.dumps({"choices": [{"delta": {"reasoning_content": "Thinking..."}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_pricing", "arguments": ""}}]}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"part\": \"X\"}"}}]}}]})}',
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, list)
        assert final.response[0].reasoning is None

    @pytest.mark.asyncio
    async def test_stream_think_false_discards_think_tags(self) -> None:
        """Streaming: think=False discards [THINK] tags from content."""
        client = _make_client("native", think=False)
        sse_lines = [
            f'data: {json.dumps({"choices": [{"delta": {"content": "[THINK]reasoning[/THINK]"}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_pricing", "arguments": ""}}]}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"part\": \"X\"}"}}]}}]})}',
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, list)
        assert final.response[0].reasoning is None

    @pytest.mark.asyncio
    async def test_stream_reasoning_content_preferred_over_tags(self) -> None:
        """Streaming: reasoning_content preferred over [THINK] in content."""
        client = _make_client("native", think=True)
        sse_lines = [
            f'data: {json.dumps({"choices": [{"delta": {"reasoning_content": "Server reasoning"}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"content": "[THINK]Content reasoning[/THINK]"}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_pricing", "arguments": ""}}]}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"part\": \"X\"}"}}]}}]})}',
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, list)
        assert final.response[0].reasoning == "Server reasoning"


# ── slot_id ────────────────────────────────────────────────────


class TestSlotId:
    """slot_id injection into request bodies."""

    def test_slot_id_stored(self) -> None:
        client = LlamafileClient(
            base_url="http://test:8080/v1", model="test", mode="native", slot_id=1
        )
        assert client._slot_id == 1

    def test_slot_id_default_none(self) -> None:
        client = LlamafileClient(
            base_url="http://test:8080/v1", model="test", mode="native"
        )
        assert client._slot_id is None

    def test_apply_slot_id_injects(self) -> None:
        client = LlamafileClient(
            base_url="http://test:8080/v1", model="test", mode="native", slot_id=1
        )
        body: dict = {"model": "test"}
        client._apply_slot_id(body)
        assert body["slot_id"] == 1

    def test_apply_slot_id_noop_when_none(self) -> None:
        client = LlamafileClient(
            base_url="http://test:8080/v1", model="test", mode="native"
        )
        body: dict = {"model": "test"}
        client._apply_slot_id(body)
        assert "slot_id" not in body

    @pytest.mark.asyncio
    async def test_native_send_includes_slot_id(self) -> None:
        client = LlamafileClient(
            base_url="http://test:8080/v1", model="test", mode="native", slot_id=1
        )
        mock_http = AsyncMock()
        client._http = mock_http

        tool_call_data = {
            "choices": [{
                "message": {
                    "content": "hello",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }],
        }
        mock_http.post.return_value = _mock_response(tool_call_data)

        await client.send(
            [{"role": "user", "content": "test"}], tools=None
        )

        call_kwargs = mock_http.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["slot_id"] == 1


class TestTemperatureOptional:
    """Issue C: temperature is optional; default constructor sends nothing."""

    @pytest.mark.asyncio
    async def test_no_temperature_when_default(self) -> None:
        """Default constructor (no temperature kwarg): outbound body has no temperature field."""
        client = _make_client(mode="native")
        client._http.post.return_value = _mock_response({
            "choices": [{
                "message": {"content": "ok", "tool_calls": None},
                "finish_reason": "stop",
            }],
        })

        await client.send([{"role": "user", "content": "hi"}], tools=None)

        call_kwargs = client._http.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "temperature" not in body

    @pytest.mark.asyncio
    async def test_explicit_temperature_in_body(self) -> None:
        """Explicit temperature kwarg appears in outbound body."""
        client = LlamafileClient(
            base_url="http://test:8080/v1",
            model="test-model",
            mode="native",
            temperature=0.5,
        )
        mock_http = AsyncMock()
        mock_http.stream = MagicMock()
        client._http = mock_http
        client._http.post.return_value = _mock_response({
            "choices": [{
                "message": {"content": "ok", "tool_calls": None},
                "finish_reason": "stop",
            }],
        })

        await client.send([{"role": "user", "content": "hi"}], tools=None)

        call_kwargs = client._http.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["temperature"] == 0.5
