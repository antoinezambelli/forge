"""Tests for forge.clients.vllm — VLLMClient with mocked HTTP."""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import BaseModel, Field

from forge.clients.base import ChunkType
from forge.clients.vllm import VLLMClient
from forge.core.workflow import TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError


# ── helpers ────────────────────────────────────────────────────


class CityParams(BaseModel):
    city: str = Field(description="City name")


def _make_spec(name: str = "get_weather") -> ToolSpec:
    return ToolSpec(name=name, description=f"Get {name}", parameters=CityParams)


def _make_client(*, think: bool = True) -> VLLMClient:
    client = VLLMClient(
        model_path="/models/gemma-4-26B-A4B-it-AWQ-4bit",
        base_url="http://test:8000/v1",
        think=think,
    )
    mock_http = AsyncMock()
    mock_http.stream = MagicMock()
    client._http = mock_http
    return client


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = data
    resp.text = json.dumps(data)
    resp.status_code = status_code
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp,
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _tool_call_response(
    name: str = "get_weather",
    args: str = '{"city": "Paris"}',
    reasoning: str | None = None,
    content: str | None = None,
    usage: dict | None = None,
) -> dict:
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    }
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _text_response(content: str = "hi", reasoning: str | None = None) -> dict:
    message = {"role": "assistant", "content": content, "tool_calls": []}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
    }


# ── constructor / identity derivation ──────────────────────────


class TestConstructor:
    def test_directory_path_derives_sampling_key_from_dirname(self) -> None:
        c = VLLMClient(model_path="/models/gemma-4-26B-A4B-it-AWQ-4bit")
        # model is the wire id (the path verbatim); sampling_key is the stem.
        assert c.model == "/models/gemma-4-26B-A4B-it-AWQ-4bit"
        assert c.sampling_key == "gemma-4-26B-A4B-it-AWQ-4bit"

    def test_hf_repo_id_derives_sampling_key_from_trailing_segment(self) -> None:
        c = VLLMClient(model_path="google/gemma-4-26B-A4B-it")
        assert c.model == "google/gemma-4-26B-A4B-it"
        assert c.sampling_key == "gemma-4-26B-A4B-it"

    def test_single_token_model_path(self) -> None:
        c = VLLMClient(model_path="some-local-name")
        assert c.model == "some-local-name"
        assert c.sampling_key == "some-local-name"

    def test_api_format_is_openai(self) -> None:
        c = VLLMClient(model_path="/models/x")
        assert c.api_format == "openai"

    def test_kwarg_only_after_model_path(self) -> None:
        """All params after model_path must be keyword-only."""
        with pytest.raises(TypeError):
            VLLMClient("/models/x", "http://other:8000")  # type: ignore[misc]


# ── send (non-streaming) ──────────────────────────────────────


class TestSend:
    @pytest.mark.asyncio
    async def test_returns_tool_call(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(_tool_call_response())
        result = await client.send(
            [{"role": "user", "content": "weather in paris?"}],
            tools=[_make_spec()],
        )
        assert isinstance(result, list)
        assert result[0].tool == "get_weather"
        assert result[0].args == {"city": "Paris"}
        assert result[0].reasoning is None

    @pytest.mark.asyncio
    async def test_returns_text_response(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(_text_response("PONG"))
        result = await client.send([{"role": "user", "content": "ping"}])
        assert isinstance(result, TextResponse)
        assert result.content == "PONG"

    @pytest.mark.asyncio
    async def test_arguments_parsed_from_json_string(self) -> None:
        """vLLM tool_calls.function.arguments arrives as a JSON-encoded string."""
        client = _make_client()
        client._http.post.return_value = _mock_response(
            _tool_call_response(args='{"city": "Paris", "units": "metric"}'),
        )
        result = await client.send(
            [{"role": "user", "content": "x"}], tools=[_make_spec()],
        )
        assert isinstance(result, list)
        assert result[0].args == {"city": "Paris", "units": "metric"}

    @pytest.mark.asyncio
    async def test_reasoning_field_captured(self) -> None:
        """vLLM 0.21 returns reasoning in `reasoning` field, not reasoning_content."""
        client = _make_client(think=True)
        client._http.post.return_value = _mock_response(
            _tool_call_response(reasoning="I should check the weather first."),
        )
        result = await client.send(
            [{"role": "user", "content": "x"}], tools=[_make_spec()],
        )
        assert isinstance(result, list)
        assert result[0].reasoning == "I should check the weather first."

    @pytest.mark.asyncio
    async def test_think_false_discards_reasoning(self) -> None:
        client = _make_client(think=False)
        client._http.post.return_value = _mock_response(
            _tool_call_response(reasoning="thought"),
        )
        result = await client.send(
            [{"role": "user", "content": "x"}], tools=[_make_spec()],
        )
        assert isinstance(result, list)
        assert result[0].reasoning is None

    @pytest.mark.asyncio
    async def test_extracts_think_tags_from_content_with_tool_call(self) -> None:
        """#110: thinking inline in content (no `reasoning` field) is captured."""
        client = _make_client(think=True)
        client._http.post.return_value = _mock_response(
            _tool_call_response(content="<think>check the weather first</think>"),
        )
        result = await client.send(
            [{"role": "user", "content": "x"}], tools=[_make_spec()],
        )
        assert isinstance(result, list)
        assert result[0].reasoning == "check the weather first"

    @pytest.mark.asyncio
    async def test_reasoning_field_preferred_over_content_tags(self) -> None:
        """Structured `reasoning` field wins over <think> tags in content."""
        client = _make_client(think=True)
        client._http.post.return_value = _mock_response(
            _tool_call_response(reasoning="structured", content="<think>inline</think>"),
        )
        result = await client.send(
            [{"role": "user", "content": "x"}], tools=[_make_spec()],
        )
        assert isinstance(result, list)
        assert result[0].reasoning == "structured"

    @pytest.mark.asyncio
    async def test_think_tags_stripped_from_text_response(self) -> None:
        """A bare text reply has <think> tags stripped from its content."""
        client = _make_client()
        client._http.post.return_value = _mock_response(
            _text_response("<think>pondering</think>The answer is 42."),
        )
        result = await client.send([{"role": "user", "content": "x"}])
        assert isinstance(result, TextResponse)
        assert result.content == "The answer is 42."

    @pytest.mark.asyncio
    async def test_thinking_only_text_response_empty_after_strip(self) -> None:
        """Thinking-only reply (no answer, no tool call) strips to empty content.

        Matches LlamafileClient; the empty TextResponse then rides the existing
        ResponseValidator retry path (covered in the validator tests).
        """
        client = _make_client()
        client._http.post.return_value = _mock_response(
            _text_response("<think>just thinking, no answer yet</think>"),
        )
        result = await client.send([{"role": "user", "content": "x"}])
        assert isinstance(result, TextResponse)
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_usage_recorded(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            _tool_call_response(
                usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            ),
        )
        await client.send([{"role": "user", "content": "x"}], tools=[_make_spec()])
        usage = client.last_usage[0]
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 120

    @pytest.mark.asyncio
    async def test_sampling_overrides_applied_per_call(self) -> None:
        client = _make_client()
        client.temperature = 1.0  # instance default
        client._http.post.return_value = _mock_response(_text_response())
        await client.send(
            [{"role": "user", "content": "x"}],
            sampling={"temperature": 0.0, "seed": 42},
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["temperature"] == 0.0
        assert body["seed"] == 42
        # Instance value not mutated
        assert client.temperature == 1.0

    @pytest.mark.asyncio
    async def test_non_200_raises_backend_error(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({"error": "bad"}, status_code=500)
        with pytest.raises(BackendError):
            await client.send([{"role": "user", "content": "x"}])

    @pytest.mark.asyncio
    async def test_read_timeout_raises_backend_error(self) -> None:
        client = _make_client()
        client._http.post.side_effect = httpx.ReadTimeout("timeout")
        with pytest.raises(BackendError):
            await client.send([{"role": "user", "content": "x"}])

    @pytest.mark.asyncio
    async def test_empty_choices_raises_backend_error(self) -> None:
        """Empty choices list — fail loud."""
        client = _make_client()
        client._http.post.return_value = _mock_response({"choices": [], "usage": {}})
        with pytest.raises(BackendError, match="no choices"):
            await client.send([{"role": "user", "content": "x"}])

    @pytest.mark.asyncio
    async def test_tools_send_with_auto_tool_choice(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(_tool_call_response())
        await client.send(
            [{"role": "user", "content": "x"}], tools=[_make_spec()],
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["tool_choice"] == "auto"
        assert len(body["tools"]) == 1
        assert body["tools"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_no_tools_omits_tools_param(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(_text_response())
        await client.send([{"role": "user", "content": "x"}])
        body = client._http.post.call_args.kwargs["json"]
        assert "tools" not in body
        assert "tool_choice" not in body


# ── send_stream ──────────────────────────────────────────────


class _MockStreamResponse:
    """Mocks the async context manager returned by httpx.AsyncClient.stream()."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self) -> "_MockStreamResponse":
        return self

    async def __aexit__(self, *args) -> None:
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}"


class TestSendStream:
    @pytest.mark.asyncio
    async def test_yields_text_delta_then_final(self) -> None:
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse([
            _sse({"choices": [{"delta": {"content": "PO"}}]}),
            _sse({"choices": [{"delta": {"content": "NG"}}]}),
            "data: [DONE]",
        ])
        chunks = []
        async for chunk in client.send_stream([{"role": "user", "content": "x"}]):
            chunks.append(chunk)
        deltas = [c for c in chunks if c.type == ChunkType.TEXT_DELTA]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert [c.content for c in deltas] == ["PO", "NG"]
        assert len(finals) == 1
        assert isinstance(finals[0].response, TextResponse)
        assert finals[0].response.content == "PONG"

    @pytest.mark.asyncio
    async def test_yields_tool_call_delta_then_final(self) -> None:
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse([
            _sse({"choices": [{"delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"name": "get_weather", "arguments": '{"city": '}
                }],
            }}]}),
            _sse({"choices": [{"delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": '"Paris"}'}
                }],
            }}]}),
            "data: [DONE]",
        ])
        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "x"}], tools=[_make_spec()],
        ):
            chunks.append(chunk)
        tc_deltas = [c for c in chunks if c.type == ChunkType.TOOL_CALL_DELTA]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(tc_deltas) == 2
        assert len(finals) == 1
        result = finals[0].response
        assert isinstance(result, list)
        assert result[0].tool == "get_weather"
        assert result[0].args == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_malformed_accumulated_args_kept_as_raw_args(self) -> None:
        # Streaming/non-streaming parity: once all fragments are accumulated,
        # an unparseable arguments string rides through on the ToolCall as raw
        # (non-dict) args so ResponseValidator routes it to the tool-error
        # channel — it must not raise out of the stream.
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse([
            _sse({"choices": [{"delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"name": "get_weather", "arguments": '{"city": '}
                }],
            }}]}),
            _sse({"choices": [{"delta": {"content": "let me try again"}}]}),
            "data: [DONE]",
        ])
        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "x"}], tools=[_make_spec()],
        ):
            chunks.append(chunk)
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        result = finals[0].response
        assert isinstance(result, list)
        assert result[0].tool == "get_weather"
        assert result[0].args == '{"city": '

    @pytest.mark.asyncio
    async def test_accumulates_reasoning_across_deltas(self) -> None:
        client = _make_client(think=True)
        client._http.stream.return_value = _MockStreamResponse([
            _sse({"choices": [{"delta": {"reasoning": "Let me "}}]}),
            _sse({"choices": [{"delta": {"reasoning": "think... "}}]}),
            _sse({"choices": [{"delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"name": "get_weather", "arguments": '{"city": "P"}'}
                }],
            }}]}),
            "data: [DONE]",
        ])
        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "x"}], tools=[_make_spec()],
        ):
            chunks.append(chunk)
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        result = finals[0].response
        assert isinstance(result, list)
        assert result[0].reasoning == "Let me think... "

    @pytest.mark.asyncio
    async def test_stream_extracts_think_tags_from_content_with_tool_call(self) -> None:
        """#110 (streaming): inline <think> in streamed content (no reasoning
        deltas) is captured on the FINAL tool call."""
        client = _make_client(think=True)
        client._http.stream.return_value = _MockStreamResponse([
            _sse({"choices": [{"delta": {"content": "<think>inline "}}]}),
            _sse({"choices": [{"delta": {"content": "plan</think>"}}]}),
            _sse({"choices": [{"delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"name": "get_weather", "arguments": '{"city": "P"}'}
                }],
            }}]}),
            "data: [DONE]",
        ])
        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "x"}], tools=[_make_spec()],
        ):
            chunks.append(chunk)
        result = [c for c in chunks if c.type == ChunkType.FINAL][0].response
        assert isinstance(result, list)
        assert result[0].reasoning == "inline plan"

    @pytest.mark.asyncio
    async def test_stream_strips_think_tags_from_text_response(self) -> None:
        """A streamed bare text reply has <think> tags stripped from FINAL."""
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse([
            _sse({"choices": [{"delta": {"content": "<think>pondering</think>"}}]}),
            _sse({"choices": [{"delta": {"content": "The answer is 42."}}]}),
            "data: [DONE]",
        ])
        chunks = []
        async for chunk in client.send_stream([{"role": "user", "content": "x"}]):
            chunks.append(chunk)
        result = [c for c in chunks if c.type == ChunkType.FINAL][0].response
        assert isinstance(result, TextResponse)
        assert result.content == "The answer is 42."

    @pytest.mark.asyncio
    async def test_non_200_raises_backend_error(self) -> None:
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse(
            ["bad gateway"], status_code=502,
        )
        with pytest.raises(BackendError):
            async for _ in client.send_stream([{"role": "user", "content": "x"}]):
                pass


# ── get_context_length ─────────────────────────────────────────


class TestGetContextLength:
    @pytest.mark.asyncio
    async def test_reads_max_model_len_from_models_endpoint(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({
            "data": [{"id": "/models/x", "max_model_len": 113000}],
        })
        ctx = await client.get_context_length()
        assert ctx == 113000

    @pytest.mark.asyncio
    async def test_empty_data_raises(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({"data": []})
        with pytest.raises(BackendError, match="no entries"):
            await client.get_context_length()

    @pytest.mark.asyncio
    async def test_missing_max_model_len_raises(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({
            "data": [{"id": "/models/x"}],  # no max_model_len
        })
        with pytest.raises(BackendError, match="missing max_model_len"):
            await client.get_context_length()


# ── get_served_model_name ──────────────────────────────────────


class TestGetServedModelName:
    @pytest.mark.asyncio
    async def test_returns_first_model_id(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({
            "data": [{"id": "local-primary", "max_model_len": 262144}],
        })
        assert await client.get_served_model_name() == "local-primary"

    @pytest.mark.asyncio
    async def test_empty_data_returns_none(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({"data": []})
        assert await client.get_served_model_name() is None

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self) -> None:
        client = _make_client()
        client._http.get.side_effect = httpx.ConnectError("refused")
        assert await client.get_served_model_name() is None


# ── discover_backend_metadata (deferred discovery) ─────────────


class TestDiscoverBackendMetadata:
    @pytest.mark.asyncio
    async def test_discovers_budget_and_adopts_identity(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({
            "data": [{"id": "google/gemma-4-26B-A4B-it", "max_model_len": 113000}],
        })
        budget = await client.discover_backend_metadata()
        assert budget == 113000
        # served id reaches the wire verbatim; registry key is the derived stem
        assert client.model == "google/gemma-4-26B-A4B-it"
        assert client.sampling_key == "gemma-4-26B-A4B-it"

    @pytest.mark.asyncio
    async def test_missing_id_raises(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({
            "data": [{"max_model_len": 113000}],  # no id
        })
        with pytest.raises(BackendError, match="missing id"):
            await client.discover_backend_metadata()

    @pytest.mark.asyncio
    async def test_missing_max_model_len_raises(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({
            "data": [{"id": "local-primary"}],  # no max_model_len
        })
        with pytest.raises(BackendError, match="missing max_model_len"):
            await client.discover_backend_metadata()

    @pytest.mark.asyncio
    async def test_empty_data_raises(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({"data": []})
        with pytest.raises(BackendError, match="no entries"):
            await client.discover_backend_metadata()

    @pytest.mark.asyncio
    async def test_non_200_raises_with_status_code(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({"error": "unauthorized"}, status_code=401)
        with pytest.raises(BackendError) as exc_info:
            await client.discover_backend_metadata()
        # status_code is carried so the proxy maps a 401 rejection → 401, not 502
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_connection_error_raises_502(self) -> None:
        client = _make_client()
        client._http.get.side_effect = httpx.ConnectError("refused")
        with pytest.raises(BackendError) as exc_info:
            await client.discover_backend_metadata()
        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_extra_headers_threaded_into_probe(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({
            "data": [{"id": "m", "max_model_len": 4096}],
        })
        extra = {"Authorization": "Bearer inbound-token"}
        await client.discover_backend_metadata(extra_headers=extra)
        # the per-request credential reaches the probe GET
        assert client._http.get.await_args.kwargs["headers"] == client._request_headers(extra)


# ── edge cases ─────────────────────────────────────────────────


class TestSendInstanceSampling:
    @pytest.mark.asyncio
    async def test_instance_sampling_applied_without_override(self) -> None:
        """Instance-level sampling fields flow into the request body."""
        client = _make_client()
        client.temperature = 0.5
        client.top_p = 0.85
        client._http.post.return_value = _mock_response(_text_response())
        await client.send([{"role": "user", "content": "x"}])
        body = client._http.post.call_args.kwargs["json"]
        assert body["temperature"] == 0.5
        assert body["top_p"] == 0.85


class TestSendStreamEdgeCases:
    @pytest.mark.asyncio
    async def test_ignores_comment_and_empty_lines(self) -> None:
        """SSE comment lines (`:keepalive`) and empty lines are skipped."""
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse([
            "",
            ":keepalive",
            _sse({"choices": [{"delta": {"content": "OK"}}]}),
            "",
            "data: [DONE]",
        ])
        chunks = []
        async for chunk in client.send_stream([{"role": "user", "content": "x"}]):
            chunks.append(chunk)
        deltas = [c for c in chunks if c.type == ChunkType.TEXT_DELTA]
        assert [c.content for c in deltas] == ["OK"]

    @pytest.mark.asyncio
    async def test_usage_only_chunk_records_usage_and_continues(self) -> None:
        """When stream_options.include_usage=True, vLLM emits a final chunk
        with `choices: []` and a `usage` block."""
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse([
            _sse({"choices": [{"delta": {"content": "hi"}}]}),
            _sse({
                "choices": [],
                "usage": {
                    "prompt_tokens": 50, "completion_tokens": 3, "total_tokens": 53,
                },
            }),
            "data: [DONE]",
        ])
        async for _ in client.send_stream([{"role": "user", "content": "x"}]):
            pass
        usage = client.last_usage[0]
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 3


class TestParseToolCalls:
    @staticmethod
    def _call(arguments: object) -> object:
        return VLLMClient._parse_tool_calls(
            [{"function": {"name": "lookup", "arguments": arguments}}],
            reasoning=None,
        )

    def test_string_args_decoded(self) -> None:
        """vLLM's native format — arguments arrive as a JSON string."""
        assert self._call('{"city": "Paris"}') == [
            ToolCall(tool="lookup", args={"city": "Paris"}),
        ]

    def test_dict_passed_through(self) -> None:
        """Some downstream wrappers send dict args directly — pass through."""
        assert self._call({"city": "Paris"}) == [
            ToolCall(tool="lookup", args={"city": "Paris"}),
        ]

    def test_empty_string_returns_empty_dict(self) -> None:
        """No-arg tool calls — empty string args is valid."""
        assert self._call("") == [ToolCall(tool="lookup", args={})]

    def test_malformed_json_kept_as_raw_args(self) -> None:
        """Malformed args are NOT coerced to {} or collapsed to a TextResponse:
        the raw (non-dict) string rides through on the ToolCall so
        ResponseValidator routes it to the tool-error channel."""
        assert self._call('{"city": ') == [
            ToolCall(tool="lookup", args='{"city": '),
        ]

    def test_unexpected_type_kept_as_raw_args(self) -> None:
        """Unknown shape (list, int, etc.) is a non-dict — kept for the
        validator's args-shape check, not raised as a BackendError."""
        assert self._call(123) == [ToolCall(tool="lookup", args=123)]

    def test_missing_function_is_defensive(self) -> None:
        """A broken tool-call entry (no "function") must not KeyError."""
        assert VLLMClient._parse_tool_calls(
            [{}], reasoning=None,
        ) == [ToolCall(tool="", args={})]

    def test_reasoning_attached_to_first_call_only(self) -> None:
        result = VLLMClient._parse_tool_calls(
            [
                {"function": {"name": "a", "arguments": "{}"}},
                {"function": {"name": "b", "arguments": "{}"}},
            ],
            reasoning="because",
        )
        assert result == [
            ToolCall(tool="a", args={}, reasoning="because"),
            ToolCall(tool="b", args={}, reasoning=None),
        ]
