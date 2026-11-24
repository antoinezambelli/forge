from forge.core.messages import RichMessage, ReasoningPart
from forge.core.codecs import argument_value
from tests.canonical import send, send_stream, calls as reply_calls, visible, reasoning

"""Tests for forge.clients.vllm — VLLMClient with mocked HTTP."""
import json
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest
from forge.clients.base import format_tool, ChunkType
from forge.clients.vllm import VLLMClient
from forge.proxy.handler import _extract_tool_definitions
from forge.core.workflow import ToolDefinition
from forge.errors import BackendError

pytestmark = pytest.mark.usefixtures("mock_httpx_client_constructor")
CITY_SCHEMA = {
    "properties": {"city": {"description": "City name", "title": "City", "type": "string"}},
    "required": ["city"],
    "title": "CityParams",
    "type": "object",
}


def _make_spec(name: str = "get_weather") -> ToolDefinition:
    return ToolDefinition(name=name, description=f"Get {name}", input_schema=CITY_SCHEMA)


def _make_client(*, think: bool = True) -> VLLMClient:
    return VLLMClient(
        model_path="/models/gemma-4-26B-A4B-it-AWQ-4bit",
        base_url="http://test:8000/v1",
        think=think,
    )


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = data
    resp.text = json.dumps(data)
    resp.status_code = status_code
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
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
            {"id": "call_1", "type": "function", "function": {"name": name, "arguments": args}}
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


class TestConstructor:
    def test_directory_path_derives_sampling_key_from_dirname(self) -> None:
        c = VLLMClient(model_path="/models/gemma-4-26B-A4B-it-AWQ-4bit")
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
            VLLMClient("/models/x", "http://other:8000")


class TestSend:
    @pytest.mark.asyncio
    async def test_custom_direct_api_base_is_literal_on_wire(self) -> None:
        client = VLLMClient(model_path="model", base_url="https://gateway.example/deploy/api")
        client._http = AsyncMock()
        client._http.headers = {}
        client._http.post.return_value = _mock_response(_text_response("ok"))
        await send(client, [{"role": "user", "content": "test"}])
        assert (
            client._http.post.await_args.args[0]
            == "https://gateway.example/deploy/api/chat/completions"
        )

    @pytest.mark.asyncio
    async def test_returns_tool_call(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(_tool_call_response())
        result = await send(
            client, [{"role": "user", "content": "weather in paris?"}], tools=[_make_spec()]
        )
        assert client._http.post.await_args.args[0] == "http://test:8000/v1/chat/completions"
        assert isinstance(result, RichMessage)
        assert reply_calls(result)[0].name == "get_weather"
        assert argument_value(reply_calls(result)[0]) == {"city": "Paris"}
        assert reasoning(result) == ""

    @pytest.mark.asyncio
    async def test_returns_text_response(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(_text_response("PONG"))
        result = await send(client, [{"role": "user", "content": "ping"}])
        assert isinstance(result, RichMessage)
        assert visible(result) == "PONG"

    @pytest.mark.asyncio
    async def test_arguments_parsed_from_json_string(self) -> None:
        """vLLM tool_calls.function.arguments arrives as a JSON-encoded string."""
        client = _make_client()
        client._http.post.return_value = _mock_response(
            _tool_call_response(args='{"city": "Paris", "units": "metric"}')
        )
        result = await send(client, [{"role": "user", "content": "x"}], tools=[_make_spec()])
        assert isinstance(result, RichMessage)
        assert argument_value(reply_calls(result)[0]) == {"city": "Paris", "units": "metric"}

    @pytest.mark.asyncio
    async def test_reasoning_field_captured(self) -> None:
        """vLLM 0.21 returns reasoning in `reasoning` field, not reasoning_content."""
        client = _make_client(think=True)
        client._http.post.return_value = _mock_response(
            _tool_call_response(reasoning="I should check the weather first.")
        )
        result = await send(client, [{"role": "user", "content": "x"}], tools=[_make_spec()])
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "I should check the weather first."

    @pytest.mark.asyncio
    async def test_think_false_discards_reasoning(self) -> None:
        client = _make_client(think=False)
        client._http.post.return_value = _mock_response(_tool_call_response(reasoning="thought"))
        result = await send(client, [{"role": "user", "content": "x"}], tools=[_make_spec()])
        assert isinstance(result, RichMessage)
        assert reasoning(result) == ""

    @pytest.mark.asyncio
    async def test_extracts_think_tags_from_content_with_tool_call(self) -> None:
        """#110: thinking inline in content (no `reasoning` field) is captured."""
        client = _make_client(think=True)
        client._http.post.return_value = _mock_response(
            _tool_call_response(content="<think>check the weather first</think>")
        )
        result = await send(client, [{"role": "user", "content": "x"}], tools=[_make_spec()])
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "check the weather first"

    @pytest.mark.asyncio
    async def test_reasoning_field_preferred_over_content_tags(self) -> None:
        """Structured `reasoning` field wins over <think> tags in content."""
        client = _make_client(think=True)
        client._http.post.return_value = _mock_response(
            _tool_call_response(reasoning="structured", content="<think>inline</think>")
        )
        result = await send(client, [{"role": "user", "content": "x"}], tools=[_make_spec()])
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "structured"

    @pytest.mark.asyncio
    async def test_think_tags_stripped_from_text_response(self) -> None:
        """A bare text reply has <think> tags stripped from its content."""
        client = _make_client()
        client._http.post.return_value = _mock_response(
            _text_response("<think>pondering</think>The answer is 42.")
        )
        result = await send(client, [{"role": "user", "content": "x"}])
        assert isinstance(result, RichMessage)
        assert visible(result) == "The answer is 42."

    @pytest.mark.asyncio
    async def test_thinking_only_text_response_empty_after_strip(self) -> None:
        """Thinking-only reply (no answer, no tool call) strips to empty content.

        Matches LlamafileClient; the empty TextResponse then rides the existing
        ResponseValidator retry path (covered in the validator tests).
        """
        client = _make_client()
        client._http.post.return_value = _mock_response(
            _text_response("<think>just thinking, no answer yet</think>")
        )
        result = await send(client, [{"role": "user", "content": "x"}])
        assert isinstance(result, RichMessage)
        assert visible(result) == ""

    @pytest.mark.asyncio
    async def test_usage_recorded(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            _tool_call_response(
                usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
            )
        )
        await send(client, [{"role": "user", "content": "x"}], tools=[_make_spec()])
        usage = client.last_usage[0]
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 120

    @pytest.mark.asyncio
    async def test_sampling_overrides_applied_per_call(self) -> None:
        client = _make_client()
        client.temperature = 1.0
        client._http.post.return_value = _mock_response(_text_response())
        await send(
            client, [{"role": "user", "content": "x"}], sampling={"temperature": 0.0, "seed": 42}
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["temperature"] == 0.0
        assert body["seed"] == 42
        assert client.temperature == 1.0

    @pytest.mark.asyncio
    async def test_non_200_raises_backend_error(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({"error": "bad"}, status_code=500)
        with pytest.raises(BackendError):
            await send(client, [{"role": "user", "content": "x"}])

    @pytest.mark.asyncio
    async def test_read_timeout_raises_backend_error(self) -> None:
        client = _make_client()
        client._http.post.side_effect = httpx.ReadTimeout("timeout")
        with pytest.raises(BackendError):
            await send(client, [{"role": "user", "content": "x"}])

    @pytest.mark.asyncio
    async def test_empty_choices_raises_backend_error(self) -> None:
        """Empty choices list — fail loud."""
        client = _make_client()
        client._http.post.return_value = _mock_response({"choices": [], "usage": {}})
        with pytest.raises(BackendError, match="no choices"):
            await send(client, [{"role": "user", "content": "x"}])

    @pytest.mark.asyncio
    async def test_tools_send_with_auto_tool_choice(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(_tool_call_response())
        await send(client, [{"role": "user", "content": "x"}], tools=[_make_spec()])
        body = client._http.post.call_args.kwargs["json"]
        assert body["tool_choice"] == "auto"
        assert len(body["tools"]) == 1
        assert body["tools"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_no_tools_omits_tools_param(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(_text_response())
        await send(client, [{"role": "user", "content": "x"}])
        body = client._http.post.call_args.kwargs["json"]
        assert "tools" not in body
        assert "tool_choice" not in body

    @pytest.mark.asyncio
    async def test_passthrough_raw_tools_and_owned_fields_are_applied(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(_text_response())
        messages = [{"role": "user", "content": "actual"}]
        nested_extension = {"route": "fast"}
        passthrough = {
            "model": "caller-alias",
            "messages": [{"role": "user", "content": "stale"}],
            "stream": True,
            "stream_options": {"include_usage": False},
            "max_tokens": 321,
            "stop": ["END"],
            "provider_extension": nested_extension,
            "temperature": 1.5,
            "tool_choice": "none",
            "tools": [{"stale": True}],
        }
        passthrough_before = deepcopy(passthrough)
        raw_tools = [
            {
                "type": "function",
                "function": {
                    "name": "custom",
                    "parameters": {"type": "object", "x-provider": "kept"},
                },
            }
        ]
        await send(
            client,
            messages,
            sampling={"temperature": 0.25},
            passthrough=passthrough,
            tools=_extract_tool_definitions(raw_tools),
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["model"] == client.model
        assert body["messages"] == messages
        assert body["stream"] is False
        assert "stream_options" not in body
        assert body["max_tokens"] == 321
        assert body["stop"] == ["END"]
        assert body["provider_extension"] == nested_extension
        assert body["temperature"] == 0.25
        assert body["tools"] == [format_tool(t) for t in _extract_tool_definitions(raw_tools)]
        assert body["tool_choice"] == "none"
        assert passthrough == passthrough_before

    @pytest.mark.asyncio
    async def test_authoritative_empty_raw_tools_does_not_default_choice(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(_text_response())
        await send(
            client,
            [{"role": "user", "content": "x"}],
            tools=[],
            passthrough={"tools": [{"stale": True}]},
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["tools"] == []
        assert "tool_choice" not in body

    @pytest.mark.asyncio
    async def test_no_tool_source_removes_stale_tools_but_keeps_choice(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(_text_response())
        await send(
            client,
            [{"role": "user", "content": "x"}],
            passthrough={
                "tools": [{"stale": True}],
                "tool_choice": {"type": "function", "function": {"name": "later"}},
            },
        )
        body = client._http.post.call_args.kwargs["json"]
        assert "tools" not in body
        assert body["tool_choice"]["function"]["name"] == "later"


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

    async def aread(self):
        return "".join([line async for line in self.aiter_lines()]).encode()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}"


class TestSendStream:
    @pytest.mark.asyncio
    async def test_yields_text_delta_then_final(self) -> None:
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse(
            [
                _sse({"choices": [{"delta": {"content": "PO"}}]}),
                _sse({"choices": [{"delta": {"content": "NG"}}]}),
                "data: [DONE]",
            ]
        )
        chunks = []
        async for chunk in send_stream(client, [{"role": "user", "content": "x"}]):
            chunks.append(chunk)
        assert client._http.stream.call_args.args[:2] == (
            "POST",
            "http://test:8000/v1/chat/completions",
        )
        deltas = [c for c in chunks if c.type == ChunkType.TEXT_DELTA]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert [c.content for c in deltas] == ["PO", "NG"]
        assert len(finals) == 1
        assert isinstance(finals[0].response, RichMessage)
        assert visible(finals[0].response) == "PONG"

    @pytest.mark.asyncio
    async def test_passthrough_raw_tools_and_stream_state_are_applied(self) -> None:
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse(
            [_sse({"choices": [{"delta": {"content": "ok"}}]}), "data: [DONE]"]
        )
        nested_extension = {"priority": 7}
        passthrough = {
            "model": "caller-alias",
            "stream": False,
            "stream_options": {"include_usage": False, "stale": True},
            "max_tokens": 222,
            "provider_extension": nested_extension,
            "tool_choice": "required",
        }
        passthrough_before = deepcopy(passthrough)
        raw_tools = [{"type": "function", "function": {"name": "raw"}}]
        async for _ in send_stream(
            client,
            [{"role": "user", "content": "x"}],
            passthrough=passthrough,
            tools=_extract_tool_definitions(raw_tools),
        ):
            pass
        body = client._http.stream.call_args.kwargs["json"]
        assert body["model"] == client.model
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert body["max_tokens"] == 222
        assert body["provider_extension"] == nested_extension
        assert body["tools"] == [format_tool(t) for t in _extract_tool_definitions(raw_tools)]
        assert body["tool_choice"] == "required"
        assert passthrough == passthrough_before

    @pytest.mark.asyncio
    async def test_yields_tool_call_delta_then_final(self) -> None:
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse(
            [
                _sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": '{"city": ',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ),
                _sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": '"Paris"}'}}
                                    ]
                                }
                            }
                        ]
                    }
                ),
                "data: [DONE]",
            ]
        )
        chunks = []
        async for chunk in send_stream(
            client, [{"role": "user", "content": "x"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)
        tc_deltas = [c for c in chunks if c.type == ChunkType.TOOL_CALL_DELTA]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(tc_deltas) == 2
        assert len(finals) == 1
        result = finals[0].response
        assert isinstance(result, RichMessage)
        assert reply_calls(result)[0].name == "get_weather"
        assert argument_value(reply_calls(result)[0]) == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_malformed_accumulated_args_kept_as_raw_args(self) -> None:
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse(
            [
                _sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": '{"city": ',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ),
                _sse({"choices": [{"delta": {"content": "let me try again"}}]}),
                "data: [DONE]",
            ]
        )
        chunks = []
        async for chunk in send_stream(
            client, [{"role": "user", "content": "x"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        result = finals[0].response
        assert isinstance(result, RichMessage)
        assert reply_calls(result)[0].name == "get_weather"
        assert argument_value(reply_calls(result)[0]) == '{"city": '

    @pytest.mark.asyncio
    async def test_accumulates_reasoning_across_deltas(self) -> None:
        client = _make_client(think=True)
        client._http.stream.return_value = _MockStreamResponse(
            [
                _sse({"choices": [{"delta": {"reasoning": "Let me "}}]}),
                _sse({"choices": [{"delta": {"reasoning": "think... "}}]}),
                _sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": '{"city": "P"}',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ),
                "data: [DONE]",
            ]
        )
        chunks = []
        async for chunk in send_stream(
            client, [{"role": "user", "content": "x"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        result = finals[0].response
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "Let me think... "

    @pytest.mark.asyncio
    async def test_stream_extracts_think_tags_from_content_with_tool_call(self) -> None:
        """#110 (streaming): inline <think> in streamed content (no reasoning
        deltas) is captured on the FINAL tool call."""
        client = _make_client(think=True)
        client._http.stream.return_value = _MockStreamResponse(
            [
                _sse({"choices": [{"delta": {"content": "<think>inline "}}]}),
                _sse({"choices": [{"delta": {"content": "plan</think>"}}]}),
                _sse(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": '{"city": "P"}',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ),
                "data: [DONE]",
            ]
        )
        chunks = []
        async for chunk in send_stream(
            client, [{"role": "user", "content": "x"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)
        result = [c for c in chunks if c.type == ChunkType.FINAL][0].response
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "inline plan"

    @pytest.mark.asyncio
    async def test_stream_strips_think_tags_from_text_response(self) -> None:
        """A streamed bare text reply has <think> tags stripped from FINAL."""
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse(
            [
                _sse({"choices": [{"delta": {"content": "<think>pondering</think>"}}]}),
                _sse({"choices": [{"delta": {"content": "The answer is 42."}}]}),
                "data: [DONE]",
            ]
        )
        chunks = []
        async for chunk in send_stream(client, [{"role": "user", "content": "x"}]):
            chunks.append(chunk)
        result = [c for c in chunks if c.type == ChunkType.FINAL][0].response
        assert isinstance(result, RichMessage)
        assert visible(result) == "The answer is 42."

    @pytest.mark.asyncio
    async def test_non_200_raises_backend_error(self) -> None:
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse(["bad gateway"], status_code=502)
        with pytest.raises(BackendError):
            async for _ in send_stream(client, [{"role": "user", "content": "x"}]):
                pass


class TestGetContextLength:
    @pytest.mark.asyncio
    async def test_exact_matches_model_in_multi_entry_catalog(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response(
            {
                "data": [
                    {"id": "other", "max_model_len": 8192},
                    {"id": "/models/gemma-4-26B-A4B-it-AWQ-4bit", "max_model_len": 113000},
                ]
            }
        )
        assert await client.get_context_length() == 113000

    @pytest.mark.asyncio
    async def test_missing_exact_model_returns_none(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response(
            {"data": [{"id": "other", "max_model_len": 8192}]}
        )
        assert await client.get_context_length() is None

    @pytest.mark.asyncio
    async def test_empty_catalog_returns_none(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({"data": []})
        assert await client.get_context_length() is None

    @pytest.mark.asyncio
    async def test_context_lookup_is_independent_of_invalid_first_identity(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response(
            {
                "data": [
                    {"max_model_len": 8192},
                    {"id": "/models/gemma-4-26B-A4B-it-AWQ-4bit", "max_model_len": 113000},
                ]
            }
        )
        assert await client.get_context_length() == 113000

    @pytest.mark.asyncio
    @pytest.mark.parametrize("window", [None, 0, -1, True, "113000", 1.5])
    async def test_unusable_exact_window_returns_none(self, window: object) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response(
            {"data": [{"id": "/models/gemma-4-26B-A4B-it-AWQ-4bit", "max_model_len": window}]}
        )
        assert await client.get_context_length() is None

    @pytest.mark.asyncio
    async def test_malformed_envelope_raises_safe_backend_error(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({"models": []})
        with pytest.raises(BackendError, match="malformed data"):
            await client.get_context_length()


class TestGetServedModelName:
    @pytest.mark.asyncio
    async def test_returns_first_model_id(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response(
            {"data": [{"id": "local-primary", "max_model_len": 262144}]}
        )
        assert await client.get_served_model_name() == "local-primary"
        client._http.get.assert_awaited_once_with("http://test:8000/v1/models")

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


class TestSendInstanceSampling:
    @pytest.mark.asyncio
    async def test_instance_sampling_applied_without_override(self) -> None:
        """Instance-level sampling fields flow into the request body."""
        client = _make_client()
        client.temperature = 0.5
        client.top_p = 0.85
        client._http.post.return_value = _mock_response(_text_response())
        await send(client, [{"role": "user", "content": "x"}])
        body = client._http.post.call_args.kwargs["json"]
        assert body["temperature"] == 0.5
        assert body["top_p"] == 0.85


class TestSendStreamEdgeCases:
    @pytest.mark.asyncio
    async def test_ignores_comment_and_empty_lines(self) -> None:
        """SSE comment lines (`:keepalive`) and empty lines are skipped."""
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse(
            [
                "",
                ":keepalive",
                _sse({"choices": [{"delta": {"content": "OK"}}]}),
                "",
                "data: [DONE]",
            ]
        )
        chunks = []
        async for chunk in send_stream(client, [{"role": "user", "content": "x"}]):
            chunks.append(chunk)
        deltas = [c for c in chunks if c.type == ChunkType.TEXT_DELTA]
        assert [c.content for c in deltas] == ["OK"]

    @pytest.mark.asyncio
    async def test_usage_only_chunk_records_usage_and_continues(self) -> None:
        """When stream_options.include_usage=True, vLLM emits a final chunk
        with `choices: []` and a `usage` block."""
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse(
            [
                _sse({"choices": [{"delta": {"content": "hi"}}]}),
                _sse(
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 50, "completion_tokens": 3, "total_tokens": 53},
                    }
                ),
                "data: [DONE]",
            ]
        )
        async for _ in send_stream(client, [{"role": "user", "content": "x"}]):
            pass
        usage = client.last_usage[0]
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 3
