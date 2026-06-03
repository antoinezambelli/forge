"""Tests for forge.clients.openai_compat — OpenAICompatClient with mocked HTTP."""

import json

import pytest
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock, MagicMock

from forge.clients.openai_compat import OpenAICompatClient
from forge.clients.base import ChunkType
from forge.core.workflow import TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError


class PartParams(BaseModel):
    part: str = Field(description="Part number")


def _make_spec(name: str = "get_pricing") -> ToolSpec:
    return ToolSpec(name=name, description=f"Get {name}", parameters=PartParams)


def _make_client(model: str = "test-model", api_key: str = "tok") -> OpenAICompatClient:
    client = OpenAICompatClient(
        base_url="https://api.example.com/v1", model=model, api_key=api_key
    )
    mock_http = AsyncMock()
    mock_http.stream = MagicMock()  # sync method returning async context manager
    client._http = mock_http
    return client


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


class _MockStreamResponse:
    """Mock for httpx streaming response with aiter_lines / aread."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return "".join(self._lines).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# ── send ─────────────────────────────────────────────────────────


class TestSend:
    @pytest.mark.asyncio
    async def test_returns_tool_call(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_pricing", "arguments": '{"part": "X123"}'},
                    }],
                }
            }]
        })
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].tool == "get_pricing"
        assert result[0].args == {"part": "X123"}

    @pytest.mark.asyncio
    async def test_returns_text_response(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "I need more info"}}]
        })
        result = await client.send([{"role": "user", "content": "test"}])
        assert isinstance(result, TextResponse)
        assert result.content == "I need more info"

    @pytest.mark.asyncio
    async def test_missing_choices_raises_backend_error(self) -> None:
        # A broken provider envelope (200 with no choices) is a contract
        # violation, not a model mistake — fail loud and consistent rather
        # than KeyError/IndexError on data["choices"][0].
        client = _make_client()
        client._http.post.return_value = _mock_response({"object": "error"})
        with pytest.raises(BackendError, match="response has no choices"):
            await client.send([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_null_content_returns_empty_text(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{"message": {"role": "assistant", "content": None}}]
        })
        result = await client.send([{"role": "user", "content": "test"}])
        assert isinstance(result, TextResponse)
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_malformed_tool_args_kept_as_raw_args(self) -> None:
        # Fail-loud: malformed argument JSON must NOT become an executable
        # empty-args tool call. The raw (non-dict) string rides through on the
        # ToolCall so ResponseValidator routes it to the tool-error channel
        # rather than collapsing to a retry-nudge TextResponse.
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "function": {"name": "get_pricing", "arguments": "{not json"},
                    }],
                }
            }]
        })
        result = await client.send([{"role": "user", "content": "test"}])
        assert isinstance(result, list)
        assert result[0].tool == "get_pricing"
        assert result[0].args == "{not json"

    @pytest.mark.asyncio
    async def test_malformed_tool_args_kept_even_with_assistant_text(self) -> None:
        # tool_calls present means we parse tool calls: a malformed call rides
        # through as a bad-args ToolCall regardless of any sibling assistant
        # text (the text is not consulted), so the validator can correct it.
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Let me look that up.",
                    "tool_calls": [{
                        "function": {"name": "get_pricing", "arguments": "{not json"},
                    }],
                }
            }]
        })
        result = await client.send([{"role": "user", "content": "test"}])
        assert isinstance(result, list)
        assert result[0].args == "{not json"

    @pytest.mark.asyncio
    async def test_malformed_among_several_kept_per_call(self) -> None:
        # Parallel tool calls: a malformed sibling no longer collapses the whole
        # batch to text. Each call is parsed independently — the good one keeps
        # its dict args, the bad one keeps raw (non-dict) args for the validator.
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "get_pricing", "arguments": '{"part": "A"}'}},
                        {"function": {"name": "get_pricing", "arguments": "{broken"}},
                    ],
                }
            }]
        })
        result = await client.send([{"role": "user", "content": "test"}])
        assert isinstance(result, list)
        assert result[0].args == {"part": "A"}
        assert result[1].args == "{broken"

    @pytest.mark.asyncio
    async def test_dict_tool_args_accepted(self) -> None:
        # A provider that returns already-parsed (non-string) arguments is
        # non-compliant but unambiguous — accept the dict rather than failing.
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "function": {"name": "get_pricing", "arguments": {"part": "X123"}},
                    }],
                }
            }]
        })
        result = await client.send([{"role": "user", "content": "test"}])
        assert isinstance(result, list)
        assert result[0].args == {"part": "X123"}

    @pytest.mark.asyncio
    async def test_formats_tools_in_request(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })
        await client.send([{"role": "user", "content": "test"}], tools=[_make_spec()])

        body = client._http.post.call_args.kwargs["json"]
        assert "tools" in body
        tool = body["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_pricing"
        assert "parameters" in tool["function"]

    @pytest.mark.asyncio
    async def test_request_body_structure(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })
        await client.send([{"role": "user", "content": "hi"}])

        body = client._http.post.call_args.kwargs["json"]
        assert body["model"] == "test-model"
        assert body["stream"] is False
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        # No temperature passed → not in body
        assert "temperature" not in body

    @pytest.mark.asyncio
    async def test_posts_to_chat_completions(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })
        await client.send([{"role": "user", "content": "hi"}])
        url = client._http.post.call_args.args[0]
        assert url == "https://api.example.com/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_sampling_override(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })
        await client.send(
            [{"role": "user", "content": "hi"}], sampling={"temperature": 0.2}
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_http_error_raises_backend_error(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({"error": "bad"}, status_code=401)
        with pytest.raises(BackendError) as exc:
            await client.send([{"role": "user", "content": "test"}])
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_records_usage(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
        await client.send([{"role": "user", "content": "test"}])
        assert client.last_usage[0].prompt_tokens == 10
        assert client.last_usage[0].completion_tokens == 5
        assert client.last_usage[0].total_tokens == 15


# ── auth ─────────────────────────────────────────────────────────


class TestAuth:
    def test_bearer_header_set_when_key_provided(self) -> None:
        client = OpenAICompatClient(
            base_url="https://x/v1", model="m", api_key="secret"
        )
        assert client._http.headers["Authorization"] == "Bearer secret"

    def test_no_auth_header_when_no_key(self) -> None:
        client = OpenAICompatClient(base_url="https://x/v1", model="m")
        assert "Authorization" not in client._http.headers

    def test_base_url_trailing_slash_stripped(self) -> None:
        client = OpenAICompatClient(base_url="https://x/v1/", model="m")
        assert client.base_url == "https://x/v1"


# ── send_stream ──────────────────────────────────────────────────


class TestSendStream:
    @pytest.mark.asyncio
    async def test_yields_text_deltas_and_final(self) -> None:
        client = _make_client()
        lines = [
            'data: ' + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
            'data: ' + json.dumps({"choices": [{"delta": {"content": " world"}}]}),
            'data: [DONE]',
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)

        chunks = []
        async for chunk in client.send_stream([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        text_deltas = [c for c in chunks if c.type == ChunkType.TEXT_DELTA]
        assert [c.content for c in text_deltas] == ["Hello", " world"]

        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, TextResponse)
        assert finals[0].response.content == "Hello world"

    @pytest.mark.asyncio
    async def test_yields_final_with_tool_call(self) -> None:
        client = _make_client()
        lines = [
            'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "get_pricing", "arguments": ""}}
            ]}}]}),
            'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"part": '}}
            ]}}]}),
            'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '"X"}'}}
            ]}}]}),
            'data: [DONE]',
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, list)
        assert finals[0].response[0].tool == "get_pricing"
        assert finals[0].response[0].args == {"part": "X"}

    @pytest.mark.asyncio
    async def test_stream_malformed_tool_args_kept_as_raw_args(self) -> None:
        # Streaming counterpart: arg fragments that never assemble into valid
        # JSON ("{not" + " json") ride through as raw (non-dict) args on the
        # ToolCall — not a TextResponse final, and not a {}-args call.
        client = _make_client()
        lines = [
            'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "get_pricing", "arguments": "{not"}}
            ]}}]}),
            'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": " json"}}
            ]}}]}),
            'data: [DONE]',
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)

        chunks = [c async for c in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        result = finals[0].response
        assert isinstance(result, list)
        assert result[0].tool == "get_pricing"
        assert result[0].args == "{not json"

    @pytest.mark.asyncio
    async def test_stream_non_string_arg_fragment_not_dropped(self) -> None:
        # A non-compliant provider that streams the whole arguments object as a
        # single non-string fragment must not be silently skipped (which would
        # leave empty args). It is serialized into the buffer and recovered.
        client = _make_client()
        lines = [
            'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "get_pricing", "arguments": {"part": "X9"}}}
            ]}}]}),
            'data: [DONE]',
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)

        chunks = [c async for c in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, list)
        assert finals[0].response[0].args == {"part": "X9"}

    @pytest.mark.asyncio
    async def test_stream_http_error_raises(self) -> None:
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse(
            ['{"error": "nope"}'], status_code=500
        )
        with pytest.raises(BackendError) as exc:
            async for _ in client.send_stream([{"role": "user", "content": "x"}]):
                pass
        assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_ignores_non_data_lines(self) -> None:
        client = _make_client()
        lines = [
            '',
            ': keep-alive comment',
            'data: ' + json.dumps({"choices": [{"delta": {"content": "hi"}}]}),
            'data: [DONE]',
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = [c async for c in client.send_stream([{"role": "user", "content": "x"}])]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert finals[0].response.content == "hi"


class TestContextLength:
    @pytest.mark.asyncio
    async def test_returns_none(self) -> None:
        client = _make_client()
        assert await client.get_context_length() is None


# ── constructor ──────────────────────────────────────────────────


class TestConstructor:
    def test_extra_headers_merged_alongside_bearer(self) -> None:
        client = OpenAICompatClient(
            model="test-model",
            base_url="https://x/v1",
            api_key="tok",
            extra_headers={"HTTP-Referer": "https://example.com", "X-Title": "MyApp"},
        )
        # httpx normalizes header names to lowercase.
        assert client._http.headers["authorization"] == "Bearer tok"
        assert client._http.headers["http-referer"] == "https://example.com"
        assert client._http.headers["x-title"] == "MyApp"

    def test_extra_headers_can_override_authorization(self) -> None:
        client = OpenAICompatClient(
            model="test-model",
            base_url="https://x/v1",
            api_key="ignored",
            extra_headers={"Authorization": "ApiKey custom-scheme"},
        )
        assert client._http.headers["authorization"] == "ApiKey custom-scheme"

    def test_sampling_kwargs_stored_as_instance_fields(self) -> None:
        client = OpenAICompatClient(
            model="test-model",
            base_url="https://x/v1",
            top_k=40,
            min_p=0.05,
            repeat_penalty=1.1,
            presence_penalty=0.2,
            chat_template_kwargs={"enable_thinking": True},
        )
        assert client.top_k == 40
        assert client.min_p == 0.05
        assert client.repeat_penalty == 1.1
        assert client.presence_penalty == 0.2
        assert client.chat_template_kwargs == {"enable_thinking": True}

    def test_recommended_sampling_off_is_silent_for_unknown_model(self) -> None:
        # Default behavior: unknown model -> empty defaults, explicit kwargs flow through.
        client = OpenAICompatClient(
            model="definitely-not-in-registry-zzz",
            base_url="https://x/v1",
        )
        assert client.temperature is None
        assert client.top_p is None
        assert client.top_k is None

    def test_recommended_sampling_on_raises_for_unknown_model(self) -> None:
        from forge.errors import UnsupportedModelError
        with pytest.raises(UnsupportedModelError):
            OpenAICompatClient(
                model="definitely-not-in-registry-zzz",
                base_url="https://x/v1",
                recommended_sampling=True,
            )


# ── instance sampling flows into request body ────────────────────


class TestSendInstanceSampling:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("field,value", [
        ("top_k", 40),
        ("min_p", 0.05),
        ("repeat_penalty", 1.1),
        ("presence_penalty", 0.2),
        ("chat_template_kwargs", {"enable_thinking": True}),
    ])
    async def test_instance_sampling_flows_into_body(self, field, value) -> None:
        client = OpenAICompatClient(
            model="test-model",
            base_url="https://x/v1",
            api_key="tok",
            **{field: value},
        )
        mock_http = AsyncMock()
        client._http = mock_http
        mock_http.post.return_value = _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })
        await client.send([{"role": "user", "content": "hi"}])
        body = mock_http.post.call_args.kwargs["json"]
        assert body[field] == value


# ── passthrough + inbound_anthropic_body ─────────────────────────


class TestPassthrough:
    @pytest.mark.asyncio
    async def test_passthrough_fields_appear_in_body(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })
        await client.send(
            [{"role": "user", "content": "hi"}],
            passthrough={"max_tokens": 512, "stop": ["END"], "tool_choice": "auto"},
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["max_tokens"] == 512
        assert body["stop"] == ["END"]
        assert body["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_forge_owned_fields_override_passthrough(self) -> None:
        # Proxy may include "model"/"messages"/"stream" in passthrough; forge's
        # values must win to keep its invariants.
        client = _make_client(model="forge-model")
        client._http.post.return_value = _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })
        await client.send(
            [{"role": "user", "content": "hi"}],
            passthrough={
                "model": "evil",
                "messages": [{"role": "system", "content": "evil"}],
                "stream": True,
            },
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["model"] == "forge-model"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["stream"] is False

    @pytest.mark.asyncio
    async def test_inbound_anthropic_body_accepted_and_ignored(self) -> None:
        # Protocol shape compatibility: accept the kwarg, never let it leak
        # into the outbound OpenAI-shape body.
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })
        await client.send(
            [{"role": "user", "content": "hi"}],
            inbound_anthropic_body={"some_anthropic_field": "value", "shape": "data"},
        )
        body = client._http.post.call_args.kwargs["json"]
        assert "some_anthropic_field" not in body
        assert "shape" not in body

    @pytest.mark.asyncio
    async def test_send_stream_accepts_passthrough_and_inbound(self) -> None:
        client = _make_client()
        lines = [
            'data: ' + json.dumps({"choices": [{"delta": {"content": "ok"}}]}),
            'data: [DONE]',
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = [c async for c in client.send_stream(
            [{"role": "user", "content": "x"}],
            passthrough={"max_tokens": 64},
            inbound_anthropic_body={"ignored": True},
        )]
        # If the kwargs were rejected, we'd never get here.
        assert any(c.type == ChunkType.FINAL for c in chunks)


# ── aclose ───────────────────────────────────────────────────────


class TestAclose:
    @pytest.mark.asyncio
    async def test_aclose_closes_http_pool(self) -> None:
        client = _make_client()
        await client.aclose()
        client._http.aclose.assert_awaited_once()
