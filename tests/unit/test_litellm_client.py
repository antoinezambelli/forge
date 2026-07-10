"""Tests for forge.clients.litellm — LiteLLMClient with mocked litellm SDK."""

import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from forge.clients.litellm import LiteLLMClient
from forge.clients.base import ChunkType
from forge.core.workflow import TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError


class PartParams(BaseModel):
    part: str = Field(description="Part number")


def _make_spec(name: str = "get_pricing") -> ToolSpec:
    return ToolSpec(name=name, description=f"Get {name}", parameters=PartParams)


def _make_client(model: str = "openai/gpt-4o", api_key: str = "tok") -> LiteLLMClient:
    return LiteLLMClient(model=model, api_key=api_key)


def _mock_message(content=None, tool_calls=None, **extra):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.model_dump.return_value = {"content": content, **(extra or {})}
    return msg


def _mock_response(content=None, tool_calls=None, usage=None, **msg_extra):
    resp = MagicMock()
    msg = _mock_message(content=content, tool_calls=tool_calls, **msg_extra)
    choice = MagicMock()
    choice.message = msg
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _mock_tool_call(name="get_pricing", arguments='{"part": "X123"}', index=0):
    tc = MagicMock()
    tc.index = index
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments
    tc.function = fn
    return tc


# ── send ─────────────────────────────────────────────────────────


class TestSend:
    @pytest.mark.asyncio
    async def test_returns_tool_call(self) -> None:
        client = _make_client()
        tc = _mock_tool_call()
        mock_resp = _mock_response(tool_calls=[tc])

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = mock_litellm

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
        mock_resp = _mock_response(content="I need more info")

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = mock_litellm

            result = await client.send([{"role": "user", "content": "test"}])
        assert isinstance(result, TextResponse)
        assert result.content == "I need more info"

    @pytest.mark.asyncio
    async def test_missing_choices_raises_backend_error(self) -> None:
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.choices = []

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = mock_litellm

            with pytest.raises(BackendError, match="response has no choices"):
                await client.send([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_null_content_returns_empty_text(self) -> None:
        client = _make_client()
        mock_resp = _mock_response(content=None)

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = mock_litellm

            result = await client.send([{"role": "user", "content": "test"}])
        assert isinstance(result, TextResponse)
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_drop_params_true_by_default(self) -> None:
        client = _make_client()
        mock_resp = _mock_response(content="ok")

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = mock_litellm

            await client.send([{"role": "user", "content": "test"}])
            call_kwargs = mock_litellm.acompletion.call_args.kwargs
            assert call_kwargs["drop_params"] is True

    @pytest.mark.asyncio
    async def test_drop_params_opt_out(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o", drop_params=False)
        mock_resp = _mock_response(content="ok")

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = mock_litellm

            await client.send([{"role": "user", "content": "test"}])
            call_kwargs = mock_litellm.acompletion.call_args.kwargs
            assert call_kwargs["drop_params"] is False

    @pytest.mark.asyncio
    async def test_api_key_forwarded(self) -> None:
        client = _make_client(api_key="sk-test-123")
        mock_resp = _mock_response(content="ok")

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = mock_litellm

            await client.send([{"role": "user", "content": "test"}])
            call_kwargs = mock_litellm.acompletion.call_args.kwargs
            assert call_kwargs["api_key"] == "sk-test-123"

    @pytest.mark.asyncio
    async def test_api_key_omitted_when_blank(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o")
        mock_resp = _mock_response(content="ok")

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = mock_litellm

            await client.send([{"role": "user", "content": "test"}])
            call_kwargs = mock_litellm.acompletion.call_args.kwargs
            assert "api_key" not in call_kwargs

    @pytest.mark.asyncio
    async def test_litellm_error_wrapped_as_backend_error(self) -> None:
        client = _make_client()

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            exc = type("APIError", (Exception,), {"status_code": 401})()
            exc.__module__ = "litellm.exceptions"
            mock_litellm.acompletion = AsyncMock(side_effect=exc)
            mock_import.return_value = mock_litellm

            with pytest.raises(BackendError):
                await client.send([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_records_usage(self) -> None:
        client = _make_client()
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 20
        usage.total_tokens = 30
        mock_resp = _mock_response(content="ok", usage=usage)

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = mock_litellm

            await client.send([{"role": "user", "content": "test"}])
        assert 0 in client.last_usage
        assert client.last_usage[0].prompt_tokens == 10
        assert client.last_usage[0].completion_tokens == 20

    @pytest.mark.asyncio
    async def test_sampling_overrides(self) -> None:
        client = LiteLLMClient(model="openai/gpt-4o", temperature=0.5)
        mock_resp = _mock_response(content="ok")

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_import.return_value = mock_litellm

            await client.send(
                [{"role": "user", "content": "test"}],
                sampling={"temperature": 0.9, "seed": 42},
            )
            call_kwargs = mock_litellm.acompletion.call_args.kwargs
            assert call_kwargs["temperature"] == 0.9
            assert call_kwargs["seed"] == 42


# ── streaming ────────────────────────────────────────────────────


def _mock_stream_chunk(content=None, tool_calls=None, usage=None):
    chunk = MagicMock()
    chunk.usage = usage
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    delta.model_dump.return_value = {"content": content}
    choice = MagicMock()
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


class TestSendStream:
    @pytest.mark.asyncio
    async def test_streams_text(self) -> None:
        client = _make_client()

        async def _fake_stream():
            yield _mock_stream_chunk(content="Hello")
            yield _mock_stream_chunk(content=" world")

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.acompletion = AsyncMock(return_value=_fake_stream())
            mock_import.return_value = mock_litellm

            chunks = []
            async for c in client.send_stream([{"role": "user", "content": "test"}]):
                chunks.append(c)

        text_chunks = [c for c in chunks if c.type == ChunkType.TEXT_DELTA]
        assert len(text_chunks) == 2
        assert text_chunks[0].content == "Hello"
        assert text_chunks[1].content == " world"

        final = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(final) == 1
        assert isinstance(final[0].response, TextResponse)
        assert final[0].response.content == "Hello world"

    @pytest.mark.asyncio
    async def test_streams_tool_calls(self) -> None:
        client = _make_client()

        def _tc_delta(name="", arguments="", index=0):
            tc = MagicMock()
            tc.index = index
            fn = MagicMock()
            fn.name = name
            fn.arguments = arguments
            tc.function = fn
            return tc

        async def _fake_stream():
            yield _mock_stream_chunk(tool_calls=[_tc_delta(name="get_pricing")])
            yield _mock_stream_chunk(tool_calls=[_tc_delta(arguments='{"part":')])
            yield _mock_stream_chunk(tool_calls=[_tc_delta(arguments=' "X123"}')])

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.acompletion = AsyncMock(return_value=_fake_stream())
            mock_import.return_value = mock_litellm

            chunks = []
            async for c in client.send_stream(
                [{"role": "user", "content": "test"}], tools=[_make_spec()]
            ):
                chunks.append(c)

        final = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(final) == 1
        result = final[0].response
        assert isinstance(result, list)
        assert result[0].tool == "get_pricing"
        assert result[0].args == {"part": "X123"}


# ── get_context_length ───────────────────────────────────────────


class TestContextLength:
    @pytest.mark.asyncio
    async def test_returns_max_input_tokens(self) -> None:
        client = _make_client()

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.get_model_info.return_value = {"max_input_tokens": 128000}
            mock_import.return_value = mock_litellm

            result = await client.get_context_length()
        assert result == 128000

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self) -> None:
        client = _make_client()

        with patch("forge.clients.litellm._import_litellm") as mock_import:
            mock_litellm = MagicMock()
            mock_litellm.get_model_info.side_effect = Exception("unknown model")
            mock_import.return_value = mock_litellm

            result = await client.get_context_length()
        assert result is None


# ── import error ─────────────────────────────────────────────────


class TestImportError:
    def test_missing_litellm_raises_helpful_message(self) -> None:
        with patch.dict(sys.modules, {"litellm": None}):
            with pytest.raises(ImportError, match="forge-guardrails\\[litellm\\]"):
                from forge.clients.litellm import _import_litellm
                _import_litellm()
