"""Tests for forge.clients.ollama — OllamaClient with mocked HTTP."""

import json

from typing import Literal

import httpx
import pytest
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock, MagicMock

from forge.clients.ollama import OllamaClient
from forge.core.workflow import TextResponse, ToolCall, ToolSpec
from forge.clients.base import ChunkType, format_tool
from forge.errors import BackendError, ThinkingNotSupportedError


class PartParams(BaseModel):
    part: str = Field(description="Part number")


def _make_spec(name: str = "get_pricing") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Get {name}",
        parameters=PartParams,
    )


def _make_client(model: str = "test-model", think: bool | None = None) -> OllamaClient:
    """Create an OllamaClient with a mocked HTTP client."""
    client = OllamaClient(base_url="http://test:11434", model=model, think=think)
    mock_http = AsyncMock()
    # stream() is a sync method returning an async context manager, not a coroutine
    mock_http.stream = MagicMock()
    client._http = mock_http
    return client


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


# ── send ─────────────────────────────────────────────────────────


class TestOllamaSend:
    @pytest.mark.asyncio
    async def test_returns_tool_call(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_pricing", "arguments": {"part": "X123"}}}
                ],
            }
        })
        result = await client.send(
            [{"role": "user", "content": "test"}],
            tools=[_make_spec()],
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].tool == "get_pricing"
        assert result[0].args == {"part": "X123"}

    @pytest.mark.asyncio
    async def test_returns_text_response(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "message": {"role": "assistant", "content": "I need more info"}
        })
        result = await client.send([{"role": "user", "content": "test"}])
        assert isinstance(result, TextResponse)
        assert result.content == "I need more info"

    @pytest.mark.asyncio
    async def test_formats_tools_in_request(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "message": {"role": "assistant", "content": "ok"}
        })
        spec = _make_spec()
        await client.send([{"role": "user", "content": "test"}], tools=[spec])

        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "tools" in body
        tool = body["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_pricing"
        assert "parameters" in tool["function"]

    @pytest.mark.asyncio
    async def test_request_body_structure(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "message": {"role": "assistant", "content": "ok"}
        })
        await client.send([{"role": "user", "content": "hi"}])

        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["model"] == "test-model"
        assert body["stream"] is False
        assert "options" in body
        # Default constructor sends no temperature; backend default applies.
        assert "temperature" not in body["options"]

    @pytest.mark.asyncio
    async def test_tool_role_passes_through(self) -> None:
        """Messages with role='tool' are sent to Ollama unchanged."""
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "message": {"role": "assistant", "content": "ok"}
        })
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
        # Structured tool_calls pass through to Ollama
        assert sent_messages[2]["tool_calls"][0]["function"]["name"] == "fetch"
        assert sent_messages[3]["role"] == "tool"
        assert sent_messages[3]["content"] == "fetch_result"

    @pytest.mark.asyncio
    async def test_empty_tool_calls_returns_text(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "message": {"role": "assistant", "content": "thinking...", "tool_calls": []}
        })
        result = await client.send([{"role": "user", "content": "test"}])
        assert isinstance(result, TextResponse)

    @pytest.mark.asyncio
    async def test_captures_reasoning_with_tool_call(self) -> None:
        """When content accompanies tool_calls, it is captured as reasoning."""
        client = _make_client(think=True)
        client._http.post.return_value = _mock_response({
            "message": {
                "role": "assistant",
                "content": "I should look up the price for this part.",
                "tool_calls": [
                    {"function": {"name": "get_pricing", "arguments": {"part": "X123"}}}
                ],
            }
        })
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning == "I should look up the price for this part."

    @pytest.mark.asyncio
    async def test_thinking_preferred_over_content(self) -> None:
        """Thinking field (reasoning model) is preferred over content for reasoning."""
        client = _make_client(think=True)
        client._http.post.return_value = _mock_response({
            "message": {
                "role": "assistant",
                "content": "Final answer text",
                "thinking": "I need to calculate 17 * 23 first...",
                "tool_calls": [
                    {"function": {"name": "get_pricing", "arguments": {"part": "X"}}}
                ],
            }
        })
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning == "I need to calculate 17 * 23 first..."

    @pytest.mark.asyncio
    async def test_content_used_when_no_thinking(self) -> None:
        """Without thinking field, content is used as reasoning (instruct model)."""
        client = _make_client(think=True)
        client._http.post.return_value = _mock_response({
            "message": {
                "role": "assistant",
                "content": "Let me look that up.",
                "tool_calls": [
                    {"function": {"name": "get_pricing", "arguments": {"part": "X"}}}
                ],
            }
        })
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning == "Let me look that up."

    @pytest.mark.asyncio
    async def test_think_false_discards_reasoning(self) -> None:
        """think=False + content alongside tool_calls → reasoning is None."""
        client = _make_client(think=False)
        client._http.post.return_value = _mock_response({
            "message": {
                "role": "assistant",
                "content": "<think>Let me reason about this</think>",
                "tool_calls": [
                    {"function": {"name": "get_pricing", "arguments": {"part": "X"}}}
                ],
            }
        })
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning is None

    @pytest.mark.asyncio
    async def test_think_true_explicit(self) -> None:
        """think=True explicitly → always in request body."""
        client = _make_client(think=True)
        client._http.post.return_value = _mock_response({
            "message": {"role": "assistant", "content": "ok"}
        })
        await client.send([{"role": "user", "content": "test"}])
        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["think"] is True

    @pytest.mark.asyncio
    async def test_think_false_explicit(self) -> None:
        """think=False explicitly → never in request body."""
        client = _make_client(think=False)
        client._http.post.return_value = _mock_response({
            "message": {"role": "assistant", "content": "ok"}
        })
        await client.send([{"role": "user", "content": "test"}])
        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "think" not in body

    @pytest.mark.asyncio
    async def test_think_auto_heuristic_match(self) -> None:
        """think=None + 'reason' in model name → think=True in body."""
        client = _make_client(model="ministral-reasoning-14b")
        client._http.post.return_value = _mock_response({
            "message": {"role": "assistant", "content": "ok"}
        })
        await client.send([{"role": "user", "content": "test"}])
        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["think"] is True

    @pytest.mark.asyncio
    async def test_think_auto_no_match(self) -> None:
        """think=None + no keywords in model name → no think in body."""
        client = _make_client(model="ministral-instruct-14b")
        client._http.post.return_value = _mock_response({
            "message": {"role": "assistant", "content": "ok"}
        })
        await client.send([{"role": "user", "content": "test"}])
        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "think" not in body

    @pytest.mark.asyncio
    async def test_think_auto_fallback_on_unsupported(self) -> None:
        """think=None + heuristic says yes + 400 error → retries without think."""
        client = _make_client(model="ministral-reasoning-14b")
        error_resp = _mock_response(
            {"error": '"ministral-reasoning-14b" does not support thinking'},
            status_code=400,
        )
        ok_resp = _mock_response({
            "message": {"role": "assistant", "content": "ok"}
        })
        client._http.post.side_effect = [error_resp, ok_resp]
        result = await client.send([{"role": "user", "content": "test"}])
        assert isinstance(result, TextResponse)
        assert result.content == "ok"
        # Second call should not have think
        second_body = client._http.post.call_args_list[1].kwargs.get("json") or client._http.post.call_args_list[1][1].get("json")
        assert "think" not in second_body
        # State is now resolved
        assert client._think is False
        assert client._think_resolved is True

    @pytest.mark.asyncio
    async def test_think_true_explicit_raises_on_unsupported(self) -> None:
        """think=True explicit + model doesn't support thinking → ThinkingNotSupportedError."""
        client = _make_client(model="ministral-instruct-14b", think=True)
        error_resp = _mock_response(
            {"error": '"ministral-instruct-14b" does not support thinking'},
            status_code=400,
        )
        client._http.post.return_value = error_resp
        with pytest.raises(ThinkingNotSupportedError) as exc_info:
            await client.send([{"role": "user", "content": "test"}])
        assert exc_info.value.model == "ministral-instruct-14b"
        assert "does not support thinking" in str(exc_info.value)
        # Should NOT retry — only one call made
        assert client._http.post.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_content_gives_no_reasoning(self) -> None:
        """Empty string content alongside tool_calls → reasoning is None."""
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_pricing", "arguments": {"part": "X"}}}
                ],
            }
        })
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning is None

    @pytest.mark.asyncio
    async def test_missing_content_gives_no_reasoning(self) -> None:
        """No content key alongside tool_calls → reasoning is None."""
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "get_pricing", "arguments": {"part": "X"}}}
                ],
            }
        })
        result = await client.send(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, list)
        assert result[0].reasoning is None

    @pytest.mark.asyncio
    async def test_read_timeout_raises_backend_error(self) -> None:
        """httpx.ReadTimeout on send() → BackendError."""
        client = _make_client()
        client._http.post.side_effect = httpx.ReadTimeout("timed out")
        with pytest.raises(BackendError) as exc_info:
            await client.send([{"role": "user", "content": "test"}])
        assert exc_info.value.status_code == 408
        assert "Read timeout" in str(exc_info.value)


# ── send_stream ──────────────────────────────────────────────────


class _MockStreamResponse:
    """Mock for httpx streaming response with aiter_lines."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class TestOllamaSendStream:
    @pytest.mark.asyncio
    async def test_yields_text_deltas_and_final(self) -> None:
        client = _make_client()
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "Hello"}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": " world"}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}),
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)

        chunks = []
        async for chunk in client.send_stream([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        # Should have 2 text deltas + 1 final
        text_deltas = [c for c in chunks if c.type == ChunkType.TEXT_DELTA]
        assert len(text_deltas) == 2
        assert text_deltas[0].content == "Hello"
        assert text_deltas[1].content == " world"

        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, TextResponse)
        assert finals[0].response.content == "Hello world"

    @pytest.mark.asyncio
    async def test_yields_final_with_tool_call(self) -> None:
        client = _make_client()
        lines = [
            json.dumps({
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "get_pricing", "arguments": {"part": "X"}}}
                    ],
                },
                "done": True,
            }),
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].type == ChunkType.FINAL
        assert isinstance(chunks[0].response, list)
        assert chunks[0].response[0].tool == "get_pricing"

    @pytest.mark.asyncio
    async def test_tool_call_on_non_done_chunk(self) -> None:
        """Ollama sends tool_calls on a done=false chunk, then empty done=true."""
        client = _make_client()
        lines = [
            json.dumps({
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_abc", "function": {"index": 0, "name": "get_pricing", "arguments": {"part": "X"}}}
                    ],
                },
                "done": False,
            }),
            json.dumps({
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
            }),
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].type == ChunkType.FINAL
        assert isinstance(chunks[0].response, list)
        assert chunks[0].response[0].tool == "get_pricing"
        assert chunks[0].response[0].args == {"part": "X"}

    @pytest.mark.asyncio
    async def test_streaming_captures_reasoning_from_deltas(self) -> None:
        """Reasoning streamed as TEXT_DELTAs before tool_call is captured."""
        client = _make_client(think=True)
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "Let me "}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": "think..."}, "done": False}),
            json.dumps({
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "get_pricing", "arguments": {"part": "X"}}}
                    ],
                },
                "done": True,
            }),
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, list)
        assert final.response[0].reasoning == "Let me think..."

    @pytest.mark.asyncio
    async def test_streaming_thinking_preferred_over_content(self) -> None:
        """Streamed thinking tokens are preferred over content for reasoning."""
        client = _make_client(think=True)
        lines = [
            json.dumps({"message": {"role": "assistant", "thinking": "Let me "}, "done": False}),
            json.dumps({"message": {"role": "assistant", "thinking": "reason..."}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": "Final."}, "done": False}),
            json.dumps({
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "get_pricing", "arguments": {"part": "X"}}}
                    ],
                },
                "done": True,
            }),
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, list)
        assert final.response[0].reasoning == "Let me reason..."

    @pytest.mark.asyncio
    async def test_streaming_think_true_explicit(self) -> None:
        """think=True explicit → think in streaming request body."""
        client = _make_client(think=True)
        lines = [
            json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}),
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        async for _ in client.send_stream([{"role": "user", "content": "test"}]):
            pass
        call_args = client._http.stream.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["think"] is True

    @pytest.mark.asyncio
    async def test_streaming_think_false_explicit(self) -> None:
        """think=False explicit → no think in streaming request body."""
        client = _make_client(think=False)
        lines = [
            json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}),
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        async for _ in client.send_stream([{"role": "user", "content": "test"}]):
            pass
        call_args = client._http.stream.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "think" not in body

    @pytest.mark.asyncio
    async def test_streaming_think_false_discards_reasoning(self) -> None:
        """think=False + streamed content before tool_call → reasoning is None."""
        client = _make_client(think=False)
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "Let me "}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": "think..."}, "done": False}),
            json.dumps({
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "get_pricing", "arguments": {"part": "X"}}}
                    ],
                },
                "done": True,
            }),
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, list)
        assert final.response[0].reasoning is None

    @pytest.mark.asyncio
    async def test_streaming_no_reasoning_when_no_content(self) -> None:
        """No TEXT_DELTAs before tool_call → reasoning is None."""
        client = _make_client()
        lines = [
            json.dumps({
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "get_pricing", "arguments": {"part": "X"}}}
                    ],
                },
                "done": True,
            }),
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)

        chunks = []
        async for chunk in client.send_stream(
            [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)

        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, list)
        assert final.response[0].reasoning is None

    @pytest.mark.asyncio
    async def test_streaming_think_true_explicit_raises_on_unsupported(self) -> None:
        """think=True explicit + streaming + unsupported → ThinkingNotSupportedError."""
        client = _make_client(model="ministral-instruct-14b", think=True)
        error_body = json.dumps(
            {"error": '"ministral-instruct-14b" does not support thinking'}
        )
        client._http.stream.return_value = _MockStreamResponse(
            [error_body], status_code=400
        )
        with pytest.raises(ThinkingNotSupportedError) as exc_info:
            async for _ in client.send_stream([{"role": "user", "content": "test"}]):
                pass
        assert exc_info.value.model == "ministral-instruct-14b"
        assert "does not support thinking" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_streaming_think_auto_fallback_on_unsupported(self) -> None:
        """think=None + heuristic match + streaming 400 → falls back without think."""
        client = _make_client(model="ministral-reasoning-14b")
        error_body = json.dumps(
            {"error": '"ministral-reasoning-14b" does not support thinking'}
        )
        error_stream = _MockStreamResponse([error_body], status_code=400)
        ok_lines = [
            json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": True}),
        ]
        ok_stream = _MockStreamResponse(ok_lines)
        client._http.stream.side_effect = [error_stream, ok_stream]

        chunks = []
        async for chunk in client.send_stream([{"role": "user", "content": "test"}]):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert isinstance(chunks[0].response, TextResponse)
        assert chunks[0].response.content == "ok"
        assert client._think is False
        assert client._think_resolved is True

    @pytest.mark.asyncio
    async def test_streaming_read_timeout_raises_backend_error(self) -> None:
        """httpx.ReadTimeout mid-stream → BackendError."""

        class _TimeoutStreamResponse:
            status_code = 200

            async def aiter_lines(self):
                yield json.dumps({"message": {"role": "assistant", "content": "Hi"}, "done": False})
                raise httpx.ReadTimeout("timed out")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        client = _make_client()
        client._http.stream.return_value = _TimeoutStreamResponse()
        with pytest.raises(BackendError) as exc_info:
            async for _ in client.send_stream([{"role": "user", "content": "test"}]):
                pass
        assert exc_info.value.status_code == 408
        assert "timeout" in str(exc_info.value).lower()


# ── get_context_length + set_num_ctx ─────────────────────────────


class TestOllamaGetContextLength:
    """Tests for simplified get_context_length() — returns stored _num_ctx."""

    @pytest.mark.asyncio
    async def test_returns_none_by_default(self) -> None:
        """Fresh OllamaClient returns None (no budget set)."""
        client = _make_client()
        result = await client.get_context_length()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_num_ctx_when_set(self) -> None:
        """After set_num_ctx(), get_context_length() returns the value."""
        client = _make_client()
        client.set_num_ctx(8000)
        result = await client.get_context_length()
        assert result == 8000


class TestSetNumCtx:
    """Tests for set_num_ctx() public setter."""

    def test_set_value(self) -> None:
        client = _make_client()
        client.set_num_ctx(4096)
        assert client._num_ctx == 4096

    def test_set_none(self) -> None:
        client = _make_client()
        client.set_num_ctx(4096)
        client.set_num_ctx(None)
        assert client._num_ctx is None


class TestBuildOptions:
    """Tests for _build_options() with and without num_ctx."""

    def test_without_num_ctx(self) -> None:
        """Fresh client: options is empty (no temperature, no num_ctx)."""
        client = _make_client()
        opts = client._build_options()
        assert "temperature" not in opts
        assert "num_ctx" not in opts

    def test_with_num_ctx(self) -> None:
        """After setting num_ctx, it appears in options."""
        client = _make_client()
        client.set_num_ctx(8000)
        opts = client._build_options()
        assert opts["num_ctx"] == 8000

    def test_sampling_defaults_absent_when_none(self) -> None:
        """top_p/top_k/min_p/repeat_penalty/presence_penalty absent from options when unset."""
        client = _make_client()
        opts = client._build_options()
        assert "top_p" not in opts
        assert "top_k" not in opts
        assert "min_p" not in opts
        assert "repeat_penalty" not in opts
        assert "presence_penalty" not in opts

    def test_sampling_params_land_in_options(self) -> None:
        """All sampling kwargs propagate into the options dict when set."""
        client = OllamaClient(
            base_url="http://test:11434",
            model="test-model",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            repeat_penalty=1.05,
            presence_penalty=1.5,
        )
        opts = client._build_options()
        assert opts["temperature"] == 0.6
        assert opts["top_p"] == 0.95
        assert opts["top_k"] == 20
        assert opts["min_p"] == 0.0
        assert opts["repeat_penalty"] == 1.05
        assert opts["presence_penalty"] == 1.5


# ── format_tool ──────────────────────────────────────────────────


class TestFormatTool:
    def test_basic_format(self) -> None:
        spec = _make_spec()
        result = format_tool(spec)
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_pricing"
        assert "properties" in result["function"]["parameters"]
        assert "required" in result["function"]["parameters"]

    def test_enum_included(self) -> None:
        class SortOrderParams(BaseModel):
            order: Literal["asc", "desc"] = Field(description="Sort order")

        spec = ToolSpec(
            name="sort",
            description="Sort items",
            parameters=SortOrderParams,
        )
        result = format_tool(spec)
        props = result["function"]["parameters"]["properties"]
        assert props["order"]["enum"] == ["asc", "desc"]

    def test_optional_not_in_required(self) -> None:
        class SearchOptionalParams(BaseModel):
            query: str = Field(description="Query")
            limit: int | None = Field(default=None, description="Limit")

        spec = ToolSpec(
            name="search",
            description="Search",
            parameters=SearchOptionalParams,
        )
        result = format_tool(spec)
        required = result["function"]["parameters"]["required"]
        assert "query" in required
        assert "limit" not in required


class TestTemperatureOptional:
    """Issue C: temperature is optional; default constructor sends nothing."""

    @pytest.mark.asyncio
    async def test_no_temperature_when_default(self) -> None:
        """Default constructor (no temperature kwarg): options has no temperature field."""
        client = _make_client()
        client._http.post.return_value = _mock_response({
            "message": {"role": "assistant", "content": "ok"}
        })

        await client.send([{"role": "user", "content": "hi"}])

        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "temperature" not in body["options"]

    @pytest.mark.asyncio
    async def test_explicit_temperature_in_options(self) -> None:
        """Explicit temperature kwarg appears in options."""
        client = OllamaClient(
            base_url="http://test:11434", model="test-model", temperature=0.5,
        )
        mock_http = AsyncMock()
        mock_http.stream = MagicMock()
        client._http = mock_http
        client._http.post.return_value = _mock_response({
            "message": {"role": "assistant", "content": "ok"}
        })

        await client.send([{"role": "user", "content": "hi"}])

        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["options"]["temperature"] == 0.5
