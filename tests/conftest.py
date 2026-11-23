"""Shared pytest fixtures and test doubles for the forge unit suite."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Literal
from unittest.mock import MagicMock

import pytest

from forge.clients.base import ChunkType, StreamChunk
from forge.core.messages import RichMessage
from forge.core.workflow import ToolDefinition
from forge.clients.base import InferenceRequest
from forge.clients.openai_compat import OpenAICompatClient
from tests.http_mocks import patch_httpx_client_constructor


@pytest.fixture
def mock_httpx_client_constructor() -> Iterator[MagicMock]:
    """Keep client-focused unit tests from opening unused connection pools."""
    with patch_httpx_client_constructor() as constructor:
        yield constructor


# ── Shared LLM client double ─────────────────────────────────────


class MockClient:
    """Configurable mock ``LLMClient`` that replays scripted responses.

    A single behavioral superset of three doubles that had drifted apart
    across the suite. Behavior is selected entirely via constructor kwargs;
    the defaults reproduce the most common case (the runner tests).

    Constructor knobs:
        responses:
            Scripted assistant RichMessages, consumed once per send/send_stream.
        on_exhausted:
            What to do once ``responses`` is exhausted. ``"raise"`` (default)
            raises ``IndexError``. Otherwise pass a fallback response object
            (e.g. an assistant RichMessage) which is returned
            instead of raising.
        stream_mode:
            How ``send_stream`` behaves. ``"deltas"`` (default) yields a
            ``TEXT_DELTA`` chunk then a ``FINAL`` chunk. ``"final"`` yields
            only the ``FINAL`` chunk. ``"unsupported"`` raises
            ``NotImplementedError``.
        api_format:
            Wire format string exposed on the instance (default ``"ollama"``).
        context_length:
            Value returned by ``get_context_length`` (default ``None``).

    Call-spy lists ``send_calls`` and ``send_stream_calls`` are always
    populated; call sites that never read them are unaffected.
    """

    def __init__(
        self,
        responses: list[RichMessage],
        *,
        on_exhausted: Literal["raise"] | RichMessage = "raise",
        stream_mode: Literal["deltas", "final", "unsupported"] = "deltas",
        api_format: str = "ollama",
        context_length: int | None = None,
    ):
        self.responses = list(responses)
        self._call_index = 0
        self._on_exhausted = on_exhausted
        self._stream_mode = stream_mode
        self.api_format = api_format
        self._context_length = context_length
        self.send_calls: list[tuple[list[dict], list[ToolDefinition] | None]] = []
        self.send_stream_calls: list[
            tuple[list[dict], list[ToolDefinition] | None]
        ] = []

    def _next(self) -> RichMessage:
        if self._call_index >= len(self.responses):
            if self._on_exhausted == "raise":
                raise IndexError("MockClient: scripted responses exhausted")
            return self._on_exhausted
        resp = self.responses[self._call_index]
        self._call_index += 1
        return resp

    def render_request(self, state, tools, metadata, *, stream=False):
        from forge.core.codecs import render_openai, replay_messages
        from forge.clients.base import format_tool

        return InferenceRequest(
            {
                "messages": render_openai(
                    replay_messages(state, metadata.reasoning_replay)
                ),
                "tools": [format_tool(t) for t in tools or []],
                "stream": stream,
            },
            "https://mock/chat",
            metadata.extra_headers,
        )

    async def send(self, request: InferenceRequest) -> RichMessage:
        self.send_calls.append(request)
        return self._next()

    async def send_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[StreamChunk]:
        if self._stream_mode == "unsupported":
            raise NotImplementedError
        self.send_stream_calls.append(request)
        response = self._next()
        if self._stream_mode == "deltas":
            yield StreamChunk(type=ChunkType.TEXT_DELTA, content="partial...")
        yield StreamChunk(type=ChunkType.FINAL, response=response)

    async def get_context_length(self) -> int | None:
        return self._context_length
