"""Shared pytest fixtures and test doubles for the forge unit suite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

from forge.clients.base import ChunkType, StreamChunk
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec


# ── Shared LLM client double ─────────────────────────────────────


class MockClient:
    """Configurable mock ``LLMClient`` that replays scripted responses.

    A single behavioral superset of three doubles that had drifted apart
    across the suite. Behavior is selected entirely via constructor kwargs;
    the defaults reproduce the most common case (the runner tests).

    Constructor knobs:
        responses:
            Scripted ``ToolCall`` / ``TextResponse`` entries, consumed one
            per ``send`` / ``send_stream`` call. A bare ``ToolCall`` entry is
            auto-wrapped to ``[ToolCall]`` to match
            ``LLMResponse = list[ToolCall] | TextResponse``.
        on_exhausted:
            What to do once ``responses`` is exhausted. ``"raise"`` (default)
            raises ``IndexError``. Otherwise pass a fallback response object
            (e.g. ``TextResponse(content="stuck")``) which is returned
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
        responses: list[ToolCall | TextResponse],
        *,
        on_exhausted: Literal["raise"] | LLMResponse = "raise",
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
        self.send_calls: list[tuple[list[dict], list[ToolSpec] | None]] = []
        self.send_stream_calls: list[tuple[list[dict], list[ToolSpec] | None]] = []

    def _next(self) -> LLMResponse:
        if self._call_index >= len(self.responses):
            if self._on_exhausted == "raise":
                raise IndexError("MockClient: scripted responses exhausted")
            return self._on_exhausted
        resp = self.responses[self._call_index]
        self._call_index += 1
        if isinstance(resp, ToolCall):
            return [resp]
        return resp

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, object] | None = None,
        passthrough: dict[str, object] | None = None,
        inbound_anthropic_body: dict[str, object] | None = None,
    ) -> LLMResponse:
        self.send_calls.append((messages, tools))
        return self._next()

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, object] | None = None,
        passthrough: dict[str, object] | None = None,
        inbound_anthropic_body: dict[str, object] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        if self._stream_mode == "unsupported":
            raise NotImplementedError
        self.send_stream_calls.append((messages, tools))
        resp = self._next()
        if self._stream_mode == "deltas":
            yield StreamChunk(type=ChunkType.TEXT_DELTA, content="partial...")
        yield StreamChunk(type=ChunkType.FINAL, response=resp)

    async def get_context_length(self) -> int | None:
        return self._context_length
