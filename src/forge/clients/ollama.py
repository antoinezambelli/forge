"""Ollama client adapter using native function calling."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from forge._endpoint_layouts import (
    BackendOperation,
    ConnectionInputKind,
    EndpointLayout,
    normalize_connection,
    resolve_endpoint,
)
from forge.clients.base import (
    ChunkType,
    StreamChunk,
    TokenUsage,
    _record_captured_usage,
    format_tool,
    resolve_request_headers,
    static_auth_present,
)
from forge.clients.sampling_defaults import apply_sampling_defaults
from forge.core.workflow import ToolDefinition
from forge.core.messages import ConversationState, RichMessage
from forge.clients.base import RequestMetadata, InferenceRequest
from forge.core.codecs import render_openai, replay_messages
from forge.clients.requests import request_extras
from forge.clients.replies import openai_reply, text_reply
from forge.errors import BackendError, ThinkingNotSupportedError


def _is_think_unsupported_error(status_code: int, body: str) -> bool:
    """Check if a response is Ollama's 'does not support thinking' error."""
    if status_code != 400:
        return False
    try:
        data = json.loads(body)
        return "does not support thinking" in data.get("error", "")
    except (json.JSONDecodeError, TypeError):
        return False


class OllamaClient:
    """Native function calling via Ollama's tools API.

    Uses Ollama's /api/chat endpoint with the tools parameter for
    structured function calling. Primary path for Mistral models.

    think parameter controls Ollama's thinking/reasoning mode:
        None (default) — use the backend default and retain returned reasoning
        True  — always send think=True (error if model doesn't support it)
        False — never send think and omit returned reasoning
    """

    api_format: str = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        presence_penalty: float | None = None,
        timeout: float = 300.0,
        think: bool | None = None,
        recommended_sampling: bool = False,
        api_key: str = "",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url
        connection = normalize_connection(
            base_url, ConnectionInputKind.OLLAMA_DAEMON_ROOT
        )
        self._chat_url = resolve_endpoint(
            EndpointLayout.OLLAMA_NATIVE,
            BackendOperation.INFERENCE,
            connection,
        )
        self.model = model
        # sampling_key is the registry-lookup key. For Ollama the wire "model"
        # field and the lookup key are the same string (the model tag).
        self.sampling_key = self.model
        # Apply per-model recommended sampling defaults. Caller's explicit
        # (non-None) kwargs win over the map field-by-field.
        defaults = apply_sampling_defaults(
            self.sampling_key, strict=recommended_sampling
        )
        self.temperature = (
            temperature if temperature is not None else defaults.get("temperature")
        )
        self.top_p = top_p if top_p is not None else defaults.get("top_p")
        self.top_k = top_k if top_k is not None else defaults.get("top_k")
        self.min_p = min_p if min_p is not None else defaults.get("min_p")
        self.repeat_penalty = (
            repeat_penalty
            if repeat_penalty is not None
            else defaults.get("repeat_penalty")
        )
        self.presence_penalty = (
            presence_penalty
            if presence_penalty is not None
            else defaults.get("presence_penalty")
        )
        # Optional backend auth (api_key → Authorization: Bearer; extra_headers
        # ride on top). Local Ollama is usually unauthenticated, but a gateway
        # in front of it may require a credential. One credential per request:
        # validate the static config (raises on api_key + a construction auth
        # header) BEFORE opening the pool so a conflict fails fast; a per-call
        # auth header is later refused when a static one is set here.
        self._static_auth = static_auth_present(api_key, extra_headers)
        headers: dict[str, str] = {}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._http = httpx.AsyncClient(headers=headers, timeout=timeout)
        self._num_ctx: int | None = None

        self._think = think
        self.last_usage: dict[int, TokenUsage] = {}

    async def aclose(self) -> None:
        """Close the underlying httpx connection pool."""
        await self._http.aclose()

    def _request_headers(
        self,
        extra_headers: dict[str, str] | None,
    ) -> dict[str, str] | None:
        """Per-call headers to apply, enforcing the one-credential rule.

        Returns the dict to pass as httpx ``headers=`` (merged over the
        construction headers, request winning), or None. Never mutates the
        shared client's construction headers.
        """
        return resolve_request_headers(self._static_auth, extra_headers)

    # Sampling fields recognized in per-call overrides. ``seed`` is
    # accepted only as a per-call override (not an instance field).
    _SAMPLING_FIELDS = (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "seed",
    )

    def _build_options(
        self,
        sampling: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the Ollama options dict.

        Instance fields supply the base sampling values; ``sampling`` (when
        provided) overrides per call. The instance is not mutated.
        """
        opts: dict[str, Any] = {}
        for field in self._SAMPLING_FIELDS:
            override = (sampling or {}).get(field)
            if override is not None:
                opts[field] = override
                continue
            instance_val = getattr(self, field, None)
            if instance_val is not None:
                opts[field] = instance_val
        if self._num_ctx is not None:
            opts["num_ctx"] = self._num_ctx
        return opts

    def _record_usage(self, data: dict[str, Any]) -> None:
        """Extract token usage from an Ollama response."""
        prompt = data.get("prompt_eval_count")
        completion = data.get("eval_count")
        if prompt is None and completion is None:
            return
        prompt = prompt or 0
        completion = completion or 0
        normalized = TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )
        self.last_usage[0] = normalized
        _record_captured_usage(normalized)

    def render_request(
        self,
        state: ConversationState,
        tools: list[ToolDefinition] | None,
        metadata: RequestMetadata,
        *,
        stream: bool = False,
    ) -> InferenceRequest:
        body = {
            "model": self.model,
            "messages": render_openai(
                replay_messages(state, metadata.reasoning_replay), ollama=True
            ),
            "stream": stream,
            "options": self._build_options(metadata.sampling),
        }
        if self._think is True:
            body["think"] = True
        if tools:
            body["tools"] = [format_tool(t) for t in tools]
        passthrough = request_extras(metadata, "openai")
        if "litellm_session_id" in passthrough:
            body["litellm_session_id"] = passthrough["litellm_session_id"]
        headers = httpx.Headers(self._http.headers)
        headers.update(self._request_headers(metadata.extra_headers) or {})
        return InferenceRequest(
            body,
            self._chat_url,
            dict(headers),
        )

    async def send(self, request: InferenceRequest) -> RichMessage:
        body = request.payload
        try:
            resp = await self._http.post(
                request.url, json=body, headers=request.headers
            )
        except httpx.ReadTimeout as exc:
            raise BackendError(408, "Read timeout") from exc
        if self._think is True and _is_think_unsupported_error(
            resp.status_code, resp.text
        ):
            raise ThinkingNotSupportedError(self.model, resp.status_code, resp.text)
        if resp.status_code == 500:
            return text_reply(resp.text)
        if resp.status_code != 200:
            raise BackendError(resp.status_code, raw_body=resp.text)
        data = resp.json()
        self._record_usage(data)
        return openai_reply(
            data.get("message", {}),
            capture_reasoning=self._think is not False,
            ollama=True,
        )

    async def send_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[StreamChunk]:
        body = request.payload
        async with self._http.stream(
            "POST", request.url, json=body, headers=request.headers
        ) as response:
            if response.status_code == 400:
                raw = (await response.aread()).decode(errors="replace")
                if self._think is True and _is_think_unsupported_error(400, raw):
                    raise ThinkingNotSupportedError(self.model, 400, raw)
                raise BackendError(400, raw_body=raw)
            async for chunk in self._iter_stream(response):
                yield chunk

    async def _iter_stream(
        self, response: httpx.Response
    ) -> AsyncIterator[StreamChunk]:
        if response.status_code == 500:
            raw = (await response.aread()).decode(errors="replace")
            yield StreamChunk(ChunkType.FINAL, response=text_reply(raw))
            return
        content, thinking, calls = "", "", []
        try:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                msg = data.get("message", {})
                content += msg.get("content") or ""
                thinking += msg.get("thinking") or ""
                if msg.get("tool_calls"):
                    calls = msg["tool_calls"]
                if data.get("done"):
                    self._record_usage(data)
                    reply = openai_reply(
                        {"content": content, "thinking": thinking, "tool_calls": calls},
                        capture_reasoning=self._think is not False,
                        ollama=True,
                    )
                    yield StreamChunk(ChunkType.FINAL, response=reply)
                elif msg.get("content"):
                    yield StreamChunk(ChunkType.TEXT_DELTA, content=msg["content"])
        except httpx.ReadTimeout as exc:
            raise BackendError(408, "Read timeout during streaming") from exc

    def set_num_ctx(self, num_ctx: int | None) -> None:
        """Set the num_ctx override sent on every request.

        Args:
            num_ctx: Token count, or None to use Ollama's default.
        """
        self._num_ctx = num_ctx

    async def get_context_length(self) -> int | None:
        """Return num_ctx if set via set_num_ctx(), None otherwise.

        Budget resolution lives in ServerManager, not here.
        """
        return self._num_ctx
