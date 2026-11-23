"""OpenAI-compatible client adapter using native function calling.

Works with any backend that exposes the OpenAI ``/v1/chat/completions``
endpoint: llama-server's OpenAI mode, Ollama's ``/v1`` shim, Cloudflare
Workers AI, Groq, Together, Fireworks, OpenRouter, OpenAI itself, etc.

This client is provider-agnostic by design. It knows the *protocol*
(base_url + bearer key + chat/completions), not any specific provider.
The caller is responsible for constructing the ``base_url`` and supplying
the ``api_key``.
"""

from __future__ import annotations

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
from forge.clients.replies import openai_reply, openai_stream
from forge.errors import BackendError


class OpenAICompatClient:
    """Native function calling via an OpenAI-compatible chat endpoint.

    Posts to ``{base_url}/chat/completions`` with the standard OpenAI
    request shape. Bearer auth is sent when ``api_key`` is provided
    (omit it for unauthenticated local servers). Provider-specific
    headers (e.g. OpenRouter's ``HTTP-Referer``) ride on
    ``extra_headers`` without a per-provider quirks registry.

    If a provider's quirks require diverging the parse or stream path,
    file an issue rather than adding if/else branches — we'll subclass
    or extract a base at that point.
    """

    api_format: str = "openai"

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str = "",
        extra_headers: dict[str, str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        presence_penalty: float | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        timeout: float = 120.0,
        recommended_sampling: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        connection = normalize_connection(
            self.base_url,
            ConnectionInputKind.OPENAI_API_BASE,
        )
        self._chat_url = resolve_endpoint(
            EndpointLayout.OPENAI_COMPAT,
            BackendOperation.INFERENCE,
            connection,
        )
        self.model = model
        # sampling_key is the registry-lookup key. For OpenAI-compat backends
        # the wire "model" field and the lookup key are the same string.
        self.sampling_key = self.model

        # Apply per-model recommended sampling defaults. Caller's explicit
        # (non-None) kwargs win over the map field-by-field. With
        # recommended_sampling=False (default) and an unknown model stem,
        # apply_sampling_defaults returns an empty dict silently — which
        # is the common case for hosted providers whose model identifiers
        # aren't in forge's registry.
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
        # chat_template_kwargs is a nested dict of Jinja template variables
        # — whole-value replacement at this field level (no nested merge).
        self.chat_template_kwargs = (
            chat_template_kwargs
            if chat_template_kwargs is not None
            else defaults.get("chat_template_kwargs")
        )

        # One credential per request: validate the static config BEFORE opening
        # the connection pool, so a double-credential conflict (api_key AND a
        # construction auth header) fails fast without leaking an unclosed
        # client. The returned bool records whether a static credential is set,
        # so a per-call auth header is later refused as a second source.
        self._static_auth = static_auth_present(api_key, extra_headers)
        # Auth header is set when api_key is provided. Non-auth extra_headers
        # (e.g. OpenRouter's HTTP-Referer) ride on top. For a non-Bearer auth
        # scheme, pass extra_headers alone and omit api_key.
        headers: dict[str, str] = {}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._http = httpx.AsyncClient(headers=headers, timeout=timeout)
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

    # ── request building ─────────────────────────────────────────────

    # Sampling fields recognized in per-call overrides. ``seed`` is
    # accepted only as a per-call override (not an instance field).
    # ``chat_template_kwargs`` is a nested dict — whole-value replacement
    # at this field level (no nested merge).
    _SAMPLING_FIELDS = (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "seed",
        "chat_template_kwargs",
    )

    def _build_body(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None,
        sampling: dict[str, Any] | None,
        stream: bool,
        passthrough: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Passthrough fields (max_tokens, stop, tool_choice, model, etc.
        # extracted by the proxy from the inbound body) seed the outbound
        # body first. Forge-owned fields then overlay on top so the
        # client's model/messages/stream/tools/sampling invariants win.
        body: dict[str, Any] = dict(passthrough or {})
        body["model"] = self.model
        body["messages"] = messages
        body["stream"] = stream
        for field in self._SAMPLING_FIELDS:
            override = (sampling or {}).get(field)
            if override is not None:
                body[field] = override
            else:
                instance_val = getattr(self, field, None)
                if instance_val is not None:
                    body[field] = instance_val
        if tools:
            body["tools"] = [format_tool(t) for t in tools]
        return body

    def _record_usage(self, data: dict[str, Any]) -> None:
        usage = data.get("usage")
        if not usage:
            return
        prompt = usage.get("prompt_tokens") or 0
        completion = usage.get("completion_tokens") or 0
        normalized = TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=usage.get("total_tokens") or (prompt + completion),
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
        messages = render_openai(replay_messages(state, metadata.reasoning_replay))
        body = self._build_body(
            messages,
            tools,
            metadata.sampling,
            stream,
            request_extras(metadata, "openai"),
        )
        return InferenceRequest(
            body,
            self._chat_url,
            {
                **dict(self._http.headers),
                **(self._request_headers(metadata.extra_headers) or {}),
            },
        )

    async def send(self, request: InferenceRequest) -> RichMessage:
        try:
            resp = await self._http.post(
                request.url, json=request.payload, headers=request.headers
            )
        except httpx.ReadTimeout as exc:
            raise BackendError(408, "Read timeout") from exc
        if resp.status_code != 200:
            raise BackendError(resp.status_code, raw_body=resp.text)
        data = resp.json()
        self._record_usage(data)
        choices = data.get("choices") or []
        if not choices:
            raise BackendError(500, f"response has no choices: {data}")
        return openai_reply(choices[0].get("message", {}), capture_reasoning=True)

    async def send_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[StreamChunk]:
        try:
            async with self._http.stream(
                "POST", request.url, json=request.payload, headers=request.headers
            ) as response:
                if response.status_code != 200:
                    raw = await response.aread()
                    raise BackendError(
                        response.status_code, raw_body=raw.decode(errors="replace")
                    )
                async for chunk in openai_stream(
                    response,
                    self._record_usage,
                    capture_reasoning=True,
                    emit_tool_deltas=False,
                ):
                    yield chunk
        except httpx.ReadTimeout as exc:
            raise BackendError(408, "Read timeout during streaming") from exc

    async def get_context_length(self) -> int | None:
        """OpenAI-compatible endpoints don't expose context length. Returns None."""
        return None
