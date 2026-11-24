"""vLLM client adapter using native function calling.

vLLM's HTTP API is OpenAI-compatible. Tool calling and reasoning extraction
both happen server-side (via ``--tool-call-parser`` and ``--reasoning-parser``
flags at server boot). The client consumes the structured response fields
``tool_calls`` (list) and ``reasoning`` (string) directly.

Differences from LlamafileClient:
- No prompt-mode injection path. vLLM parses tool calls server-side.
- No ``--jinja`` negotiation. vLLM uses the model's bundled chat template.
- Reasoning content arrives in ``reasoning`` (vLLM 0.21), not
  ``reasoning_content`` (llama.cpp's name).
- Context length is discovered via ``/v1/models`` (``max_model_len``),
  not ``/props``.
- The model identity is a path to a model directory (or HF repo id),
  not a single ``.gguf`` file. Constructor accepts ``model_path``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from forge._backend_profiles import parse_vllm_model_catalog
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

logger = logging.getLogger("forge.clients.vllm")


class VLLMClient:
    """Native function calling via vLLM's OpenAI-compatible API.

    Requires the vLLM server to be started with ``--enable-auto-tool-choice
    --tool-call-parser <name>`` for tool calling, and (for reasoning models)
    ``--reasoning-parser <name>`` to split thinking content into a separate
    response field. Without those flags, tool calls return 400 and reasoning
    arrives inline in ``content``.
    """

    api_format: str = "openai"

    def __init__(
        self,
        model_path: str | Path,
        *,
        base_url: str = "http://localhost:8000/v1",
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        presence_penalty: float | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        timeout: float = 300.0,
        think: bool = True,
        recommended_sampling: bool = False,
        api_key: str = "",
        extra_headers: dict[str, str] | None = None,
        adopt_served_identity: bool = True,
    ) -> None:
        self.base_url = base_url
        connection = normalize_connection(base_url, ConnectionInputKind.OPENAI_API_BASE)
        self._chat_url = resolve_endpoint(
            EndpointLayout.VLLM_OPENAI,
            BackendOperation.INFERENCE,
            connection,
        )
        self._models_url = resolve_endpoint(
            EndpointLayout.VLLM_OPENAI,
            BackendOperation.MODEL_CATALOG,
            connection,
        )
        # Two identity roles, set together (see _set_model_identity):
        #   self.model        — the wire "model" field, sent verbatim. For vLLM
        #                       this is the model path / HF repo id (or the
        #                       served-model-name once discovered in external
        #                       mode), which vLLM validates the request against.
        #   self.sampling_key — the derived registry-lookup key for
        #                       apply_sampling_defaults below (must be set first).
        self._set_model_identity(model_path)
        # Whether external-mode discovery may replace the identity above with
        # the backend-reported served name. The proxy sets this False when the
        # user pinned an explicit --model (issue #122): an explicit identity is
        # never overwritten, only warned about on mismatch.
        self._adopt_served_identity = adopt_served_identity

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
        # chat_template_kwargs is a nested dict of Jinja template variables
        # (e.g. {"enable_thinking": True}) that vLLM unpacks into the chat
        # template at render time. Whole-value replacement at the field
        # level — no nested merge.
        self.chat_template_kwargs = (
            chat_template_kwargs
            if chat_template_kwargs is not None
            else defaults.get("chat_template_kwargs")
        )
        # Optional backend auth (api_key → Authorization: Bearer; extra_headers
        # ride on top). vLLM servers may be started with --api-key. One
        # credential per request: validate the static config (raises on api_key
        # + a construction auth header) BEFORE opening the pool so a conflict
        # fails fast; a per-call auth header is later refused when a static one
        # is set here.
        self._static_auth = static_auth_present(api_key, extra_headers)
        headers: dict[str, str] = {}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._http = httpx.AsyncClient(headers=headers, timeout=timeout)
        self._think: bool = think
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

    @staticmethod
    def _derive_sampling_key(wire_id: str) -> str:
        """Derive the sampling-registry lookup key from the wire model id.

        vLLM's wire id is either a local directory (safetensors + config) or an
        HF repo id (e.g. "google/gemma-4-26B-A4B-it"). The lookup key uses the
        stem so registry lookups match the existing GGUF-stem convention:
        a filesystem path → its directory name; an HF repo id (has "/") → its
        trailing segment; anything else → the string unchanged.
        """
        path_obj = Path(wire_id)
        if path_obj.is_absolute() or path_obj.exists():
            return path_obj.name
        if "/" in wire_id:
            return wire_id.split("/")[-1]
        return wire_id

    def _set_model_identity(self, wire_id: str | Path) -> None:
        """Set both identity fields atomically from one wire id.

        ``model`` is the wire "model" field (sent verbatim); ``sampling_key``
        is the derived registry-lookup key. Used by ``__init__`` and by the
        proxy's external-mode served-name adoption, so the
        ``(model, sampling_key)`` invariant holds the same way in both —
        instead of mutating the two fields separately after served-name
        discovery.
        """
        self.model = str(wire_id)
        self.sampling_key = self._derive_sampling_key(self.model)

    # Sampling fields recognized in per-call overrides. ``seed`` is
    # accepted only as a per-call override (not an instance field).
    # ``chat_template_kwargs`` is a nested dict of Jinja template variables
    # — whole-value replacement at this field level (no nested merge).
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

    def _apply_sampling(
        self,
        body: dict[str, Any],
        sampling: dict[str, Any] | None = None,
    ) -> None:
        """Inject optional sampling params into a request body.

        Instance fields supply the base sampling values; ``sampling`` (when
        provided) overrides per call. The instance is not mutated. None =
        don't send; backend default applies.
        """
        for field in self._SAMPLING_FIELDS:
            override = (sampling or {}).get(field)
            if override is not None:
                body[field] = override
                continue
            instance_val = getattr(self, field, None)
            if instance_val is not None:
                body[field] = instance_val

    def _record_usage(self, data: dict[str, Any]) -> None:
        """Extract usage from a response."""
        usage = data.get("usage")
        if not usage:
            return
        normalized = TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
        self.last_usage[0] = normalized
        _record_captured_usage(normalized)

    def _build_request_body(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None,
        sampling: dict[str, Any] | None,
        passthrough: dict[str, Any] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """Build one OpenAI-shaped vLLM request without mutating inputs."""
        body: dict[str, Any] = dict(passthrough or {})
        body.update(
            {
                "model": self.model,
                "messages": messages,
                "stream": stream,
            }
        )
        if stream:
            body["stream_options"] = {"include_usage": True}
        else:
            body.pop("stream_options", None)

        if tools is not None:
            selected_tools = [format_tool(tool) for tool in tools]
            body["tools"] = selected_tools
        else:
            selected_tools = None
            body.pop("tools", None)

        if selected_tools and "tool_choice" not in body:
            body["tool_choice"] = "auto"

        # Sampling is forge-owned and therefore wins over passthrough values.
        self._apply_sampling(body, sampling)
        return body

    def render_request(
        self,
        state: ConversationState,
        tools: list[ToolDefinition] | None,
        metadata: RequestMetadata,
        *,
        stream: bool = False,
    ) -> InferenceRequest:
        messages = render_openai(replay_messages(state, metadata.reasoning_replay))
        body = self._build_request_body(
            messages,
            tools,
            metadata.sampling,
            request_extras(metadata, "openai"),
            stream=stream,
        )
        headers = httpx.Headers(self._http.headers)
        headers.update(self._request_headers(metadata.extra_headers) or {})
        return InferenceRequest(
            body,
            self._chat_url,
            dict(headers),
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
        return openai_reply(
            choices[0].get("message", {}), capture_reasoning=self._think
        )

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
                    response, self._record_usage, capture_reasoning=self._think
                ):
                    yield chunk
        except httpx.ReadTimeout as exc:
            raise BackendError(408, "Read timeout during streaming") from exc

    async def get_context_length(self) -> int | None:
        """Return this client's exact model window from /v1/models.

        A different catalog entry never supplies the denominator. Missing or
        unusable exact-model window metadata returns None; transport, status,
        and malformed-catalog failures remain loud.
        """
        try:
            resp = await self._http.get(self._models_url)
            resp.raise_for_status()
            catalog = parse_vllm_model_catalog(resp.json())
        except httpx.HTTPError:
            raise
        except (TypeError, ValueError) as exc:
            raise BackendError(500, "vLLM /v1/models returned malformed data") from exc
        return catalog.context_length_for(str(self.model))

    async def get_served_model_name(self) -> str | None:
        """Query /v1/models for the name vLLM is actually serving.

        vLLM validates the request ``model`` field against its
        ``--served-model-name`` aliases and returns 404 for an unknown name —
        unlike llama.cpp, which ignores the field entirely. In external mode
        the proxy has no model path to send, so it discovers the served
        identity here (the first ``data[].id``) rather than guessing.

        Returns None if the endpoint reports no models or is unreachable, in
        which case the caller keeps its placeholder identity.
        """
        try:
            resp = await self._http.get(self._models_url)
            resp.raise_for_status()
            catalog = parse_vllm_model_catalog(resp.json())
        except (httpx.HTTPError, TypeError, ValueError):
            return None
        return catalog.first_served_id
