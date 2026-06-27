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

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from forge.clients.base import (
    ChunkType,
    StreamChunk,
    TokenUsage,
    decode_tool_args,
    format_tool,
    resolve_request_headers,
    static_auth_present,
)
from forge.clients.sampling_defaults import apply_sampling_defaults
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError
from forge.prompts.think_tags import extract_think_tags

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
    ) -> None:
        self.base_url = base_url
        # Two identity roles, set together (see _set_model_identity):
        #   self.model        — the wire "model" field, sent verbatim. For vLLM
        #                       this is the model path / HF repo id (or the
        #                       served-model-name once discovered in external
        #                       mode), which vLLM validates the request against.
        #   self.sampling_key — the derived registry-lookup key for
        #                       apply_sampling_defaults below (must be set first).
        self._set_model_identity(model_path)

        # Apply per-model recommended sampling defaults. Caller's explicit
        # (non-None) kwargs win over the map field-by-field.
        defaults = apply_sampling_defaults(self.sampling_key, strict=recommended_sampling)
        self.temperature = temperature if temperature is not None else defaults.get("temperature")
        self.top_p = top_p if top_p is not None else defaults.get("top_p")
        self.top_k = top_k if top_k is not None else defaults.get("top_k")
        self.min_p = min_p if min_p is not None else defaults.get("min_p")
        self.repeat_penalty = repeat_penalty if repeat_penalty is not None else defaults.get("repeat_penalty")
        self.presence_penalty = presence_penalty if presence_penalty is not None else defaults.get("presence_penalty")
        # chat_template_kwargs is a nested dict of Jinja template variables
        # (e.g. {"enable_thinking": True}) that vLLM unpacks into the chat
        # template at render time. Whole-value replacement at the field
        # level — no nested merge.
        self.chat_template_kwargs = (
            chat_template_kwargs if chat_template_kwargs is not None
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
        self, extra_headers: dict[str, str] | None,
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
        "temperature", "top_p", "top_k", "min_p",
        "repeat_penalty", "presence_penalty", "seed",
        "chat_template_kwargs",
    )

    def _apply_sampling(
        self, body: dict[str, Any], sampling: dict[str, Any] | None = None,
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
        self.last_usage[0] = TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def _resolve_reasoning(self, reasoning: str, content: str) -> str | None:
        """Build final reasoning from the structured field and content, gated
        on _think.

        vLLM 0.21 returns reasoning in the ``reasoning`` field of the assistant
        message when ``--reasoning-parser`` is enabled at server boot. When
        that parser is absent — or doesn't split a given model's output — the
        thinking instead arrives inline in ``content`` (often wrapped in
        ``<think>...</think>``). To avoid silently dropping it (issue #110) and
        to keep send() and send_stream() in lockstep with LlamafileClient, fall
        back to ``<think>``-tag extraction and then to the raw content when the
        structured field is empty. Both call sites pass the same (reasoning,
        content) pair, so the two paths resolve identically.
        """
        if not self._think:
            return None
        if reasoning:
            return reasoning
        think, _ = extract_think_tags(content)
        return think or content or None

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
        raw_openai_tools: list[dict[str, Any]] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Send messages via /v1/chat/completions and parse the response.

        ``passthrough`` / ``inbound_anthropic_body`` / ``raw_openai_tools`` are
        accepted for protocol symmetry and ignored — vLLM parses tools and
        reasoning server-side and is native-only. ``extra_headers`` carries the
        per-call credential, applied over the construction headers.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            body["tools"] = [format_tool(t) for t in tools]
            body["tool_choice"] = "auto"
        self._apply_sampling(body, sampling)

        try:
            resp = await self._http.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=self._request_headers(extra_headers),
            )
        except httpx.ReadTimeout as exc:
            raise BackendError(408, "Read timeout") from exc

        if resp.status_code != 200:
            raise BackendError(resp.status_code, raw_body=resp.text)
        data = resp.json()
        self._record_usage(data)

        choices = data.get("choices") or []
        if not choices:
            raise BackendError(500, f"vLLM response has no choices: {data}")
        message = choices[0].get("message", {})

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            return self._parse_tool_calls(
                tool_calls,
                reasoning=self._resolve_reasoning(
                    message.get("reasoning") or "", message.get("content") or "",
                ),
            )

        # No tool calls: strip any inline thinking — reasoning is only useful
        # attached to a ToolCall; a TextResponse carries clean content (parity
        # with LlamafileClient.send()).
        _, content = extract_think_tags(message.get("content") or "")
        return TextResponse(content=content)

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
        raw_openai_tools: list[dict[str, Any]] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream via SSE from /v1/chat/completions.

        ``passthrough`` / ``inbound_anthropic_body`` / ``raw_openai_tools``
        accepted for protocol symmetry and ignored (see ``send``).
        ``extra_headers`` carries the per-call credential.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = [format_tool(t) for t in tools]
            body["tool_choice"] = "auto"
        self._apply_sampling(body, sampling)

        accumulated_content = ""
        accumulated_reasoning = ""
        # Track multiple tool calls by index — OpenAI streaming sends
        # tool_calls[N] deltas with an index field.
        tool_call_parts: dict[int, dict[str, str]] = {}

        async with self._http.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=body,
            headers=self._request_headers(extra_headers),
        ) as response:
            if response.status_code != 200:
                error_body = ""
                async for line in response.aiter_lines():
                    error_body += line
                raise BackendError(response.status_code, raw_body=error_body)

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                chunk = json.loads(data_str)
                if "choices" not in chunk or not chunk["choices"]:
                    self._record_usage(chunk)
                    continue
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                if "tool_calls" in delta:
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta.get("index", 0)
                        if idx not in tool_call_parts:
                            tool_call_parts[idx] = {"name": "", "args": ""}
                        func = tc_delta.get("function", {})
                        if "name" in func:
                            tool_call_parts[idx]["name"] = func["name"]
                        if "arguments" in func:
                            tool_call_parts[idx]["args"] += func["arguments"]
                            yield StreamChunk(
                                type=ChunkType.TOOL_CALL_DELTA,
                                content=func["arguments"],
                            )

                # vLLM 0.21 streams reasoning as `reasoning` deltas (mirroring
                # the non-streaming response field). If a future vLLM renames,
                # this assignment becomes empty and tests should catch it.
                reasoning_delta = delta.get("reasoning") or ""
                if reasoning_delta:
                    accumulated_reasoning += reasoning_delta

                content = delta.get("content") or ""
                if content:
                    accumulated_content += content
                    yield StreamChunk(
                        type=ChunkType.TEXT_DELTA, content=content,
                    )

        # Build the final response. Reassemble the accumulated deltas into the
        # OpenAI tool-call shape and route through the same parser as send(), so
        # streaming and non-streaming agree on malformed-args handling: a fully
        # accumulated but unparseable arguments string rides through as raw
        # (non-dict) args on the ToolCall — which ResponseValidator routes to
        # the tool-error channel — not an exception.
        if tool_call_parts:
            reassembled = [
                {"function": {"name": part["name"], "arguments": part["args"]}}
                for part in (tool_call_parts[k] for k in sorted(tool_call_parts))
            ]
            final: LLMResponse = self._parse_tool_calls(
                reassembled,
                reasoning=self._resolve_reasoning(
                    accumulated_reasoning, accumulated_content,
                ),
            )
        else:
            # Strip inline thinking from the final text for parity with send().
            _, text = extract_think_tags(accumulated_content)
            final = TextResponse(content=text)
        yield StreamChunk(type=ChunkType.FINAL, response=final)

    @staticmethod
    def _parse_tool_calls(
        tool_calls: list[dict[str, Any]],
        reasoning: str | None,
    ) -> LLMResponse:
        """Parse vLLM ``tool_calls`` into ``ToolCall`` objects.

        Mirrors ``OpenAICompatClient`` / ``LlamafileClient`` so every
        OpenAI-shape client behaves the same. Tool-call ``arguments`` arrive as
        a JSON string (vLLM's native format) or an already-decoded dict.
        ``decode_tool_args`` is fail-loud: malformed JSON (or any non-dict
        shape) is NOT coerced into empty args — the raw value is kept so
        ``ResponseValidator`` routes it through the tool-error channel instead
        of crashing or letting the model proceed with wrong arguments.

        Defensive ``.get`` on ``function`` / ``name`` keeps a broken tool-call
        entry from raising ``KeyError``. Used by both send() and send_stream()
        for parity (the stream path reassembles deltas into this shape first).
        """
        return [
            ToolCall(
                tool=tc.get("function", {}).get("name", ""),
                args=decode_tool_args(tc.get("function", {}).get("arguments")),
                reasoning=reasoning if i == 0 else None,
            )
            for i, tc in enumerate(tool_calls)
        ]

    async def get_context_length(self) -> int | None:
        """Query the vLLM /v1/models endpoint for max_model_len.

        vLLM exposes the configured context window via the OpenAI-compat
        models endpoint. Single endpoint, single field — raises on
        unexpected response shape.
        """
        resp = await self._http.get(f"{self.base_url}/models")
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data") or []
        if not models:
            raise BackendError(500, f"/v1/models returned no entries: {data}")
        max_model_len = models[0].get("max_model_len")
        if max_model_len is None:
            raise BackendError(
                500, f"/v1/models entry missing max_model_len: {models[0]}",
            )
        return int(max_model_len)

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
            resp = await self._http.get(f"{self.base_url}/models")
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        models = resp.json().get("data") or []
        if not models:
            return None
        return models[0].get("id")

    async def discover_backend_metadata(
        self, extra_headers: dict[str, str] | None = None,
    ) -> int | None:
        """Probe /v1/models once for both budget and served identity.

        vLLM exposes ``max_model_len`` (context budget) and the served model
        ``id`` (the wire ``model`` field it validates) on the same endpoint, so
        one credentialed GET yields both — collapsing the two separate startup
        round-trips. The served id is adopted into this client immediately
        (``_set_model_identity``), and the budget is returned for the caller to
        apply to the context manager.

        Both fields are required: vLLM 404s every request without a valid served
        id, and a missing ``max_model_len`` leaves the budget undiscoverable —
        either is a loud failure (no silent ``"default"`` / no guessed budget).
        """
        try:
            resp = await self._http.get(
                f"{self.base_url}/models",
                headers=self._request_headers(extra_headers),
            )
        except httpx.HTTPError as exc:
            raise BackendError(502, f"vLLM /v1/models unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise BackendError(resp.status_code, raw_body=resp.text)

        models = resp.json().get("data") or []
        if not models:
            raise BackendError(500, "vLLM /v1/models returned no entries")
        entry = models[0]

        served = entry.get("id")
        if not served:
            raise BackendError(500, f"vLLM /v1/models entry missing id: {entry}")
        logger.info("Discovered vLLM served model name: %s", served)
        self._set_model_identity(served)

        max_model_len = entry.get("max_model_len")
        if max_model_len is None:
            raise BackendError(
                500, f"vLLM /v1/models entry missing max_model_len: {entry}",
            )
        return int(max_model_len)
