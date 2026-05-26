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
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from forge.clients.base import ChunkType, StreamChunk, TokenUsage, format_tool
from forge.clients.sampling_defaults import apply_sampling_defaults
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError


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
    ) -> None:
        self.base_url = base_url
        # model_path is the canonical identity. vLLM accepts either a local
        # directory containing safetensors + config or a HuggingFace repo id
        # (e.g. "google/gemma-4-26B-A4B-it"). We pass it through as-is in
        # the wire-format "model" field and as the sampling-defaults lookup
        # key (using the path stem for directory paths so registry lookups
        # match the existing GGUF-stem convention).
        self.model_path = str(model_path)
        path_obj = Path(self.model_path)
        # If model_path is a filesystem path, use the directory name as the
        # registry lookup key. If it's an HF repo id (no leading slash, has
        # a "/"), use the trailing segment. Otherwise the full string.
        if path_obj.is_absolute() or path_obj.exists():
            self.model = path_obj.name
        elif "/" in self.model_path:
            self.model = self.model_path.split("/")[-1]
        else:
            self.model = self.model_path

        # Apply per-model recommended sampling defaults. Caller's explicit
        # (non-None) kwargs win over the map field-by-field.
        defaults = apply_sampling_defaults(self.model, strict=recommended_sampling)
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
        self._http = httpx.AsyncClient(timeout=timeout)
        self._think: bool = think
        self.last_usage: dict[int, TokenUsage] = {}

    async def aclose(self) -> None:
        """Close the underlying httpx connection pool."""
        await self._http.aclose()

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

    def _resolve_reasoning(
        self, message_or_accum: dict[str, Any] | str, accumulated_content: str = "",
    ) -> str | None:
        """Extract reasoning, gated on _think.

        vLLM 0.21 returns reasoning in the ``reasoning`` field of the
        assistant message when ``--reasoning-parser`` is enabled at server
        boot. If thinking is disabled or the field is empty, return None.

        Accepts either a message dict (from send()) or an accumulated
        reasoning string (from send_stream()).
        """
        if not self._think:
            return None
        if isinstance(message_or_accum, dict):
            return message_or_accum.get("reasoning") or None
        return message_or_accum or accumulated_content or None

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send messages via /v1/chat/completions and parse the response."""
        body: dict[str, Any] = {
            "model": self.model_path,
            "messages": messages,
            "stream": False,
        }
        if tools:
            body["tools"] = [format_tool(t) for t in tools]
            body["tool_choice"] = "auto"
        self._apply_sampling(body, sampling)

        try:
            resp = await self._http.post(
                f"{self.base_url}/chat/completions", json=body,
            )
        except httpx.ReadTimeout as exc:
            raise BackendError(408, "Read timeout") from exc

        if resp.status_code != 200:
            raise BackendError(resp.status_code, resp.text)
        data = resp.json()
        self._record_usage(data)

        choices = data.get("choices") or []
        if not choices:
            raise BackendError(500, f"vLLM response has no choices: {data}")
        message = choices[0].get("message", {})

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            reasoning = self._resolve_reasoning(message)
            return [
                ToolCall(
                    tool=tc["function"]["name"],
                    args=self._parse_tool_args(tc["function"].get("arguments", {})),
                    reasoning=reasoning if i == 0 else None,
                )
                for i, tc in enumerate(tool_calls)
            ]

        return TextResponse(content=message.get("content") or "")

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream via SSE from /v1/chat/completions."""
        body: dict[str, Any] = {
            "model": self.model_path,
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
            "POST", f"{self.base_url}/chat/completions", json=body,
        ) as response:
            if response.status_code != 200:
                error_body = ""
                async for line in response.aiter_lines():
                    error_body += line
                raise BackendError(response.status_code, error_body)

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

        # Build the final response
        if tool_call_parts:
            reasoning = self._resolve_reasoning(
                accumulated_reasoning, accumulated_content,
            )
            final: LLMResponse = [
                ToolCall(
                    tool=part["name"],
                    args=self._parse_tool_args(part["args"]),
                    reasoning=reasoning if i == 0 else None,
                )
                for i, part in enumerate(
                    tool_call_parts[k] for k in sorted(tool_call_parts)
                )
            ]
        else:
            final = TextResponse(content=accumulated_content)
        yield StreamChunk(type=ChunkType.FINAL, response=final)

    @staticmethod
    def _parse_tool_args(raw: Any) -> dict[str, Any]:
        """Tool args from vLLM arrive as JSON-encoded string in the
        OpenAI native format. Decode to dict.
        """
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            if not raw:
                return {}
            return json.loads(raw)
        raise BackendError(500, f"unexpected tool args shape: {type(raw).__name__}")

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
