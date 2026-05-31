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

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from forge.clients.base import ChunkType, StreamChunk, TokenUsage, format_tool
from forge.clients.sampling_defaults import apply_sampling_defaults
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
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
        self.model = model

        # Apply per-model recommended sampling defaults. Caller's explicit
        # (non-None) kwargs win over the map field-by-field. With
        # recommended_sampling=False (default) and an unknown model stem,
        # apply_sampling_defaults returns an empty dict silently — which
        # is the common case for hosted providers whose model identifiers
        # aren't in forge's registry.
        defaults = apply_sampling_defaults(self.model, strict=recommended_sampling)
        self.temperature = temperature if temperature is not None else defaults.get("temperature")
        self.top_p = top_p if top_p is not None else defaults.get("top_p")
        self.top_k = top_k if top_k is not None else defaults.get("top_k")
        self.min_p = min_p if min_p is not None else defaults.get("min_p")
        self.repeat_penalty = repeat_penalty if repeat_penalty is not None else defaults.get("repeat_penalty")
        self.presence_penalty = presence_penalty if presence_penalty is not None else defaults.get("presence_penalty")
        # chat_template_kwargs is a nested dict of Jinja template variables
        # — whole-value replacement at this field level (no nested merge).
        self.chat_template_kwargs = (
            chat_template_kwargs if chat_template_kwargs is not None
            else defaults.get("chat_template_kwargs")
        )

        # Auth header is set when api_key is provided; extra_headers ride
        # on top and can override (kept open so a provider with a different
        # scheme doesn't need a new constructor kwarg).
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._http = httpx.AsyncClient(headers=headers, timeout=timeout)
        self.last_usage: dict[int, TokenUsage] = {}

    async def aclose(self) -> None:
        """Close the underlying httpx connection pool."""
        await self._http.aclose()

    # ── request building ─────────────────────────────────────────────

    # Sampling fields recognized in per-call overrides. ``seed`` is
    # accepted only as a per-call override (not an instance field).
    # ``chat_template_kwargs`` is a nested dict — whole-value replacement
    # at this field level (no nested merge).
    _SAMPLING_FIELDS = (
        "temperature", "top_p", "top_k", "min_p",
        "repeat_penalty", "presence_penalty", "seed",
        "chat_template_kwargs",
    )

    def _build_body(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None,
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
        self.last_usage[0] = TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=usage.get("total_tokens") or (prompt + completion),
        )

    @staticmethod
    def _parse_tool_calls(
        tool_calls: list[dict[str, Any]], fallback_content: str = ""
    ) -> LLMResponse:
        """Parse OpenAI ``tool_calls`` into ``ToolCall`` objects.

        Tool-call ``arguments`` arrive as JSON strings. Forge is fail-loud:
        malformed argument JSON must NOT be coerced into executable empty args,
        or a provider/model can emit invalid arguments and Forge proceeds with
        ``fn(**{})`` — exactly the quiet false success the library avoids.
        Instead we return a ``TextResponse``, which routes the response back
        into the validator's rescue-parse + retry/nudge loop, matching
        ``LlamafileClient`` (see ``llamafile.py`` ``_send_native``).

        ``fallback_content`` is the assistant message text to surface for the
        rescue attempt; we fall back to the raw malformed args when there is no
        text, so the rescue parser still has the original JSON to work with.
        """
        parsed: list[ToolCall] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    return TextResponse(content=fallback_content or raw_args)
            else:
                args = raw_args
            parsed.append(ToolCall(tool=fn.get("name", ""), args=args))
        return parsed

    # ── send ─────────────────────────────────────────────────────────

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send messages via /chat/completions and parse the response.

        ``inbound_anthropic_body`` is accepted to satisfy the LLMClient
        protocol but ignored — Path-1 Anthropic forwarding doesn't apply
        to OpenAI-shape clients.
        """
        del inbound_anthropic_body  # protocol-only, never read here
        body = self._build_body(messages, tools, sampling, stream=False, passthrough=passthrough)
        try:
            resp = await self._http.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.ReadTimeout as exc:
            raise BackendError(408, "Read timeout") from exc

        if resp.status_code != 200:
            raise BackendError(resp.status_code, resp.text)

        data = resp.json()
        self._record_usage(data)

        choices = data.get("choices") or []
        if not choices:
            raise BackendError(500, f"response has no choices: {data}")
        msg = choices[0].get("message", {})
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            return self._parse_tool_calls(tool_calls, fallback_content=msg.get("content") or "")
        return TextResponse(content=msg.get("content") or "")

    # ── streaming ────────────────────────────────────────────────────

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream via SSE from /chat/completions.

        ``inbound_anthropic_body`` is accepted to satisfy the LLMClient
        protocol but ignored — see :meth:`send`.
        """
        del inbound_anthropic_body  # protocol-only, never read here
        body = self._build_body(messages, tools, sampling, stream=True, passthrough=passthrough)

        accumulated_content = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] | None = None

        async with self._http.stream(
            "POST", f"{self.base_url}/chat/completions", json=body
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                raise BackendError(response.status_code, error_body.decode(errors="replace"))

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                content = delta.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        content = str(content)
                    if content:
                        accumulated_content += content
                        yield StreamChunk(type=ChunkType.TEXT_DELTA, content=content)

                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(
                        idx, {"function": {"name": "", "arguments": ""}}
                    )
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        slot["function"]["name"] += str(fn["name"])
                    # OpenAI streaming sends `arguments` as JSON-string
                    # fragments we concatenate into the final JSON string. A
                    # non-string fragment is a non-compliant provider; serialize
                    # it into the buffer rather than silently dropping it.
                    # Dropping leaves a gap in the assembled JSON that may parse
                    # into wrong-but-valid args (a quiet false success); folding
                    # it in instead means the single parse at stream end either
                    # recovers a whole-object fragment or fails loud into the
                    # TextResponse/retry path below, matching LlamafileClient.
                    args_frag = fn.get("arguments")
                    if args_frag is not None:
                        slot["function"]["arguments"] += (
                            args_frag if isinstance(args_frag, str) else json.dumps(args_frag)
                        )

        if usage:
            self._record_usage({"usage": usage})

        if tool_calls:
            ordered = [tool_calls[i] for i in sorted(tool_calls)]
            final: LLMResponse = self._parse_tool_calls(
                ordered, fallback_content=accumulated_content
            )
        else:
            final = TextResponse(content=accumulated_content)
        yield StreamChunk(type=ChunkType.FINAL, response=final)

    async def get_context_length(self) -> int | None:
        """OpenAI-compatible endpoints don't expose context length. Returns None."""
        return None
