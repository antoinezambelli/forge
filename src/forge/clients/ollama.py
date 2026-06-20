"""Ollama client adapter using native function calling."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from forge.clients.base import (
    ChunkType,
    StreamChunk,
    TokenUsage,
    decode_tool_args,
    flatten_content_to_text,
    format_tool,
)
from forge.clients.sampling_defaults import apply_sampling_defaults
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError, ThinkingNotSupportedError
from forge.prompts.think_tags import extract_think_tags

_THINK_HEURISTIC_KEYWORDS = ("reason", "think")


def _normalize_messages_for_ollama(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Coerce OpenAI-wire messages into Ollama's native /api/chat schema.

    Ollama's native endpoint is stricter than the OpenAI wire format the proxy
    forwards verbatim on the native-passthrough path (issues #111/#115):

      - ``content`` must be a plain string, not a multi-part array. Multi-part
        arrays are flattened text-only (images dropped, matching the converted
        path's long-standing behavior).
      - tool-call ``arguments`` must be a dict, not the JSON string OpenAI
        clients replay in assistant history. Only a value that decodes to a
        dict is rewritten; a malformed / non-object payload is left untouched
        (it rides the validator's tool-error lane rather than being coerced).

    Returns a new list; the input messages and their nested dicts are not
    mutated (direct callers may reuse their list).
    """
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        new_msg = dict(msg)
        if isinstance(new_msg.get("content"), list):
            new_msg["content"] = flatten_content_to_text(new_msg["content"])

        tool_calls = new_msg.get("tool_calls")
        if isinstance(tool_calls, list):
            new_calls: list[Any] = []
            for tc in tool_calls:
                new_tc = dict(tc) if isinstance(tc, dict) else tc
                func = new_tc.get("function") if isinstance(new_tc, dict) else None
                if isinstance(func, dict) and "arguments" in func:
                    decoded = decode_tool_args(func["arguments"])
                    if isinstance(decoded, dict):
                        new_func = dict(func)
                        new_func["arguments"] = decoded
                        new_tc["function"] = new_func
                new_calls.append(new_tc)
            new_msg["tool_calls"] = new_calls

        normalized.append(new_msg)
    return normalized


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
        None (default) — auto-detect from model name, fall back on error
        True  — always send think=True (error if model doesn't support it)
        False — never send think
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
    ) -> None:
        self.base_url = base_url
        self.model = model
        # sampling_key is the registry-lookup key. For Ollama the wire "model"
        # field and the lookup key are the same string (the model tag).
        self.sampling_key = self.model
        # Apply per-model recommended sampling defaults. Caller's explicit
        # (non-None) kwargs win over the map field-by-field.
        defaults = apply_sampling_defaults(self.sampling_key, strict=recommended_sampling)
        self.temperature = temperature if temperature is not None else defaults.get("temperature")
        self.top_p = top_p if top_p is not None else defaults.get("top_p")
        self.top_k = top_k if top_k is not None else defaults.get("top_k")
        self.min_p = min_p if min_p is not None else defaults.get("min_p")
        self.repeat_penalty = repeat_penalty if repeat_penalty is not None else defaults.get("repeat_penalty")
        self.presence_penalty = presence_penalty if presence_penalty is not None else defaults.get("presence_penalty")
        self._http = httpx.AsyncClient(timeout=timeout)
        self._num_ctx: int | None = None

        if think is not None:
            self._think: bool = think
        else:
            # Heuristic: enable for models with "reason"/"think" in name
            model_lower = model.lower()
            self._think = any(kw in model_lower for kw in _THINK_HEURISTIC_KEYWORDS)
        self._think_resolved: bool = think is not None
        self.last_usage: dict[int, TokenUsage] = {}

    async def aclose(self) -> None:
        """Close the underlying httpx connection pool."""
        await self._http.aclose()

    # Sampling fields recognized in per-call overrides. ``seed`` is
    # accepted only as a per-call override (not an instance field).
    _SAMPLING_FIELDS = (
        "temperature", "top_p", "top_k", "min_p",
        "repeat_penalty", "presence_penalty", "seed",
    )

    def _build_options(
        self, sampling: dict[str, Any] | None = None,
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

    def _resolve_reasoning(
        self,
        thinking: str,
        content: str,
    ) -> str | None:
        """Gate reasoning capture on _think flag.

        When _think is False, discard all reasoning. When True: prefer the
        structured ``thinking`` field; if absent, extract ``<think>`` tags from
        content; finally fall back to the raw content (an instruct model
        narrating before its tool call). Mirrors LlamafileClient.
        """
        if not self._think:
            return None
        if thinking:
            return thinking
        think, _ = extract_think_tags(content)
        return think or content or None

    def _record_usage(self, data: dict[str, Any]) -> None:
        """Extract token usage from an Ollama response."""
        prompt = data.get("prompt_eval_count")
        completion = data.get("eval_count")
        if prompt is None and completion is None:
            return
        prompt = prompt or 0
        completion = completion or 0
        self.last_usage[0] = TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
        raw_openai_tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send messages via /api/chat and parse the response.

        ``passthrough`` is accepted for protocol symmetry but not yet
        plumbed — Ollama is not currently a proxy-side external backend
        (forge proxy uses LlamafileClient for external mode). Adding
        Ollama passthrough is a follow-up.

        ``inbound_anthropic_body`` / ``raw_openai_tools`` accepted for protocol
        symmetry, ignored (Ollama is OpenAI-shape only).
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": _normalize_messages_for_ollama(messages),
            "stream": False,
            "options": self._build_options(sampling),
        }
        if self._think:
            body["think"] = True
        if tools:
            body["tools"] = [format_tool(t) for t in tools]

        try:
            resp = await self._http.post(f"{self.base_url}/api/chat", json=body)
        except httpx.ReadTimeout as exc:
            raise BackendError(408, "Read timeout") from exc

        # Think unsupported: fail fast if explicit, fall back if auto-detected
        if _is_think_unsupported_error(resp.status_code, resp.text):
            if self._think_resolved:
                raise ThinkingNotSupportedError(self.model, resp.status_code, resp.text)
            self._think = False
            self._think_resolved = True
            del body["think"]
            resp = await self._http.post(f"{self.base_url}/api/chat", json=body)

        if resp.status_code == 500:
            return TextResponse(content=resp.text)
        if resp.status_code != 200:
            raise BackendError(resp.status_code, resp.text)
        data = resp.json()
        self._record_usage(data)

        if not self._think_resolved:
            self._think_resolved = True

        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            reasoning = self._resolve_reasoning(
                msg.get("thinking", ""), msg.get("content", ""),
            )
            # Ollama returns tool-call arguments already decoded as a dict
            # (unlike vLLM/llama.cpp, which send a JSON string) — no json.loads
            # needed. Defensive .get on function/name so a broken tool-call
            # entry degrades to empty rather than raising KeyError.
            return [
                ToolCall(
                    tool=tc.get("function", {}).get("name", ""),
                    args=tc.get("function", {}).get("arguments", {}),
                    reasoning=reasoning if i == 0 else None,
                )
                for i, tc in enumerate(tool_calls)
            ]

        # No tool calls: strip inline thinking so the TextResponse carries
        # clean content (parity with LlamafileClient).
        _, content = extract_think_tags(msg.get("content", ""))
        return TextResponse(content=content)

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
        raw_openai_tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream via NDJSON from /api/chat.

        ``passthrough`` / ``inbound_anthropic_body`` / ``raw_openai_tools``
        accepted for protocol symmetry; see ``send`` notes.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": _normalize_messages_for_ollama(messages),
            "stream": True,
            "options": self._build_options(sampling),
        }
        if self._think:
            body["think"] = True
        if tools:
            body["tools"] = [format_tool(t) for t in tools]

        async with self._http.stream(
            "POST", f"{self.base_url}/api/chat", json=body
        ) as response:
            # Think unsupported: fail fast if explicit, fall back if auto-detected
            if response.status_code == 400:
                error_body = ""
                async for line in response.aiter_lines():
                    error_body += line
                if _is_think_unsupported_error(400, error_body):
                    if self._think_resolved:
                        raise ThinkingNotSupportedError(self.model, 400, error_body)
                    self._think = False
                    self._think_resolved = True
                    del body["think"]
                    # Fall through to retry below
                else:
                    raise BackendError(400, error_body)

            if not self._think_resolved:
                self._think_resolved = True

            # If we just disabled think, we need a new stream
            if "think" not in body and self._think is False and response.status_code == 400:
                pass  # Exit context manager, retry below
            else:
                async for chunk in self._iter_stream(response):
                    yield chunk
                return

        # Retry stream without think
        async with self._http.stream(
            "POST", f"{self.base_url}/api/chat", json=body
        ) as response:
            async for chunk in self._iter_stream(response):
                yield chunk

    async def _iter_stream(
        self, response: httpx.Response
    ) -> AsyncIterator[StreamChunk]:
        """Parse NDJSON stream chunks from an Ollama response."""
        if response.status_code == 500:
            error_body = ""
            async for line in response.aiter_lines():
                error_body += line
            yield StreamChunk(
                type=ChunkType.FINAL,
                response=TextResponse(content=error_body),
            )
            return

        accumulated_content = ""
        accumulated_thinking = ""
        pending_tool_calls: list[dict[str, Any]] | None = None
        try:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                msg = data.get("message", {})

                if data.get("done"):
                    self._record_usage(data)
                    tool_calls = msg.get("tool_calls") or pending_tool_calls
                    if tool_calls:
                        reasoning = self._resolve_reasoning(
                            accumulated_thinking,
                            accumulated_content or msg.get("content", ""),
                        )
                        final: LLMResponse = [
                            ToolCall(
                                tool=tc.get("function", {}).get("name", ""),
                                args=tc.get("function", {}).get("arguments", {}),
                                reasoning=reasoning if i == 0 else None,
                            )
                            for i, tc in enumerate(tool_calls)
                        ]
                    else:
                        content = msg.get("content", "")
                        if content:
                            accumulated_content += content
                        _, text = extract_think_tags(accumulated_content)
                        final = TextResponse(content=text)
                    yield StreamChunk(type=ChunkType.FINAL, response=final)
                else:
                    tool_calls = msg.get("tool_calls")
                    if tool_calls:
                        pending_tool_calls = tool_calls
                    thinking = msg.get("thinking", "")
                    if thinking:
                        accumulated_thinking += thinking
                    content = msg.get("content", "")
                    if content:
                        accumulated_content += content
                        yield StreamChunk(
                            type=ChunkType.TEXT_DELTA, content=content
                        )
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
