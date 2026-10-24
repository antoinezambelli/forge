"""Anthropic API client adapter for frontier model baselines.

Translates between forge's OpenAI-style message format (what the runner
produces) and Anthropic's native Messages API format internally.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import os
import random
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from forge.clients.base import (
    ChunkType,
    StreamChunk,
    TokenUsage,
    _record_captured_usage,
    decode_tool_args,
    has_auth_header,
    resolve_request_headers,
    static_auth_present,
)
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError, MissingCredentialError, MissingModelError

log = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"

# Preserve the existing exclusion of SDK control fields from caller bodies.
# They are never HTTP transport controls, even on the verbatim path.
_SDK_CONTROL_KWARGS = ("extra_headers", "extra_body", "extra_query", "timeout")


def _prepare_anthropic_headers(
    headers: dict[str, str] | None,
) -> dict[str, str] | None:
    """Copy headers case-insensitively, retaining the existing beta exclusion."""
    if not headers:
        return None
    prepared: dict[str, str] = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl == "anthropic-beta":
            continue
        prepared[kl] = v
    return prepared or None


class AnthropicClient:
    """Anthropic Messages API client for Claude models.

    Uses HTTP directly. The runner serializes messages in
    OpenAI format (``api_format = "openai"``); this client converts them
    to Anthropic format before each API call.
    """

    api_format: str = "openai"

    def __init__(
        self,
        model: str | None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 300.0,
        max_retries: int = 3,
        tool_choice: str | None = None,
        recommended_sampling: bool = False,
        base_url: str | None = None,
        prompt_caching: bool = False,
        thinking: dict[str, Any] | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._tool_choice = tool_choice  # "auto", "any", or None (default=auto)
        # Opt-in Anthropic prompt caching (billing-only). When on, the rebuild
        # path marks a static cache breakpoint over the tool defs + system
        # prompt (re-sent verbatim every turn). Off by default so the proxy
        # verbatim path and existing request shape are untouched. See
        # _apply_static_cache for why caching is static-only here.
        self._prompt_caching = prompt_caching
        # Extended-thinking request config, e.g. {"type": "adaptive"}. When set,
        # merged into every messages.create call (and a forced tool_choice is
        # suppressed — Anthropic requires tool_choice="auto" with thinking on).
        # None = thinking off; the proxy passthrough path can still carry its
        # own ``thinking`` via ``passthrough``.
        self._thinking = thinking
        # Accepted for API symmetry across clients but currently a no-op:
        # AnthropicClient does not expose sampling kwargs through forge today.
        # Sampling passthrough remains governed by the existing body paths.
        if recommended_sampling:
            log.debug(
                "AnthropicClient ignores recommended_sampling=True — no sampling kwargs are exposed."
            )
        # One credential per request. ``static_auth`` reflects an explicitly
        # configured static credential (an ``api_key`` argument or an auth
        # header in ``default_headers``); a per-call auth header is then refused
        # as a second source.
        self._static_auth = static_auth_present(api_key, default_headers)
        # None retains direct-client environment auth; "" is the proxy's
        # credential-free passthrough mode. Explicit keys never read ambient
        # auth. Resolve locally rather than temporarily mutating os.environ.
        self._api_key = os.environ.get("ANTHROPIC_API_KEY") if api_key is None else api_key or None
        self._auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN") if api_key is None else None
        headers = httpx.Headers({
            "content-type": "application/json",
            "anthropic-version": DEFAULT_ANTHROPIC_VERSION,
        })
        if self._api_key is not None:
            headers["x-api-key"] = self._api_key
        if self._auth_token is not None:
            headers["authorization"] = f"Bearer {self._auth_token}"
        prepared_default_headers = _prepare_anthropic_headers(default_headers)
        if prepared_default_headers:
            headers.update(prepared_default_headers)
        root = base_url if base_url is not None else os.environ.get(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com",
        )
        self._http = httpx.AsyncClient(
            base_url=root.rstrip("/") + "/", headers=headers,
            timeout=timeout, follow_redirects=True,
        )
        self._max_retries = max_retries
        # Populated after each send()/send_stream() call. Slot-keyed
        # ``{id_slot: TokenUsage}`` to preserve the direct-client compatibility
        # mirror used by LlamafileClient / OllamaClient. There is no shared
        # inference slot, so we always use slot 0 — same
        # convention as OllamaClient.
        self.last_usage: dict[int, TokenUsage] = {}

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._http.aclose()

    def _request_headers(
        self, extra_headers: dict[str, str] | None,
    ) -> dict[str, str] | None:
        """Resolve request-local auth/version without mutating client defaults."""
        headers = _prepare_anthropic_headers(
            resolve_request_headers(self._static_auth, extra_headers)
        )
        self._ensure_credential(headers)
        return headers

    def _ensure_credential(self, headers: dict[str, str] | None) -> None:
        """Fail loud if this request would reach the backend with no credential.

        Retains the existing resolved-key/token or per-call-header check.
        """
        if (
            self._api_key is None
            and self._auth_token is None
            and not has_auth_header(headers)
        ):
            raise MissingCredentialError("Anthropic backend")

    @staticmethod
    def _prepare_body(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Keep the existing body exclusions; gateway extras remain JSON."""
        for control_kwarg in _SDK_CONTROL_KWARGS:
            kwargs.pop(control_kwarg, None)
        return kwargs

    @staticmethod
    def _should_retry(response: httpx.Response) -> bool:
        directive = response.headers.get("x-should-retry")
        if directive in ("true", "false"):
            return directive == "true"
        return response.status_code in (408, 409, 429) or response.status_code >= 500

    @staticmethod
    def _retry_delay(attempt: int, headers: httpx.Headers | None) -> float:
        # Preserve the SDK's bounded Retry-After handling and backoff.
        if headers:
            try:
                if "retry-after-ms" in headers:
                    delay = float(headers["retry-after-ms"]) / 1000
                else:
                    delay = float(headers["retry-after"])
                if 0 < delay <= 60:
                    return delay
            except (KeyError, ValueError):
                try:
                    when = parsedate_to_datetime(headers.get("retry-after", ""))
                    delay = (when - datetime.now(timezone.utc)).total_seconds()
                    if 0 < delay <= 60:
                        return delay
                except (TypeError, ValueError, OverflowError):
                    pass
        return min(0.5 * (2 ** min(attempt, 1000)), 8.0) * (1 - 0.25 * random.random())

    @asynccontextmanager
    async def _request(
        self, method: str, path: str, *, body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        stream: bool = False,
    ) -> AsyncIterator[httpx.Response]:
        headers = self._request_headers(extra_headers)
        attempt = 0
        while True:
            started = False
            response_headers = None
            try:
                async with self._http.stream(method, path, json=body, headers=headers) as response:
                    if response.is_success:
                        if not stream:
                            await response.aread()
                        started = True
                        yield response
                        return
                    await response.aread()
                    if attempt >= self._max_retries or not self._should_retry(response):
                        raise BackendError(response.status_code, raw_body=response.text)
                    response_headers = response.headers
            except httpx.TransportError as exc:
                # Never replay a stream after any response has been exposed.
                if started or attempt >= self._max_retries:
                    raise BackendError(0, "Anthropic backend request failed", raw_body=str(exc)) from exc
            await asyncio.sleep(self._retry_delay(attempt, response_headers))
            attempt += 1

    @staticmethod
    async def _events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        data: list[str] = []
        async for line in response.aiter_lines():
            if not line:
                if data:
                    yield json.loads("\n".join(data))
                    data = []
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            yield json.loads("\n".join(data))

    def _record_usage(self, usage: dict[str, Any]) -> None:
        normalized = TokenUsage(
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0) or 0,
        )
        self.last_usage = {0: normalized}
        _record_captured_usage(normalized)

    # ── Tool schema conversion ───────────────────────────────────

    @staticmethod
    def _convert_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """ToolSpec list → Anthropic tool definitions."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.get_json_schema(),
            }
            for spec in tools
        ]

    # ── Message format conversion ────────────────────────────────

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """OpenAI-format dicts → (system_prompt, anthropic_messages).

        Handles:
        - System message extraction (→ separate ``system=`` kwarg)
        - assistant tool_calls → ``tool_use`` content blocks
        - role=tool → ``tool_result`` content blocks inside user messages
        - Unpaired tool_use (step/unknown-tool nudges) → synthetic error
          tool_results injected before the next user text
        - Consecutive same-role merging (Anthropic requires strict alternation)
        """
        system: str | None = None
        converted: list[dict[str, Any]] = []
        # Track tool_use IDs that haven't received a tool_result yet.
        pending_tool_use_ids: list[str] = []

        for msg in messages:
            role = msg["role"]

            if role == "system":
                system = msg["content"]
                continue

            if role == "assistant":
                if "tool_calls" in msg:
                    blocks: list[dict[str, Any]] = []
                    content = msg.get("content", "")
                    if content:
                        blocks.append({"type": "text", "text": content})
                    for tc in msg["tool_calls"]:
                        func = tc["function"]
                        args = func.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError as exc:
                                raise ValueError(
                                    f"Malformed JSON in tool_call arguments for "
                                    f"{func.get('name', '?')!r}: {args!r}"
                                ) from exc
                        tc_id = tc.get("id", f"toolu_{len(converted)}")
                        blocks.append({
                            "type": "tool_use",
                            "id": tc_id,
                            "name": func["name"],
                            "input": args,
                        })
                        pending_tool_use_ids.append(tc_id)
                    converted.append({"role": "assistant", "content": blocks})
                else:
                    converted.append({
                        "role": "assistant",
                        "content": msg.get("content", ""),
                    })
                continue

            if role == "tool":
                tc_id = msg.get("tool_call_id", "unknown")
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tc_id,
                    "content": msg.get("content", ""),
                }
                if tc_id in pending_tool_use_ids:
                    pending_tool_use_ids.remove(tc_id)
                converted.append({"role": "user", "content": [block]})
                continue

            if role == "user":
                # Inject error tool_results for any unpaired tool_use blocks
                # (e.g. step nudge or unknown-tool nudge — tool was never executed).
                if pending_tool_use_ids:
                    blocks = []
                    for tc_id in pending_tool_use_ids:
                        blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tc_id,
                            "content": "Not executed.",
                            "is_error": True,
                        })
                    pending_tool_use_ids.clear()
                    text = msg.get("content", "")
                    if text:
                        blocks.append({"type": "text", "text": text})
                    converted.append({"role": "user", "content": blocks})
                else:
                    converted.append({
                        "role": "user",
                        "content": msg.get("content", ""),
                    })
                continue

        # Merge consecutive same-role messages (Anthropic requires strict
        # user/assistant alternation).
        merged: list[dict[str, Any]] = []
        for msg in converted:
            if merged and merged[-1]["role"] == msg["role"]:
                prev_content = merged[-1]["content"]
                curr_content = msg["content"]
                # Normalise to list-of-blocks
                if isinstance(prev_content, str):
                    prev_blocks = [{"type": "text", "text": prev_content}]
                else:
                    prev_blocks = list(prev_content)
                if isinstance(curr_content, str):
                    curr_blocks = [{"type": "text", "text": curr_content}]
                else:
                    curr_blocks = list(curr_content)
                merged[-1] = {
                    "role": msg["role"],
                    "content": prev_blocks + curr_blocks,
                }
            else:
                merged.append(msg)

        return system, merged

    # ── Response parsing ─────────────────────────────────────────

    @staticmethod
    def _resolve_reasoning(thinking: str, text: str) -> str | None:
        """Prefer native thinking, preserving the existing text fallback."""
        return thinking or text or None

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> LLMResponse:
        """Anthropic Message → list[ToolCall] or TextResponse."""
        tool_uses: list[Any] = []
        thinking_parts: list[str] = []
        text_parts: list[str] = []

        for block in response["content"]:
            if block["type"] == "tool_use":
                tool_uses.append(block)
            elif block["type"] == "thinking":
                thinking_parts.append(block["thinking"])
            elif block["type"] == "text":
                text_parts.append(block["text"])

        if tool_uses:
            reasoning = AnthropicClient._resolve_reasoning(
                "\n".join(thinking_parts), "\n".join(text_parts),
            )
            return [
                ToolCall(
                    tool=tu["name"],
                    args=dict(tu["input"]),
                    reasoning=reasoning if i == 0 else None,
                )
                for i, tu in enumerate(tool_uses)
            ]
        return TextResponse(content="\n".join(text_parts))

    # ── API methods ──────────────────────────────────────────────

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the Messages request body for send / send_stream.

        ``passthrough`` (proxy use) supplies inbound-body fields the proxy
        didn't deconstruct (``max_tokens``, ``stop_sequences``, ``tool_choice``,
        ``metadata``, ``thinking``, etc.). Forge-owned fields (``model``,
        ``messages``, ``system``, ``tools``) overlay it.

        ``inbound_anthropic_body`` (proxy path 1) bypasses the deconstruct/
        rebuild path: the backend receives the original inbound body
        verbatim, preserving block-level fields like ``cache_control`` that
        forge.Message doesn't represent. The runner only passes this on the
        clean first-attempt call (no compaction, no retries) — see ADR-015.
        """
        if inbound_anthropic_body is not None:
            # Verbatim emit. Drop the proxy-internal ``stream`` field; the
            # caller (send vs send_stream) selects streaming. A fixed client
            # model is authoritative; a dynamic
            # proxy client keeps the already-resolved request-local model.
            kwargs = dict(inbound_anthropic_body)
            kwargs.pop("stream", None)
            if self.model is not None:
                kwargs["model"] = self.model
            if kwargs.get("model") in (None, ""):
                raise MissingModelError()
            return kwargs

        system, converted = self._convert_messages(messages)
        kwargs = dict(passthrough or {})
        model = self.model if self.model is not None else kwargs.get("model")
        if model in (None, ""):
            raise MissingModelError()
        kwargs.update({
            "model": model,
            "messages": converted,
        })
        kwargs.setdefault("max_tokens", self.max_tokens)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            # Extended thinking is incompatible with a forced tool_choice;
            # Anthropic requires "auto" (the default) when thinking is on.
            if self._tool_choice and not self._thinking and "tool_choice" not in kwargs:
                kwargs["tool_choice"] = {"type": self._tool_choice}
        if self._thinking and "thinking" not in kwargs:
            kwargs["thinking"] = self._thinking
        if self._prompt_caching:
            self._apply_static_cache(kwargs)
        return kwargs

    @staticmethod
    def _apply_static_cache(kwargs: dict[str, Any]) -> None:
        """Mark a static ephemeral cache breakpoint over tool defs + system.

        The tool block and system prompt are byte-identical on every turn of a
        run, so this prefix reliably read-hits (at 0.1×) from turn 2 onward
        instead of re-billing the re-sent schema + prompt at full price.

        Static-only on purpose: a *rolling* per-turn breakpoint over the growing
        conversation is NOT placed here. Under ``reasoning_replay="keep-last"``
        earlier tool-call messages re-serialize differently each turn (only the
        latest reasoning is kept), which busts a rolling prefix cache — you'd
        pay 1.25× writes with no reads. The conversation prefix is only stable
        under ``none``/``full``, and ``reasoning_replay`` is a measured eval
        variable we won't pin, so caching is confined to the always-stable
        tools+system region.

        The cached prefix is ordered tools → system → messages, so a single
        breakpoint on the system block subsumes the tools; we additionally mark
        the last tool so the tool prefix still caches when ``system`` is absent.
        """
        ephemeral = {"type": "ephemeral"}
        tools = kwargs.get("tools")
        if tools:
            tools[-1]["cache_control"] = ephemeral
        system = kwargs.get("system")
        if isinstance(system, str) and system:
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": ephemeral}
            ]

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
        """Send JSON with the existing body conversion and sampling policy."""
        if sampling:
            log.debug(
                "AnthropicClient ignores per-call sampling overrides: %s",
                sorted(sampling.keys()),
            )
        kwargs = self._build_kwargs(
            messages, tools, passthrough, inbound_anthropic_body,
        )
        async with self._request(
            "POST", "v1/messages", body=self._prepare_body(kwargs),
            extra_headers=extra_headers,
        ) as response:
            payload = response.json()
        self._record_usage(payload["usage"])
        return self._parse_response(payload)

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
        """Consume native Messages SSE without replaying an interrupted stream."""
        if sampling:
            log.debug(
                "AnthropicClient ignores per-call sampling overrides: %s",
                sorted(sampling.keys()),
            )
        kwargs = self._build_kwargs(
            messages, tools, passthrough, inbound_anthropic_body,
        )
        accumulated_text = ""
        accumulated_thinking = ""
        tool_blocks: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] = {}
        kwargs = self._prepare_body(kwargs)
        kwargs["stream"] = True
        async with self._request(
            "POST", "v1/messages", body=kwargs, extra_headers=extra_headers,
            stream=True,
        ) as response:
            async for event in self._events(response):
                kind = event["type"]
                if kind == "error":
                    raise BackendError(502, "Anthropic stream failed", raw_body=json.dumps(event))
                if kind == "message_start":
                    usage.update(event["message"].get("usage", {}))
                elif kind == "message_delta":
                    usage.update({
                        key: value for key, value in event.get("usage", {}).items()
                        if value is not None
                    })
                elif kind == "content_block_start":
                    block = event["content_block"]
                    if block["type"] == "tool_use":
                        tool_blocks[event["index"]] = {"name": block["name"], "args": ""}
                elif kind == "content_block_delta":
                    delta = event["delta"]
                    if delta["type"] == "text_delta":
                        accumulated_text += delta["text"]
                        yield StreamChunk(type=ChunkType.TEXT_DELTA, content=delta["text"])
                    elif delta["type"] == "thinking_delta":
                        accumulated_thinking += delta["thinking"]
                    elif delta["type"] == "input_json_delta" and event["index"] in tool_blocks:
                        tool_blocks[event["index"]]["args"] += delta["partial_json"]
                        yield StreamChunk(type=ChunkType.TOOL_CALL_DELTA, content=delta["partial_json"])
                elif kind == "message_stop":
                    self._record_usage(usage)
                    if tool_blocks:
                        reasoning = self._resolve_reasoning(accumulated_thinking, accumulated_text)
                        final: LLMResponse = [
                            ToolCall(
                                tool=tb["name"], args=decode_tool_args(tb["args"]),
                                reasoning=reasoning if i == 0 else None,
                            )
                            for i, tb in enumerate(tool_blocks.values())
                        ]
                    else:
                        final = TextResponse(content=accumulated_text)
                    yield StreamChunk(type=ChunkType.FINAL, response=final)
                    return
        raise BackendError(502, "Anthropic stream ended before message_stop")

    async def get_context_length(self) -> int | None:
        """Return exact official model metadata."""
        if not self.model:
            return None
        return await self._get_context_length_for_model(self.model, None)

    async def _get_context_length_for_model(
        self,
        model: str,
        extra_headers: dict[str, str] | None,
    ) -> int | None:
        """Return canonical Anthropic metadata for one exact request model."""

        if not self._uses_official_metadata_root():
            return None
        try:
            async with self._request(
                "GET", f"v1/models/{quote(model, safe='')}", extra_headers=extra_headers,
            ) as response:
                metadata = response.json()
        except BackendError as exc:
            raise BackendError(
                exc.status_code or 502,
                "Anthropic model metadata request failed",
            ) from exc
        window = metadata.get("max_input_tokens")
        if (
            not isinstance(window, int)
            or isinstance(window, bool)
            or window <= 0
        ):
            return None
        return window

    def _uses_official_metadata_root(self) -> bool:
        """Whether the HTTP client targets canonical Anthropic."""
        target = urlsplit(str(self._http.base_url))
        return (
            target.scheme.lower() == "https"
            and target.hostname == "api.anthropic.com"
            and target.port in (None, 443)
            and target.path.rstrip("/") == ""
            and not target.query
            and not target.fragment
            and target.username is None
            and target.password is None
        )
