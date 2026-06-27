"""Anthropic API client adapter for frontier model baselines.

Translates between forge's OpenAI-style message format (what the runner
produces) and Anthropic's native Messages API format internally.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import anthropic

from forge.clients.base import (
    ChunkType,
    StreamChunk,
    TokenUsage,
    decode_tool_args,
    has_auth_header,
    resolve_request_headers,
    static_auth_present,
)
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError, MissingCredentialError

log = logging.getLogger(__name__)

# The Anthropic SDK pins these headers itself — ``anthropic-version`` is part
# of its wire contract. A forwarded inbound value must not clobber them, so we
# drop them from any forge-forwarded header set.
_ANTHROPIC_PINNED_HEADERS = frozenset({"anthropic-version", "anthropic-beta"})

# The SDK's auth preflight (``_validate_headers``) reads these keys
# case-SENSITIVELY from a plain dict before the request goes out. The proxy
# lowercases all inbound header keys, so a relocated ``x-api-key`` would fail
# the preflight ("Could not resolve authentication method") unless re-cased to
# exactly what the SDK expects. forge never inspects the credential value.
_SDK_AUTH_CASING = {"authorization": "Authorization", "x-api-key": "X-Api-Key"}

# SDK request-control kwargs that must never be sourced from an inbound body or
# passthrough — they are client-side controls (headers/query/timeout), not
# Anthropic Messages API fields. In particular a body-supplied ``extra_headers``
# would smuggle an auth header into ``messages.create`` past the one-credential
# gate, so these are stripped before forge sets its own per-call credential.
_SDK_CONTROL_KWARGS = ("extra_headers", "extra_body", "extra_query", "timeout")

# Env vars the Anthropic SDK reads for auth when api_key/auth_token are unset.
# Suppressed while forge constructs a client it owns the credential for, so an
# ambient value can't inject a hidden second credential the per-request gate
# never sees.
_AMBIENT_ANTHROPIC_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


@contextlib.contextmanager
def _suppressed_ambient_credentials() -> Iterator[None]:
    """Temporarily clear Anthropic credential env vars, then restore them.

    Used only around SDK construction for forge-owned-credential clients (the
    proxy). forge may itself run evals that rely on these env vars in another
    client/process, so they are restored immediately after construction.
    """
    saved = {k: os.environ.pop(k, None) for k in _AMBIENT_ANTHROPIC_ENV}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def _prepare_anthropic_headers(
    headers: dict[str, str] | None,
) -> dict[str, str] | None:
    """Canonicalize auth-header casing for the SDK and drop SDK-pinned headers.

    Returns a new dict (or None). Auth headers are re-cased to the SDK's
    case-sensitive expectation; ``anthropic-version``/``anthropic-beta`` are
    dropped so the SDK's own pinned version wins.
    """
    if not headers:
        return None
    prepared: dict[str, str] = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in _ANTHROPIC_PINNED_HEADERS:
            continue
        prepared[_SDK_AUTH_CASING.get(kl, k)] = v
    return prepared or None


class AnthropicClient:
    """Anthropic Messages API client for Claude models.

    Uses the official anthropic SDK.  The runner serializes messages in
    OpenAI format (``api_format = "openai"``); this client converts them
    to Anthropic format before each API call.
    """

    api_format: str = "openai"

    def __init__(
        self,
        model: str,
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
        # The Anthropic SDK manages sampling internally.
        if recommended_sampling:
            log.debug(
                "AnthropicClient ignores recommended_sampling=True — no sampling kwargs are exposed."
            )
        # One credential per request. ``static_auth`` reflects an explicitly
        # configured static credential (an ``api_key`` argument or an auth
        # header in ``default_headers``); a per-call auth header is then refused
        # as a second source.
        self._static_auth = static_auth_present(api_key, default_headers)
        # Credential ownership has three modes, keyed on ``api_key``:
        #   None  — WorkflowRunner direct use: defer to the SDK's own env-based
        #           auth (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN), the dev's
        #           deliberate single credential. forge stays out of it.
        #   ""    — proxy pure passthrough: forge holds NO static credential and
        #           the per-call forwarded header is the sole auth. Map to None
        #           so the SDK emits no auth header at all (api_key="" would emit
        #           a spurious empty X-Api-Key that collides with a per-call
        #           Authorization on the OAuth path).
        #   "key" — explicit static credential (proxy --backend-api-key, or an
        #           explicit WR key).
        # Whenever forge owns the credential (api_key is not None, including
        # ""), suppress ambient ANTHROPIC_* env during construction so a stray
        # ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN can't become a silent second
        # credential the per-request gate never sees.
        forge_owns_credential = api_key is not None
        sdk_kwargs: dict[str, Any] = {
            "api_key": api_key or None,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        prepared_default_headers = _prepare_anthropic_headers(default_headers)
        if prepared_default_headers:
            sdk_kwargs["default_headers"] = prepared_default_headers
        # base_url retargets the SDK at an Anthropic-shape downstream
        # (LiteLLM, a self-hosted proxy, etc.) — proxy path 1.
        if base_url is not None:
            sdk_kwargs["base_url"] = base_url
        if forge_owns_credential:
            with _suppressed_ambient_credentials():
                self._client = anthropic.AsyncAnthropic(**sdk_kwargs)
        else:
            self._client = anthropic.AsyncAnthropic(**sdk_kwargs)
        # Populated after each send()/send_stream() call. Slot-keyed
        # ``{slot_id: TokenUsage}`` to match LlamafileClient / OllamaClient so
        # ``inference._get_usage`` reads every client uniformly. The Anthropic
        # SDK is per-call (no shared inference slot), so we always use slot 0 —
        # same convention as OllamaClient.
        self.last_usage: dict[int, TokenUsage] = {}

    async def aclose(self) -> None:
        """Close the underlying Anthropic SDK client (and its httpx pool)."""
        await self._client.close()

    def _sdk_extra_headers(
        self, extra_headers: dict[str, str] | None,
    ) -> dict[str, str] | None:
        """Per-call SDK headers, enforcing the one-credential rule.

        Validates against a static construction credential (raises on a second
        source), then re-cases auth headers to the SDK's case-sensitive
        expectation and drops SDK-pinned version headers. Passed via the SDK's
        per-call ``extra_headers=`` — never set as ``default_headers`` (the
        proxy shares one client instance, so a per-request credential must not
        leak into later requests).
        """
        return _prepare_anthropic_headers(
            resolve_request_headers(self._static_auth, extra_headers)
        )

    def _ensure_credential(self, sdk_headers: dict[str, str] | None) -> None:
        """Fail loud if this request would reach the backend with no credential.

        A credential is present iff the SDK client resolved a construction key
        (``api_key`` or ``auth_token``, incl. ambient env at build time) or this
        call carries a per-call auth header. With none, the Anthropic SDK refuses
        the request with an opaque error surfaced as HTTP 502; forge raises a
        clear MissingCredentialError (mapped to 401) instead. Never inspects the
        secret value.
        """
        if (
            self._client.api_key is None
            and self._client.auth_token is None
            and not has_auth_header(sdk_headers)
        ):
            raise MissingCredentialError("Anthropic backend")

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
    def _parse_response(response: Any) -> LLMResponse:
        """Anthropic Message → list[ToolCall] or TextResponse."""
        tool_uses: list[Any] = []
        text_parts: list[str] = []

        for block in response.content:
            if block.type == "tool_use":
                tool_uses.append(block)
            elif block.type == "text":
                text_parts.append(block.text)

        if tool_uses:
            reasoning = "\n".join(text_parts) if text_parts else None
            return [
                ToolCall(
                    tool=tu.name,
                    args=dict(tu.input),
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
        """Build kwargs dict for messages.create / messages.stream.

        ``passthrough`` (proxy use) supplies inbound-body fields the proxy
        didn't deconstruct (``max_tokens``, ``stop_sequences``, ``tool_choice``,
        ``metadata``, ``thinking``, etc.). Forge-owned fields (``model``,
        ``messages``, ``system``, ``tools``) overlay it.

        ``inbound_anthropic_body`` (proxy path 1) bypasses the deconstruct/
        rebuild path: the SDK is called with the original inbound body
        verbatim, preserving block-level fields like ``cache_control`` that
        forge.Message doesn't represent. The runner only passes this on the
        clean first-attempt call (no compaction, no retries) — see ADR-015.
        """
        if inbound_anthropic_body is not None:
            # Verbatim emit. Drop the proxy-internal ``stream`` field; the
            # SDK call shape (messages.create vs messages.stream) selects
            # streaming. ``model`` defaults to the inbound value but the
            # client's configured model wins if the inbound omitted it.
            kwargs = dict(inbound_anthropic_body)
            kwargs.pop("stream", None)
            kwargs.setdefault("model", self.model)
            return kwargs

        system, converted = self._convert_messages(messages)
        kwargs = dict(passthrough or {})
        kwargs.update({
            "model": self.model,
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
        """Send messages via the Anthropic Messages API.

        ``sampling`` is accepted for protocol symmetry but ignored —
        AnthropicClient does not currently expose sampling kwargs through
        forge. ``passthrough`` merges inbound-body extras into the SDK call.
        ``inbound_anthropic_body`` (path 1) triggers verbatim emit — see
        ADR-015 for the cache_control preservation rationale.
        ``raw_openai_tools`` accepted for protocol symmetry, ignored
        (Anthropic uses its own tool conversion). ``extra_headers`` carries the
        per-call credential, re-cased for the SDK and forwarded via the SDK's
        per-call ``extra_headers=``.
        """
        if sampling:
            log.debug(
                "AnthropicClient ignores per-call sampling overrides: %s",
                sorted(sampling.keys()),
            )
        kwargs = self._build_kwargs(
            messages, tools, passthrough, inbound_anthropic_body,
        )
        # Strip SDK control kwargs that a verbatim/passthrough body could carry
        # before forge installs its own per-call credential.
        for control_kwarg in _SDK_CONTROL_KWARGS:
            kwargs.pop(control_kwarg, None)
        sdk_headers = self._sdk_extra_headers(extra_headers)
        self._ensure_credential(sdk_headers)
        if sdk_headers:
            kwargs["extra_headers"] = sdk_headers
        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise BackendError(
                getattr(exc, "status_code", 0), str(exc)
            ) from exc
        self.last_usage = {
            0: TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cache_creation_input_tokens=getattr(
                    response.usage, "cache_creation_input_tokens", 0
                ) or 0,
                cache_read_input_tokens=getattr(
                    response.usage, "cache_read_input_tokens", 0
                ) or 0,
            )
        }
        return self._parse_response(response)

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
        """Stream via the Anthropic Messages API.

        ``sampling`` is accepted for protocol symmetry but ignored.
        ``passthrough`` merges inbound-body extras into the SDK call.
        ``inbound_anthropic_body`` (path 1) triggers verbatim emit; see ADR-015.
        ``raw_openai_tools`` accepted for protocol symmetry, ignored.
        ``extra_headers`` carries the per-call credential (see :meth:`send`).
        """
        if sampling:
            log.debug(
                "AnthropicClient ignores per-call sampling overrides: %s",
                sorted(sampling.keys()),
            )
        kwargs = self._build_kwargs(
            messages, tools, passthrough, inbound_anthropic_body,
        )
        # Strip SDK control kwargs that a verbatim/passthrough body could carry
        # before forge installs its own per-call credential.
        for control_kwarg in _SDK_CONTROL_KWARGS:
            kwargs.pop(control_kwarg, None)
        sdk_headers = self._sdk_extra_headers(extra_headers)
        self._ensure_credential(sdk_headers)
        if sdk_headers:
            kwargs["extra_headers"] = sdk_headers

        accumulated_text = ""
        # Track multiple tool_use blocks by index.
        tool_blocks: list[dict[str, str]] = []  # [{name, args}, ...]
        _current_tool_idx: int = -1
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            tool_blocks.append({
                                "name": event.content_block.name,
                                "args": "",
                            })
                            _current_tool_idx = len(tool_blocks) - 1
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            accumulated_text += event.delta.text
                            yield StreamChunk(
                                type=ChunkType.TEXT_DELTA,
                                content=event.delta.text,
                            )
                        elif event.delta.type == "input_json_delta" and _current_tool_idx >= 0:
                            tool_blocks[_current_tool_idx]["args"] += event.delta.partial_json
                            yield StreamChunk(
                                type=ChunkType.TOOL_CALL_DELTA,
                                content=event.delta.partial_json,
                            )
                    elif event.type == "content_block_stop":
                        # Reset current tool index when a block finishes
                        _current_tool_idx = -1
                    elif event.type == "message_stop":
                        if tool_blocks:
                            reasoning = accumulated_text or None
                            final: LLMResponse = [
                                ToolCall(
                                    tool=tb["name"],
                                    args=decode_tool_args(tb["args"]),
                                    reasoning=reasoning if i == 0 else None,
                                )
                                for i, tb in enumerate(tool_blocks)
                            ]
                        else:
                            final = TextResponse(content=accumulated_text)
                        yield StreamChunk(
                            type=ChunkType.FINAL, response=final
                        )
                # Grab usage from the final accumulated message.
                final_message = await stream.get_final_message()
                self.last_usage = {
                    0: TokenUsage(
                        prompt_tokens=final_message.usage.input_tokens,
                        completion_tokens=final_message.usage.output_tokens,
                        total_tokens=final_message.usage.input_tokens
                        + final_message.usage.output_tokens,
                        cache_creation_input_tokens=getattr(
                            final_message.usage, "cache_creation_input_tokens", 0
                        ) or 0,
                        cache_read_input_tokens=getattr(
                            final_message.usage, "cache_read_input_tokens", 0
                        ) or 0,
                    )
                }
        except anthropic.APIError as exc:
            raise BackendError(
                getattr(exc, "status_code", 0), str(exc)
            ) from exc

    async def get_context_length(self) -> int | None:
        """Claude models have 200K context."""
        return 200_000

    async def discover_backend_metadata(
        self, extra_headers: dict[str, str] | None = None,
    ) -> int | None:
        """Static 200K context, no probe and no identity to adopt.

        The Anthropic path reports a known context length without a network
        call, so it is never deferred — this exists for Protocol uniformity and
        ignores ``extra_headers``.
        """
        return 200_000
