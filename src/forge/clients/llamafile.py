"""Llamafile client adapter with native FC and prompt-injected fallback."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
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
from forge.core.messages import (
    ConversationState,
    RichMessage,
    ToolCallPart,
    ValidArguments,
    TextPart,
    ReasoningPart,
    MessageRole,
)
from forge.clients.base import RequestMetadata, InferenceRequest
from forge.core.codecs import render_openai, replay_messages
from forge.core.reasoning import REASONING_MESSAGE_FIELDS
from forge.clients.requests import request_extras
from forge.clients.replies import openai_reply, openai_stream, text_reply
from forge.errors import BackendError, ContextDiscoveryError
from forge.prompts.templates import build_tool_prompt, extract_tool_call

# Re-exported under the historical private name so existing imports
# (`from forge.clients.llamafile import _extract_think_tags`) keep working.
from forge.prompts.think_tags import extract_think_tags as _extract_think_tags

# Multi-shard GGUF naming convention: "<stem>-00001-of-00003.gguf". The shard
# index is filesystem layout, not model identity, so strip it for the
# sampling-defaults registry key.
_SHARD_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}$")

# Known file extensions for GGUF/llamafile model files. These are the only
# suffixes we strip from the filename when deriving the model identity string.
# ``Path.stem`` is NOT used here because it strips whatever follows the LAST
# dot whether or not it is a real extension — harmless for actual
# ``*.gguf``/``*.llamafile`` files, but it silently truncates BARE dotted
# model names as they arrive in proxy mode (e.g. ``mimo-v2.5`` → ``mimo-v2``).
_KNOWN_GGUF_EXTENSIONS: tuple[str, ...] = (".gguf", ".llamafile")


# A 500 whose body is llama.cpp's tool-call parser rejecting MALFORMED/INCOMPLETE
# model output (e.g. a `write` missing `content`, or a duplicated/incomplete call)
# — NOT an arbitrary backend error. This one is a transient sampling artifact and
# recoverable by re-sampling, so we surface it as a retryable text response (the
# run_inference retry loop nudges the model to re-emit a clean call) instead of
# echoing the raw error JSON into the conversation. Every OTHER 500 cascades.
#
# The gate is STRUCTURAL, not phrase-based: a 500 is rescuable iff its body
# carries generated tool-call XML (``<tool_call``/``<function=``). Only the
# model generating such a call and the backend choking on it puts that syntax in
# a 500 body — genuine faults (OOM/slot/context/CUDA) never echo generation, and
# a request-parse failure dumps JSON, not this XML. ``<tool_call`` is
# deliberately unclosed: an echo truncated mid-open-tag (EOS or token budget
# landing inside the tag) is the same re-sampleable artifact as a complete one,
# and the leading ``<`` keeps the gate structural — request and schema dumps
# carry ``"tool_calls"`` as quoted JSON, never the tag.
#
# VERSION BOUNDARY: rescue only fires on llama.cpp builds that echo the raw
# rejected generation into the 500 body. That ended at commit 581e8eca8
# ("chat: harden peg-native tool call parsing", PR #24329, first in tag b9656,
# 2026-06-15): the exception became a generic "...does not match the expected
# <format> format" and the raw generation moved to server logs only. On b9656+
# this gate never matches, and grammar hardening in the same series makes the
# malformed call rare anyway, so the 500 just cascades. Effective only on <= b9647.
def _is_malformed_tool_call_500(body: str) -> bool:
    return "<tool_call" in body or "<function=" in body


_MALFORMED_TOOL_CALL_RETRY_TEXT = (
    "(The previous tool call was malformed and rejected by the parser — likely a "
    "missing required parameter or a duplicated/incomplete call. Re-emitting a "
    "single, complete, well-formed tool call.)"
)

# Used when the 500 body demonstrably contains <tool_call> block(s) but none
# survive rescue — either mangled syntax / unknown tool names, or (the common
# case) skeleton/preview blocks dropped for missing a required parameter. A
# malformed-500 without visible blocks keeps the generic text above; we don't
# name a stutter we haven't seen. Complete blocks are returned as real
# ToolCalls and never reach this text (see _rescue_tool_calls).
_STUTTER_RETRY_TEXT = (
    "(The previous response contained a malformed tool-call block — a "
    "preview/skeleton call missing its required parameters, or a duplicated "
    "call. Re-emitting exactly ONE complete tool-call block with EVERY "
    "required parameter filled in, with no preview or skeleton block before "
    "it.)"
)

log = logging.getLogger(__name__)

# llama.cpp's "Failed to parse input at pos N: ..." message embeds the raw
# rejected generation from the failure position onward — including, in the
# stutter case, the complete well-formed call the model emitted right after
# the skeleton block. These parse that embedded text (Qwen-coder XML tool
# format). Caveat: a parameter value containing a literal "</parameter>"
# line would be truncated at it; rescue validation below rejects the block
# if that loses a required parameter.
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_TOOL_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _tool_requirements(
    tools_array: list[dict[str, Any]] | None,
) -> dict[str, tuple[set[str], dict[str, str | None]]]:
    """Map tool name -> (required param names, param JSON types) from an
    OpenAI-shape ``tools`` array (what the request body actually carried)."""
    reqs: dict[str, tuple[set[str], dict[str, str | None]]] = {}
    for entry in tools_array or []:
        fn = entry.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        required = set(params.get("required") or [])
        types = {
            key: (prop or {}).get("type")
            for key, prop in (params.get("properties") or {}).items()
        }
        reqs[name] = (required, types)
    return reqs


def _coerce_param(value: str, json_type: str | None) -> Any:
    """Coerce an XML-format (stringly-typed) param to its schema type.

    Mirrors llama.cpp's own coercion. Raises ValueError when the value
    cannot satisfy the declared type — the caller drops the block.
    """
    if json_type == "integer":
        return int(value.strip())
    if json_type == "number":
        return float(value.strip())
    if json_type == "boolean":
        lowered = value.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        raise ValueError(f"not a boolean: {value!r}")
    return value


def _rescue_tool_calls(
    error_body: str,
    tools_array: list[dict[str, Any]] | None,
) -> tuple[list[ToolCallPart], bool]:
    """Re-parse the COMPLETE tool calls out of a malformed-tool-call 500 body.

    Returns ``(calls, saw_blocks)``. A block is returned only when it parses,
    names a tool present in the request's ``tools`` array, AND carries every
    parameter that tool marks required. Skeleton/preview blocks missing a
    required param are DROPPED — dispatching them would raise a
    ``[ToolError] TypeError`` on the tool channel and, in the write-stutter
    case (skeleton immediately followed by the complete call), drive the model
    into consecutive-tool-error exhaustion. The complete call from that same
    stutter survives the filter and is returned alone; a skeleton-only body
    yields no calls, so the caller falls through to a clean re-emit nudge.

    Unknown tool names are never returned (nothing we can check against the
    request), exact duplicates are deduped, and param values are coerced to
    their declared schema types where possible (kept as raw strings when
    coercion fails, matching the native path — a present-but-malformed arg
    still satisfies the required-set check and the tool decides). ``saw_blocks``
    reports whether any ``<tool_call>`` block was visible at all — it selects
    the stutter-specific retry text when parsing comes up empty.
    """
    try:
        message = json.loads(error_body).get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        message = error_body
    if not isinstance(message, str) or not message:
        message = error_body

    requirements = _tool_requirements(tools_array)
    calls: list[ToolCallPart] = []
    seen: set[tuple[str, str]] = set()
    blocks = _TOOL_CALL_BLOCK_RE.findall(message)
    for name, params_text in blocks:
        spec = requirements.get(name)
        if spec is None:
            continue  # unknown tool — never fabricate a call we can't check
        required, types = spec
        args: dict[str, Any] = {}
        for key, raw_value in _TOOL_PARAM_RE.findall(params_text):
            try:
                args[key] = _coerce_param(raw_value, types.get(key))
            except ValueError:
                args[key] = raw_value
        if not required.issubset(args):
            continue  # skeleton/preview block missing a required param — drop it
        dedupe_key = (name, json.dumps(args, sort_keys=True, default=str))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        calls.append(ToolCallPart(name, ValidArguments(args)))
    return calls, bool(blocks)


def _merge_consecutive(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure strict user/assistant alternation for Jinja parity checker.

    llama-server's Mistral Jinja template counts only plain user and plain
    assistant messages (absent, null, or empty tool_calls). Messages with
    nonempty tool_calls or role="tool" are invisible to the checker.
    When two plain messages of the same role
    would appear at consecutive visible positions, merge them to avoid a 500.

    This handles:
    - Adjacent same-role messages (retry nudge after user input)
    - Same-role messages separated by invisible messages (step nudge after
      user → assistant(tc) → tool cycles)
    """
    if not messages:
        return messages

    result: list[dict[str, Any]] = [messages[0]]
    for m in messages[1:]:
        role = m.get("role")
        is_plain = role in ("user", "assistant") and not m.get("tool_calls")

        if is_plain:
            # Find the last visible (plain user/assistant) message in result
            last_visible_idx = None
            for i in range(len(result) - 1, -1, -1):
                r = result[i]
                if r.get("role") in ("user", "assistant") and not r.get("tool_calls"):
                    last_visible_idx = i
                    break

            if (
                last_visible_idx is not None
                and result[last_visible_idx].get("role") == role
            ):
                # Same role at consecutive visible positions — merge.
                target = result[last_visible_idx]
                left, right = target.get("content", ""), m.get("content", "")
                if left is None:
                    left = ""
                if right is None:
                    right = ""
                if isinstance(left, str) and isinstance(right, str):
                    content = left + "\n\n" + right
                else:
                    left = (
                        [{"type": "text", "text": left}]
                        if isinstance(left, str)
                        else left
                    )
                    right = (
                        [{"type": "text", "text": right}]
                        if isinstance(right, str)
                        else right
                    )
                    content = left + [{"type": "text", "text": "\n\n"}] + right
                merged = {**target, "content": content}
                if role == "assistant":
                    # These are lowered typed parts, not first-message-only extras.
                    for key in REASONING_MESSAGE_FIELDS:
                        if key in m:
                            merged[key] = (
                                merged[key] + "\n" + m[key] if key in merged else m[key]
                            )
                result[last_visible_idx] = merged
                continue

        result.append(m)
    return result


def _prompt_text(content: Any) -> str:
    """Textualize supported content for the existing prompt prefix."""
    if isinstance(content, list):
        return "\n".join(
            block if isinstance(block, str) else block["text"] for block in content
        )
    return content if content is not None else ""


def _downgrade_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downgrade messages for llamafile prompt-injected compatibility.

    - role='tool' → role='user' (backend doesn't support tool role)
    - Structured tool_calls on assistant messages → JSON tool call format
      matching the prompt instruction format, so history acts as few-shot
      examples of the expected output.
    """
    result: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "tool":
            result.append({**m, "role": "user"})
        elif m.get("tool_calls"):
            parts: list[str] = []
            for tc_entry in m["tool_calls"]:
                tc = tc_entry["function"]
                args = tc["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                parts.append(json.dumps({"tool": tc["name"], "args": args}))
            result.append(
                {
                    "role": m["role"],
                    "content": "\n".join(parts),
                }
            )
        else:
            result.append(m)
    return result


class LlamafileClient:
    """OpenAI-compatible client for Llamafile / llama.cpp.

    The capability is declared once at construction and frozen — there is no
    runtime auto-detection. ``mode`` is one of:

    - ``"native"`` (default): forwards tools via the ``tools`` parameter
      (requires a backend with native function calling — llama.cpp ``--jinja``).
    - ``"prompt"``: injects tool descriptions into the prompt and parses the
      JSON tool call back out; for backends without native FC.

    Native-first is the default because function-calling support across local
    models has matured to the point where it is the more reliable path.
    Prompt-injection remains fully supported as an explicit opt-in: it is the
    theoretically correct fallback when a backend can't do native FC, but be
    aware that on more complex, multi-step interactions models tend to struggle
    to drive the prompt-injected protocol reliably. Choose ``"prompt"`` only
    when the backend leaves no alternative.
    """

    api_format: str = "openai"

    @staticmethod
    def _derive_sampling_key(wire_id: str | Path) -> str:
        """Derive the sampling-registry lookup key from the wire model id.

        The wire id is a GGUF/llamafile filename (e.g. ``mimo-v2.5.gguf``) or,
        in proxy mode, a bare model name (e.g. ``mimo-v2.5``).  Known file
        extensions are stripped explicitly; shard suffixes are then removed.

        Interior dots are preserved — only a trailing ``.gguf`` / ``.llamafile``
        is consumed.  This stays separate from
        ``VLLMClient._derive_sampling_key`` (``vllm.py:135``) because the two
        backends use different artifact conventions (GGUF extension + shard
        suffix vs. filesystem dirs / HF repo ids), not shared logic.

        For llamafile the wire id and the registry key are the same string, so
        ``model == sampling_key`` is an invariant of ``LlamafileClient``.
        """
        path = Path(wire_id)
        name = path.name
        for ext in _KNOWN_GGUF_EXTENSIONS:
            name = name.removesuffix(ext)
        return _SHARD_SUFFIX_RE.sub("", name)

    def __init__(
        self,
        gguf_path: str | Path,
        base_url: str = "http://localhost:8080/v1",
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        presence_penalty: float | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        mode: str = "native",
        timeout: float = 300.0,
        think: bool | None = None,
        cache_prompt: bool = True,
        id_slot: int | None = None,
        recommended_sampling: bool = False,
        api_key: str = "",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if mode not in ("native", "prompt"):
            raise ValueError(
                f"mode must be 'native' or 'prompt', got {mode!r}. "
                "Runtime auto-detection was removed — declare the backend "
                "capability explicitly (native-first; 'prompt' for non-FC "
                "backends)."
            )
        self.base_url = base_url
        connection = normalize_connection(base_url, ConnectionInputKind.OPENAI_API_BASE)
        self._chat_url = resolve_endpoint(
            EndpointLayout.LLAMA_OPENAI,
            BackendOperation.INFERENCE,
            connection,
        )
        self._props_url = resolve_endpoint(
            EndpointLayout.LLAMA_OPENAI,
            BackendOperation.PROPERTIES,
            connection,
        )
        # Count of tool calls salvaged from malformed-tool-call 500 bodies —
        # eval drivers can surface this so rescued runs stay auditable.
        self.rescued_tool_calls = 0
        # gguf_path is the source path. self.model is the derived identity
        # (filename minus .gguf/.llamafile suffix, minus shard index) used as
        # the wire "model" field (llama-server ignores it but it flows into
        # eval JSONL rows). sampling_key is the registry-lookup key; for
        # llamafile it equals the model, so the wire id and the lookup key
        # are the same string.
        self.gguf_path = Path(gguf_path)
        self.model = self._derive_sampling_key(self.gguf_path)
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
        # chat_template_kwargs is a nested dict of Jinja template variables
        # (e.g. {"reasoning_effort": "high", "enable_thinking": False}) that
        # llama-server unpacks into the chat template at render time.
        # Whole-value replacement at the field level — no nested merge.
        self.chat_template_kwargs = (
            chat_template_kwargs
            if chat_template_kwargs is not None
            else defaults.get("chat_template_kwargs")
        )
        self.mode = mode
        # Optional backend auth (api_key → Authorization: Bearer; extra_headers
        # ride on top). This is the proxy's OpenAI-shape external client, so a
        # gateway in front of llama.cpp may require a credential. One credential
        # per request: validate the static config (raises on api_key + a
        # construction auth header) BEFORE opening the pool so a conflict fails
        # fast; a per-call auth header is later refused when a static one is set.
        self._static_auth = static_auth_present(api_key, extra_headers)
        headers: dict[str, str] = {}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._http = httpx.AsyncClient(headers=headers, timeout=timeout)
        self._think: bool = think if think is not None else True  # think=None → capture
        self._cache_prompt = cache_prompt
        self._id_slot = id_slot

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

    def _apply_id_slot(self, body: dict[str, Any]) -> None:
        """Inject id_slot into a request body if configured."""
        if self._id_slot is not None:
            body["id_slot"] = self._id_slot

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

        llama-server accepts temperature/top_p/top_k/min_p/repeat_penalty/
        presence_penalty/seed as top-level OpenAI-compatible body fields.
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
        """Extract usage from a response and store it keyed by slot ID."""
        usage = data.get("usage")
        if not usage:
            return
        slot = self._id_slot if self._id_slot is not None else 0
        normalized = TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
        self.last_usage[slot] = normalized
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
        prepared = _merge_consecutive(
            messages if self.mode == "native" else _downgrade_messages(messages)
        )
        body = request_extras(metadata, "openai")
        body.update(messages=prepared, cache_prompt=self._cache_prompt)
        body.setdefault("model", self.model)
        self._apply_id_slot(body)
        self._apply_sampling(body, metadata.sampling)
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        if self.mode == "native" and tools is not None:
            body["tools"] = [format_tool(t) for t in tools]
        elif self.mode == "prompt" and tools:
            prepared[0] = {
                **prepared[0],
                "content": build_tool_prompt(tools)
                + "\n\n"
                + _prompt_text(prepared[0]["content"]),
            }
        headers = httpx.Headers(self._http.headers)
        headers.update(self._request_headers(metadata.extra_headers) or {})
        return InferenceRequest(
            body,
            self._chat_url,
            dict(headers),
            tuple(t.name for t in tools or []),
        )

    def _prompt_reply(
        self, reply: RichMessage, request: InferenceRequest
    ) -> RichMessage:
        if (
            self.mode == "prompt"
            and request.tool_names
            and not reply.tool_calls
        ):
            calls = extract_tool_call(
                "\n".join(p.text for p in reply.parts if isinstance(p, TextPart)),
                list(request.tool_names),
            )
            if calls:
                reply.parts = [
                    p for p in reply.parts if isinstance(p, ReasoningPart)
                ] + calls
        return reply

    def _malformed_reply(self, body: str, request: InferenceRequest) -> RichMessage:
        calls, saw_blocks = _rescue_tool_calls(body, request.payload.get("tools"))
        if calls:
            self.rescued_tool_calls += len(calls)
            log.warning(
                "[rescue] re-parsed %d tool call(s) from a malformed-tool-call 500",
                len(calls),
            )
            return RichMessage(MessageRole.ASSISTANT, calls)
        return text_reply(
            _STUTTER_RETRY_TEXT if saw_blocks else _MALFORMED_TOOL_CALL_RETRY_TEXT
        )

    async def send(self, request: InferenceRequest) -> RichMessage:
        resp = await self._http.post(
            request.url, json=request.payload, headers=request.headers
        )
        if (
            self.mode == "native"
            and resp.status_code == 500
            and _is_malformed_tool_call_500(resp.text)
        ):
            return self._malformed_reply(resp.text, request)
        if resp.status_code != 200:
            raise BackendError(resp.status_code, raw_body=resp.text)
        data = resp.json()
        self._record_usage(data)
        choices = data.get("choices") or []
        if not choices:
            raise BackendError(
                500, "response has no choices", raw_body=json.dumps(data, default=str)
            )
        return self._prompt_reply(
            openai_reply(choices[0].get("message", {}), capture_reasoning=self._think),
            request,
        )

    async def send_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[StreamChunk]:
        async with self._http.stream(
            "POST", request.url, json=request.payload, headers=request.headers
        ) as response:
            if response.status_code != 200:
                raw = (await response.aread()).decode(errors="replace")
                if response.status_code == 500 and _is_malformed_tool_call_500(raw):
                    yield StreamChunk(
                        ChunkType.FINAL, response=self._malformed_reply(raw, request)
                    )
                    return
                raise BackendError(response.status_code, raw_body=raw)
            async for chunk in openai_stream(
                response, self._record_usage, capture_reasoning=self._think
            ):
                if chunk.type == ChunkType.FINAL:
                    chunk = StreamChunk(
                        ChunkType.FINAL,
                        response=self._prompt_reply(chunk.response, request),
                    )
                yield chunk

    async def get_context_length(self) -> int | None:
        """Query the Llamafile /props endpoint for configured context length.

        The /props endpoint is on the base server URL, NOT on the /v1 prefix.
        Parses default_generation_settings.n_ctx from the response.
        """
        resp = await self._http.get(self._props_url)
        resp.raise_for_status()
        data = resp.json()

        try:
            n_ctx = data.get("default_generation_settings", {}).get("n_ctx")
            return int(n_ctx) if n_ctx is not None else None
        except (ValueError, KeyError, TypeError) as exc:
            raise ContextDiscoveryError(exc) from exc
