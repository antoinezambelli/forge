"""Llamafile client adapter with native FC and prompt-injected fallback."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from forge.clients.base import (
    ChunkType,
    RawOpenAITools,
    StreamChunk,
    TokenUsage,
    decode_tool_args,
    format_tool,
    resolve_request_headers,
    static_auth_present,
)
from forge.clients.sampling_defaults import apply_sampling_defaults
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
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
_TOOL_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL
)


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
) -> tuple[list[ToolCall], bool]:
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
    calls: list[ToolCall] = []
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
        calls.append(ToolCall(tool=name, args=args, reasoning=None))
    return calls, bool(blocks)


def _merge_consecutive(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure strict user/assistant alternation for Jinja parity checker.

    llama-server's Mistral Jinja template counts only plain user and plain
    assistant messages (no tool_calls). Messages with tool_calls or role="tool"
    are invisible to the checker. When two plain messages of the same role
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
        is_plain = role in ("user", "assistant") and "tool_calls" not in m

        if is_plain:
            # Find the last visible (plain user/assistant) message in result
            last_visible_idx = None
            for i in range(len(result) - 1, -1, -1):
                r = result[i]
                if r.get("role") in ("user", "assistant") and "tool_calls" not in r:
                    last_visible_idx = i
                    break

            if last_visible_idx is not None and result[last_visible_idx].get("role") == role:
                # Same role at consecutive visible positions — merge
                target = result[last_visible_idx]
                result[last_visible_idx] = {
                    **target,
                    "content": target.get("content", "") + "\n\n" + m.get("content", ""),
                }
                continue

        result.append(m)
    return result


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
        elif "tool_calls" in m:
            parts: list[str] = []
            for tc_entry in m["tool_calls"]:
                tc = tc_entry["function"]
                args = tc["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                parts.append(json.dumps({"tool": tc["name"], "args": args}))
            result.append({
                "role": m["role"],
                "content": "\n".join(parts),
            })
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
        slot_id: int | None = None,
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
        defaults = apply_sampling_defaults(self.sampling_key, strict=recommended_sampling)
        self.temperature = temperature if temperature is not None else defaults.get("temperature")
        self.top_p = top_p if top_p is not None else defaults.get("top_p")
        self.top_k = top_k if top_k is not None else defaults.get("top_k")
        self.min_p = min_p if min_p is not None else defaults.get("min_p")
        self.repeat_penalty = repeat_penalty if repeat_penalty is not None else defaults.get("repeat_penalty")
        self.presence_penalty = presence_penalty if presence_penalty is not None else defaults.get("presence_penalty")
        # chat_template_kwargs is a nested dict of Jinja template variables
        # (e.g. {"reasoning_effort": "high", "enable_thinking": False}) that
        # llama-server unpacks into the chat template at render time.
        # Whole-value replacement at the field level — no nested merge.
        self.chat_template_kwargs = (
            chat_template_kwargs if chat_template_kwargs is not None
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
        self._slot_id = slot_id

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

    def _apply_slot_id(self, body: dict[str, Any]) -> None:
        """Inject slot_id into a request body if configured."""
        if self._slot_id is not None:
            body["slot_id"] = self._slot_id

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
        slot = self._slot_id if self._slot_id is not None else 0
        self.last_usage[slot] = TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def _resolve_reasoning(
        self, accumulated_reasoning: str, accumulated_content: str
    ) -> str | None:
        """Build final reasoning from accumulated streams, respecting _think flag.

        Priority: reasoning_content field > [THINK] tags in content > content fallback.
        When _think is False, discard all reasoning.
        """
        if not self._think:
            return None

        # Server already parsed reasoning_content — use it directly
        if accumulated_reasoning:
            return accumulated_reasoning

        # Try client-side [THINK] tag extraction from content
        if accumulated_content:
            think_text, _ = _extract_think_tags(accumulated_content)
            if think_text:
                return think_text
            # Content fallback (instruct model narrating before tool call)
            return accumulated_content

        return None

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
        raw_openai_tools: RawOpenAITools | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Dispatch to the native or prompt-injected path per the declared mode.

        ``inbound_anthropic_body`` is accepted for protocol symmetry and
        silently ignored — LlamafileClient only speaks OpenAI shape.

        ``raw_openai_tools`` (proxy use) is forwarded verbatim as the
        backend's ``tools`` array on the native path; the prompt path
        accepts and ignores it (it keeps forge's prompt-injection format).

        ``extra_headers`` carries the per-call credential and is forwarded to
        both the native and prompt paths.
        """
        if self.mode == "native":
            return await self._send_native(
                messages, tools, sampling, passthrough, raw_openai_tools,
                extra_headers,
            )
        return await self._send_prompt(
            messages, tools, sampling, passthrough, extra_headers,
        )

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
        raw_openai_tools: RawOpenAITools | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream via SSE, handling both native FC and prompt-injected paths.

        ``inbound_anthropic_body`` accepted for protocol symmetry, ignored.
        ``raw_openai_tools`` (proxy use) is forwarded verbatim on the native
        path; ignored on the prompt path. ``extra_headers`` carries the
        per-call credential.
        """
        mode = self.mode

        body: dict[str, Any] = dict(passthrough or {})
        body.update({
            "stream": True,
            "stream_options": {"include_usage": True},
            "cache_prompt": self._cache_prompt,
        })
        body.setdefault("model", self.model)
        self._apply_slot_id(body)
        self._apply_sampling(body, sampling)

        if mode == "native":
            prepared = _merge_consecutive(messages)
        else:
            prepared = _merge_consecutive(_downgrade_messages(messages))
        if mode == "native" and (raw_openai_tools is not None or tools):
            body["tools"] = (
                raw_openai_tools
                if raw_openai_tools is not None
                else [format_tool(t) for t in tools]
            )
            body["messages"] = prepared
        elif mode == "prompt" and tools:
            tool_prompt = build_tool_prompt(tools)
            prepared[0] = {
                **prepared[0],
                "content": tool_prompt + "\n\n" + prepared[0]["content"],
            }
            body["messages"] = prepared
        else:
            body["messages"] = prepared

        accumulated_content = ""
        accumulated_reasoning = ""
        # Track multiple tool calls by index — OpenAI streaming sends
        # tool_calls[N] deltas with an index field.
        tool_call_parts: dict[int, dict[str, str]] = {}  # idx -> {name, args}

        async with self._http.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=body,
            headers=self._request_headers(extra_headers),
        ) as response:
            if response.status_code == 500:
                error_body = ""
                async for line in response.aiter_lines():
                    error_body += line
                if _is_malformed_tool_call_500(error_body):
                    # Re-parse the model's COMPLETE calls out of the 500 body
                    # and hand them downstream to execute; skeletons missing a
                    # required param are dropped inside _rescue_tool_calls so
                    # they never reach dispatch.
                    calls, saw_blocks = _rescue_tool_calls(
                        error_body, body.get("tools")
                    )
                    if calls:
                        self.rescued_tool_calls += len(calls)
                        log.warning(
                            "[rescue] re-parsed %d tool call(s) from a "
                            "malformed-tool-call 500: %s",
                            len(calls), [(c.tool, sorted(c.args)) for c in calls],
                        )
                        yield StreamChunk(type=ChunkType.FINAL, response=calls)
                        return
                    # Recoverable: re-sample. Clean nudge, not the raw 500 JSON.
                    # Name the stutter only when blocks were actually seen.
                    yield StreamChunk(
                        type=ChunkType.FINAL,
                        response=TextResponse(
                            content=_STUTTER_RETRY_TEXT if saw_blocks
                            else _MALFORMED_TOOL_CALL_RETRY_TEXT
                        ),
                    )
                    return
                # Arbitrary backend 500 — cascade. Raw body off the message.
                raise BackendError(500, raw_body=error_body)
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

                reasoning_content = delta.get("reasoning_content") or ""
                if reasoning_content:
                    accumulated_reasoning += reasoning_content

                content = delta.get("content") or ""
                if content:
                    accumulated_content += content
                    yield StreamChunk(
                        type=ChunkType.TEXT_DELTA, content=content
                    )

            # Stream ended — build and yield FINAL response.
            if tool_call_parts:
                reasoning = self._resolve_reasoning(
                    accumulated_reasoning, accumulated_content
                )
                result_calls: list[ToolCall] = []
                for idx in sorted(tool_call_parts):
                    part = tool_call_parts[idx]
                    result_calls.append(ToolCall(
                        tool=part["name"],
                        args=decode_tool_args(part["args"]),
                        reasoning=reasoning if idx == 0 else None,
                    ))
                final: LLMResponse = result_calls
            elif mode == "prompt" and tools:
                think_text, cleaned = _extract_think_tags(
                    accumulated_content
                )
                tool_names = [t.name for t in tools]
                extracted = extract_tool_call(cleaned, tool_names)
                if extracted:
                    extracted[0].reasoning = self._resolve_reasoning(
                        accumulated_reasoning, think_text
                    )
                    final = extracted
                else:
                    final = TextResponse(content=cleaned)
            else:
                final = TextResponse(content=accumulated_content)
            yield StreamChunk(type=ChunkType.FINAL, response=final)

    async def get_context_length(self) -> int | None:
        """Query the Llamafile /props endpoint for configured context length.

        The /props endpoint is on the base server URL, NOT on the /v1 prefix.
        Parses default_generation_settings.n_ctx from the response.
        """
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]

        resp = await self._http.get(f"{base}/props")
        resp.raise_for_status()
        data = resp.json()

        try:
            n_ctx = data.get("default_generation_settings", {}).get("n_ctx")
            return int(n_ctx) if n_ctx is not None else None
        except (ValueError, KeyError, TypeError) as exc:
            raise ContextDiscoveryError(exc) from exc

    async def discover_backend_metadata(
        self, extra_headers: dict[str, str] | None = None,
    ) -> int | None:
        """Probe /props once for the context budget, credentialed.

        llama.cpp ignores the wire ``model`` field, so there is no identity to
        adopt — this returns the context length (``n_ctx``) and nothing else.
        Carries ``extra_headers`` so a gateway in front of llama.cpp can
        authenticate the probe on the first request. Raises ``BackendError`` on
        a rejected, unreachable, or unparseable probe (so the proxy maps it to a
        clean status); returns None when /props reports no ``n_ctx`` (the caller
        fails loud rather than guessing a budget).
        """
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]

        try:
            resp = await self._http.get(
                f"{base}/props", headers=self._request_headers(extra_headers),
            )
        except httpx.HTTPError as exc:
            raise BackendError(502, f"llama.cpp /props unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise BackendError(resp.status_code, raw_body=resp.text)

        try:
            body = resp.json()
        except ValueError as exc:
            raise BackendError(502, f"llama.cpp /props returned non-JSON: {exc}") from exc
        settings = body.get("default_generation_settings", {}) if isinstance(body, dict) else {}
        n_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
        if n_ctx is None:
            return None
        try:
            return int(n_ctx)
        except (ValueError, TypeError) as exc:
            raise BackendError(502, f"llama.cpp /props n_ctx not an integer: {exc}") from exc

    async def _send_native(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        raw_openai_tools: RawOpenAITools | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Send using native function calling (OpenAI tools parameter).

        When ``raw_openai_tools`` is supplied (proxy native passthrough), it is
        sent as the ``tools`` array verbatim so the backend sees the client's
        original schema instead of forge's re-emitted ``format_tool(spec)``.
        ``extra_headers`` carries the per-call credential.
        """
        merged = _merge_consecutive(messages)
        body: dict[str, Any] = dict(passthrough or {})
        body.update({
            "messages": merged,
            "cache_prompt": self._cache_prompt,
        })
        body.setdefault("model", self.model)
        self._apply_slot_id(body)
        self._apply_sampling(body, sampling)
        if raw_openai_tools is not None:
            body["tools"] = raw_openai_tools
        elif tools:
            body["tools"] = [format_tool(t) for t in tools]

        resp = await self._http.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers=self._request_headers(extra_headers),
        )
        if resp.status_code == 500:
            is_parse = _is_malformed_tool_call_500(resp.text)
            if is_parse:
                # Re-parse the model's COMPLETE calls out of the 500 body and
                # hand them downstream to execute; skeletons missing a required
                # param are dropped inside _rescue_tool_calls so they never
                # reach dispatch.
                calls, saw_blocks = _rescue_tool_calls(resp.text, body.get("tools"))
                if calls:
                    self.rescued_tool_calls += len(calls)
                    log.warning(
                        "[rescue] re-parsed %d tool call(s) from a "
                        "malformed-tool-call 500: %s",
                        len(calls), [(c.tool, sorted(c.args)) for c in calls],
                    )
                    return calls
                # Recoverable: re-sample. Return a clean nudge, not the raw 500
                # JSON. Name the stutter only when blocks were actually seen.
                return TextResponse(
                    content=_STUTTER_RETRY_TEXT if saw_blocks
                    else _MALFORMED_TOOL_CALL_RETRY_TEXT
                )
            # Arbitrary backend 500 — cascade. Raw body off the message.
            raise BackendError(500, raw_body=resp.text)
        if resp.status_code != 200:
            raise BackendError(resp.status_code, raw_body=resp.text)
        data = resp.json()
        self._record_usage(data)

        choices = data.get("choices") or []
        if not choices:
            # Raw envelope off the message (a gateway could echo a credential
            # into a 200 body too); kept on exc.body for debugging.
            raise BackendError(
                500, "response has no choices", raw_body=json.dumps(data, default=str)
            )
        choice = choices[0].get("message", {})
        raw_tool_calls = choice.get("tool_calls")
        if raw_tool_calls:
            reasoning = self._resolve_reasoning(
                choice.get("reasoning_content", ""),
                choice.get("content", ""),
            )
            result_calls: list[ToolCall] = []
            for i, tc_entry in enumerate(raw_tool_calls):
                tc_func = tc_entry.get("function", {})
                result_calls.append(ToolCall(
                    tool=tc_func.get("name", ""),
                    args=decode_tool_args(tc_func.get("arguments")),
                    reasoning=reasoning if i == 0 else None,
                ))
            return result_calls

        content = choice.get("content", "")
        # Strip [THINK] tags from text responses — reasoning is only
        # useful on ToolCall, TextResponse just gets clean content
        if content:
            _, content = _extract_think_tags(content)
        return TextResponse(content=content)

    async def _send_prompt(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Send using prompt-injected tool calling.

        ``extra_headers`` carries the per-call credential.
        """
        prepared = _merge_consecutive(_downgrade_messages(messages))
        if tools:
            tool_prompt = build_tool_prompt(tools)
            prepared[0] = {
                **prepared[0],
                "content": tool_prompt + "\n\n" + prepared[0]["content"],
            }

        body: dict[str, Any] = dict(passthrough or {})
        body.update({
            "messages": prepared,
            "cache_prompt": self._cache_prompt,
        })
        body.setdefault("model", self.model)
        self._apply_slot_id(body)
        self._apply_sampling(body, sampling)

        resp = await self._http.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers=self._request_headers(extra_headers),
        )
        resp.raise_for_status()
        data = resp.json()
        self._record_usage(data)

        top_choice = data["choices"][0]
        content = top_choice["message"].get("content", "")
        reasoning_content = top_choice["message"].get("reasoning_content", "")
        if tools:
            think_text, cleaned = _extract_think_tags(content)
            tool_names = [t.name for t in tools]
            tc_list = extract_tool_call(cleaned, tool_names)
            if tc_list:
                tc_list[0].reasoning = self._resolve_reasoning(
                    reasoning_content, think_text
                )
                return tc_list

        # Strip think tags from TextResponse — clean content only
        if content:
            _, content = _extract_think_tags(content)
        return TextResponse(content=content)
