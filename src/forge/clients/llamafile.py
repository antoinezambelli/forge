"""Llamafile client adapter with native FC and prompt-injected fallback."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from forge.clients.base import ChunkType, StreamChunk, TokenUsage, format_tool
from forge.clients.sampling_defaults import apply_sampling_defaults
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError, ContextDiscoveryError
from forge.prompts.templates import build_tool_prompt, extract_tool_call

# Model-specific thinking tag formats. Extend this list when adding new model
# families. If a model library/registry is added later, move these patterns
# into per-model profiles instead of hard-coding here.
#   - [THINK]...[/THINK]  — Mistral (Ministral Reasoning)
#   - <think>...</think>   — Qwen3, DeepSeek
_THINK_TAG_RE = re.compile(
    r"\[THINK\](.*?)\[/THINK\]|<think>(.*?)</think>", re.DOTALL
)


def _extract_think_tags(text: str) -> tuple[str, str]:
    """Extract thinking blocks from text.

    Supports [THINK]...[/THINK] (Mistral) and <think>...</think> (Qwen/DeepSeek).
    Returns (reasoning, remaining_content).
    """
    reasoning_parts: list[str] = []
    remaining = text
    for m in _THINK_TAG_RE.finditer(text):
        # group(1) is [THINK] match, group(2) is <think> match
        content = (m.group(1) or m.group(2) or "").strip()
        reasoning_parts.append(content)
    if reasoning_parts:
        remaining = _THINK_TAG_RE.sub("", text).strip()
    return "\n\n".join(reasoning_parts), remaining


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
    """OpenAI-compatible client for Llamafile.

    mode="native" uses the tools parameter (requires Llamafile with FC support).
    mode="prompt" injects tool descriptions into the prompt and extracts JSON.
    mode="auto" tries native first, falls back to prompt on failure — with
        an explicit warning log and resolved_mode set for caller inspection.
    """

    api_format: str = "openai"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        presence_penalty: float | None = None,
        mode: str = "auto",
        timeout: float = 300.0,
        think: bool | None = None,
        cache_prompt: bool = True,
        slot_id: int | None = None,
        recommended_sampling: bool = False,
    ) -> None:
        self.base_url = base_url
        self.model = model
        # Apply per-model recommended sampling defaults. Caller's explicit
        # (non-None) kwargs win over the map field-by-field.
        defaults = apply_sampling_defaults(model, strict=recommended_sampling)
        self.temperature = temperature if temperature is not None else defaults.get("temperature")
        self.top_p = top_p if top_p is not None else defaults.get("top_p")
        self.top_k = top_k if top_k is not None else defaults.get("top_k")
        self.min_p = min_p if min_p is not None else defaults.get("min_p")
        self.repeat_penalty = repeat_penalty if repeat_penalty is not None else defaults.get("repeat_penalty")
        self.presence_penalty = presence_penalty if presence_penalty is not None else defaults.get("presence_penalty")
        self.mode = mode
        self._http = httpx.AsyncClient(timeout=timeout)
        self._think: bool = think if think is not None else True  # auto = capture
        self._cache_prompt = cache_prompt
        self._slot_id = slot_id

        self.last_usage: dict[int, TokenUsage] = {}

        if mode in ("native", "prompt"):
            self.resolved_mode: str | None = mode
        else:
            self.resolved_mode = None

    def _apply_slot_id(self, body: dict[str, Any]) -> None:
        """Inject slot_id into a request body if configured."""
        if self._slot_id is not None:
            body["slot_id"] = self._slot_id

    # Sampling fields recognized in per-call overrides. ``seed`` is
    # accepted only as a per-call override (not an instance field).
    _SAMPLING_FIELDS = (
        "temperature", "top_p", "top_k", "min_p",
        "repeat_penalty", "presence_penalty", "seed",
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
            return accumulated_content or None

        return None

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Resolve mode on first call with tools, then dispatch."""
        if self.resolved_mode is None:
            return await self._resolve_and_send(messages, tools, sampling)
        elif self.resolved_mode == "native":
            return await self._send_native(messages, tools, sampling)
        else:
            return await self._send_prompt(messages, tools, sampling)

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream via SSE, handling both native FC and prompt-injected paths."""
        if self.resolved_mode is None:
            # Probe with a non-streaming call to resolve native vs prompt.
            # Result is discarded — the runner will use the streamed response.
            await self._resolve_and_send(messages, tools, sampling)
        mode = self.resolved_mode

        body: dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "cache_prompt": self._cache_prompt,
        }
        self._apply_slot_id(body)
        self._apply_sampling(body, sampling)

        if mode == "native":
            prepared = _merge_consecutive(messages)
        else:
            prepared = _merge_consecutive(_downgrade_messages(messages))
        if mode == "native" and tools:
            body["tools"] = [format_tool(t) for t in tools]
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
            "POST", f"{self.base_url}/chat/completions", json=body
        ) as response:
            if response.status_code == 500:
                error_body = ""
                async for line in response.aiter_lines():
                    error_body += line
                yield StreamChunk(
                    type=ChunkType.FINAL,
                    response=TextResponse(content=error_body),
                )
                return
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
                bad_args = False
                for idx in sorted(tool_call_parts):
                    part = tool_call_parts[idx]
                    try:
                        args = json.loads(part["args"]) if part["args"] else {}
                    except json.JSONDecodeError:
                        bad_args = True
                        break
                    result_calls.append(ToolCall(
                        tool=part["name"],
                        args=args,
                        reasoning=reasoning if idx == 0 else None,
                    ))
                if bad_args:
                    final: LLMResponse = TextResponse(
                        content=accumulated_content or part["args"],
                    )
                else:
                    final = result_calls
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

    async def _resolve_and_send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None,
        sampling: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Auto-resolve mode on first send with tools.

        Only falls back to prompt-injected mode on an HTTP error (backend
        doesn't support the tools parameter). A TextResponse with tools
        provided is not a fallback signal — it means native FC is supported
        but the model chose not to call a tool. The runner's retry logic
        handles that case.
        """
        if not tools:
            # No tools to test with — send without tools, defer resolution
            self.resolved_mode = "native"
            return await self._send_native(messages, tools, sampling)

        try:
            result = await self._send_native(messages, tools, sampling)
            self.resolved_mode = "native"
            return result
        except (httpx.HTTPStatusError, BackendError):
            self.resolved_mode = "prompt"
            return await self._send_prompt(messages, tools, sampling)

    async def _send_native(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None,
        sampling: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send using native function calling (OpenAI tools parameter)."""
        merged = _merge_consecutive(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": merged,
            "cache_prompt": self._cache_prompt,
        }
        self._apply_slot_id(body)
        self._apply_sampling(body, sampling)
        if tools:
            body["tools"] = [format_tool(t) for t in tools]

        resp = await self._http.post(
            f"{self.base_url}/chat/completions", json=body
        )
        if resp.status_code == 500:
            return TextResponse(content=resp.text)
        if resp.status_code != 200:
            raise BackendError(resp.status_code, resp.text)
        data = resp.json()
        self._record_usage(data)

        top_choice = data["choices"][0]
        choice = top_choice["message"]
        raw_tool_calls = choice.get("tool_calls")
        if raw_tool_calls:
            reasoning = self._resolve_reasoning(
                choice.get("reasoning_content", ""),
                choice.get("content", ""),
            )
            result_calls: list[ToolCall] = []
            for i, tc_entry in enumerate(raw_tool_calls):
                tc_func = tc_entry["function"]
                args = tc_func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        return TextResponse(content=choice.get("content", args))
                result_calls.append(ToolCall(
                    tool=tc_func["name"],
                    args=args,
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
    ) -> LLMResponse:
        """Send using prompt-injected tool calling."""
        prepared = _merge_consecutive(_downgrade_messages(messages))
        if tools:
            tool_prompt = build_tool_prompt(tools)
            prepared[0] = {
                **prepared[0],
                "content": tool_prompt + "\n\n" + prepared[0]["content"],
            }

        body: dict[str, Any] = {
            "model": self.model,
            "messages": prepared,
            "cache_prompt": self._cache_prompt,
        }
        self._apply_slot_id(body)
        self._apply_sampling(body, sampling)

        resp = await self._http.post(
            f"{self.base_url}/chat/completions", json=body
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
