"""LiteLLM client adapter — access 100+ LLM providers through one interface.

Works with any model supported by LiteLLM (OpenAI, Anthropic, Azure,
Bedrock, Vertex, Groq, Together, Mistral, Cohere, etc.) via the
``litellm`` Python SDK. Unlike ``OpenAICompatClient`` (which speaks raw
HTTP to a single OpenAI-compatible endpoint), this client delegates
provider routing, auth, and format translation to ``litellm.acompletion``
— so the caller supplies a LiteLLM model string (e.g.
``anthropic/claude-sonnet-4-6``) and an API key, and the SDK handles the
rest.

Requires the ``litellm`` optional extra::

    pip install forge-guardrails[litellm]
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from forge.clients.base import (
    ChunkType,
    StreamChunk,
    TokenUsage,
    decode_tool_args,
    format_tool,
)
from forge.core.reasoning import REASONING_MESSAGE_FIELDS
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError
from forge.prompts.think_tags import extract_think_tags


def _import_litellm():
    """Lazy-import ``litellm`` so the module loads without the optional extra."""
    try:
        import litellm
        return litellm
    except ImportError as exc:
        raise ImportError(
            "litellm is required for LiteLLMClient. "
            "Install it with: pip install forge-guardrails[litellm]"
        ) from exc


class LiteLLMClient:
    """LLM client that routes through the ``litellm`` Python SDK.

    Accepts any model identifier that ``litellm.acompletion`` supports
    (e.g. ``openai/gpt-4o``, ``anthropic/claude-sonnet-4-6``,
    ``bedrock/anthropic.claude-3-sonnet``). Provider-specific auth is
    supplied via ``api_key`` (forwarded as the provider's native key) or
    via environment variables that ``litellm`` reads automatically
    (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.).

    ``drop_params=True`` is the default so provider-unsupported kwargs
    (``seed``, ``presence_penalty``, ``strict``, etc.) are silently
    dropped instead of causing 400 errors on providers that reject them.
    """

    api_format: str = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        drop_params: bool = True,
        timeout: float = 120.0,
    ) -> None:
        _import_litellm()
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.drop_params = drop_params
        self.timeout = timeout
        self.last_usage: dict[int, TokenUsage] = {}

    async def aclose(self) -> None:
        """No persistent connection pool to close."""

    def _base_kwargs(self) -> dict[str, Any]:
        """Common kwargs for every ``litellm.acompletion`` call."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "drop_params": self.drop_params,
            "timeout": self.timeout,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return kwargs

    def _sampling_kwargs(
        self, sampling: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Resolve sampling params (per-call overrides win over instance)."""
        out: dict[str, Any] = {}
        for field in ("temperature", "top_p", "presence_penalty"):
            override = (sampling or {}).get(field)
            if override is not None:
                out[field] = override
            else:
                val = getattr(self, field, None)
                if val is not None:
                    out[field] = val
        seed = (sampling or {}).get("seed")
        if seed is not None:
            out["seed"] = seed
        return out

    def _record_usage(self, usage: Any) -> None:
        if not usage:
            return
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        self.last_usage[0] = TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=getattr(usage, "total_tokens", 0) or (prompt + completion),
        )

    @staticmethod
    def _structured_reasoning(msg: dict[str, Any]) -> str:
        for field in REASONING_MESSAGE_FIELDS:
            val = msg.get(field)
            if not val:
                continue
            if not isinstance(val, str):
                raise BackendError(
                    500,
                    f"reasoning field {field!r} is {type(val).__name__}, not a "
                    f"string: {val!r}",
                )
            return val
        return ""

    @staticmethod
    def _resolve_reasoning(structured: str, content: str) -> str | None:
        if structured:
            return structured
        think, _ = extract_think_tags(content)
        return think or None

    @staticmethod
    def _parse_tool_calls(
        tool_calls: list[Any],
        *,
        reasoning: str | None = None,
    ) -> LLMResponse:
        parsed: list[ToolCall] = []
        for i, tc in enumerate(tool_calls):
            fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
            name = fn.name if hasattr(fn, "name") else fn.get("name", "")
            raw_args = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments")
            parsed.append(ToolCall(
                tool=name,
                args=decode_tool_args(raw_args),
                reasoning=reasoning if i == 0 else None,
            ))
        return parsed

    # ── send ─────────────────────────────────────────────────────────

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
        del inbound_anthropic_body, raw_openai_tools
        litellm = _import_litellm()

        kwargs = {**self._base_kwargs(), **self._sampling_kwargs(sampling)}
        kwargs["messages"] = messages
        if tools:
            kwargs["tools"] = [format_tool(t) for t in tools]
        if passthrough:
            for k, v in passthrough.items():
                if k not in kwargs:
                    kwargs[k] = v

        try:
            resp = await litellm.acompletion(**kwargs)
        except Exception as exc:
            qualname = f"{type(exc).__module__}.{type(exc).__name__}"
            if "litellm" in qualname:
                status = getattr(exc, "status_code", 500) or 500
                raise BackendError(status, str(exc)) from exc
            raise

        self._record_usage(getattr(resp, "usage", None))

        choices = resp.choices or []
        if not choices:
            raise BackendError(500, f"response has no choices: {resp}")
        msg = choices[0].message
        msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            return self._parse_tool_calls(
                tool_calls,
                reasoning=self._resolve_reasoning(
                    self._structured_reasoning(msg_dict),
                    getattr(msg, "content", None) or "",
                ),
            )
        _, content = extract_think_tags(getattr(msg, "content", None) or "")
        return TextResponse(content=content)

    # ── streaming ────────────────────────────────────────────────────

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
        del inbound_anthropic_body, raw_openai_tools
        litellm = _import_litellm()

        kwargs = {**self._base_kwargs(), **self._sampling_kwargs(sampling)}
        kwargs["messages"] = messages
        kwargs["stream"] = True
        if tools:
            kwargs["tools"] = [format_tool(t) for t in tools]
        if passthrough:
            for k, v in passthrough.items():
                if k not in kwargs:
                    kwargs[k] = v

        accumulated_content = ""
        accumulated_reasoning = ""
        tool_calls: dict[int, dict[str, Any]] = {}

        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:
            qualname = f"{type(exc).__module__}.{type(exc).__name__}"
            if "litellm" in qualname:
                status = getattr(exc, "status_code", 500) or 500
                raise BackendError(status, str(exc)) from exc
            raise

        async for chunk in response:
            usage = getattr(chunk, "usage", None)
            if usage:
                self._record_usage(usage)

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta

            content = getattr(delta, "content", None)
            if content:
                accumulated_content += content
                yield StreamChunk(type=ChunkType.TEXT_DELTA, content=content)

            delta_dict = delta.model_dump() if hasattr(delta, "model_dump") else {}
            for field in REASONING_MESSAGE_FIELDS:
                frag = delta_dict.get(field)
                if not frag:
                    continue
                if not isinstance(frag, str):
                    raise BackendError(
                        500,
                        f"streamed reasoning field {field!r} is "
                        f"{type(frag).__name__}, not a string: {frag!r}",
                    )
                accumulated_reasoning += frag

            for tc in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc, "index", 0) or 0
                slot = tool_calls.setdefault(
                    idx, {"function": {"name": "", "arguments": ""}}
                )
                fn = tc.function if hasattr(tc, "function") else {}
                name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else "")
                if name:
                    slot["function"]["name"] += str(name)
                args_frag = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else None)
                if args_frag is not None:
                    slot["function"]["arguments"] += (
                        args_frag if isinstance(args_frag, str) else json.dumps(args_frag)
                    )

        if tool_calls:
            ordered = [tool_calls[i] for i in sorted(tool_calls)]
            final: LLMResponse = self._parse_tool_calls(
                ordered,
                reasoning=self._resolve_reasoning(
                    accumulated_reasoning, accumulated_content,
                ),
            )
        else:
            _, text = extract_think_tags(accumulated_content)
            final = TextResponse(content=text)
        yield StreamChunk(type=ChunkType.FINAL, response=final)

    async def get_context_length(self) -> int | None:
        """Query litellm's model info for the context window size."""
        litellm = _import_litellm()
        try:
            info = litellm.get_model_info(self.model)
            return info.get("max_input_tokens") or info.get("max_tokens")
        except Exception:
            return None

    async def discover_backend_metadata(
        self, extra_headers: dict[str, str] | None = None,
    ) -> int | None:
        return await self.get_context_length()
