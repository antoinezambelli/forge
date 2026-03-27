"""Ollama client adapter using native function calling."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from forge.clients.base import ChunkType, StreamChunk, TokenUsage, format_tool
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError, ThinkingNotSupportedError

_THINK_HEURISTIC_KEYWORDS = ("reason", "think")


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
        temperature: float = 0.7,
        timeout: float = 300.0,
        think: bool | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
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

    def _build_options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {"temperature": self.temperature}
        if self._num_ctx is not None:
            opts["num_ctx"] = self._num_ctx
        return opts

    def _resolve_reasoning(
        self,
        thinking: str,
        content: str,
    ) -> str | None:
        """Gate reasoning capture on _think flag.

        When _think is False, discard all reasoning.
        When True: prefer thinking field, fall back to content.
        """
        if not self._think:
            return None
        return thinking or content or None

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
    ) -> LLMResponse:
        """Send messages via /api/chat and parse the response."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._build_options(),
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
            return [
                ToolCall(
                    tool=tc["function"]["name"],
                    args=tc["function"].get("arguments", {}),
                    reasoning=reasoning if i == 0 else None,
                )
                for i, tc in enumerate(tool_calls)
            ]

        # No tool_calls and done=true → model intentionally chose text
        done = data.get("done", False)
        return TextResponse(content=msg.get("content", ""), intentional=done)

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream via NDJSON from /api/chat."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self._build_options(),
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
                                tool=tc["function"]["name"],
                                args=tc["function"].get("arguments", {}),
                                reasoning=reasoning if i == 0 else None,
                            )
                            for i, tc in enumerate(tool_calls)
                        ]
                    else:
                        content = msg.get("content", "")
                        if content:
                            accumulated_content += content
                        final = TextResponse(content=accumulated_content, intentional=True)
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
