"""Anthropic API client adapter for frontier model baselines.

Translates between forge's OpenAI-style message format (what the runner
produces) and Anthropic's native Messages API format internally.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from forge.clients.base import ChunkType, StreamChunk
from forge.core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from forge.errors import BackendError


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
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 300.0,
        max_retries: int = 3,
        tool_choice: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._tool_choice = tool_choice  # "auto", "any", or None (default=auto)
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        # Populated after each send()/send_stream() call.
        self.last_usage: dict[str, int] | None = None

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
                            args = json.loads(args)
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
        intentional = getattr(response, "stop_reason", None) == "end_turn"
        return TextResponse(content="\n".join(text_parts), intentional=intentional)

    # ── API methods ──────────────────────────────────────────────

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
    ) -> dict[str, Any]:
        """Build kwargs dict for messages.create / messages.stream."""
        system, converted = self._convert_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            if self._tool_choice:
                kwargs["tool_choice"] = {"type": self._tool_choice}
        return kwargs

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Send messages via the Anthropic Messages API."""
        kwargs = self._build_kwargs(messages, tools)
        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise BackendError(
                getattr(exc, "status_code", 0), str(exc)
            ) from exc
        self.last_usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return self._parse_response(response)

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream via the Anthropic Messages API."""
        kwargs = self._build_kwargs(messages, tools)

        accumulated_text = ""
        # Track multiple tool_use blocks by index.
        tool_blocks: list[dict[str, str]] = []  # [{name, args}, ...]
        _current_tool_idx: int = -1
        stream_stop_reason: str | None = None

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
                    elif event.type == "message_delta":
                        stream_stop_reason = getattr(event.delta, "stop_reason", None)
                    elif event.type == "message_stop":
                        if tool_blocks:
                            reasoning = accumulated_text or None
                            final: LLMResponse = [
                                ToolCall(
                                    tool=tb["name"],
                                    args=json.loads(tb["args"]) if tb["args"] else {},
                                    reasoning=reasoning if i == 0 else None,
                                )
                                for i, tb in enumerate(tool_blocks)
                            ]
                        else:
                            final = TextResponse(
                                content=accumulated_text,
                                intentional=stream_stop_reason == "end_turn",
                            )
                        yield StreamChunk(
                            type=ChunkType.FINAL, response=final
                        )
                # Grab usage from the final accumulated message.
                final_message = await stream.get_final_message()
                self.last_usage = {
                    "input_tokens": final_message.usage.input_tokens,
                    "output_tokens": final_message.usage.output_tokens,
                }
        except anthropic.APIError as exc:
            raise BackendError(
                getattr(exc, "status_code", 0), str(exc)
            ) from exc

    async def get_context_length(self) -> int | None:
        """Claude models have 200K context."""
        return 200_000
