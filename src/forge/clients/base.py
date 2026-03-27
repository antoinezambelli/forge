"""Streaming types and LLM client protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from forge.core.workflow import LLMResponse, ToolCall, TextResponse, ToolSpec


@dataclass(frozen=True)
class TokenUsage:
    """Token counts from a single LLM response.

    Populated from the server's ``usage`` field when available (e.g.
    llama-server).  Backends that don't report usage leave the client's
    ``last_usage`` empty and the context manager falls back to heuristic
    estimation.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# Both Ollama and llama-server use the OpenAI tool schema format today.
# If a backend diverges, move this back into the relevant client module.
def format_tool(spec: ToolSpec) -> dict[str, Any]:
    """Convert a ToolSpec into the OpenAI-compatible tool schema."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.get_json_schema(),
        },
    }


class ChunkType(str, Enum):
    """What kind of partial data a stream chunk carries."""

    TEXT_DELTA = "text_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    FINAL = "final"
    RETRY = "retry"


@dataclass(frozen=True)
class StreamChunk:
    """A single chunk from a streaming LLM response.

    Consumers (UI, logging) process TEXT_DELTA and TOOL_CALL_DELTA as they
    arrive. The runner ignores all chunks except FINAL, which carries the
    resolved response. On RETRY, consumers should discard the partial output
    from the failed attempt.
    """

    type: ChunkType
    content: str = ""
    response: LLMResponse | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Interface that client adapters implement.

    The client is responsible for:
    1. Sending messages to the LLM backend
    2. Parsing the response into ToolCall or TextResponse
    3. Handling native FC or prompt-injected calling internally
    4. Optionally streaming partial responses via send_stream()

    The client does NOT retry. Retry logic lives in the WorkflowRunner.
    """

    api_format: str
    """Wire format for Message.to_api_dict(): 'ollama' or 'openai'."""

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Send messages and return a parsed response.

        Returns list[ToolCall] if the model produced valid tool invocations.
        Returns TextResponse if the model produced text (reasoning, refusal,
        or malformed output that couldn't be parsed as a tool call).

        The runner inspects the response and decides whether to retry.
        """
        ...

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Send messages and yield streaming chunks.

        Yields TEXT_DELTA or TOOL_CALL_DELTA chunks as they arrive.
        The final chunk has type FINAL and carries the resolved LLMResponse
        (same list[ToolCall] | TextResponse as send() would return).

        The runner forwards chunks to its on_chunk callback for UI/logging,
        then inspects the FINAL chunk and decides whether to retry.
        """
        ...

    async def get_context_length(self) -> int | None:
        """Query the backend for its configured context window size."""
        ...
