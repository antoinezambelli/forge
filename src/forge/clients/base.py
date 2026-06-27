"""Streaming types and LLM client protocol."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from forge.core.workflow import LLMResponse, ToolSpec
from forge.errors import MultipleCredentialsError

# Verbatim OpenAI-shape payloads forwarded by the proxy. The proxy hands the
# client the user's original ``tools`` array so the backend sees the exact
# schema the client authored, instead of forge's reconstructed ToolSpec.
RawOpenAITools = list[dict[str, Any]]
RawOpenAIMessages = list[dict[str, Any]]


# ── Auth credential helpers ──────────────────────────────────────────
#
# forge carries exactly ONE credential to the backend, placed in the
# backend's native auth header. It does not validate the credential, manage
# its lifecycle, or form any opinion on its value — it only detects presence
# and relocates the header to the target protocol's canonical slot. Two
# credentials present anywhere is a hard error (Design Principle #1: fail
# loud, never silently merge or pick a winner).

# Header names that carry a backend credential (lowercased for
# case-insensitive comparison).
AUTH_HEADER_NAMES = frozenset({"authorization", "x-api-key"})


def has_auth_header(headers: Mapping[str, str] | None) -> bool:
    """True if any key in ``headers`` is a recognized auth header."""
    if not headers:
        return False
    return any(name.lower() in AUTH_HEADER_NAMES for name in headers)


def static_auth_present(
    api_key: str | None,
    construction_headers: Mapping[str, str] | None,
) -> bool:
    """Whether a static credential was configured, refusing two static sources.

    A construction ``api_key`` AND a construction auth header — or two auth
    headers in the construction set — is more than one credential, all of which
    would ride on every request. Refuse it at construction (fail loud, Design
    Principle #1) rather than silently sending several. A blank/whitespace
    ``api_key`` (or scheme-only auth header) is not a credential and is ignored.
    Returns True when exactly one static credential is present, False when none.
    """
    count = (1 if (api_key and api_key.strip()) else 0) + count_auth_credentials(
        construction_headers,
    )
    if count > 1:
        raise MultipleCredentialsError(
            "more than one static credential at construction "
            "(api_key and/or auth headers)"
        )
    return count == 1


def resolve_request_headers(
    static_auth_present: bool,
    extra_headers: Mapping[str, str] | None,
) -> dict[str, str] | None:
    """Validate the one-credential rule and return per-call headers to apply.

    If the client already holds a static auth credential (set at
    construction) AND this call supplies its own auth header, that is two
    credentials — refuse it (fail loud; never merge, never pick a winner).

    Returns a plain-dict copy of ``extra_headers`` to pass as the per-call
    ``headers=`` (httpx merges it over the construction headers, request
    winning), or ``None`` when there are no per-call headers. Callers MUST
    pass these per call and MUST NOT mutate the shared client's construction
    headers — the proxy reuses one client instance across serialized
    requests, so a mutated credential would leak into later calls.
    """
    if not extra_headers:
        return None
    per_call = count_auth_credentials(extra_headers)
    if per_call > 1:
        raise MultipleCredentialsError("more than one per-call auth header")
    if static_auth_present and per_call >= 1:
        raise MultipleCredentialsError(
            "client construction credential + per-call auth header"
        )
    return dict(extra_headers)


def redact_auth_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Return a copy of ``headers`` with auth values masked for logging.

    Never log a raw secret — not even a prefix of an opaque token. The header
    *name* is preserved (so logs show which slot carried the credential); the
    value is replaced wholesale with ``***``.
    """
    if not headers:
        return {}
    return {
        k: ("***" if k.lower() in AUTH_HEADER_NAMES else v)
        for k, v in headers.items()
    }


BEARER_PREFIX = "bearer "


def auth_credential_token(name: str, value: str) -> str:
    """The secret token carried by an auth header value (``''`` if none).

    Strips a ``Bearer `` scheme from an ``Authorization`` value; ``x-api-key`` is
    already the raw token. A scheme-only ``Bearer `` or a blank/whitespace value
    yields ``''`` — i.e. the header is present but carries no credential, so it
    must not be counted or forwarded as one.
    """
    if name.lower() == "authorization" and value[: len(BEARER_PREFIX)].lower() == BEARER_PREFIX:
        return value[len(BEARER_PREFIX):].strip()
    return value.strip()


def count_auth_credentials(headers: Mapping[str, str] | None) -> int:
    """Number of headers that actually carry an auth credential.

    Counts recognized auth headers whose value resolves to a non-empty token;
    blank or scheme-only auth headers do not count. forge carries exactly one
    credential, so callers refuse a bag that yields more than one.
    """
    if not headers:
        return 0
    return sum(
        1 for name, value in headers.items()
        if name.lower() in AUTH_HEADER_NAMES and value and auth_credential_token(name, value)
    )


@dataclass(frozen=True)
class TokenUsage:
    """Token counts from a single LLM response.

    Populated from the server's ``usage`` field when available (e.g.
    llama-server).  Backends that don't report usage leave the client's
    ``last_usage`` empty and the context manager falls back to heuristic
    estimation.

    ``cache_creation_input_tokens`` / ``cache_read_input_tokens`` are
    Anthropic prompt-cache counters (0 for backends without caching, or when
    caching is off). ``prompt_tokens`` stays the *uncached* input sliver and
    ``total_tokens`` stays ``prompt + completion`` — the cache counters are
    carried separately so cost can price them (write 1.25×, read 0.1× of the
    input rate) without shifting any existing consumer's semantics.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


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


def flatten_content_to_text(content: Any) -> str:
    """Flatten OpenAI multi-part ``content`` blocks into a plain string.

    OpenAI allows ``content`` as a list of parts (e.g.
    ``[{"type": "text", "text": "..."}]``). Text parts and bare-string parts
    are joined with newlines; non-text dict blocks (images, audio, …) are
    dropped — forge is text-only on this path today. A string passes through
    unchanged; ``None`` (and any other shape) degrades to ``""``/``str()``.
    """
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    if content is None:
        return ""
    return content if isinstance(content, str) else str(content)


def decode_tool_args(raw: Any) -> Any:
    """Decode a tool-call ``arguments`` payload, fail-loud.

    JSON-string args are parsed; on malformed JSON the raw string is returned
    unchanged (a non-dict). ``ResponseValidator``'s args-shape check then routes
    it through the tool-error channel instead of crashing the parser or coercing
    to ``{}`` — so a structural arg failure rides the same lane as a runtime
    tool error rather than a trailing retry nudge.

    Non-string payloads (an already-decoded dict from Ollama / the Anthropic
    SDK, or any other shape) pass through untouched for the validator to judge.
    A missing or empty payload is a no-arg call (``{}``).
    """
    if raw is None:
        return {}
    if not isinstance(raw, str):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


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

    model: str
    """The backend model identity, sent verbatim as the wire "model" field
    (the served-model-name, gguf stem, or model tag depending on backend).
    Distinct from any sampling-registry lookup key a client also derives."""

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
        """Send messages and return a parsed response.

        Returns list[ToolCall] if the model produced valid tool invocations.
        Returns TextResponse if the model produced text (reasoning, refusal,
        or malformed output that couldn't be parsed as a tool call).

        The runner inspects the response and decides whether to retry.

        Args:
            messages: API-format messages to send.
            tools: Tool specs to include with the request.
            sampling: Optional per-call sampling overrides
                (``temperature``, ``top_p``, ``top_k``, ``min_p``,
                ``repeat_penalty``, ``presence_penalty``, ``seed``).
                Per-call values win over instance state for this call only;
                the client's instance fields are not mutated.
            passthrough: Optional dict of inbound body fields forge doesn't
                own. The client merges these into the outbound body before
                overlaying its own fields (model, messages, tools, sampling).
                Used by the proxy to preserve user intent (max_tokens, stop,
                tool_choice, etc.) without forge having to enumerate every
                supported field. None = no extras to merge.
            inbound_anthropic_body: Path-1 only — when set, the AnthropicClient
                will send this body verbatim (bypassing its deconstruct/rebuild
                path) to preserve block-level Anthropic fields like
                ``cache_control``. The runner clears this kwarg on any
                forge-mutation (retry / compaction / context warning) so
                only the clean first-attempt call rides verbatim. Other
                clients accept and ignore. See ADR-015.
            raw_openai_tools: Proxy-only — the client's verbatim OpenAI
                ``tools`` array. When set, LlamafileClient's native path sends
                it as-is instead of re-emitting ``format_tool(spec)``, so the
                backend sees the original schema (no name/schema drift). Other
                clients accept and ignore.
            extra_headers: Per-call HTTP headers, primarily the one credential
                forge forwards to the backend (proxy: a relocated inbound auth
                header; WorkflowRunner: a rotating SSO token). Applied per call
                over the construction headers (request wins). If the client
                already holds a static auth credential, supplying an auth
                header here raises ``MultipleCredentialsError`` — exactly one
                credential reaches the backend. None = no per-call headers.
        """
        ...

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
        """Send messages and yield streaming chunks.

        Yields TEXT_DELTA or TOOL_CALL_DELTA chunks as they arrive.
        The final chunk has type FINAL and carries the resolved LLMResponse
        (same list[ToolCall] | TextResponse as send() would return).

        The runner forwards chunks to its on_chunk callback for UI/logging,
        then inspects the FINAL chunk and decides whether to retry.

        Args:
            messages: API-format messages to send.
            tools: Tool specs to include with the request.
            sampling: Optional per-call sampling overrides (see ``send``).
                Per-call values win over instance state without mutating self.
            passthrough: Optional inbound-body extras dict (see ``send``).
            inbound_anthropic_body: Optional path-1 verbatim body (see ``send``).
            raw_openai_tools: Optional verbatim OpenAI tools array (see ``send``).
            extra_headers: Optional per-call credential header (see ``send``).
        """
        ...

    async def get_context_length(self) -> int | None:
        """Query the backend for its configured context window size."""
        ...

    async def discover_backend_metadata(
        self, extra_headers: dict[str, str] | None = None,
    ) -> int | None:
        """Probe the backend once, credentialed by ``extra_headers``.

        The deferred counterpart to startup discovery: adopt any backend-owned
        wire identity into this client (e.g. vLLM's served model name) as a
        side effect, and return the discovered context budget — or None when the
        backend exposes no context length. Carrying ``extra_headers`` lets the
        proxy run this lazily on the first request so it authenticates with that
        request's inbound credential (external passthrough mode). Raises
        ``BackendError`` if the probe is rejected or returns an unusable shape.
        """
        ...

    async def aclose(self) -> None:
        """Release held network resources (e.g. the httpx connection pool)."""
        ...
