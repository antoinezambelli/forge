"""Proxy credential handling (forge v0.8.0 Phase C).

Covers forge.proxy.auth (inbound extraction, cross-protocol relocation, the
two-source rule) and the handler threading the resolved credential to the
backend client.
"""

from unittest.mock import AsyncMock

import pytest

from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.core.workflow import TextResponse, ToolCall
from forge.errors import MultipleCredentialsError
from forge.proxy.auth import (
    DUPLICATE_AUTH_MARKER,
    extract_inbound_credential,
    relocate_credential,
    resolve_inbound_credential,
)
from forge.proxy.handler import handle_chat_completions


# ── extract_inbound_credential ───────────────────────────────────────


class TestExtractInboundCredential:
    def test_none_present(self):
        assert extract_inbound_credential({"content-type": "application/json"}) == (None, None)
        assert extract_inbound_credential(None) == (None, None)
        assert extract_inbound_credential({}) == (None, None)

    def test_authorization(self):
        assert extract_inbound_credential({"authorization": "Bearer x"}) == (
            "authorization", "Bearer x",
        )

    def test_x_api_key(self):
        assert extract_inbound_credential({"x-api-key": "k"}) == ("x-api-key", "k")

    def test_case_insensitive_name(self):
        # The proxy lowercases keys, but be robust to mixed case anyway.
        assert extract_inbound_credential({"Authorization": "Bearer x"}) == (
            "authorization", "Bearer x",
        )

    def test_two_distinct_auth_headers_raises(self):
        with pytest.raises(MultipleCredentialsError):
            extract_inbound_credential({"authorization": "Bearer x", "x-api-key": "k"})

    def test_duplicate_same_name_marker_raises(self):
        # The proxy reader injects the marker when one auth name repeats.
        with pytest.raises(MultipleCredentialsError):
            extract_inbound_credential(
                {"authorization": "Bearer x", DUPLICATE_AUTH_MARKER: "1"},
            )

    def test_empty_or_whitespace_value_is_not_a_credential(self):
        assert extract_inbound_credential({"authorization": ""}) == (None, None)
        assert extract_inbound_credential({"x-api-key": "   "}) == (None, None)

    def test_scheme_only_bearer_is_not_a_credential(self):
        # "Bearer " (scheme, no token) is non-empty as a raw string but carries
        # no credential — must be treated as absent, not forwarded as empty.
        assert extract_inbound_credential({"authorization": "Bearer "}) == (None, None)
        assert extract_inbound_credential({"authorization": "Bearer    "}) == (None, None)
        assert extract_inbound_credential({"Authorization": "bearer "}) == (None, None)

    def test_scheme_only_bearer_fails_loud_through_resolve(self):
        # End-to-end: a token-less Bearer must not relocate to an empty
        # x-api-key / "Bearer " — resolve returns None (→ fail loud downstream).
        assert resolve_inbound_credential(
            {"authorization": "Bearer "}, "openai", "anthropic", False,
        ) is None

    def test_real_bearer_token_still_extracted(self):
        # Guard against over-stripping: a real token survives.
        assert extract_inbound_credential({"authorization": "Bearer sk-abc123"}) == (
            "authorization", "Bearer sk-abc123",
        )


# ── relocate_credential ──────────────────────────────────────────────


class TestRelocateCredential:
    def test_same_protocol_openai_verbatim(self):
        assert relocate_credential("authorization", "Bearer x", "openai", "openai") == {
            "authorization": "Bearer x",
        }

    def test_same_protocol_anthropic_verbatim(self):
        assert relocate_credential("x-api-key", "k", "anthropic", "anthropic") == {
            "x-api-key": "k",
        }

    def test_cross_anthropic_to_openai_wraps_bearer(self):
        # CC (x-api-key) → OpenAI backend → Authorization: Bearer
        assert relocate_credential("x-api-key", "k", "anthropic", "openai") == {
            "Authorization": "Bearer k",
        }

    def test_cross_openai_to_anthropic_strips_bearer(self):
        # OpenAI client (Authorization Bearer) → Anthropic backend → x-api-key
        assert relocate_credential("authorization", "Bearer k", "openai", "anthropic") == {
            "x-api-key": "k",
        }

    def test_cross_bearer_strip_is_case_insensitive(self):
        assert relocate_credential("authorization", "BEARER k", "openai", "anthropic") == {
            "x-api-key": "k",
        }

    def test_cross_non_bearer_authorization_passes_whole(self):
        # A non-Bearer scheme cross-relocated is passed as-is (documented:
        # cross-protocol assumes Bearer/x-api-key token semantics).
        assert relocate_credential("authorization", "Basic abc", "openai", "anthropic") == {
            "x-api-key": "Basic abc",
        }

    def test_no_double_scheme_on_round_trip(self):
        # x-api-key → openai (Bearer k) is the only scheme-adding path; verify
        # it never doubles even when value coincidentally starts with "Bearer".
        assert relocate_credential("x-api-key", "Bearer-like", "anthropic", "openai") == {
            "Authorization": "Bearer Bearer-like",
        }


# ── resolve_inbound_credential ───────────────────────────────────────


class TestResolveInboundCredential:
    def test_no_inbound_returns_none(self):
        assert resolve_inbound_credential({}, "openai", "openai", False) is None

    def test_inbound_plus_static_key_raises(self):
        with pytest.raises(MultipleCredentialsError):
            resolve_inbound_credential(
                {"authorization": "Bearer x"}, "openai", "openai",
                backend_api_key_present=True,
            )

    def test_inbound_relocated_when_no_static(self):
        out = resolve_inbound_credential(
            {"x-api-key": "k"}, "anthropic", "openai", backend_api_key_present=False,
        )
        assert out == {"Authorization": "Bearer k"}

    def test_two_inbound_auth_headers_raises(self):
        with pytest.raises(MultipleCredentialsError):
            resolve_inbound_credential(
                {"authorization": "Bearer x", "x-api-key": "k"},
                "openai", "openai", False,
            )

    def test_empty_inbound_value_returns_none(self):
        # An empty/whitespace auth value is not a credential, so it must not
        # spuriously trip the two-source block against a static backend key.
        assert resolve_inbound_credential(
            {"authorization": ""}, "openai", "openai", backend_api_key_present=True,
        ) is None


# ── handler threading ────────────────────────────────────────────────


def _mock_client(response):
    client = AsyncMock()
    client.api_format = "ollama"
    client.send = AsyncMock(return_value=response)
    client.last_usage = {}
    client._slot_id = 0
    return client


def _ctx():
    return ContextManager(strategy=NoCompact(), budget_tokens=8192)


def _body(tools=None):
    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    if tools:
        body["tools"] = tools
    return body


_SEARCH_TOOL = [{
    "type": "function",
    "function": {"name": "search", "description": "", "parameters": {"type": "object", "properties": {}}},
}]


@pytest.mark.asyncio
async def test_handler_no_tools_relocates_inbound_credential():
    client = _mock_client(TextResponse(content="ok"))
    await handle_chat_completions(
        body=_body(),
        client=client,
        context_manager=_ctx(),
        protocol="openai",
        backend_protocol="anthropic",
        headers={"authorization": "Bearer INBOUND"},
    )
    # cross openai→anthropic: Bearer stripped, relocated to x-api-key
    assert client.send.call_args.kwargs["extra_headers"] == {"x-api-key": "INBOUND"}


@pytest.mark.asyncio
async def test_handler_with_tools_threads_credential_via_run_inference():
    client = _mock_client([ToolCall(tool="search", args={})])
    await handle_chat_completions(
        body=_body(tools=_SEARCH_TOOL),
        client=client,
        context_manager=_ctx(),
        protocol="anthropic",
        backend_protocol="openai",
        headers={"x-api-key": "INBOUND"},
    )
    # cross anthropic→openai: wrapped as Authorization: Bearer
    assert client.send.call_args.kwargs["extra_headers"] == {"Authorization": "Bearer INBOUND"}


@pytest.mark.asyncio
async def test_handler_no_inbound_credential_forwards_none():
    client = _mock_client(TextResponse(content="ok"))
    await handle_chat_completions(
        body=_body(),
        client=client,
        context_manager=_ctx(),
        protocol="openai",
        backend_protocol="openai",
        headers={},
    )
    assert client.send.call_args.kwargs["extra_headers"] is None


@pytest.mark.asyncio
async def test_handler_logs_redacted_credential(caplog):
    import logging

    client = _mock_client(TextResponse(content="ok"))
    with caplog.at_level(logging.DEBUG, logger="forge.proxy"):
        await handle_chat_completions(
            body=_body(),
            client=client,
            context_manager=_ctx(),
            protocol="openai",
            backend_protocol="anthropic",
            headers={"authorization": "Bearer SUPERSECRET"},
        )
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "SUPERSECRET" not in text  # never log a raw secret
    assert "***" in text
    assert "x-api-key" in text  # the relocated slot name is shown


@pytest.mark.asyncio
async def test_handler_inbound_plus_static_key_raises():
    client = _mock_client(TextResponse(content="ok"))
    with pytest.raises(MultipleCredentialsError):
        await handle_chat_completions(
            body=_body(),
            client=client,
            context_manager=_ctx(),
            protocol="openai",
            backend_protocol="openai",
            headers={"authorization": "Bearer INBOUND"},
            backend_api_key_present=True,
        )
    client.send.assert_not_awaited()  # refused before dispatch
