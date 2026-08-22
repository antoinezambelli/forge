"""Auth-credential tests for the client layer (forge v0.8.0).

forge carries exactly ONE credential to the backend, in the backend's native
auth header. These tests assert the real wire behavior via httpx.MockTransport
(for the four httpx clients) and the Anthropic SDK's own request pipeline (for
AnthropicClient), so the credential's final on-the-wire placement — not just
what forge passes to httpx — is what's verified.

Covers design §12: construction key → canonical header; per-call extra_headers
reaches the wire; two credentials on one call → raises; the shared-instance
no-leak property; cross-protocol re-casing for the case-sensitive Anthropic SDK
preflight; and the anthropic-version/beta filter.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import httpx
import pytest

from forge.clients.anthropic import AnthropicClient, _prepare_anthropic_headers
from forge.clients.base import (
    AUTH_HEADER_NAMES,
    has_auth_header,
    redact_auth_headers,
    resolve_request_headers,
    static_auth_present,
)
from forge.clients.llamafile import LlamafileClient
from forge.clients.ollama import OllamaClient
from forge.clients.openai_compat import OpenAICompatClient
from forge.clients.vllm import VLLMClient
from forge.errors import MissingCredentialError, MultipleCredentialsError
from tests.http_mocks import build_with_mock_http

_USER_MSG = [{"role": "user", "content": "hi"}]

_OPENAI_OK = {"choices": [{"message": {"content": "ok"}}]}


# ── base.py shared helpers ───────────────────────────────────────────


class TestResolveRequestHeaders:
    def test_success_cases_return_fresh_headers_or_none(self) -> None:
        cases = [
            ("static with no extra", True, None, None),
            ("no static with no extra", False, None, None),
            ("empty extra", True, {}, None),
            (
                "per-call auth",
                False,
                {"Authorization": "Bearer x"},
                {"Authorization": "Bearer x"},
            ),
            ("static plus trace", True, {"X-Trace-Id": "1"}, {"X-Trace-Id": "1"}),
            ("blank auth", True, {"Authorization": ""}, {"Authorization": ""}),
            (
                "scheme-only auth",
                True,
                {"Authorization": "Bearer "},
                {"Authorization": "Bearer "},
            ),
        ]
        for case, static, extra, expected in cases:
            out = resolve_request_headers(static, extra)
            assert out == expected, case
            if out is not None:
                assert out is not extra, case

    def test_static_plus_per_call_auth_raises(self) -> None:
        with pytest.raises(MultipleCredentialsError):
            resolve_request_headers(True, {"x-api-key": "k"})
        with pytest.raises(MultipleCredentialsError):
            resolve_request_headers(True, {"Authorization": "Bearer k"})

    def test_two_per_call_auth_headers_raises(self) -> None:
        # One credential per request — two auth headers in one call is refused
        # (the proxy never hands two, but a direct library caller could).
        with pytest.raises(MultipleCredentialsError):
            resolve_request_headers(False, {"Authorization": "Bearer A", "x-api-key": "B"})


class TestStaticAuthPresent:
    def test_reports_presence_for_each_zero_or_one_source_shape(self) -> None:
        cases = [
            ("no sources", None, None, False),
            ("empty sources", "", {}, False),
            ("api key", "sk-x", None, True),
            ("header", None, {"x-api-key": "k"}, True),
            ("blank key", "   ", None, False),
            ("blank key with header", "   ", {"x-api-key": "k"}, True),
            ("blank header", None, {"Authorization": ""}, False),
        ]
        for case, api_key, headers, expected in cases:
            assert static_auth_present(api_key, headers) is expected, case

    def test_key_plus_header_raises(self) -> None:
        with pytest.raises(MultipleCredentialsError):
            static_auth_present("sk-x", {"x-api-key": "k"})

    def test_two_construction_headers_raises(self) -> None:
        with pytest.raises(MultipleCredentialsError):
            static_auth_present(None, {"Authorization": "Bearer A", "x-api-key": "B"})


class TestHasAuthHeader:
    def test_recognizes_auth_header_names_case_insensitively(self) -> None:
        cases = [
            ("uppercase authorization", {"AUTHORIZATION": "x"}, True),
            ("mixed-case api key", {"X-Api-Key": "x"}, True),
            ("lowercase api key", {"x-api-key": "x"}, True),
            ("none", None, False),
            ("empty", {}, False),
            ("non-auth header", {"X-Trace-Id": "1"}, False),
        ]
        for case, headers, expected in cases:
            assert has_auth_header(headers) is expected, case

    def test_constant(self) -> None:
        assert AUTH_HEADER_NAMES == {"authorization", "x-api-key"}


class TestRedactAuthHeaders:
    def test_masks_auth_values_and_preserves_other_headers(self) -> None:
        cases = [
            (
                "authorization",
                {"Authorization": "Bearer secret", "X-Trace": "1"},
                {"Authorization": "***", "X-Trace": "1"},
            ),
            ("api key", {"x-api-key": "sk-secret"}, {"x-api-key": "***"}),
            ("none", None, {}),
        ]
        for case, headers, expected in cases:
            assert redact_auth_headers(headers) == expected, case


# ── httpx clients: real wire headers via MockTransport ───────────────
#
# Each factory builds the client normally (so its construction headers come
# from the real api_key/extra_headers path), then swaps only the transport —
# preserving the exact construction headers — so the captured request shows the
# genuine httpx merge of construction + per-call headers.


def _capturing_http(construction_headers: httpx.Headers, captured: dict) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json=_OPENAI_OK)

    return httpx.AsyncClient(
        headers=construction_headers,
        transport=httpx.MockTransport(handler),
    )


def _swap_transport(client, handler) -> None:
    """Swap a built client's httpx transport for a capturing one, keeping its
    real construction headers."""
    client._http = httpx.AsyncClient(
        headers=client._http.headers, transport=httpx.MockTransport(handler)
    )


def _wire(build):
    """Turn a client constructor into a ``(**kw) -> (client, cap)`` factory whose
    client captures the merged request headers."""

    def factory(**kw) -> tuple[object, dict]:
        c = build_with_mock_http(build, **kw)
        cap: dict = {}
        c._http = _capturing_http(c._http.headers, cap)
        return c, cap

    return factory


_openai = _wire(lambda **kw: OpenAICompatClient(base_url="https://b/v1", model="m", **kw))
_ollama = _wire(lambda **kw: OllamaClient(model="m", base_url="https://b", **kw))
_llamafile = _wire(lambda **kw: LlamafileClient(gguf_path="m.gguf", base_url="https://b/v1", **kw))
_vllm = _wire(lambda **kw: VLLMClient(model_path="m", base_url="https://b/v1", **kw))


_FACTORIES = [
    pytest.param(_openai, id="openai_compat"),
    pytest.param(_ollama, id="ollama"),
    pytest.param(_llamafile, id="llamafile"),
    pytest.param(_vllm, id="vllm"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", _FACTORIES)
async def test_construction_api_key_sets_bearer_on_wire(factory) -> None:
    client, cap = factory(api_key="STATICKEY")
    await client.send(_USER_MSG)
    assert cap["request"].headers["authorization"] == "Bearer STATICKEY"


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", _FACTORIES)
async def test_per_call_extra_headers_reach_the_wire(factory) -> None:
    client, cap = factory(api_key="")  # no static credential
    await client.send(_USER_MSG, extra_headers={"Authorization": "Bearer INBOUND"})
    assert cap["request"].headers["authorization"] == "Bearer INBOUND"


@pytest.mark.asyncio
async def test_static_plus_per_call_auth_raises() -> None:
    factory = _openai
    client, _ = factory(api_key="STATICKEY")
    with pytest.raises(MultipleCredentialsError):
        await client.send(_USER_MSG, extra_headers={"x-api-key": "INBOUND"})


@pytest.mark.asyncio
async def test_construction_extra_headers_auth_also_blocks_per_call() -> None:
    # §10.3 must trip even when the static credential came via extra_headers
    # (not api_key=), so static_present can't be derived from api_key alone.
    client, _ = _openai(extra_headers={"Authorization": "Bearer STATIC"})
    with pytest.raises(MultipleCredentialsError):
        await client.send(_USER_MSG, extra_headers={"Authorization": "Bearer INBOUND"})


@pytest.mark.asyncio
async def test_no_credential_leak_across_serialized_requests() -> None:
    # The proxy reuses one client instance across requests. A per-call inbound
    # credential must NOT persist into a later request that carries none.
    client, cap = _openai(api_key="")
    await client.send(_USER_MSG, extra_headers={"Authorization": "Bearer A"})
    assert cap["request"].headers["authorization"] == "Bearer A"
    await client.send(_USER_MSG)  # no per-call credential this time
    assert "authorization" not in cap["request"].headers


@pytest.mark.asyncio
async def test_ollama_think_retry_keeps_credential() -> None:
    # The think-unsupported fallback re-POSTs; that second request must still
    # carry the per-call credential (M6).
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                400, json={"error": "registry.ollama.ai does not support thinking"}
            )
        return httpx.Response(200, json={"message": {"content": "ok"}})

    client = OllamaClient(model="reasoner", base_url="https://b", think=None)
    _swap_transport(client, handler)
    await client.send(_USER_MSG, extra_headers={"Authorization": "Bearer INBOUND"})
    assert len(requests) == 2  # initial + retry
    assert requests[1].headers["authorization"] == "Bearer INBOUND"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(_openai, id="openai_compat"),
        pytest.param(_vllm, id="vllm"),
        pytest.param(_llamafile, id="llamafile"),
    ],
)
async def test_sse_stream_forwards_credential(factory) -> None:
    # The three OpenAI-SSE clients must forward the per-call credential on the
    # streaming path too (M6), not just send().
    client, cap = factory(api_key="")
    sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        cap["request"] = request
        return httpx.Response(200, content=sse)

    _swap_transport(client, handler)
    chunks = [
        c async for c in client.send_stream(
            _USER_MSG, extra_headers={"Authorization": "Bearer INBOUND"}
        )
    ]
    assert chunks  # stream produced output
    assert cap["request"].headers["authorization"] == "Bearer INBOUND"


@pytest.mark.asyncio
async def test_ollama_stream_forwards_credential() -> None:
    cap: dict = {}
    ndjson = b'{"message":{"content":"hi"},"done":false}\n{"done":true}\n'

    def handler(request: httpx.Request) -> httpx.Response:
        cap["request"] = request
        return httpx.Response(200, content=ndjson)

    client = OllamaClient(model="m", base_url="https://b", think=False)
    _swap_transport(client, handler)
    _ = [
        c async for c in client.send_stream(
            _USER_MSG, extra_headers={"Authorization": "Bearer INBOUND"}
        )
    ]
    assert cap["request"].headers["authorization"] == "Bearer INBOUND"


@pytest.mark.asyncio
async def test_llamafile_prompt_mode_forwards_credential() -> None:
    # Prompt capability routes through _send_prompt, a distinct outbound body
    # from _send_native — verify it carries the credential too.
    client, cap = _llamafile(api_key="STATICKEY", mode="prompt")
    await client.send(_USER_MSG)
    assert cap["request"].headers["authorization"] == "Bearer STATICKEY"


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", _FACTORIES)
async def test_construction_api_key_plus_auth_header_raises(factory) -> None:
    # Two static credentials at construction (api_key AND an auth header) is
    # itself two credentials → refused at construction (fail loud).
    with pytest.raises(MultipleCredentialsError):
        factory(api_key="STATICKEY", extra_headers={"x-api-key": "OTHER"})


# ── Anthropic SDK seam ───────────────────────────────────────────────


class TestPrepareAnthropicHeaders:
    def test_normalizes_every_supported_header_shape(self) -> None:
        cases = [
            ("api key", {"x-api-key": "k"}, {"X-Api-Key": "k"}),
            (
                "authorization",
                {"authorization": "Bearer k"},
                {"Authorization": "Bearer k"},
            ),
            (
                "pinned SDK headers",
                {
                    "x-api-key": "k",
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "x",
                },
                {"X-Api-Key": "k"},
            ),
            ("unknown", {"X-Trace": "1"}, {"X-Trace": "1"}),
            ("none", None, None),
            ("empty", {}, None),
            ("all dropped", {"anthropic-version": "x"}, None),
        ]
        for case, headers, expected in cases:
            assert _prepare_anthropic_headers(headers) == expected, case


def _anthropic_capturing(
    client: AnthropicClient, captured: dict, api_key: str = "",
) -> None:
    """Swap the client's SDK for one whose transport captures the request.

    ``api_key`` mirrors the construction credential so the SDK's own auth
    (e.g. a static ``x-api-key``) is exercised on the captured request.
    """
    import anthropic

    if int(anthropic.__version__.split(".", 1)[0]) >= 1:
        import httpx2 as sdk_http
    else:
        sdk_http = httpx

    def handler(request):
        captured["request"] = request
        return sdk_http.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client._client = anthropic.AsyncAnthropic(
        api_key=api_key,
        base_url="https://b",
        http_client=sdk_http.AsyncClient(transport=sdk_http.MockTransport(handler)),
    )


def test_anthropic_empty_key_is_pure_passthrough_no_static_auth() -> None:
    # The proxy constructs with api_key="" for pure inbound passthrough: no
    # static credential, and api_key is mapped to None so the SDK emits no auth
    # header at all (no spurious empty X-Api-Key).
    client = AnthropicClient(model="claude", api_key="")
    assert client._static_auth is False
    assert client._client.api_key is None
    assert client._client.auth_token is None


def test_anthropic_passthrough_suppresses_ambient_env(monkeypatch) -> None:
    # B1/B2: with ambient ANTHROPIC_* set, a pure-passthrough client must carry
    # NO ambient credential, and the env must be restored after construction.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-auth-token")
    client = AnthropicClient(model="claude", api_key="")
    assert client._client.api_key is None
    assert client._client.auth_token is None
    # restored for forge's own (eval) clients / other processes
    assert os.environ["ANTHROPIC_API_KEY"] == "env-api-key"
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "env-auth-token"


def test_anthropic_static_key_suppresses_ambient_auth_token(monkeypatch) -> None:
    # A static --backend-api-key must be the ONLY credential — ambient
    # ANTHROPIC_AUTH_TOKEN must not add a second (Authorization: Bearer).
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-auth-token")
    client = AnthropicClient(model="claude", api_key="STATIC")
    assert client._client.api_key == "STATIC"
    assert client._client.auth_token is None
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "env-auth-token"


def test_anthropic_none_key_defers_to_env(monkeypatch) -> None:
    # WR direct use: api_key=None reads ANTHROPIC_API_KEY (the dev's deliberate
    # single credential); forge does not suppress it.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-api-key")
    client = AnthropicClient(model="claude")  # api_key=None default
    assert client._client.api_key == "env-api-key"


@pytest.mark.asyncio
async def test_anthropic_inbound_lowercase_xapikey_reaches_wire() -> None:
    # The proxy lowercases inbound keys; forge must re-case to X-Api-Key so the
    # SDK's case-sensitive preflight passes AND the wire carries the credential
    # exactly once. (Regression guard for blocker B1.)
    client = AnthropicClient(model="claude", api_key="")
    cap: dict = {}
    _anthropic_capturing(client, cap)
    # lowercase, as it would arrive from the proxy's header dict
    await client.send(_USER_MSG, extra_headers={"x-api-key": "REALKEY"})
    wire = cap["request"].headers
    assert wire["x-api-key"] == "REALKEY"  # httpx.Headers lookup is case-insensitive
    # exactly one auth credential on the wire
    assert wire.get_list("x-api-key") == ["REALKEY"]
    assert "authorization" not in wire


@pytest.mark.asyncio
async def test_anthropic_static_plus_per_call_auth_raises() -> None:
    client = AnthropicClient(model="claude", api_key="STATIC")
    assert client._static_auth is True
    with pytest.raises(MultipleCredentialsError):
        await client.send(_USER_MSG, extra_headers={"x-api-key": "INBOUND"})


def test_anthropic_construction_api_key_plus_default_header_raises() -> None:
    with pytest.raises(MultipleCredentialsError):
        AnthropicClient(
            model="claude", api_key="STATIC",
            default_headers={"x-api-key": "OTHER"},
        )


def test_anthropic_construction_default_headers_recased_for_sdk() -> None:
    # Construction default_headers must reach the SDK re-cased to X-Api-Key
    # (the SDK's case-sensitive preflight slot).
    client = AnthropicClient(
        model="claude", api_key="", default_headers={"x-api-key": "STATICKEY"},
    )
    assert client._static_auth is True
    stored = {k.lower(): v for k, v in dict(client._client.default_headers).items()}
    assert stored.get("x-api-key") == "STATICKEY"


@pytest.mark.asyncio
async def test_anthropic_verbatim_body_cannot_smuggle_extra_headers() -> None:
    # A verbatim inbound body may carry top-level ``extra_headers`` without a
    # Forge per-call credential. Use a static key so the request proceeds, then
    # assert that only the static credential reaches the wire.
    client = AnthropicClient(model="claude", api_key="STATIC")
    cap: dict = {}
    _anthropic_capturing(client, cap, api_key="STATIC")
    malicious_body = {
        "model": "claude",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "extra_headers": {"x-api-key": "SMUGGLED"},
    }
    await client.send(_USER_MSG, inbound_anthropic_body=malicious_body)
    values = cap["request"].headers.get_list("x-api-key")
    assert "SMUGGLED" not in values
    assert values == ["STATIC"]  # exactly the static credential, nothing smuggled


@pytest.mark.asyncio
async def test_anthropic_passthrough_cannot_smuggle_extra_headers() -> None:
    # Same guard on the non-verbatim path: a passthrough dict carrying
    # ``extra_headers`` must not inject auth either.
    client = AnthropicClient(model="claude", api_key="STATIC")
    cap: dict = {}
    _anthropic_capturing(client, cap, api_key="STATIC")
    await client.send(
        _USER_MSG,
        passthrough={"extra_headers": {"x-api-key": "SMUGGLED"}, "max_tokens": 16},
    )
    values = cap["request"].headers.get_list("x-api-key")
    assert "SMUGGLED" not in values
    assert values == ["STATIC"]


@pytest.mark.asyncio
async def test_anthropic_stream_installs_recased_credential() -> None:
    # send_stream shares the credential install + re-casing with send; verify
    # the re-cased header lands in the SDK stream call.
    client = AnthropicClient(model="claude", api_key="")
    captured: dict = {}

    class _FakeStream:
        async def __aenter__(self_):
            return self_

        async def __aexit__(self_, *args):
            return False

        def __aiter__(self_):
            async def _gen():
                if False:
                    yield  # pragma: no cover - empty event stream
            return _gen()

        async def get_final_message(self_):
            msg = MagicMock()
            msg.usage.input_tokens = 1
            msg.usage.output_tokens = 1
            msg.usage.cache_creation_input_tokens = 0
            msg.usage.cache_read_input_tokens = 0
            return msg

    def fake_stream(**kwargs):
        captured["kwargs"] = kwargs
        return _FakeStream()

    client._client = MagicMock()
    client._client.messages.stream = fake_stream
    _ = [
        c async for c in client.send_stream(
            _USER_MSG, extra_headers={"x-api-key": "REALKEY"}
        )
    ]
    assert captured["kwargs"]["extra_headers"] == {"X-Api-Key": "REALKEY"}


# ── zero-credential fail-loud (no credential at all) ─────────────────


@pytest.mark.asyncio
async def test_anthropic_zero_credential_send_raises_missing(monkeypatch) -> None:
    # Pure passthrough (api_key="" -> None), no ambient env, no per-call auth:
    # forge must fail loud BEFORE the SDK's opaque "could not resolve
    # authentication" error (which surfaces as HTTP 502). Raised pre-dispatch,
    # so no network call is attempted.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    client = AnthropicClient(model="claude", api_key="")
    assert client._client.api_key is None and client._client.auth_token is None
    with pytest.raises(MissingCredentialError):
        await client.send(_USER_MSG)


@pytest.mark.asyncio
async def test_anthropic_zero_credential_stream_raises_missing(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    client = AnthropicClient(model="claude", api_key="")
    with pytest.raises(MissingCredentialError):
        _ = [c async for c in client.send_stream(_USER_MSG)]


def test_anthropic_ambient_env_satisfies_credential_requirement(monkeypatch) -> None:
    # WR direct use: api_key=None reads ANTHROPIC_API_KEY at construction, so the
    # zero-credential guard must NOT trip even with no per-call header.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    client = AnthropicClient(model="claude")  # api_key=None default
    assert client._client.api_key == "env-key"
    client._ensure_credential(None)  # no raise
