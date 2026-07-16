"""Unit tests for shared client helpers in forge.clients.base."""

from __future__ import annotations

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock

from forge.clients.anthropic import AnthropicClient
from forge.clients.base import decode_tool_args, open_backend_forward
from forge.clients.llamafile import LlamafileClient
from forge.clients.ollama import OllamaClient
from forge.clients.openai_compat import OpenAICompatClient
from forge.clients.vllm import VLLMClient
from forge.errors import BackendError


class TestDecodeToolArgs:
    """decode_tool_args: parse JSON-string args, fail-loud on malformed.

    Contract: return a dict for well-formed object args; return the raw
    (non-dict) value untouched for anything else, so ResponseValidator's
    args-shape check can route it to the tool-error channel. Never coerce a
    malformed payload to ``{}`` and never raise.
    """

    def test_valid_json_object_decoded(self) -> None:
        assert decode_tool_args('{"city": "Paris"}') == {"city": "Paris"}

    def test_empty_string_is_no_arg_call(self) -> None:
        assert decode_tool_args("") == {}

    def test_none_is_no_arg_call(self) -> None:
        # Missing "arguments" key — a no-arg call, not a failure.
        assert decode_tool_args(None) == {}

    def test_already_decoded_dict_passes_through(self) -> None:
        # Ollama and the Anthropic SDK hand back parsed dicts.
        d = {"city": "Paris"}
        assert decode_tool_args(d) is d

    def test_malformed_json_kept_as_raw_string(self) -> None:
        # The crux: malformed args are NOT coerced to {} and do NOT raise —
        # the raw string (a non-dict) survives for the validator to catch.
        assert decode_tool_args('{"city": ') == '{"city": '

    def test_valid_json_non_object_kept_as_is(self) -> None:
        # Parseable but not an object (list / scalar) — a non-dict the
        # validator must reject, so it rides through unchanged.
        assert decode_tool_args("[1, 2]") == [1, 2]
        assert decode_tool_args("42") == 42
        assert decode_tool_args('"bare"') == "bare"

    def test_non_string_non_dict_passes_through(self) -> None:
        # Any other already-decoded shape is left for the validator to judge.
        assert decode_tool_args(123) == 123
        assert decode_tool_args([1, 2]) == [1, 2]


# ── forward_request / open_backend_forward (proxy passthrough) ────
#
# The passthrough engine is implemented ONCE (open_backend_forward) and each
# OpenAI-shape client wires it to its own server root, so the engine contract
# is tested once and the per-client wiring is a parameterized matrix.
class _FakeStreamCtx:
    """httpx stream CM stand-in; optionally raises on enter (connect fail)."""

    def __init__(self, status: int = 200, enter_exc: Exception | None = None):
        self.status_code = status
        self._enter_exc = enter_exc

    async def __aenter__(self):
        if self._enter_exc is not None:
            raise self._enter_exc
        return self

    async def __aexit__(self, *args):
        return False


def _mock_http(status: int = 200, enter_exc: Exception | None = None) -> AsyncMock:
    http = AsyncMock()
    http.stream = MagicMock(return_value=_FakeStreamCtx(status, enter_exc))
    return http


class TestOpenBackendForward:
    """The shared passthrough engine: verbatim, fail-loud only on transport.

    Contract: the backend's response object is yielded whatever its status
    (an HTTP error status is the backend's answer, never raised); only a
    connection-level failure raises BackendError(502); the read timeout is
    disabled for stream=True (SSE feeds are silent between events) and left
    at the pool default otherwise.
    """

    @pytest.mark.asyncio
    async def test_yields_response_and_wires_request(self) -> None:
        http = _mock_http()
        cm = open_backend_forward(
            http, "http://t:8080/props?x=1", "GET", b"", {"h": "v"},
        )
        async with cm as resp:
            assert resp.status_code == 200
        args = http.stream.call_args
        assert args.args == ("GET", "http://t:8080/props?x=1")
        assert args.kwargs["headers"] == {"h": "v"}
        assert args.kwargs["content"] is None  # empty body → no body

    @pytest.mark.asyncio
    async def test_body_bytes_ride_through(self) -> None:
        http = _mock_http()
        async with open_backend_forward(
            http, "http://t:8080/models/load", "POST", b'{"model": "x"}',
        ):
            pass
        assert http.stream.call_args.kwargs["content"] == b'{"model": "x"}'

    @pytest.mark.asyncio
    async def test_error_status_yielded_not_raised(self) -> None:
        http = _mock_http(status=404)
        async with open_backend_forward(http, "http://t:8080/nope", "GET") as resp:
            assert resp.status_code == 404  # the backend's answer, not a fault

    @pytest.mark.asyncio
    async def test_connect_error_raises_502(self) -> None:
        http = _mock_http(enter_exc=httpx.ConnectError("refused"))
        with pytest.raises(BackendError) as exc_info:
            async with open_backend_forward(http, "http://t:8080/props", "GET"):
                pass
        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_stream_disables_read_timeout(self) -> None:
        http = _mock_http()
        async with open_backend_forward(
            http, "http://t:8080/models/sse", "GET", stream=True,
        ):
            pass
        assert http.stream.call_args.kwargs["timeout"].read is None

    @pytest.mark.asyncio
    async def test_buffered_keeps_pool_timeout(self) -> None:
        http = _mock_http()
        async with open_backend_forward(http, "http://t:8080/props", "GET"):
            pass
        assert http.stream.call_args.kwargs["timeout"] is httpx.USE_CLIENT_DEFAULT


# One row per OpenAI-shape client: factory and the server root its
# forward_request must derive from base_url (/v1 suffix stripped; Ollama's
# base_url is already the root).
_CLIENT_ROOTS = [
    pytest.param(
        lambda: LlamafileClient(gguf_path="m", base_url="http://t:8080/v1"),
        "http://t:8080", id="llamafile",
    ),
    pytest.param(
        lambda: OpenAICompatClient(model="m", base_url="https://api.example.com/v1"),
        "https://api.example.com", id="openai_compat",
    ),
    pytest.param(
        lambda: VLLMClient(model_path="m", base_url="http://t:8000/v1"),
        "http://t:8000", id="vllm",
    ),
    pytest.param(
        lambda: OllamaClient(model="m", base_url="http://t:11434"),
        "http://t:11434", id="ollama",
    ),
]


class TestForwardRequestWiring:
    """Per-client forward_request wiring over the shared engine."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("make_client,root", _CLIENT_ROOTS)
    async def test_target_appended_to_server_root(self, make_client, root) -> None:
        client = make_client()
        client._http = _mock_http()
        async with client.forward_request("GET", "/props?model=X&autoload=false"):
            pass
        args = client._http.stream.call_args
        assert args.args == ("GET", f"{root}/props?model=X&autoload=false")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("make_client,root", _CLIENT_ROOTS)
    async def test_v1_target_keeps_its_own_prefix(self, make_client, root) -> None:
        # /v1/models carries its own prefix — the root derivation must not
        # eat it (nor double it on clients whose base_url ends in /v1).
        client = make_client()
        client._http = _mock_http()
        async with client.forward_request("GET", "/v1/models"):
            pass
        assert client._http.stream.call_args.args == ("GET", f"{root}/v1/models")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("make_client,root", _CLIENT_ROOTS)
    async def test_extra_headers_threaded(self, make_client, root) -> None:
        client = make_client()
        client._http = _mock_http()
        extra = {"Authorization": "Bearer inbound-token"}
        async with client.forward_request("GET", "/props", extra_headers=extra):
            pass
        sent = client._http.stream.call_args.kwargs["headers"]
        assert sent == client._request_headers(extra)


class TestAnthropicForwardRequest:
    def test_returns_none(self) -> None:
        # The Anthropic client speaks through the SDK — no raw HTTP surface
        # to forward to. The proxy answers 404 itself (synthesized entry for
        # the /v1/models family). Sync method: called without await.
        client = AnthropicClient(model="claude-test", api_key="dummy")
        assert client.forward_request("GET", "/props") is None
