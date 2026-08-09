"""Raw asyncio HTTP server for the proxy.

No framework dependencies — uses asyncio.start_server directly.
Handles routing, request queuing (single-GPU serialization), health
checks, SSE streaming, and client disconnect detection.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from forge._backend_profiles import ClientAdapter, MetadataFormat, ModelCatalog
from forge.clients.base import AUTH_HEADER_NAMES, LLMClient
from forge.context.manager import ContextManager
from forge.context.observations import ContextUsage
from forge.core.reasoning import DEFAULT_REASONING_REPLAY, ReasoningReplay, validate_reasoning_replay
from forge.errors import (
    BackendDiscoveryError,
    BackendError,
    MissingCredentialError,
    MissingModelError,
    MultipleCredentialsError,
)
from forge.proxy.auth import (
    DUPLICATE_AUTH_MARKER,
    resolve_inbound_credential,
    _resolve_metadata_credential,
)
from forge.proxy.handler import (
    LazyDiscovery,
    RequestFacts,
    handle_chat_completions,
    observe_request_context,
    resolve_effective_model,
    run_lazy_discovery,
    validate_request_model,
)

logger = logging.getLogger("forge.proxy")

# Maximum request body size (16 MB)
_MAX_BODY = 16 * 1024 * 1024
_FORWARDED_METADATA_PATHS = frozenset({
    "/health",
    "/v1/health",
    "/v1/models",
    "/models",
    "/props",
})


@dataclass(frozen=True)
class _MetadataResponse:
    """Buffered backend facts safe for the raw HTTP response writer."""

    status: int
    body: bytes
    content_type: str | None


class _MetadataCourier:
    """Proxy-owned transport for the closed read-only metadata surface."""

    def __init__(
        self,
        mount_root: str,
        target_protocol: str,
        backend_api_key: str | None,
        timeout: float,
    ) -> None:
        self._mount_root = mount_root
        self._target_protocol = target_protocol
        self._backend_api_key = backend_api_key
        self._http = httpx.AsyncClient(timeout=timeout)
        self._closed = False

    async def forward(
        self,
        raw_target: str,
        inbound_headers: dict[str, str],
    ) -> _MetadataResponse:
        """Fetch one approved raw request target without parsing its body."""
        headers = _resolve_metadata_credential(
            inbound_headers,
            target_protocol=self._target_protocol,
            backend_api_key=self._backend_api_key,
        )
        return await self._get(self._url_for(raw_target), headers)

    async def fetch_private(
        self,
        absolute_url: str,
        inbound_headers: dict[str, str],
    ) -> _MetadataResponse:
        """Fetch one trusted internal absolute URL without mount rejoining."""
        headers = _resolve_metadata_credential(
            inbound_headers,
            target_protocol=self._target_protocol,
            backend_api_key=self._backend_api_key,
        )
        return await self._get(absolute_url, headers)

    async def _get(
        self, url: str, headers: dict[str, str] | None,
    ) -> _MetadataResponse:
        """Perform the shared buffered credentialed GET."""
        response = await self._http.get(url, headers=headers)
        return _MetadataResponse(
            status=response.status_code,
            body=response.content,
            content_type=response.headers.get("content-type"),
        )

    def _url_for(self, raw_target: str) -> str:
        """Join a raw target to the mount while replacing any root query."""
        raw_path, separator, raw_query = raw_target.partition("?")
        root = urlsplit(self._mount_root)
        url = urlunsplit(root._replace(
            path=f"{root.path.rstrip('/')}{raw_path}",
            query=raw_query if separator else "",
            fragment="",
        ))
        return f"{url}?" if separator and not raw_query else url

    async def close(self) -> None:
        """Release the private pool exactly once."""
        if not self._closed:
            self._closed = True
            await self._http.aclose()


@dataclass
class _QueueItem:
    """A request waiting to be processed by the inference worker."""

    body: dict[str, Any]
    protocol: str = "openai"
    # Per-request inbound headers (lowercased). Carries the inbound credential
    # the handler relocates to the backend. Per-item, never shared.
    headers: dict[str, str] = field(default_factory=dict)
    facts: RequestFacts = field(default_factory=RequestFacts)
    future: asyncio.Future = field(default=None)  # type: ignore[assignment]
    cancelled: bool = False

    def __post_init__(self) -> None:
        if self.future is None:
            self.future = asyncio.get_running_loop().create_future()


@dataclass(frozen=True)
class _PredispatchResult:
    """Tagged streaming predispatch outcome and its request-local facts."""

    error: Exception | None
    facts: RequestFacts


@dataclass(frozen=True)
class _ContextReportingConfig:
    """Private denominator facts selected once by Proxy composition."""

    managed: bool
    context_window_tokens: int | None
    metadata_format: MetadataFormat
    metadata_url: str | None = None


class HTTPServer:
    """Raw asyncio HTTP server with OpenAI-compatible routing."""

    def __init__(
        self,
        client: LLMClient,
        context_manager: ContextManager,
        client_adapter: ClientAdapter,
        host: str = "127.0.0.1",
        port: int = 8081,
        serialize_requests: bool = True,
        max_retries: int = 3,
        max_tool_errors: int = 2,
        rescue_enabled: bool = True,
        native_passthrough: bool = True,
        inject_respond_tool: bool = False,
        reasoning_replay: ReasoningReplay = DEFAULT_REASONING_REPLAY,
        backend_protocol: str = "openai",
        backend_api_key_present: bool = False,
        lazy_discovery: LazyDiscovery | None = None,
    ) -> None:
        self._client = client
        self._context_manager = context_manager
        self._lazy_discovery = lazy_discovery
        self._host = host
        self._port = port
        self._max_retries = max_retries
        self._max_tool_errors = max_tool_errors
        self._rescue_enabled = rescue_enabled
        self._native_passthrough = native_passthrough
        self._inject_respond_tool = inject_respond_tool
        self._reasoning_replay = validate_reasoning_replay(reasoning_replay)
        # Target wire protocol of the backend (relocation target) and whether a
        # static --backend-api-key is configured (for the two-source check).
        # The raw key itself never reaches the handler — it is baked into the
        # backend client at construction; the handler only needs to know it
        # exists to refuse an inbound credential alongside it.
        self._backend_protocol = backend_protocol
        self._client_adapter = client_adapter
        self._backend_api_key_present = backend_api_key_present
        self._metadata_courier: _MetadataCourier | None = None
        self._private_catalog_url: str | None = None
        self._catalog_parser: Callable[[Any], ModelCatalog] | None = None
        self._context_reporting: _ContextReportingConfig | None = None
        self._server: asyncio.Server | None = None
        self._serialize = serialize_requests
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def _configure_metadata_courier(
        self,
        mount_root: str,
        backend_api_key: str | None,
        timeout: float,
        private_catalog_url: str | None = None,
        catalog_parser: Callable[[Any], ModelCatalog] | None = None,
    ) -> None:
        """Install the private courier before the HTTP listener starts."""
        self._metadata_courier = _MetadataCourier(
            mount_root=mount_root,
            target_protocol=self._backend_protocol,
            backend_api_key=backend_api_key,
            timeout=timeout,
        )
        self._private_catalog_url = private_catalog_url
        self._catalog_parser = catalog_parser

    def _configure_context_reporting(
        self,
        *,
        managed: bool,
        context_window_tokens: int | None,
        metadata_format: MetadataFormat,
        metadata_url: str | None = None,
    ) -> None:
        """Install the narrow completed-request reporting policy."""

        self._context_reporting = _ContextReportingConfig(
            managed=managed,
            context_window_tokens=context_window_tokens,
            metadata_format=metadata_format,
            metadata_url=metadata_url,
        )

    async def start(self) -> None:
        """Start listening for connections."""
        if self._serialize:
            self._worker_task = asyncio.create_task(self._inference_worker())
        try:
            self._server = await asyncio.start_server(
                self._handle_connection, self._host, self._port,
            )
        except Exception:
            await self._stop_worker()
            if self._metadata_courier is not None:
                await self._metadata_courier.close()
            raise
        logger.info("Proxy listening on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the server."""
        await self._stop_worker()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._metadata_courier is not None:
            await self._metadata_courier.close()

    async def _stop_worker(self) -> None:
        """Cancel the inference worker if startup or shutdown created one."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def _inference_worker(self) -> None:
        """Single worker that pulls requests off the queue and processes them.

        Ensures only one inference runs at a time (single-GPU constraint).
        """
        while True:
            item = await self._queue.get()
            try:
                if item.cancelled or item.future.cancelled():
                    logger.info("   Skipping cancelled request")
                    continue
                result = await self._run_handler(
                    item.body, item.protocol, item.headers, item.facts,
                )
                if not item.future.done():
                    item.future.set_result(result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not item.future.done():
                    item.future.set_result(exc)
            finally:
                self._queue.task_done()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single HTTP connection."""
        try:
            # Read request line
            request_line = await asyncio.wait_for(
                reader.readline(), timeout=30.0,
            )
            if not request_line:
                return

            request_str = request_line.decode("utf-8", errors="replace").strip()
            parts = request_str.split(" ", 2)
            if len(parts) < 2:
                await self._send_error(writer, 400, "Bad request")
                return

            method, raw_path = parts[0], parts[1]
            logger.info(">> %s %s", method, raw_path)
            # Strip the query string before routing. Real clients append
            # query params (e.g. Claude Code POSTs /v1/messages?beta=true);
            # exact-matching the raw target would 404 every such request.
            path = raw_path.split("?", 1)[0]

            # Read headers
            headers = await self._read_headers(reader)
            try:
                content_length = int(headers.get("content-length", "0"))
            except ValueError:
                await self._send_error(writer, 400, "Invalid Content-Length")
                return

            # Read body
            body_bytes = b""
            if content_length > 0:
                if content_length > _MAX_BODY:
                    await self._send_error(writer, 413, "Request too large")
                    return
                body_bytes = await asyncio.wait_for(
                    reader.readexactly(content_length), timeout=60.0,
                )

            # Route
            if method == "GET" and path == "/forge/health":
                await self._handle_forge_health(writer)
            elif method == "GET" and path == "/forge/usage":
                await self._handle_forge_usage(writer)
            elif method == "GET" and path in _FORWARDED_METADATA_PATHS:
                await self._handle_metadata(writer, raw_path, headers)
            elif method == "POST" and path in ("/v1/chat/completions", "/chat/completions"):
                # llama.cpp serves the OpenAI chat endpoint on both spellings;
                # llama.cpp-native clients (pi-llama-cpp) POST the unprefixed
                # one, so a transparent front must accept it too.
                await self._handle_completions(
                    writer, body_bytes, protocol="openai", headers=headers,
                )
            elif method == "POST" and path == "/v1/messages":
                await self._handle_completions(
                    writer, body_bytes, protocol="anthropic", headers=headers,
                )
            elif method == "OPTIONS":
                await self._send_cors_preflight(writer)
            else:
                await self._send_error(writer, 404, "Not found")

        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception:
            logger.exception("Unhandled error in connection handler")
            try:
                await self._send_error(writer, 500, "Internal server error")
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_headers(self, reader: asyncio.StreamReader) -> dict[str, str]:
        """Read HTTP headers until blank line."""
        headers: dict[str, str] = {}
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=30.0)
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                break
            if ":" in decoded:
                key, value = decoded.split(":", 1)
                key = key.strip().lower()
                # A repeated auth header name would collapse to last-wins in a
                # plain dict — forge must never silently pick a credential
                # winner, so flag it for the credential resolver to refuse.
                if key in AUTH_HEADER_NAMES and key in headers:
                    headers[DUPLICATE_AUTH_MARKER] = "1"
                headers[key] = value.strip()
        return headers

    async def _handle_forge_health(self, writer: asyncio.StreamWriter) -> None:
        """GET /forge/health -- return Forge process liveness locally."""
        body = '{"status":"ok"}'
        await self._send_json(writer, 200, body)

    async def _handle_forge_usage(self, writer: asyncio.StreamWriter) -> None:
        """GET /forge/usage -- serialize only the published local snapshot."""

        usage = self._context_manager.published_usage
        if usage is None or usage.context_window_tokens is None:
            await self._send_no_content(writer)
            return
        payload: dict[str, Any] = {
            "current_usage_tokens": usage.current_usage_tokens,
            "context_window_tokens": usage.context_window_tokens,
            "usage_percent": (
                usage.current_usage_tokens
                / usage.context_window_tokens
                * 100
            ),
            "model": usage.model,
            "context_window_source": usage.context_window_source,
            "observed_at": usage.observed_at.isoformat().replace(
                "+00:00", "Z",
            ),
        }
        if usage.session is not None:
            payload["session"] = {
                "id": usage.session.id,
                "source": usage.session.source,
            }
        await self._send_json(writer, 200, json.dumps(payload))

    async def _handle_metadata(
        self,
        writer: asyncio.StreamWriter,
        raw_target: str,
        headers: dict[str, str],
    ) -> None:
        """Courier one approved metadata GET to the resolved backend mount."""
        if self._metadata_courier is None:
            await self._send_error(writer, 502, "Backend request failed")
            return
        try:
            response = await self._metadata_courier.forward(raw_target, headers)
        except MultipleCredentialsError as exc:
            await self._send_error(writer, 400, str(exc))
            return
        except httpx.HTTPError:
            await self._send_error(writer, 502, "Backend request failed")
            return
        await self._send_bytes(
            writer,
            response.status,
            response.body,
            content_type=response.content_type,
        )

    async def _fetch_private_catalog(
        self, inbound_headers: dict[str, str],
    ) -> ModelCatalog:
        """Fetch and parse trusted private catalog facts for vLLM identity."""
        if (
            self._metadata_courier is None
            or self._private_catalog_url is None
            or self._catalog_parser is None
        ):
            raise BackendError(502, "Backend model catalog is not configured")
        try:
            response = await self._metadata_courier.fetch_private(
                self._private_catalog_url, inbound_headers,
            )
        except MultipleCredentialsError:
            raise
        except httpx.HTTPError as exc:
            raise BackendError(502, "Backend model catalog request failed") from exc
        if response.status != 200:
            raise BackendError(response.status)
        try:
            payload = json.loads(response.body)
            return self._catalog_parser(payload)
        except (TypeError, ValueError) as exc:
            raise BackendError(502, "Backend model catalog was unusable") from exc

    async def _fetch_reporting_json(
        self,
        url: str,
        inbound_headers: dict[str, str],
    ) -> Any:
        """Fetch one private reporting document after response delivery."""

        if self._metadata_courier is None:
            raise BackendError(502, "Backend reporting metadata is not configured")
        try:
            response = await self._metadata_courier.fetch_private(
                url, inbound_headers,
            )
        except MultipleCredentialsError:
            raise
        except httpx.HTTPError as exc:
            raise BackendError(502, "Backend reporting metadata request failed") from exc
        if response.status != 200:
            raise BackendError(response.status)
        try:
            return json.loads(response.body)
        except (TypeError, ValueError) as exc:
            raise BackendError(502, "Backend reporting metadata was unusable") from exc

    async def _resolve_context_window(
        self,
        facts: RequestFacts,
        inbound_headers: dict[str, str],
        inbound_protocol: str,
    ) -> tuple[int, str] | None:
        """Resolve one exact-model reporting denominator by closed precedence."""

        config = self._context_reporting
        if config is None:
            return None
        configured = config.context_window_tokens
        if (
            isinstance(configured, int)
            and not isinstance(configured, bool)
            and configured > 0
        ):
            source = "managed_backend" if config.managed else "operator_config"
            return configured, source
        if config.managed:
            return None

        model = facts.effective_model
        if not isinstance(model, str) or not model:
            return None

        current = self._context_manager.published_usage
        if (
            current is not None
            and current.model == model
            and current.context_window_source == "backend_metadata"
            and current.context_window_tokens is not None
        ):
            return current.context_window_tokens, "backend_metadata"

        metadata_format = config.metadata_format
        if metadata_format == MetadataFormat.VLLM_MODELS:
            catalog = facts.model_catalog
            if catalog is None:
                if config.metadata_url is None:
                    return None
                payload = await self._fetch_reporting_json(
                    config.metadata_url, inbound_headers,
                )
                if self._catalog_parser is None:
                    return None
                catalog = self._catalog_parser(payload)
            window = catalog.context_length_for(model)
        elif metadata_format == MetadataFormat.LLAMA_PROPERTIES:
            if config.metadata_url is None:
                return None
            payload = await self._fetch_reporting_json(
                config.metadata_url, inbound_headers,
            )
            settings = (
                payload.get("default_generation_settings")
                if isinstance(payload, dict)
                else None
            )
            window = settings.get("n_ctx") if isinstance(settings, dict) else None
        elif metadata_format == MetadataFormat.ANTHROPIC_MODELS:
            resolver = getattr(self._client, "_get_context_length_for_model", None)
            if not callable(resolver):
                return None
            extra_headers = resolve_inbound_credential(
                inbound_headers,
                source_protocol=inbound_protocol,
                target_protocol=self._backend_protocol,
                backend_api_key_present=self._backend_api_key_present,
            )
            window = await resolver(model, extra_headers)
        else:
            return None

        if (
            not isinstance(window, int)
            or isinstance(window, bool)
            or window <= 0
        ):
            return None
        return window, "backend_metadata"

    async def _finalize_context_report(
        self,
        facts: RequestFacts,
        inbound_headers: dict[str, str],
        inbound_protocol: str,
    ) -> None:
        """Publish or clear one eligible report only after complete delivery."""

        if not facts.reporting_eligible or not facts.completed:
            return
        usage = facts.usage
        if usage is None:
            self._context_manager.clear_published_usage()
            return
        numerator = (
            usage.prompt_tokens
            + usage.cache_creation_input_tokens
            + usage.cache_read_input_tokens
        )
        if not isinstance(numerator, int) or isinstance(numerator, bool) or numerator < 0:
            self._context_manager.clear_published_usage()
            return
        try:
            resolved = await self._resolve_context_window(
                facts, inbound_headers, inbound_protocol,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.info("Context reporting metadata unavailable", exc_info=True)
            resolved = None
        if resolved is None:
            self._context_manager.clear_published_usage()
            return
        window, source = resolved
        self._context_manager.record_published_usage(ContextUsage(
            current_usage_tokens=numerator,
            context_window_tokens=window,
            model=facts.effective_model,
            context_window_source=source,
            observed_at=datetime.now(timezone.utc),
            session=facts.session,
        ))

    async def _handle_completions(
        self,
        writer: asyncio.StreamWriter,
        body_bytes: bytes,
        protocol: str = "openai",
        headers: dict[str, str] | None = None,
    ) -> None:
        """POST /v1/chat/completions (or /v1/messages) — the main proxy endpoint."""
        headers = headers or {}
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            await self._send_error(writer, 400, "Invalid JSON")
            return

        if not isinstance(body, dict):
            await self._send_error(writer, 400, "Request body must be a JSON object")
            return

        is_stream = body.get("stream", False)
        request_facts = RequestFacts()
        observe_request_context(body, headers, request_facts)
        msg_count = len(body.get("messages", []))
        tool_count = len(body.get("tools", []))
        logger.info(
            "   proto=%s stream=%s messages=%d tools=%d model=%s",
            protocol, is_stream, msg_count, tool_count, body.get("model", "?"),
        )

        # Streaming responses flush a 200 + SSE header before the handler runs
        # (so a queued client knows the connection is alive). Run the checks that
        # must be able to fail with a real HTTP status — model and credential
        # resolution plus the first-request discovery probe — BEFORE that flush,
        # so a bad request returns 400/401 rather than 200 + an SSE error event.
        # On success discovery latches; both resolvers are pure and repeat
        # harmlessly in the handler.
        # Non-streaming needs no pre-check — it never flushes early, so its
        # errors already carry a real status.
        if is_stream:
            predispatch = await self._predispatch(body, protocol, headers)
            request_facts = predispatch.facts
            if predispatch.error is not None:
                await self._send_exception(
                    writer, predispatch.error, protocol, as_stream=False,
                )
                return

        if self._serialize:
            # Queue the request and wait for the worker to process it
            item = _QueueItem(
                body=body, protocol=protocol, headers=headers, facts=request_facts,
            )
            queue_depth = self._queue.qsize()
            if queue_depth > 0:
                logger.info("   Queued (depth=%d)", queue_depth + 1)

            # For streaming requests, send SSE headers immediately so the
            # client knows we're alive while waiting in the queue
            if is_stream:
                await self._send_sse_header(writer)

            self._queue.put_nowait(item)

            # Wait for result, monitoring for client disconnect
            result = await self._await_with_disconnect(item, writer)
        else:
            if is_stream:
                await self._send_sse_header(writer)
            result = await self._run_handler(
                body, protocol, headers, request_facts,
            )

        if result is None:
            # Client disconnected
            logger.info("<< Client disconnected, discarding result")
            return

        if isinstance(result, Exception):
            await self._send_exception(writer, result, protocol, as_stream=is_stream)
            return

        if writer.is_closing():
            return
        if is_stream:
            logger.info("<< SSE %d events", len(result))
            delivered = await self._send_sse_body(writer, result, protocol=protocol)
            if not delivered:
                return
        else:
            logger.info("<< JSON 200")
            await self._send_json(writer, 200, json.dumps(result))
            if writer.is_closing():
                return
        await self._finalize_context_report(request_facts, headers, protocol)

    async def _predispatch(
        self, body: dict[str, Any], protocol: str, headers: dict[str, str],
    ) -> _PredispatchResult:
        """Pre-flush validation for a streaming request.

        Validates request-local model requirements and the inbound credential,
        then runs first-request backend discovery. The effective wire model is
        selected only after discovery. These checks must be able to fail with a
        real HTTP status, so this returns the Exception to surface, or None to
        proceed. The handler reuses the populated request-local facts.
        """
        facts = RequestFacts()
        observe_request_context(body, headers, facts)
        try:
            validate_request_model(
                body,
                self._client,
                self._client_adapter,
            )
            resolve_inbound_credential(
                headers,
                source_protocol=protocol,
                target_protocol=self._backend_protocol,
                backend_api_key_present=self._backend_api_key_present,
            )
            await run_lazy_discovery(
                self._client,
                self._lazy_discovery,
                headers,
                self._fetch_private_catalog,
                facts,
            )
            facts.effective_model = resolve_effective_model(
                body,
                self._client,
                self._client_adapter,
            )
            return _PredispatchResult(error=None, facts=facts)
        except Exception as exc:
            return _PredispatchResult(error=exc, facts=facts)

    async def _send_exception(
        self,
        writer: asyncio.StreamWriter,
        exc: Exception,
        protocol: str,
        as_stream: bool,
    ) -> None:
        """Send an exception as the response.

        ``as_stream`` True → an SSE error event (the 200 + SSE header was already
        flushed, e.g. a backend fault mid-generation); False → a real HTTP error
        status. Exception messages are safe to log/return by construction —
        forge never authors a secret into one, and ``BackendError`` keeps the raw
        backend body off its message (on ``exc.body`` instead).
        """
        error_msg = str(exc)
        logger.info("<< ERROR: %s", error_msg[:120])
        # Missing models and credential conflicts are client errors (400); no
        # credential for an auth-required backend is 401, not a backend fault.
        # These messages carry no secret values.
        if isinstance(exc, (MissingModelError, MultipleCredentialsError)):
            status = 400
        elif isinstance(exc, MissingCredentialError):
            status = 401
        elif isinstance(exc, BackendDiscoveryError):
            # Deferred discovery failed: a backend auth rejection is the caller's
            # 401; any other cause (backend down, bad shape) is a 502.
            status = 401 if exc.status_code in (401, 403) else 502
        elif isinstance(exc, BackendError) and exc.status_code in (401, 403):
            # A backend auth rejection during normal dispatch (a later zero-cred
            # request to a gated backend, or a bad inbound key) is the caller's
            # 401, not a forge/backend fault.
            status = 401
        else:
            status = 502
        if as_stream:
            await self._send_sse_body(writer, [{"error": error_msg}], protocol=protocol)
        else:
            await self._send_error(writer, status, error_msg)

    async def _await_with_disconnect(
        self,
        item: _QueueItem,
        writer: asyncio.StreamWriter,
    ) -> dict[str, Any] | list[dict[str, Any]] | Exception | None:
        """Wait for a queued item's result, checking for client disconnect.

        Returns None if the client disconnected.
        """
        while not item.future.done():
            if writer.is_closing():
                item.cancelled = True
                logger.info("   Client disconnected, cancelling queued request")
                return None
            try:
                await asyncio.wait_for(
                    asyncio.shield(item.future), timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
        return item.future.result()

    async def _run_handler(
        self,
        body: dict[str, Any],
        protocol: str = "openai",
        headers: dict[str, str] | None = None,
        request_facts: RequestFacts | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]] | Exception:
        """Run the handler, catching errors."""
        try:
            return await handle_chat_completions(
                body=body,
                client=self._client,
                context_manager=self._context_manager,
                max_retries=self._max_retries,
                max_tool_errors=self._max_tool_errors,
                rescue_enabled=self._rescue_enabled,
                native_passthrough=self._native_passthrough,
                inject_respond_tool=self._inject_respond_tool,
                protocol=protocol,
                reasoning_replay=self._reasoning_replay,
                headers=headers,
                backend_protocol=self._backend_protocol,
                client_adapter=self._client_adapter,
                backend_api_key_present=self._backend_api_key_present,
                lazy_discovery=self._lazy_discovery,
                request_facts=request_facts or RequestFacts(),
                catalog_fetcher=self._fetch_private_catalog,
            )
        except Exception as exc:
            logger.exception("Handler error")
            return exc

    async def _send_json(
        self, writer: asyncio.StreamWriter, status: int, body: str,
    ) -> None:
        """Send a JSON HTTP response."""
        response = (
            f"HTTP/1.1 {status} {_status_text(status)}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body.encode())}\r\n"
            f"Connection: close\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"\r\n"
            f"{body}"
        )
        writer.write(response.encode())
        await writer.drain()

    async def _send_no_content(self, writer: asyncio.StreamWriter) -> None:
        """Send an exact empty 204 response."""

        response = (
            "HTTP/1.1 204 No Content\r\n"
            "Content-Length: 0\r\n"
            "Connection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n"
        )
        writer.write(response.encode())
        await writer.drain()

    async def _send_bytes(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        body: bytes,
        content_type: str | None = None,
    ) -> None:
        """Send buffered backend bytes with only approved response headers."""
        response_headers = [f"HTTP/1.1 {status} {_status_text(status)}"]
        if content_type is not None:
            response_headers.append(f"Content-Type: {content_type}")
        response_headers.extend([
            f"Content-Length: {len(body)}",
            "Connection: close",
            "Access-Control-Allow-Origin: *",
            "",
            "",
        ])
        writer.write("\r\n".join(response_headers).encode("latin-1") + body)
        await writer.drain()

    async def _send_sse_header(self, writer: asyncio.StreamWriter) -> None:
        """Send SSE response headers immediately (before body is ready)."""
        header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Transfer-Encoding: chunked\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: keep-alive\r\n"
            "\r\n"
        )
        writer.write(header.encode())
        await writer.drain()

    async def _send_sse_body(
        self,
        writer: asyncio.StreamWriter,
        events: list[dict[str, Any]],
        protocol: str = "openai",
    ) -> bool:
        """Send SSE event data and terminator. Headers must already be sent.

        OpenAI wire format: ``data: {json}\\n\\n`` per event, terminated by
        ``data: [DONE]\\n\\n``.

        Anthropic wire format: ``event: <type>\\ndata: {json}\\n\\n`` per
        event (type read from the event's top-level ``type`` field). No
        ``[DONE]`` terminator — the ``message_stop`` event ends the stream.
        """
        for event in events:
            if writer.is_closing():
                return False
            if protocol == "anthropic":
                event_type = event.get("type", "")
                payload = f"event: {event_type}\ndata: {json.dumps(event)}\n\n".encode()
            else:
                payload = f"data: {json.dumps(event)}\n\n".encode()
            writer.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            await writer.drain()

        if writer.is_closing():
            return False
        if protocol == "openai":
            done = b"data: [DONE]\n\n"
            writer.write(f"{len(done):x}\r\n".encode() + done + b"\r\n")

        # Terminating zero-length chunk
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        if writer.is_closing():
            return False
        logger.info("<< SSE complete (%s)", protocol)
        return True

    async def _send_error(
        self, writer: asyncio.StreamWriter, status: int, message: str,
    ) -> None:
        """Send an error JSON response."""
        body = json.dumps({"error": {"message": message, "type": "proxy_error"}})
        await self._send_json(writer, status, body)

    async def _send_cors_preflight(self, writer: asyncio.StreamWriter) -> None:
        """Handle CORS preflight."""
        response = (
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            # x-api-key is a first-class inbound credential slot (Anthropic-wire);
            # browser clients must be allowed to preflight it. anthropic-version /
            # anthropic-beta are standard Anthropic client headers.
            "Access-Control-Allow-Headers: Content-Type, Authorization, X-Api-Key, "
            "anthropic-version, anthropic-beta\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        writer.write(response.encode())
        await writer.drain()


def _status_text(code: int) -> str:
    """HTTP status code to text."""
    return {
        200: "OK",
        204: "No Content",
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        413: "Payload Too Large",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
    }.get(code, "Error")
