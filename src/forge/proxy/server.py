"""Raw ASGI application for the forge proxy server.

No framework dependencies -- just an async function that speaks HTTP.

Supports two modes:
  - External: connects to an existing backend (--backend-url)
  - Managed: starts/stops the backend via ServerManager (--backend)

Endpoints:
    POST /v1/chat/completions  -- Core proxy (guardrails + forward)
    GET  /v1/models            -- Pass-through to backend
    GET  /health               -- Proxy health check
    *    /*                    -- Pass-through to backend
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from forge.proxy.handler import ChatHandler

logger = logging.getLogger("forge.proxy")


def create_app(
    backend_url: str,
    managed_backend: str | None = None,
    gguf_path: str | None = None,
    backend_port: int = 8080,
    budget_tokens: int | None = None,
    keep_recent: int = 2,
    max_retries: int = 3,
    rescue_enabled: bool = True,
    compact_enabled: bool = True,
    timeout: float = 300.0,
) -> Any:
    """Create the proxy ASGI application.

    Args:
        backend_url: Base URL of the model server.
        managed_backend: If set ("llamaserver" or "llamafile"), forge manages
            the backend process via ServerManager.
        gguf_path: Path to GGUF model file (required for managed mode).
        backend_port: Port for the managed backend.
        budget_tokens: Override context budget. None = auto-detect.
        keep_recent: Compaction: recent iterations to preserve.
        max_retries: Max guardrail retry attempts per request.
        rescue_enabled: Attempt to parse tool calls from plain text.
        compact_enabled: Enable context compaction.
        timeout: HTTP timeout for backend requests in seconds.

    Returns:
        An ASGI application callable.
    """
    # These are initialized during lifespan startup (async context needed).
    state: dict[str, Any] = {
        "handler": None,
        "passthrough_client": None,
        "server_manager": None,
    }

    backend = backend_url.rstrip("/")

    async def _startup() -> None:
        from forge.context.manager import ContextManager
        from forge.context.strategies import TieredCompact

        server_manager = None
        context_manager = None

        if managed_backend:
            # Managed mode: start backend via ServerManager
            from forge.server import ServerManager, BudgetMode

            server_manager = ServerManager(
                backend=managed_backend,
                port=backend_port,
            )

            if budget_tokens is not None:
                budget_mode = BudgetMode.MANUAL
            else:
                budget_mode = BudgetMode.FORGE_FULL

            budget = await server_manager.start_with_budget(
                model=gguf_path,
                gguf_path=gguf_path,
                mode="native",
                budget_mode=budget_mode,
                manual_tokens=budget_tokens,
            )

            logger.info("Managed backend started, budget: %d tokens", budget)

            if compact_enabled:
                context_manager = ContextManager(
                    strategy=TieredCompact(keep_recent=keep_recent),
                    budget_tokens=budget,
                )
        else:
            # External mode: auto-detect budget from backend
            if compact_enabled:
                if budget_tokens is not None:
                    budget = budget_tokens
                    logger.info("Using manual context budget: %d tokens", budget)
                else:
                    budget = await _detect_budget(backend, timeout)
                    logger.info("Auto-detected context budget: %d tokens", budget)

                context_manager = ContextManager(
                    strategy=TieredCompact(keep_recent=keep_recent),
                    budget_tokens=budget,
                )

        state["server_manager"] = server_manager
        state["handler"] = ChatHandler(
            backend_url=backend,
            context_manager=context_manager,
            max_retries=max_retries,
            rescue_enabled=rescue_enabled,
            timeout=timeout,
        )
        state["passthrough_client"] = httpx.AsyncClient(timeout=timeout)

    async def _shutdown() -> None:
        if state["handler"]:
            await state["handler"].close()
        if state["passthrough_client"]:
            await state["passthrough_client"].aclose()
        if state["server_manager"]:
            logger.info("Stopping managed backend...")
            await state["server_manager"].stop()
            logger.info("Managed backend stopped")

    async def app(scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            await _handle_lifespan(scope, receive, send)
            return

        if scope["type"] != "http":
            return

        handler = state["handler"]
        passthrough_client = state["passthrough_client"]
        path = scope["path"]
        method = scope["method"]

        if path == "/health" and method == "GET":
            await _send_json(send, 200, {"status": "ok"})

        elif path == "/v1/chat/completions" and method == "POST":
            body = await _read_body(receive)
            parsed = json.loads(body)
            is_streaming = parsed.get("stream", False)

            chunks, status = await handler.handle(parsed)

            if status != 200 or not is_streaming:
                await _send_response(
                    send, status, chunks[0],
                    content_type=b"application/json",
                )
            else:
                await _send_sse(send, chunks)

        elif path == "/v1/models" and method == "GET":
            response = await passthrough_client.get(f"{backend}/v1/models")
            await _send_response(
                send, response.status_code, response.content,
                content_type=b"application/json",
            )

        else:
            body = await _read_body(receive)
            headers = _extract_headers(scope)
            response = await passthrough_client.request(
                method=method,
                url=f"{backend}{path}",
                content=body,
                headers=headers,
            )
            ct = response.headers.get("content-type", "application/json")
            await _send_response(
                send, response.status_code, response.content,
                content_type=ct.encode(),
            )

    async def _handle_lifespan(scope: dict, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    await _startup()
                    await send({"type": "lifespan.startup.complete"})
                except Exception as e:
                    logger.error("Startup failed: %s", e)
                    await send({"type": "lifespan.startup.failed", "message": str(e)})
                    return
            elif message["type"] == "lifespan.shutdown":
                await _shutdown()
                await send({"type": "lifespan.shutdown.complete"})
                return

    return app


async def _detect_budget(backend_url: str, timeout: float) -> int:
    """Auto-detect context budget from the backend's /props endpoint.

    Raises BudgetResolutionError if detection fails.
    """
    from forge.errors import BudgetResolutionError

    base = backend_url.rstrip("/")
    # Strip /v1 suffix if present
    if base.endswith("/v1"):
        base = base[:-3]

    url = f"{base}/props"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        raise BudgetResolutionError(exc)

    n_ctx = data.get("default_generation_settings", {}).get("n_ctx")
    if n_ctx is None:
        raise BudgetResolutionError(
            Exception(f"Backend /props response missing n_ctx: {data}")
        )

    return int(n_ctx)


async def _read_body(receive: Any) -> bytes:
    body = b""
    while True:
        msg = await receive()
        body += msg.get("body", b"")
        if not msg.get("more_body", False):
            break
    return body


def _extract_headers(scope: dict) -> dict[str, str]:
    skip = {b"host", b"content-length", b"transfer-encoding"}
    return {
        k.decode(): v.decode()
        for k, v in scope.get("headers", [])
        if k.lower() not in skip
    }


async def _send_json(send: Any, status: int, data: Any) -> None:
    body = json.dumps(data).encode()
    await _send_response(send, status, body, content_type=b"application/json")


async def _send_response(
    send: Any, status: int, body: bytes, content_type: bytes = b"application/json"
) -> None:
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            [b"content-type", content_type],
            [b"content-length", str(len(body)).encode()],
        ],
    })
    await send({
        "type": "http.response.body",
        "body": body,
    })


async def _send_sse(send: Any, chunks: list[bytes]) -> None:
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [
            [b"content-type", b"text/event-stream"],
            [b"cache-control", b"no-cache"],
            [b"connection", b"keep-alive"],
        ],
    })
    for chunk in chunks:
        await send({
            "type": "http.response.body",
            "body": chunk,
            "more_body": True,
        })
    await send({
        "type": "http.response.body",
        "body": b"",
    })
