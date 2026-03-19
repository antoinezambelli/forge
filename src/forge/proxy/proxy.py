"""ProxyServer — programmatic API for the forge proxy.

Start and stop the proxy from Python code:

    from forge.proxy import ProxyServer

    proxy = ProxyServer(backend_url="http://localhost:8080")
    proxy.start()  # non-blocking
    # ... your code hits http://localhost:8081/v1 ...
    proxy.stop()   # clean shutdown

Managed mode (forge starts llama-server):

    proxy = ProxyServer(backend="llamaserver", gguf="path/to/model.gguf")
    proxy.start()
    # ... llama-server + proxy both running ...
    proxy.stop()   # stops both
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger("forge.proxy")


class ProxyServer:
    """Forge guardrail proxy with programmatic start/stop.

    Args:
        backend_url: Connect to an existing backend. Mutually exclusive
            with ``backend``.
        backend: Managed mode — "llamaserver" or "llamafile". Requires
            ``gguf``. Mutually exclusive with ``backend_url``.
        gguf: Path to GGUF model file (managed mode only).
        port: Proxy listen port.
        host: Proxy bind address.
        backend_port: Port for the managed backend.
        budget: Context budget override in tokens. None = auto-detect.
        keep_recent: Compaction: recent iterations to preserve.
        max_retries: Max guardrail retry attempts per request.
        rescue_enabled: Attempt to parse tool calls from plain text.
        compact_enabled: Enable context compaction.
        timeout: Backend request timeout in seconds.
    """

    def __init__(
        self,
        backend_url: str | None = None,
        backend: str | None = None,
        gguf: str | None = None,
        port: int = 8081,
        host: str = "127.0.0.1",
        backend_port: int = 8080,
        budget: int | None = None,
        keep_recent: int = 2,
        max_retries: int = 3,
        rescue_enabled: bool = True,
        compact_enabled: bool = True,
        timeout: float = 300.0,
    ) -> None:
        if not backend_url and not backend:
            raise ValueError("Provide either backend_url or backend")
        if backend_url and backend:
            raise ValueError("Provide backend_url or backend, not both")
        if backend and not gguf:
            raise ValueError("backend requires gguf path")

        self._backend_url = backend_url or f"http://localhost:{backend_port}"
        self._managed_backend = backend
        self._gguf = gguf
        self._port = port
        self._host = host
        self._backend_port = backend_port
        self._budget = budget
        self._keep_recent = keep_recent
        self._max_retries = max_retries
        self._rescue_enabled = rescue_enabled
        self._compact_enabled = compact_enabled
        self._timeout = timeout

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any = None
        self._started = threading.Event()
        self._stop_event: asyncio.Event | None = None

    @property
    def url(self) -> str:
        """Base URL of the running proxy (e.g. http://127.0.0.1:8081)."""
        return f"http://{self._host}:{self._port}"

    def start(self) -> None:
        """Start the proxy in a background thread. Non-blocking.

        Blocks briefly until the server is ready to accept connections.
        """
        if self._thread is not None:
            raise RuntimeError("Proxy already running")

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started.wait(timeout=120)

        if not self._started.is_set():
            raise RuntimeError("Proxy failed to start within 120 seconds")

    def stop(self) -> None:
        """Stop the proxy and clean up (including managed backend)."""
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)

        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None

        self._loop = None
        self._server = None
        self._started.clear()

    def _run(self) -> None:
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        """Async server loop."""
        from forge.proxy.server import create_app

        app = create_app(
            backend_url=self._backend_url,
            managed_backend=self._managed_backend,
            gguf_path=self._gguf,
            backend_port=self._backend_port,
            budget_tokens=self._budget,
            keep_recent=self._keep_recent,
            max_retries=self._max_retries,
            rescue_enabled=self._rescue_enabled,
            compact_enabled=self._compact_enabled,
            timeout=self._timeout,
        )

        # Run lifespan startup (initializes handler, starts managed backend)
        startup_complete = asyncio.Event()
        shutdown_complete = asyncio.Event()
        self._stop_event = asyncio.Event()

        async def lifespan_send(message: dict) -> None:
            if message["type"] == "lifespan.startup.complete":
                startup_complete.set()
            elif message["type"] == "lifespan.startup.failed":
                logger.error("Startup failed: %s", message.get("message", ""))
                startup_complete.set()  # unblock, but with failure
            elif message["type"] == "lifespan.shutdown.complete":
                shutdown_complete.set()

        lifespan_messages: asyncio.Queue = asyncio.Queue()
        await lifespan_messages.put({"type": "lifespan.startup"})

        async def lifespan_receive() -> dict:
            return await lifespan_messages.get()

        # Start lifespan
        lifespan_task = asyncio.create_task(
            app({"type": "lifespan", "asgi": {"version": "3.0"}}, lifespan_receive, lifespan_send)
        )
        await startup_complete.wait()

        # Start HTTP server
        async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await _handle_http(app, reader, writer)

        server = await asyncio.start_server(
            handle_connection, self._host, self._port
        )
        self._server = server
        self._started.set()

        logger.info("Proxy serving on %s:%d", self._host, self._port)

        # Wait for stop signal
        await self._stop_event.wait()

        # Shutdown
        server.close()
        await server.wait_closed()

        await lifespan_messages.put({"type": "lifespan.shutdown"})
        await shutdown_complete.wait()
        lifespan_task.cancel()


async def _handle_http(
    app: Any, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Minimal HTTP/1.1 handler that bridges raw sockets to the ASGI app."""
    try:
        # Read request line
        request_line = await reader.readline()
        if not request_line:
            writer.close()
            return

        parts = request_line.decode().strip().split(" ")
        if len(parts) < 3:
            writer.close()
            return

        method, path, _ = parts[0], parts[1], parts[2]

        # Read headers
        headers: list[tuple[bytes, bytes]] = []
        content_length = 0
        while True:
            line = await reader.readline()
            if line == b"\r\n" or line == b"\n" or not line:
                break
            decoded = line.decode().strip()
            if ":" in decoded:
                key, value = decoded.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                headers.append((key.encode(), value.encode()))
                if key == "content-length":
                    content_length = int(value)

        # Read body
        body = b""
        if content_length > 0:
            body = await reader.readexactly(content_length)

        # Build ASGI scope
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "path": path,
            "headers": headers,
        }

        # ASGI receive
        body_sent = False

        async def receive() -> dict:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        # ASGI send
        response_started = False
        response_status = 200
        response_headers: list[tuple[bytes, bytes]] = []

        async def send(message: dict) -> None:
            nonlocal response_started, response_status, response_headers

            if message["type"] == "http.response.start":
                response_started = True
                response_status = message["status"]
                response_headers = message.get("headers", [])

            elif message["type"] == "http.response.body":
                resp_body = message.get("body", b"")
                more_body = message.get("more_body", False)

                if not more_body or resp_body:
                    # Build HTTP response
                    if response_started:
                        status_line = f"HTTP/1.1 {response_status} OK\r\n"
                        writer.write(status_line.encode())
                        for key, value in response_headers:
                            writer.write(key + b": " + value + b"\r\n")
                        writer.write(b"\r\n")
                        response_started = False  # headers sent

                    if resp_body:
                        writer.write(resp_body)
                        await writer.drain()

        await app(scope, receive, send)

    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
