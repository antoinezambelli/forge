"""Server lifecycle management and budget resolution.

ServerManager owns backend lifecycle (start/stop processes, health polling)
and resolves context budgets based on BudgetMode.  It is the single point
of truth for "how much context can I use?" — clients just send messages.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from forge.context.hardware import detect_hardware
from forge.context.manager import CompactEvent, ContextManager
from forge.context.strategies import TieredCompact
from forge.errors import BackendError, BudgetResolutionError


class BudgetMode(str, Enum):
    """How to determine the context budget for compaction."""

    BACKEND = "backend"  # Trust the backend's default. No override sent.
    MANUAL = "manual"  # User specifies exact token count.
    FORGE_FULL = "forge-full"  # Max safe context (server auto-tune / Ollama tier).
    FORGE_FAST = "forge-fast"  # Half of full. Trades context for faster attention.


class ServerManager:
    """Manages backend lifecycle and resolves context budgets.

    For llama-server/llamafile: starts/stops processes, health polling,
    /props query for actual n_ctx.
    For Ollama: ``ollama stop`` for clean VRAM unloads between model switches.
    """

    def __init__(
        self,
        backend: str,
        port: int = 8080,
        models_dir: str | Path | None = None,
    ) -> None:
        """
        Args:
            backend: Which backend this manager controls
                     (``"ollama"`` | ``"llamaserver"`` | ``"llamafile"``).
            port: Server port (llama-server / llamafile only).
            models_dir: Directory containing GGUF files.
        """
        self._backend = backend
        self._port = port
        self._models_dir = Path(models_dir) if models_dir is not None else None

        self._proc: subprocess.Popen | None = None
        self._current_model: str | None = None
        self._current_mode: str | None = None
        self._current_ctx: int | None = None
        self._current_flags: tuple[str, ...] = ()

    # ── start / stop ────────────────────────────────────────────

    async def start(
        self,
        model: str,
        gguf_path: str | Path,
        mode: str = "native",
        extra_flags: list[str] | None = None,
        ctx_override: int | None = None,
    ) -> None:
        """Start a llama-server/llamafile process.

        No-op if the same model + mode + ctx + extra_flags is already running.
        For ``backend="ollama"`` this is always a no-op.

        For ``backend="llamafile"``, the llamafile runtime binary is
        located automatically in the same directory as *gguf_path*
        (glob ``llamafile-*``).

        Args:
            model: Canonical model name.
            gguf_path: Path to the GGUF or llamafile model file.
            mode: ``"native"`` or ``"prompt"``.
            extra_flags: Additional CLI flags (e.g. ``["--reasoning-format", "auto"]``).
            ctx_override: If set, pass ``-c <value>`` to the server.
        """
        if self._backend == "ollama":
            return

        # Reuse if same configuration is already running
        flags = tuple(extra_flags) if extra_flags else ()
        if (
            self._current_model == model
            and self._current_mode == mode
            and self._current_ctx == ctx_override
            and self._current_flags == flags
        ):
            return

        await self.stop()

        if self._backend == "llamafile":
            runtime = self._find_llamafile_runtime(Path(gguf_path).parent)
            cmd: list[str] = [
                str(runtime),
                "--server",
                "--nobrowser",
                "-m",
                str(gguf_path),
                "-ngl",
                "999",
                "--port",
                str(self._port),
            ]
        else:
            cmd = [
                "llama-server",
                "-m",
                str(gguf_path),
                "-ngl",
                "999",
                "--port",
                str(self._port),
            ]
        if mode == "native":
            cmd.append("--jinja")
        if extra_flags:
            cmd.extend(extra_flags)
        if ctx_override is not None:
            cmd.extend(["-c", str(ctx_override)])

        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        await self._wait_healthy()

        self._current_model = model
        self._current_mode = mode
        self._current_ctx = ctx_override
        self._current_flags = flags

    async def stop(self) -> None:
        """Stop the current server / unload the Ollama model."""
        if self._backend == "ollama":
            if self._current_model is not None:
                subprocess.run(["ollama", "stop", self._current_model])
                self._current_model = None
            return

        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
            self._current_model = None
            self._current_mode = None
            self._current_ctx = None
            self._current_flags = ()
            await asyncio.sleep(3)  # let VRAM clear

    # ── /props + context ────────────────────────────────────────

    async def query_props(self) -> dict[str, Any]:
        """Query the llama-server ``/props`` endpoint.

        Returns:
            Parsed JSON from the response.

        Raises:
            BackendError: On non-200 response.
        """
        url = f"http://localhost:{self._port}/props"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise BackendError(resp.status_code, resp.text)
            return resp.json()

    async def get_server_context(self) -> int:
        """Read the actual n_ctx from the running server.

        Returns:
            The context length.

        Raises:
            BudgetResolutionError: Server unreachable, returned an error,
                or response missing the n_ctx field.
        """
        try:
            props = await self.query_props()
        except (httpx.HTTPError, BackendError) as exc:
            raise BudgetResolutionError(cause=exc) from exc
        ctx = props.get("default_generation_settings", {}).get("n_ctx")
        if ctx is None:
            raise BudgetResolutionError()
        return ctx

    # ── budget resolution ───────────────────────────────────────

    async def resolve_budget(
        self,
        mode: BudgetMode,
        manual_tokens: int | None = None,
    ) -> int:
        """Resolve the ContextManager budget for the given mode.

        Args:
            mode: The budget mode to use.
            manual_tokens: Required when ``mode`` is ``MANUAL`` and
                           backend is ``"ollama"``.

        Returns:
            Budget in tokens.

        Raises:
            ValueError: ``MANUAL`` mode without ``manual_tokens``.
            BudgetResolutionError: Server can't provide a context value.
        """
        if mode == BudgetMode.MANUAL:
            if self._backend == "ollama":
                if manual_tokens is None:
                    raise ValueError("manual mode requires manual_tokens")
                return manual_tokens
            # llamaserver / llamafile: server was started with -c
            return await self.get_server_context()

        if self._backend == "ollama":
            full = self._ollama_vram_tier_budget()
            if mode == BudgetMode.FORGE_FAST:
                return full // 2
            return full

        # llamaserver / llamafile — all non-manual modes read /props
        return await self.get_server_context()

    async def start_with_budget(
        self,
        model: str,
        gguf_path: str | Path,
        mode: str = "native",
        budget_mode: BudgetMode = BudgetMode.BACKEND,
        manual_tokens: int | None = None,
        extra_flags: list[str] | None = None,
    ) -> int:
        """Start server with the specified budget mode and return the resolved budget.

        Handles the mode-specific startup dance:
        - BACKEND/FORGE_FULL: start without -c, read /props
        - MANUAL: start with -c = manual_tokens, read /props
        - FORGE_FAST: start without -c, read /props for max,
                      restart with -c = max // 2, read /props again

        For Ollama: ignores gguf_path, doesn't start a process.
        Returns VRAM tier budget.

        Args:
            model: Model name (Ollama-style canonical name).
            gguf_path: Path to GGUF file (llamaserver/llamafile only).
            mode: FC mode - ``"native"`` or ``"prompt"``.
            budget_mode: How to determine context budget.
            manual_tokens: Required for MANUAL mode.
            extra_flags: Additional server CLI flags.

        Returns:
            Resolved budget in tokens (ready for ContextManager).

        Raises:
            ValueError: MANUAL mode without manual_tokens.
            BudgetResolutionError: Server can't provide context info.
        """
        if budget_mode == BudgetMode.MANUAL and manual_tokens is None:
            raise ValueError("manual mode requires manual_tokens")

        if self._backend == "ollama":
            self._current_model = model
            return await self.resolve_budget(budget_mode, manual_tokens)

        if budget_mode == BudgetMode.FORGE_FAST:
            # Phase 1: start with auto-tune to discover max
            await self.start(model, gguf_path, mode, extra_flags, ctx_override=None)
            full_ctx = await self.get_server_context()
            half_ctx = full_ctx // 2

            # Phase 2: restart with half context
            await self.start(model, gguf_path, mode, extra_flags, ctx_override=half_ctx)
            return await self.resolve_budget(budget_mode)

        # BACKEND / FORGE_FULL / MANUAL
        ctx_override = manual_tokens if budget_mode == BudgetMode.MANUAL else None
        await self.start(model, gguf_path, mode, extra_flags, ctx_override=ctx_override)
        return await self.resolve_budget(budget_mode, manual_tokens)

    def _ollama_vram_tier_budget(self) -> int:
        """Published Ollama defaults based on total VRAM."""
        hw = detect_hardware()
        if hw is None:
            return 4096
        vram_gb = hw.vram_total_gb
        if vram_gb >= 48:
            return 262_144
        elif vram_gb >= 24:
            return 32_768
        else:
            return 4_096

    @staticmethod
    def _find_llamafile_runtime(directory: Path) -> Path:
        """Find the llamafile runtime binary (``llamafile-*``) in *directory*."""
        hits = sorted(directory.glob("llamafile-*"))
        if not hits:
            raise FileNotFoundError(
                f"No llamafile runtime found in {directory} "
                "(expected a file matching llamafile-*)"
            )
        return hits[-1]  # highest version

    # ── health polling ──────────────────────────────────────────

    async def _wait_healthy(self, timeout: float = 120.0) -> None:
        """Poll ``/health`` until the server is ready.

        Raises:
            RuntimeError: If the server doesn't become healthy within *timeout*.
        """
        url = f"http://localhost:{self._port}/health"
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("status") == "ok":
                            return
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(2)
        raise RuntimeError(
            f"Server did not become healthy within {timeout}s"
        )


async def setup_backend(
    backend: str,
    model: str,
    budget_mode: BudgetMode = BudgetMode.BACKEND,
    manual_tokens: int | None = None,
    client: Any | None = None,
    gguf_path: str | Path | None = None,
    mode: str = "native",
    port: int = 8080,
    extra_flags: list[str] | None = None,
    compact_threshold: float = 0.75,
    on_compact: Callable[[CompactEvent], None] | None = None,
) -> tuple[ServerManager, ContextManager]:
    """One-call setup: start backend, resolve budget, create ContextManager.

    For Ollama backends, pass the ``client`` so that ``set_num_ctx()`` is
    called automatically — keeping the client's per-request ``num_ctx``
    in sync with the resolved budget.  For llama-server / llamafile the
    context size is baked into the server process via ``-c``, so the
    client parameter is ignored.

    Example usage::

        client = OllamaClient(model=model)
        server, ctx = await setup_backend(
            backend="ollama",
            model="ministral-3:14b-instruct-2512-q4_K_M",
            budget_mode=BudgetMode.FORGE_FAST,
            client=client,
        )
        runner = WorkflowRunner(client=client, context_manager=ctx)
        # ... run workflows ...
        await server.stop()

    Returns:
        (ServerManager, ContextManager) tuple. Caller is responsible
        for calling ``server.stop()`` when done.
    """
    server = ServerManager(backend=backend, port=port)
    budget = await server.start_with_budget(
        model=model,
        gguf_path=gguf_path or "",
        mode=mode,
        budget_mode=budget_mode,
        manual_tokens=manual_tokens,
        extra_flags=extra_flags,
    )

    # Ollama: wire num_ctx so every request uses the resolved budget
    if backend == "ollama" and client is not None and hasattr(client, "set_num_ctx"):
        client.set_num_ctx(budget)

    ctx_manager = ContextManager(
        strategy=TieredCompact(),
        budget_tokens=budget,
        compact_threshold=compact_threshold,
        on_compact=on_compact,
    )
    return server, ctx_manager
