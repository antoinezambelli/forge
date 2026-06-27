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
                     (``"ollama"`` | ``"llamaserver"`` | ``"llamafile"`` |
                     ``"vllm"``).
            port: Server port (llama-server / llamafile / vllm only).
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
        self._current_cache_type_k: str | None = None
        self._current_cache_type_v: str | None = None
        self._current_n_slots: int | None = None
        self._current_kv_unified: bool = False

    # ── start / stop ────────────────────────────────────────────

    async def start(
        self,
        model: str,
        *,
        gguf_path: str | Path | None = None,
        model_path: str | Path | None = None,
        mode: str = "native",
        extra_flags: list[str] | None = None,
        ctx_override: int | None = None,
        cache_type_k: str | None = None,
        cache_type_v: str | None = None,
        n_slots: int | None = None,
        kv_unified: bool = False,
    ) -> None:
        """Start a llama-server/llamafile/vllm process.

        No-op if the same model + mode + ctx + extra_flags + cache types
        + slots + kv_unified is already running.
        For ``backend="ollama"`` this is always a no-op.

        Path argument is backend-specific (mutually exclusive, validated):
        - ``gguf_path`` required for ``llamaserver`` / ``llamafile``
          (single .gguf file).
        - ``model_path`` required for ``vllm`` (directory containing
          safetensors or HuggingFace repo id).

        For ``backend="llamafile"``, the llamafile runtime binary is
        located automatically in the same directory as *gguf_path*
        (glob ``llamafile-*``).

        Args:
            model: Canonical model name.
            gguf_path: Path to the GGUF or llamafile model file
                       (llamaserver / llamafile only).
            model_path: Path to model directory or HF repo id (vllm only).
            mode: ``"native"`` or ``"prompt"``. vLLM ignores this — its
                  chat template comes from the model and tool/reasoning
                  parsing is configured at server boot via extra_flags.
            extra_flags: Additional CLI flags.
            ctx_override: If set, pass ``-c <value>`` (llama-server /
                          llamafile) or ``--max-model-len <value>``
                          (vllm).
            cache_type_k: KV cache quantization type for keys
                          (e.g. ``"q8_0"``, ``"q4_0"``). llama-server /
                          llamafile only.
            cache_type_v: KV cache quantization type for values
                          (e.g. ``"q8_0"``, ``"q4_0"``). llama-server /
                          llamafile only.
            n_slots: Number of concurrent slots (each with its own KV
                     cache). llama-server / llamafile only.
            kv_unified: If True, use a single unified KV cache shared
                        across all slots. llama-server / llamafile only.
        """
        if self._backend == "ollama":
            return

        # Per-backend path validation (fail-fast on misuse).
        if self._backend in ("llamaserver", "llamafile"):
            if model_path is not None:
                raise ValueError(
                    f"backend={self._backend!r} does not accept model_path "
                    "(use gguf_path)"
                )
            if not gguf_path:
                raise ValueError(
                    f"backend={self._backend!r} requires gguf_path"
                )
        elif self._backend == "vllm":
            if gguf_path is not None:
                raise ValueError(
                    "backend='vllm' does not accept gguf_path (use model_path)"
                )
            if not model_path:
                raise ValueError("backend='vllm' requires model_path")
            if cache_type_k is not None or cache_type_v is not None:
                raise ValueError(
                    "backend='vllm' does not support cache_type_k/cache_type_v "
                    "(quantization is baked into the model artifact)"
                )
            if n_slots is not None or kv_unified:
                raise ValueError(
                    "backend='vllm' does not support n_slots/kv_unified "
                    "(vLLM has its own scheduler concepts)"
                )
        else:
            raise ValueError(f"unsupported backend: {self._backend!r}")

        # Reuse if same configuration is already running
        flags = tuple(extra_flags) if extra_flags else ()
        if (
            self._current_model == model
            and self._current_mode == mode
            and self._current_ctx == ctx_override
            and self._current_flags == flags
            and self._current_cache_type_k == cache_type_k
            and self._current_cache_type_v == cache_type_v
            and self._current_n_slots == n_slots
            and self._current_kv_unified == kv_unified
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
            if mode == "native":
                cmd.append("--jinja")
            if extra_flags:
                cmd.extend(extra_flags)
            if ctx_override is not None:
                cmd.extend(["-c", str(ctx_override)])
            if cache_type_k is not None:
                cmd.extend(["--cache-type-k", cache_type_k])
            if cache_type_v is not None:
                cmd.extend(["--cache-type-v", cache_type_v])
            if n_slots is not None:
                cmd.extend(["--parallel", str(n_slots)])
            if kv_unified:
                cmd.append("--kv-unified")
        elif self._backend == "llamaserver":
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
            if cache_type_k is not None:
                cmd.extend(["--cache-type-k", cache_type_k])
            if cache_type_v is not None:
                cmd.extend(["--cache-type-v", cache_type_v])
            if n_slots is not None:
                cmd.extend(["--parallel", str(n_slots)])
            if kv_unified:
                cmd.append("--kv-unified")
        else:  # vllm
            cmd = [
                "vllm",
                "serve",
                str(model_path),
                "--port",
                str(self._port),
            ]
            if extra_flags:
                cmd.extend(extra_flags)
            if ctx_override is not None:
                cmd.extend(["--max-model-len", str(ctx_override)])

        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        await self._wait_healthy()

        self._current_model = model
        self._current_mode = mode
        self._current_ctx = ctx_override
        self._current_flags = flags
        self._current_cache_type_k = cache_type_k
        self._current_cache_type_v = cache_type_v
        self._current_n_slots = n_slots
        self._current_kv_unified = kv_unified

    async def stop(self) -> None:
        """Stop the current server / unload the Ollama model."""
        if self._backend == "ollama":
            if self._current_model is not None:
                wombat = await asyncio.create_subprocess_exec(
                    "ollama", "stop", self._current_model
                )
                await wombat.wait()
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
            self._current_cache_type_k = None
            self._current_cache_type_v = None
            self._current_n_slots = None
            self._current_kv_unified = False
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
                raise BackendError(resp.status_code, raw_body=resp.text)
            return resp.json()

    async def get_server_context(self) -> int:
        """Read the actual context length from the running server.

        llamaserver/llamafile: reads from ``/props``. Without
        ``--kv-unified`` this is the per-slot partition; with it, the
        full pool. Either is the correct compaction budget.

        vllm: reads ``max_model_len`` from ``/v1/models``.

        Raises:
            BudgetResolutionError: Server unreachable, returned an error,
                or response missing the expected field.
        """
        if self._backend == "vllm":
            url = f"http://localhost:{self._port}/v1/models"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        raise BackendError(resp.status_code, raw_body=resp.text)
                    data = resp.json()
            except (httpx.HTTPError, BackendError) as exc:
                raise BudgetResolutionError(cause=exc) from exc
            models = data.get("data") or []
            if not models:
                raise BudgetResolutionError()
            ctx = models[0].get("max_model_len")
            if ctx is None:
                raise BudgetResolutionError()
            return int(ctx)

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

        # llamaserver / llamafile — all non-manual modes read /props.
        # With kv_unified, /props already reports the full available context
        # (each slot can use the whole pool). Without it, /props reports the
        # per-slot partition — which is the correct budget for compaction.
        return await self.get_server_context()

    async def start_with_budget(
        self,
        model: str,
        *,
        gguf_path: str | Path | None = None,
        model_path: str | Path | None = None,
        mode: str = "native",
        budget_mode: BudgetMode = BudgetMode.BACKEND,
        manual_tokens: int | None = None,
        extra_flags: list[str] | None = None,
        cache_type_k: str | None = None,
        cache_type_v: str | None = None,
        n_slots: int | None = None,
        kv_unified: bool = False,
    ) -> int:
        """Start server with the specified budget mode and return the resolved budget.

        Handles the mode-specific startup dance:
        - BACKEND/FORGE_FULL: start without -c, read /props
        - MANUAL: start with -c = manual_tokens, read /props
        - FORGE_FAST: start without -c, read /props for max,
                      restart with -c = max // 2, read /props again

        For Ollama: ignores gguf_path, doesn't start a process.
        Returns VRAM tier budget.

        The returned budget accounts for slot configuration:
        - Non-unified (default): per-slot context (what ContextManager
          should use for compaction — the slot can only use this much).
        - Unified (``kv_unified=True``): total context across all slots
          (each slot can use up to the full amount).

        Args:
            model: Model name (Ollama-style canonical name).
            gguf_path: Path to GGUF file (llamaserver/llamafile only).
            mode: FC mode - ``"native"`` or ``"prompt"``.
            budget_mode: How to determine context budget.
            manual_tokens: Required for MANUAL mode.
            extra_flags: Additional server CLI flags.
            cache_type_k: KV cache quantization type for keys
                          (e.g. ``"q8_0"``, ``"q4_0"``).
            cache_type_v: KV cache quantization type for values
                          (e.g. ``"q8_0"``, ``"q4_0"``).
            n_slots: Number of concurrent slots.
            kv_unified: If True, use a single unified KV cache shared
                        across all slots.

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
            await self.start(
                model, gguf_path=gguf_path, model_path=model_path,
                mode=mode, extra_flags=extra_flags, ctx_override=None,
                cache_type_k=cache_type_k, cache_type_v=cache_type_v,
                n_slots=n_slots, kv_unified=kv_unified,
            )
            # /props reports per-slot context (non-unified) or full context
            # (unified). Either way, recover the total for -c math.
            reported_ctx = await self.get_server_context()
            if kv_unified or not n_slots or n_slots <= 1:
                total_ctx = reported_ctx
            else:
                total_ctx = reported_ctx * n_slots
            half_total = total_ctx // 2

            # Phase 2: restart with half total context
            await self.start(
                model, gguf_path=gguf_path, model_path=model_path,
                mode=mode, extra_flags=extra_flags, ctx_override=half_total,
                cache_type_k=cache_type_k, cache_type_v=cache_type_v,
                n_slots=n_slots, kv_unified=kv_unified,
            )
            return await self.resolve_budget(budget_mode)

        # BACKEND / FORGE_FULL / MANUAL
        ctx_override = manual_tokens if budget_mode == BudgetMode.MANUAL else None
        await self.start(
            model, gguf_path=gguf_path, model_path=model_path,
            mode=mode, extra_flags=extra_flags, ctx_override=ctx_override,
            cache_type_k=cache_type_k, cache_type_v=cache_type_v,
            n_slots=n_slots, kv_unified=kv_unified,
        )
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

    async def _wait_healthy(self, timeout: float | None = None) -> None:
        """Poll until the server is fully ready.

        llamaserver/llamafile: polls ``/props`` (which gates ``is_ready``
        AND confirms model loaded). 180s default.

        vllm: polls ``/v1/models`` (returns the loaded model entry only
        after the engine is fully initialized — strictly stronger than
        ``/health`` which can flip true mid-load). 300s default to
        accommodate vLLM's 2-3 min cold-start with tensor parallel.

        Raises:
            RuntimeError: If the server doesn't become ready within *timeout*.
        """
        if self._backend == "vllm":
            url = f"http://localhost:{self._port}/v1/models"
            effective_timeout = timeout if timeout is not None else 300.0
            readiness_check = lambda data: bool(data.get("data"))
        else:
            url = f"http://localhost:{self._port}/props"
            effective_timeout = timeout if timeout is not None else 180.0
            readiness_check = lambda data: "default_generation_settings" in data

        deadline = time.monotonic() + effective_timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        if readiness_check(data):
                            return
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(2)
        raise RuntimeError(
            f"Server did not become ready within {effective_timeout}s"
        )


async def setup_backend(
    backend: str,
    model: str | None = None,
    budget_mode: BudgetMode = BudgetMode.BACKEND,
    manual_tokens: int | None = None,
    client: Any | None = None,
    gguf_path: str | Path | None = None,
    model_path: str | Path | None = None,
    mode: str = "native",
    port: int = 8080,
    extra_flags: list[str] | None = None,
    on_compact: Callable[[CompactEvent], None] | None = None,
    compact_threshold: float = 0.75,
    phase_thresholds: tuple[float, float, float] | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    n_slots: int | None = None,
    kv_unified: bool = False,
    context_thresholds: list[float] | None = None,
    on_context_threshold: Callable[[int, int, float], str | None] | None = None,
) -> tuple[ServerManager, ContextManager]:
    """One-call setup: start backend, resolve budget, create ContextManager.

    Identity rules (mutually exclusive, enforced at call time):

    - ``backend="ollama"``: ``model`` required; ``gguf_path`` and
      ``model_path`` rejected. The Ollama runtime is keyed by the model
      string.
    - ``backend in ("llamaserver", "llamafile")``: ``gguf_path`` required;
      ``model`` and ``model_path`` rejected. The model file *is* the
      identity.
    - ``backend="vllm"``: ``model_path`` required; ``model`` and
      ``gguf_path`` rejected. ``model_path`` is a directory containing
      model weights/config (safetensors) or a HuggingFace repo id.

    For Ollama backends, pass the ``client`` so that ``set_num_ctx()`` is
    called automatically — keeping the client's per-request ``num_ctx``
    in sync with the resolved budget.  For llama-server / llamafile the
    context size is baked into the server process via ``-c``, so the
    client parameter is ignored. For vllm, context size is baked in via
    ``--max-model-len``; the client parameter is ignored.

    KV cache quantization (``cache_type_k`` / ``cache_type_v``) reduces
    VRAM usage per token. Llama-server / llamafile only — vLLM rejects
    these (quantization is baked into the model artifact).

    When ``kv_unified=True``, all slots share a single KV cache pool.
    Llama-server / llamafile only — vLLM has its own scheduler concepts
    and rejects ``n_slots`` / ``kv_unified``.

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
    if backend == "ollama":
        if gguf_path is not None:
            raise ValueError("backend='ollama' does not accept gguf_path (use model)")
        if model_path is not None:
            raise ValueError("backend='ollama' does not accept model_path (use model)")
        if not model:
            raise ValueError("backend='ollama' requires model")
        identity = model
    elif backend == "vllm":
        if gguf_path is not None:
            raise ValueError("backend='vllm' does not accept gguf_path (use model_path)")
        if model is not None:
            raise ValueError("backend='vllm' does not accept model (use model_path)")
        if not model_path:
            raise ValueError("backend='vllm' requires model_path")
        identity = str(model_path)
    elif backend in ("llamaserver", "llamafile"):
        if model is not None:
            raise ValueError(f"backend={backend!r} does not accept model (use gguf_path)")
        if model_path is not None:
            raise ValueError(f"backend={backend!r} does not accept model_path (use gguf_path)")
        if not gguf_path:
            raise ValueError(f"backend={backend!r} requires gguf_path")
        # ServerManager's cache-equality check keys off the identity string.
        # For non-Ollama backends the GGUF path *is* the identity, so feed
        # str(gguf_path) into ServerManager's `model` param. The wire format
        # 'model' field is set elsewhere (LlamafileClient derives it from
        # gguf_path stem); ServerManager only needs equality semantics.
        identity = str(gguf_path)
    else:
        raise ValueError(f"unsupported backend: {backend!r}")

    server = ServerManager(backend=backend, port=port)
    budget = await server.start_with_budget(
        model=identity,
        gguf_path=gguf_path,
        model_path=model_path,
        mode=mode,
        budget_mode=budget_mode,
        manual_tokens=manual_tokens,
        extra_flags=extra_flags,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        n_slots=n_slots,
        kv_unified=kv_unified,
    )

    # Ollama: wire num_ctx so every request uses the resolved budget
    if backend == "ollama" and client is not None and hasattr(client, "set_num_ctx"):
        client.set_num_ctx(budget)

    ctx_manager = ContextManager(
        strategy=TieredCompact(
            compact_threshold=compact_threshold,
            phase_thresholds=phase_thresholds,
        ),
        budget_tokens=budget,
        on_compact=on_compact,
        context_thresholds=context_thresholds,
        on_context_threshold=on_context_threshold,
    )
    return server, ctx_manager
