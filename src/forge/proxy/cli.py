"""CLI entry point for `python -m forge.proxy` (forge serve)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


DEFAULT_COMPACT_THRESHOLD = 0.75
DEFAULT_KEEP_RECENT = 2


async def _detect_budget(backend_url: str, timeout: float) -> int:
    """Auto-detect context budget from the backend's /props endpoint.

    Queries the llama-server /props endpoint for n_ctx. Raises
    BudgetResolutionError if the backend doesn't respond or the
    response can't be parsed.
    """
    import httpx
    from forge.errors import BudgetResolutionError

    base = backend_url.rstrip("/")
    # Strip /v1 suffix if present (llama-server serves /props at root)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="forge-proxy",
        description="OpenAI-compatible proxy server with forge guardrails.",
    )
    parser.add_argument(
        "--backend-url",
        required=True,
        help="Base URL of the model server (e.g. http://localhost:8080)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port to listen on (default: 8081)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Context budget in tokens (default: auto-detect from backend)",
    )
    parser.add_argument(
        "--keep-recent",
        type=int,
        default=DEFAULT_KEEP_RECENT,
        help=f"Recent iterations to preserve during compaction (default: {DEFAULT_KEEP_RECENT})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max guardrail retry attempts per request (default: 3)",
    )
    parser.add_argument(
        "--no-rescue",
        action="store_true",
        default=False,
        help="Disable rescue parsing (don't extract tool calls from text)",
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        default=False,
        help="Disable context compaction",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Backend request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("forge.proxy")

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is required for the proxy server.\n"
            "Install with: pip install -e '.[proxy]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from forge.context.manager import ContextManager
    from forge.context.strategies import TieredCompact
    from forge.proxy.server import create_app

    # Resolve context budget
    context_manager = None

    if not args.no_compact:
        if args.budget is not None:
            budget = args.budget
            log.info("Using manual context budget: %d tokens", budget)
        else:
            budget = asyncio.run(_detect_budget(args.backend_url, args.timeout))
            log.info("Auto-detected context budget: %d tokens", budget)

        context_manager = ContextManager(
            strategy=TieredCompact(keep_recent=args.keep_recent),
            budget_tokens=budget,
        )

    app = create_app(
        backend_url=args.backend_url,
        context_manager=context_manager,
        max_retries=args.max_retries,
        rescue_enabled=not args.no_rescue,
        timeout=args.timeout,
    )

    log.info(
        "Starting forge proxy on %s:%d -> %s",
        args.host,
        args.port,
        args.backend_url,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
