"""CLI entry point for `python -m forge.proxy`.

Thin wrapper around ProxyServer — the programmatic API is primary.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="forge-proxy",
        description="OpenAI-compatible proxy server with forge guardrails.",
    )

    # Backend: either managed or external (mutually exclusive)
    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument(
        "--backend-url",
        help="Connect to an existing backend (e.g. http://localhost:8080)",
    )
    backend_group.add_argument(
        "--backend",
        choices=["llamaserver", "llamafile"],
        help="Managed mode: forge starts and manages the backend process",
    )

    parser.add_argument(
        "--gguf",
        help="Path to GGUF model file (required for managed mode)",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8080,
        help="Port for the managed backend (default: 8080)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Proxy listen port (default: 8081)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Proxy bind address (default: 127.0.0.1)",
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
        default=2,
        help="Recent iterations to preserve during compaction (default: 2)",
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
        help="Disable rescue parsing",
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

    if args.backend and not args.gguf:
        parser.error("--gguf is required when using --backend (managed mode)")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from forge.proxy import ProxyServer

    proxy = ProxyServer(
        backend_url=args.backend_url,
        backend=args.backend,
        gguf=args.gguf,
        port=args.port,
        host=args.host,
        backend_port=args.backend_port,
        budget=args.budget,
        keep_recent=args.keep_recent,
        max_retries=args.max_retries,
        rescue_enabled=not args.no_rescue,
        compact_enabled=not args.no_compact,
        timeout=args.timeout,
    )

    proxy.start()
    print(f"Forge proxy running on {proxy.url}")
    print("Press Ctrl+C to stop.")

    try:
        signal.pause()
    except AttributeError:
        # Windows doesn't have signal.pause — block on input
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass

    proxy.stop()
    print("Proxy stopped.")


if __name__ == "__main__":
    main()
