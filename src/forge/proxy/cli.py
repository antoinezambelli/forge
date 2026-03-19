"""CLI entry point for `python -m forge.proxy` (forge serve)."""

from __future__ import annotations

import argparse
import logging
import sys


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
        "--max-retries",
        type=int,
        default=3,
        help="Max guardrail retry attempts per request (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Backend request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--no-rescue",
        action="store_true",
        default=False,
        help="Disable rescue parsing (don't extract tool calls from text)",
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

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is required for the proxy server.\n"
            "Install with: pip install -e '.[proxy]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from forge.proxy.server import create_app

    app = create_app(
        backend_url=args.backend_url,
        max_retries=args.max_retries,
        rescue_enabled=not args.no_rescue,
        timeout=args.timeout,
    )

    logger = logging.getLogger("forge.proxy")
    logger.info(
        "Starting forge proxy on %s:%d -> %s",
        args.host,
        args.port,
        args.backend_url,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
