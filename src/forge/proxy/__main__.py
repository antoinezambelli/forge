"""CLI entry point: python -m forge.proxy"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from forge.proxy.proxy import ProxyServer
from forge.server import BudgetMode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="forge proxy — OpenAI-compatible proxy with guardrails",
    )

    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--backend-url",
        help="URL of externally managed backend (external mode)",
    )
    group.add_argument(
        "--backend",
        choices=["llamaserver", "llamafile", "ollama"],
        help="Backend type (managed mode)",
    )

    # Managed mode options
    parser.add_argument("--model", help="Model name (required for ollama)")
    parser.add_argument("--gguf", help="Path to GGUF file (llamaserver/llamafile)")
    parser.add_argument("--backend-port", type=int, default=8080, help="Backend port (default: 8080)")
    parser.add_argument(
        "--budget-mode",
        choices=["backend", "manual", "forge-full", "forge-fast"],
        default="backend",
        help="Context budget mode (default: backend)",
    )
    parser.add_argument("--budget-tokens", type=int, help="Manual token budget")
    parser.add_argument("--extra-flags", nargs="*", help="Additional backend CLI flags")

    # Proxy options
    parser.add_argument("--host", default="127.0.0.1", help="Proxy listen host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8081, help="Proxy listen port (default: 8081)")
    parser.add_argument("--serialize", action="store_true", default=None, help="Force request serialization")
    parser.add_argument("--no-serialize", action="store_true", help="Disable request serialization")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per request (default: 3)")
    parser.add_argument("--no-rescue", action="store_true", help="Disable rescue parsing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve serialize flag
    serialize = None
    if args.serialize:
        serialize = True
    elif args.no_serialize:
        serialize = False

    proxy = ProxyServer(
        backend_url=args.backend_url,
        backend=args.backend,
        model=args.model,
        gguf=args.gguf,
        backend_port=args.backend_port,
        budget_mode=BudgetMode(args.budget_mode),
        budget_tokens=args.budget_tokens,
        extra_flags=args.extra_flags,
        host=args.host,
        port=args.port,
        serialize=serialize,
        max_retries=args.max_retries,
        rescue_enabled=not args.no_rescue,
    )

    def _shutdown(sig: int, _frame: object) -> None:
        print("\nShutting down...")
        proxy.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    proxy.start()
    print(f"forge proxy running at {proxy.url}")
    print(f"  Point your client at {proxy.url}/v1/chat/completions")
    print("  Ctrl+C to stop")

    # Block main thread. Use a timed loop so Python can deliver
    # signals between iterations (Event.wait() without timeout
    # blocks signal handling on Windows).
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        _shutdown(0, None)


if __name__ == "__main__":
    main()
