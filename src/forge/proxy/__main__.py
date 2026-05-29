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

    # Mode selection. External mode uses --backend-url; managed mode uses
    # --backend (+ an identity flag). For an external vLLM server, pass both
    # --backend-url and --backend vllm so the proxy selects the vLLM adapter.
    # ProxyServer enforces "exactly one of url/backend" and the per-backend rules.
    parser.add_argument(
        "--backend-url",
        help="URL of an externally managed backend (external mode)",
    )
    parser.add_argument(
        "--backend",
        choices=["llamaserver", "llamafile", "ollama", "vllm"],
        help="Backend type. Required for managed mode; in external mode use "
             "'vllm' to select the vLLM adapter (default adapter is llama.cpp).",
    )

    # Managed mode options
    parser.add_argument("--model", help="Model name (required for ollama)")
    parser.add_argument("--gguf", help="Path to GGUF file (llamaserver/llamafile)")
    parser.add_argument("--model-path", help="Model directory or HF repo id (vllm, managed mode)")
    parser.add_argument("--backend-port", type=int, default=8080, help="Backend port (default: 8080)")
    parser.add_argument(
        "--budget-mode",
        choices=["backend", "manual", "forge-full", "forge-fast"],
        default="backend",
        help="Context budget mode (default: backend)",
    )
    parser.add_argument("--budget-tokens", type=int, help="Manual token budget")
    parser.add_argument("--extra-flags", nargs="*", help="Additional backend CLI flags")
    parser.add_argument(
        "--mode",
        choices=["native", "prompt"],
        default="native",
        help="Function-calling mode (default: native). Use 'prompt' for "
             "OpenAI-compatible backends without a function-calling template.",
    )
    parser.add_argument(
        "--backend-protocol",
        choices=["openai", "anthropic"],
        default="openai",
        help="Wire format of the external backend (default: openai). Use "
             "'anthropic' for Anthropic-shape downstreams (LiteLLM /v1/messages, "
             "real Anthropic API, self-hosted Anthropic proxy). External mode only.",
    )

    # Proxy options
    parser.add_argument("--host", default="127.0.0.1", help="Proxy listen host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8081, help="Proxy listen port (default: 8081)")
    parser.add_argument("--serialize", action="store_true", default=None, help="Force request serialization")
    parser.add_argument("--no-serialize", action="store_true", help="Disable request serialization")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per request (default: 3)")
    parser.add_argument(
        "--backend-timeout",
        type=float,
        default=300.0,
        help="Backend response timeout in seconds (default: 300)",
    )
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
        model_path=args.model_path,
        backend_port=args.backend_port,
        budget_mode=BudgetMode(args.budget_mode),
        budget_tokens=args.budget_tokens,
        extra_flags=args.extra_flags,
        host=args.host,
        port=args.port,
        serialize=serialize,
        max_retries=args.max_retries,
        rescue_enabled=not args.no_rescue,
        mode=args.mode,
        backend_protocol=args.backend_protocol,
        backend_timeout=args.backend_timeout,
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
