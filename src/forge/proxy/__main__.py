"""CLI entry point: python -m forge.proxy"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from collections.abc import Sequence

from forge._backend_profiles import proxy_backend_selectors
from forge.core.reasoning import DEFAULT_REASONING_REPLAY, REASONING_REPLAY_CHOICES
from forge.proxy.proxy import ProxyServer
from forge.server import BudgetMode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="forge proxy — OpenAI- and Anthropic-compatible proxy with guardrails",
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
        choices=proxy_backend_selectors(),
        help="Managed backend or unmanaged wire/profile selector.",
    )

    # Managed mode options
    parser.add_argument(
        "--model",
        help="Model name (required for managed ollama). External generic "
             "OpenAI/llama profiles use it as a fallback when the request "
             "omits model; external vLLM and Anthropic profiles use it as a "
             "wire-model pin. It does not provide or suppress context-window "
             "reporting metadata.",
    )
    parser.add_argument("--gguf", help="Path to GGUF file (llamaserver/llamafile)")
    parser.add_argument("--model-path", help="Model directory or HF repo id (vllm, managed mode)")
    parser.add_argument("--backend-port", type=int, help="Backend target port")
    parser.add_argument(
        "--budget-mode",
        choices=["backend", "manual", "forge-full", "forge-fast"],
        help="Managed context budget mode (default: backend)",
    )
    parser.add_argument(
        "--budget-tokens",
        type=int,
        help="Positive managed manual allocation with --budget-mode manual; "
             "in unmanaged mode, reporting denominator only (never compacts "
             "or enforces caller history)",
    )
    parser.add_argument(
        "--extra-flags",
        nargs=argparse.REMAINDER,
        help="Terminal argv tail for a Forge-spawned llama-server, llamafile, "
             "or vLLM backend; rejected for Ollama and unmanaged mode; all "
             "Forge options must precede it.",
    )

    # Proxy options
    parser.add_argument("--host", default="127.0.0.1", help="Proxy listen host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8081, help="Proxy listen port (default: 8081)")
    serialization = parser.add_mutually_exclusive_group()
    serialization.add_argument(
        "--serialize", dest="serialize", action="store_true",
        help="Force request serialization",
    )
    serialization.add_argument(
        "--no-serialize", dest="serialize", action="store_false",
        help="Disable request serialization",
    )
    parser.set_defaults(serialize=None)
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per request (default: 3)")
    parser.add_argument("--max-tool-errors", type=int, default=2, help="Max consecutive tool-call errors per request (default: 2)")
    parser.add_argument(
        "--backend-timeout",
        type=float,
        default=300.0,
        help="Backend response timeout in seconds (default: 300)",
    )
    parser.add_argument("--no-rescue", action="store_true", help="Disable rescue parsing")
    parser.add_argument(
        "--backend-api-key",
        default=os.environ.get("FORGE_BACKEND_API_KEY"),
        help="Static credential forge sends to the backend in its native auth "
             "header (LM Studio, hosted providers, service accounts). forge "
             "relocates it to the backend's protocol slot. When set, an inbound "
             "auth header is refused as a second credential (at most one "
             "credential per "
             "request). This is backend authentication, not caller authorization; "
             "Proxy does not authenticate callers. Defaults to the "
             "FORGE_BACKEND_API_KEY env var.",
    )
    parser.add_argument(
        "--backend-capability",
        choices=["native", "prompt"],
        default="native",
        help="Tool-calling protocol for the backend (default: native). "
             "'native' uses the selected adapter's structured-tool path; "
             "compatible OpenAI-shaped clean paths preserve raw tool fields, "
             "while other adapters convert or rebuild them. 'prompt' opts into "
             "prompt-injection for non-FC llama.cpp/llamafile backends "
             "(strips tools into the prompt, parses the JSON call back). "
             "Frozen at startup — never probed or switched mid-stream.",
    )
    parser.add_argument(
        "--inject-respond-tool",
        action="store_true",
        help="Inject forge's synthetic respond() tool when the client sends "
             "tools (keeps small models in tool-calling mode). Default off.",
    )
    parser.add_argument(
        "--reasoning-replay",
        choices=REASONING_REPLAY_CHOICES,
        default=DEFAULT_REASONING_REPLAY,
        help="How much captured reasoning to replay to the backend "
             "(default: none).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    return parser


def _proxy_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> ProxyServer:
    try:
        return ProxyServer(
            backend_url=args.backend_url,
            backend=args.backend,
            model=args.model,
            gguf=args.gguf,
            model_path=args.model_path,
            backend_port=args.backend_port,
            budget_mode=(BudgetMode(args.budget_mode) if args.budget_mode else None),
            budget_tokens=args.budget_tokens,
            extra_flags=args.extra_flags,
            host=args.host,
            port=args.port,
            serialize=args.serialize,
            max_retries=args.max_retries,
            max_tool_errors=args.max_tool_errors,
            rescue_enabled=not args.no_rescue,
            backend_capability=args.backend_capability,
            inject_respond_tool=args.inject_respond_tool,
            backend_timeout=args.backend_timeout,
            reasoning_replay=args.reasoning_replay,
            backend_api_key=args.backend_api_key,
        )
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    proxy = _proxy_from_args(parser, args)

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
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
