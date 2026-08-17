"""CLI entry point: python -m forge.proxy"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.metadata
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from forge import __version__
from forge._backend_profiles import ClientAdapter, find_managed_profile
from forge.clients.base import LLMClient
from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.proxy._config import _RawProxyConfig
from forge.proxy._options import add_proxy_options, supplied_proxy_options
from forge.proxy._profiles import (
    _load_profile,
    _managed_profile_path,
    _managed_profile_root,
    _managed_profiles,
    _parse_profile_document,
    _profile_bytes,
    _validate_profile_name,
    _write_managed_profile,
)
from forge.proxy.proxy import ProxyServer
from forge.proxy.server import HTTPServer
from forge.server import BudgetMode


_SOURCE_GUIDANCE = """Configuration sources (choose exactly one):
  forge-proxy --profile NAME
  forge-proxy --config PATH
  forge-proxy --backend-url URL [PROXY OPTIONS]

With no options, forge-proxy discovers the managed default.toml profile.
Create it with: forge-proxy init

Commands:
  forge-proxy init [OPTIONS]  Create a managed profile.
  forge-proxy check           Validate managed profiles and local health.
  forge-proxy install-artifact --version X.Y.Z --sha256 HEX [--no-init]
                              Install this standalone artifact.
  forge-proxy update [--version X.Y.Z]
                              Install a newer standalone release.
  forge-proxy uninstall       Remove the owned standalone installation.

Proxy guidance:
  https://github.com/antoinezambelli/forge/blob/main/docs/PROXY_INSTALLATION.md
  https://github.com/antoinezambelli/forge#proxy-server
  https://github.com/antoinezambelli/forge/blob/main/docs/USER_GUIDE.md
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="forge proxy — OpenAI- and Anthropic-compatible proxy with guardrails",
        epilog=_SOURCE_GUIDANCE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument(
        "--profile",
        metavar="NAME",
        help="Load a Forge-managed named profile",
    )
    selectors.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help="Load an externally owned TOML profile without rewriting it",
    )
    add_proxy_options(parser)
    return parser


def _raw_from_args(args: argparse.Namespace) -> _RawProxyConfig:
    return _RawProxyConfig(
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


def _proxy_from_raw(
    parser: argparse.ArgumentParser,
    raw: _RawProxyConfig,
) -> ProxyServer:
    try:
        return ProxyServer(
            backend_url=raw.backend_url,
            backend=raw.backend,
            model=raw.model,
            gguf=raw.gguf,
            model_path=raw.model_path,
            backend_port=raw.backend_port,
            budget_mode=raw.budget_mode,
            budget_tokens=raw.budget_tokens,
            extra_flags=raw.extra_flags,
            host=raw.host,
            port=raw.port,
            serialize=raw.serialize,
            max_retries=raw.max_retries,
            max_tool_errors=raw.max_tool_errors,
            rescue_enabled=raw.rescue_enabled,
            backend_capability=raw.backend_capability,
            inject_respond_tool=raw.inject_respond_tool,
            backend_timeout=raw.backend_timeout,
            reasoning_replay=raw.reasoning_replay,
            backend_api_key=raw.backend_api_key,
        )
    except ValueError as exc:
        parser.error(str(exc))


def _proxy_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> ProxyServer:
    return _proxy_from_raw(parser, _raw_from_args(args))


def _selected_launch(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str],
) -> tuple[_RawProxyConfig, bool, bool]:
    supplied = supplied_proxy_options(argv)
    if (args.profile is not None or args.config is not None) and supplied:
        parser.error(
            "profile/config selectors cannot be combined with Proxy configuration "
            "flags. Use either 'forge-proxy --profile NAME' or "
            "'forge-proxy --backend-url URL [PROXY OPTIONS]'."
        )

    if args.profile is not None:
        try:
            path = _managed_profile_path(args.profile)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.config is not None:
        path = args.config
    elif supplied:
        return _raw_from_args(args), bool(args.verbose), True
    else:
        path = _managed_profile_path("default")
        if not path.is_file():
            parser.error(
                f"no default profile found at {path}. Run 'forge-proxy init' or "
                "launch with flags, for example 'forge-proxy --backend-url URL'."
            )

    try:
        launch = _load_profile(path)
    except (OSError, ValueError) as exc:
        parser.error(f"cannot load profile {path}: {exc}")
    return launch.raw, launch.verbose, False


def _build_init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge-proxy init",
        description="Create or replace one Forge-managed Proxy profile.",
    )
    parser.add_argument("--profile", default="default", metavar="NAME")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--force", action="store_true")
    add_proxy_options(parser, suppress_defaults=True, include_credentials=False)
    return parser


def _profile_launch_command(name: str) -> str:
    arguments = ["forge-proxy", "--profile", name]
    return (
        subprocess.list2cmdline(arguments) if os.name == "nt" else shlex.join(arguments)
    )


def _prompt_required(prompt: str) -> str:
    value = input(prompt)
    if not value:
        raise ValueError("a value is required")
    return value


def _interactive_values(values: dict[str, object]) -> dict[str, object]:
    values = dict(values)
    backend = cast(str | None, values.get("backend"))
    unmanaged = (
        "backend_url" in values
        or backend is not None
        and find_managed_profile(backend) is None
    )
    if not unmanaged and "backend" not in values:
        mode = _prompt_required("Backend ownership [managed/unmanaged]: ").lower()
        if mode not in {"managed", "unmanaged"}:
            raise ValueError("backend ownership must be 'managed' or 'unmanaged'")
        unmanaged = mode == "unmanaged"

    if unmanaged:
        if "backend_url" not in values:
            values["backend_url"] = _prompt_required("Backend URL: ")
        if "backend" not in values:
            selector = input("Backend selector [openai]: ")
            if selector:
                values["backend"] = selector
    else:
        if "backend" not in values:
            values["backend"] = _prompt_required(
                "Managed backend [llamaserver/llamafile/ollama/vllm]: "
            )
        backend = cast(str, values["backend"])
        profile = find_managed_profile(backend)
        if profile is not None:
            identity = {
                "model-tag": "model",
                "gguf-path": "gguf",
                "model-path": "model_path",
            }[profile.required_identity.value]
            if identity not in values:
                values[identity] = _prompt_required(
                    f"{identity.replace('_', ' ').title()}: "
                )

    if "host" not in values:
        host = input("Proxy host [127.0.0.1]: ")
        if host:
            values["host"] = host
    if "port" not in values:
        port = input("Proxy port [8081]: ")
        if port:
            values["port"] = int(port)
    return values


def _run_init(argv: list[str]) -> None:
    parser = _build_init_parser()
    args = parser.parse_args(argv)
    try:
        _validate_profile_name(args.profile)
        controls = {"profile", "non_interactive", "force"}
        explicit = {
            name: value for name, value in vars(args).items() if name not in controls
        }
        if not args.non_interactive:
            explicit = _interactive_values(explicit)
        document = {"schema_version": 1, **explicit}
        _parse_profile_document(document)
        content = _profile_bytes(explicit)
        print(content.decode("utf-8"), end="")
        path = _managed_profile_path(args.profile)
        changed = _write_managed_profile(path, content, force=args.force)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{'Wrote' if changed else 'Unchanged'} profile: {path}")
    print(f"Launch with: {_profile_launch_command(args.profile)}")


def _build_check_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="forge-proxy check",
        description="Validate all managed profiles and one local Forge health listener.",
    )


def _runtime_check() -> None:
    for module in (
        "forge.clients.anthropic",
        "pydantic",
        "httpx",
        "anthropic",
        "tomli_w",
    ):
        importlib.import_module(module)
    installed = importlib.metadata.version("forge-guardrails")
    if installed != __version__:
        raise ValueError(
            f"runtime version {__version__} does not match package metadata {installed}"
        )


async def _local_health_check() -> None:
    server = HTTPServer(
        client=cast(LLMClient, object()),
        context_manager=ContextManager(strategy=NoCompact(), budget_tokens=None),
        client_adapter=ClientAdapter.LLAMAFILE,
        host="127.0.0.1",
        port=0,
        serialize_requests=False,
    )
    await server.start()
    try:
        assert server._server is not None
        port = server._server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            b"GET /forge/health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
        if b" 200 " not in response.partition(b"\r\n")[0] or not response.endswith(
            b'{"status":"ok"}'
        ):
            raise ValueError("unexpected /forge/health response")
    finally:
        await server.stop()


def _run_check(argv: list[str]) -> None:
    parser = _build_check_parser()
    parser.parse_args(argv)
    passed = True
    try:
        _runtime_check()
        print(f"OK runtime {__version__}")
    except Exception as exc:
        passed = False
        print(f"ERROR runtime: {exc}")

    profiles = _managed_profiles()
    if not profiles:
        passed = False
        print(
            f"ERROR profiles: no managed profiles in {_managed_profile_root()}; "
            "run 'forge-proxy init'"
        )
    for path in profiles:
        try:
            _validate_profile_name(path.stem)
            _load_profile(path)
            print(f"OK profile {path.stem}")
        except (OSError, ValueError) as exc:
            passed = False
            print(f"ERROR profile {path.stem}: {exc}")

    try:
        asyncio.run(_local_health_check())
        print("OK local /forge/health")
    except Exception as exc:
        passed = False
        print(f"ERROR local /forge/health: {exc}")
    if not passed:
        raise SystemExit(1)


def _run_installer_self_check(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="forge-proxy _installer-self-check")
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args(argv)
    if args.expected_version != __version__:
        parser.error(
            f"artifact version {__version__} does not match requested version "
            f"{args.expected_version}"
        )
    try:
        _runtime_check()
        asyncio.run(_local_health_check())
    except Exception as exc:
        parser.error(str(exc))


def _run_installer_profile_check(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="forge-proxy _installer-profile-check")
    parser.parse_args(argv)
    compatible = True
    for path in _managed_profiles():
        try:
            _load_profile(path)
            print(f"Compatible managed profile: {path.stem}")
        except (OSError, ValueError) as exc:
            compatible = False
            print(f"Incompatible managed profile {path.stem}: {exc}")
    if not compatible:
        raise SystemExit(1)


def _run_install_artifact(argv: list[str]) -> None:
    from forge.proxy import _installer

    parser = argparse.ArgumentParser(
        prog="forge-proxy install-artifact",
        description="Install the currently executing standalone artifact.",
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--no-init", action="store_true")
    parser.add_argument("--install-root", type=Path)
    args = parser.parse_args(argv)
    try:
        _installer.install_artifact(
            _installer.current_artifact(),
            args.version,
            args.sha256,
            install_root=args.install_root,
            no_init=args.no_init,
        )
    except (OSError, _installer.InstallerError) as exc:
        parser.error(str(exc))


def _run_update(argv: list[str]) -> None:
    from forge.proxy import _installer

    parser = argparse.ArgumentParser(
        prog="forge-proxy update",
        description="Install a newer stable or exact standalone Proxy release.",
    )
    parser.add_argument("--version")
    args = parser.parse_args(argv)
    release_urls: dict[str, str] = {}
    if os.environ.get("_FORGE_PROXY_INSTALLER_TESTING") == "1":
        pointer_url = os.environ.get("_FORGE_PROXY_INSTALLER_POINTER_URL")
        release_base = os.environ.get("_FORGE_PROXY_INSTALLER_RELEASE_BASE_URL")
        if pointer_url:
            release_urls["pointer_url"] = pointer_url
        if release_base:
            release_base = release_base.rstrip("/")
            release_urls["manifest_url"] = (
                f"{release_base}/v{{version}}/proxy-{{version}}.json"
            )
            release_urls["asset_url"] = f"{release_base}/v{{version}}/{{name}}"
    try:
        _installer.update(args.version, **release_urls)
    except (OSError, _installer.InstallerError) as exc:
        parser.error(str(exc))


def _run_uninstall(argv: list[str]) -> None:
    from forge.proxy import _installer

    parser = argparse.ArgumentParser(
        prog="forge-proxy uninstall",
        description="Delegate removal to the installed native uninstaller.",
    )
    parser.parse_args(argv)
    try:
        _installer.delegate_uninstall()
    except (OSError, _installer.InstallerError) as exc:
        parser.error(str(exc))


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_installer-self-check":
        _run_installer_self_check(arguments[1:])
        return
    if arguments and arguments[0] == "_installer-profile-check":
        _run_installer_profile_check(arguments[1:])
        return
    if arguments and arguments[0] == "install-artifact":
        _run_install_artifact(arguments[1:])
        return
    if arguments and arguments[0] == "update":
        _run_update(arguments[1:])
        return
    if arguments and arguments[0] == "uninstall":
        _run_uninstall(arguments[1:])
        return
    if arguments and arguments[0] == "init":
        _run_init(arguments[1:])
        return
    if arguments and arguments[0] == "check":
        _run_check(arguments[1:])
        return

    parser = _build_parser()
    args = parser.parse_args(arguments)
    raw, verbose, is_flag_only = _selected_launch(parser, args, arguments)
    proxy = (
        _proxy_from_args(parser, args) if is_flag_only else _proxy_from_raw(parser, raw)
    )

    level = logging.DEBUG if verbose else logging.INFO
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
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _shutdown)

    proxy.start()
    print(f"forge proxy running at {proxy.url}")
    print(f"  Point your client at {proxy.url}/v1/chat/completions")
    print("  Ctrl+C to stop")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        _shutdown(0, None)


if __name__ == "__main__":
    main()
