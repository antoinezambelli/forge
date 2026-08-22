"""CLI parsing and shared Proxy validation coverage."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

from forge.proxy import __main__ as proxy_cli
from forge.proxy import _installer
from forge.proxy.__main__ import _build_parser, _proxy_from_args


def test_python_distribution_keeps_module_cli_without_global_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    with pytest.raises(SystemExit) as exc:
        proxy_cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == project["project"]["version"]
    distribution = importlib.metadata.distribution("forge-guardrails")
    assert not any(
        entry.group == "console_scripts" and entry.name == "forge-proxy"
        for entry in distribution.entry_points
    )


def test_help_lists_installed_lifecycle_without_out_of_scope_modes() -> None:
    help_text = _build_parser().format_help()
    assert "init" in help_text
    assert "check" in help_text
    assert "install-artifact" in help_text
    assert "update [--version X.Y.Z]" in help_text
    assert "uninstall" in help_text
    assert "--stable" not in help_text
    assert "rollback" not in help_text


def test_install_artifact_dispatches_current_artifact_and_custom_root(
    tmp_path: Path,
) -> None:
    current = tmp_path / "forge-proxy.exe"
    root = tmp_path / "custom root"
    with (
        patch.object(_installer, "current_artifact", return_value=current),
        patch.object(_installer, "install_artifact") as install_artifact,
    ):
        proxy_cli.main(
            [
                "install-artifact",
                "--version",
                "1.2.3",
                "--sha256",
                "a" * 64,
                "--no-init",
                "--install-root",
                str(root),
            ]
        )
    install_artifact.assert_called_once_with(
        current,
        "1.2.3",
        "a" * 64,
        install_root=root,
        no_init=True,
    )


def test_update_and_uninstall_dispatch_before_launch_parser() -> None:
    with patch.object(_installer, "update") as update:
        proxy_cli.main(["update", "--version", "1.2.3"])
    update.assert_called_once_with("1.2.3")
    with patch.object(_installer, "delegate_uninstall") as uninstall:
        proxy_cli.main(["uninstall"])
    uninstall.assert_called_once_with()


def test_release_gate_can_point_installed_update_at_local_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("_FORGE_PROXY_INSTALLER_TESTING", "1")
    monkeypatch.setenv(
        "_FORGE_PROXY_INSTALLER_POINTER_URL", "http://127.0.0.1:1234/pointer"
    )
    monkeypatch.setenv(
        "_FORGE_PROXY_INSTALLER_RELEASE_BASE_URL", "http://127.0.0.1:1234/"
    )
    with patch.object(_installer, "update") as update:
        proxy_cli.main(["update", "--version", "1.2.3"])
    update.assert_called_once_with(
        "1.2.3",
        pointer_url="http://127.0.0.1:1234/pointer",
        manifest_url="http://127.0.0.1:1234/v{version}/proxy-{version}.json",
        asset_url="http://127.0.0.1:1234/v{version}/{name}",
    )


def test_private_self_check_rejects_requested_version_mismatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        proxy_cli.main(
            [
                "_installer-self-check",
                "--expected-version",
                "999.0.0",
            ]
        )
    assert exc.value.code == 2
    assert "does not match requested version" in capsys.readouterr().err


def test_backend_choices_come_from_complete_selector_set() -> None:
    parser = _build_parser()
    backend_action = next(
        action for action in parser._actions if action.dest == "backend"
    )
    assert backend_action.choices == (
        "llamaserver",
        "llamafile",
        "ollama",
        "vllm",
        "openai",
        "anthropic",
    )


def test_removed_backend_protocol_is_unrecognized(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            ["--backend-url", "http://host", "--backend-protocol", "openai"]
        )
    assert exc.value.code == 2
    assert "unrecognized arguments: --backend-protocol" in capsys.readouterr().err


def test_serialization_switches_are_mutually_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "--backend-url",
                "http://host",
                "--serialize",
                "--no-serialize",
            ]
        )
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_extra_flags_are_a_terminal_remainder() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--backend",
            "vllm",
            "--model-path",
            "/m",
            "--port",
            "9000",
            "--extra-flags",
            "--verbose",
            "backend-value",
            "--no-rescue",
        ]
    )
    assert args.port == 9000
    assert args.no_rescue is False
    assert args.extra_flags == ["--verbose", "backend-value", "--no-rescue"]
    proxy = _proxy_from_args(parser, args)
    assert proxy._extra_flags == args.extra_flags


def test_help_describes_budget_and_auth_boundaries() -> None:
    help_text = _build_parser().format_help()
    normalized = " ".join(help_text.split())
    assert "generic OpenAI/llama profiles use it as a fallback" in normalized
    assert "vLLM and Anthropic profiles use it as a wire-model pin" in normalized
    assert "reporting denominator only" in normalized
    assert "never compacts or enforces caller history" in normalized
    assert "not caller authorization" in normalized
    assert "all Forge options must precede it" in normalized


def test_empty_cli_extra_flags_normalize_as_absent() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--backend",
            "ollama",
            "--model",
            "tag",
            "--extra-flags",
        ]
    )
    assert args.extra_flags == []
    assert _proxy_from_args(parser, args)._extra_flags is None


def test_constructor_value_error_becomes_argparse_error_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _build_parser()
    args = parser.parse_args(["--backend", "ollama"])
    with pytest.raises(SystemExit) as exc:
        _proxy_from_args(parser, args)
    stderr = capsys.readouterr().err
    assert exc.value.code == 2
    assert "backend='ollama' requires model" in stderr
    assert "Traceback" not in stderr


def test_invalid_cli_invocation_exits_two_without_traceback() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "forge.proxy", "--backend", "ollama"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "backend='ollama' requires model" in result.stderr
    assert "Traceback" not in result.stderr


def test_main_starts_and_stops_on_keyboard_interrupt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    proxy = MagicMock()
    proxy.url = "http://127.0.0.1:8081"

    with (
        patch.object(proxy_cli, "_proxy_from_args", return_value=proxy),
        patch.object(proxy_cli.logging, "basicConfig") as configure_logging,
        patch.object(proxy_cli.signal, "signal") as register_signal,
        patch.object(proxy_cli.time, "sleep", side_effect=KeyboardInterrupt),
        pytest.raises(SystemExit) as exc,
    ):
        proxy_cli.main(["--backend-url", "http://backend"])

    assert exc.value.code == 0
    configure_logging.assert_called_once()
    supported_signals = [proxy_cli.signal.SIGINT]
    if hasattr(proxy_cli.signal, "SIGTERM"):
        supported_signals.append(proxy_cli.signal.SIGTERM)
    if hasattr(proxy_cli.signal, "SIGBREAK"):
        supported_signals.append(proxy_cli.signal.SIGBREAK)
    for supported_signal in supported_signals:
        register_signal.assert_any_call(supported_signal, ANY)
    proxy.start.assert_called_once_with()
    proxy.stop.assert_called_once_with()
    output = capsys.readouterr().out
    assert "forge proxy running at http://127.0.0.1:8081" in output
    assert "Shutting down..." in output


@pytest.mark.parametrize(
    "shutdown_signal",
    [
        proxy_cli.signal.SIGINT,
        *([proxy_cli.signal.SIGTERM] if hasattr(proxy_cli.signal, "SIGTERM") else []),
        *([proxy_cli.signal.SIGBREAK] if hasattr(proxy_cli.signal, "SIGBREAK") else []),
    ],
)
def test_supported_signal_reaches_stop_and_exit(shutdown_signal: int) -> None:
    proxy = MagicMock()
    proxy.url = "http://127.0.0.1:8081"
    handlers: dict[int, object] = {}

    def register(sig: int, handler: object) -> None:
        handlers[sig] = handler

    def deliver_signal(_seconds: float) -> None:
        handler = handlers[shutdown_signal]
        assert callable(handler)
        handler(shutdown_signal, None)

    with (
        patch.object(proxy_cli, "_proxy_from_args", return_value=proxy),
        patch.object(proxy_cli.signal, "signal", side_effect=register),
        patch.object(proxy_cli.time, "sleep", side_effect=deliver_signal),
        pytest.raises(SystemExit) as exc,
    ):
        proxy_cli.main(["--backend-url", "http://backend"])

    assert exc.value.code == 0
    proxy.stop.assert_called_once_with()
