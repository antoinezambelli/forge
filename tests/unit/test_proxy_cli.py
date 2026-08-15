"""CLI parsing and shared Proxy validation coverage."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import ANY, MagicMock, patch

import pytest

from forge.proxy import __main__ as proxy_cli
from forge.proxy.__main__ import _build_parser, _proxy_from_args


def test_backend_choices_come_from_complete_selector_set() -> None:
    parser = _build_parser()
    backend_action = next(
        action for action in parser._actions if action.dest == "backend"
    )
    assert backend_action.choices == (
        "llamaserver", "llamafile", "ollama", "vllm", "openai", "anthropic",
    )


def test_removed_backend_protocol_is_unrecognized(capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--backend-url", "http://host", "--backend-protocol", "openai"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --backend-protocol" in capsys.readouterr().err


def test_serialization_switches_are_mutually_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([
            "--backend-url", "http://host", "--serialize", "--no-serialize",
        ])
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_extra_flags_are_a_terminal_remainder() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "--backend", "vllm", "--model-path", "/m", "--port", "9000",
        "--extra-flags", "--verbose", "backend-value", "--no-rescue",
    ])
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
    args = parser.parse_args([
        "--backend", "ollama", "--model", "tag", "--extra-flags",
    ])
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
    register_signal.assert_any_call(proxy_cli.signal.SIGINT, ANY)
    proxy.start.assert_called_once_with()
    proxy.stop.assert_called_once_with()
    output = capsys.readouterr().out
    assert "forge proxy running at http://127.0.0.1:8081" in output
    assert "Shutting down..." in output
