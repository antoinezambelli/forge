"""Focused Proxy profile, init, source-selection, and check coverage."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from forge.proxy import __main__ as proxy_cli
from forge.proxy import _profiles as profiles
from forge.proxy._config import _normalize_proxy_config
from forge.proxy._options import supplied_proxy_options


def _document(**values: object) -> dict[str, object]:
    return {"schema_version": 1, **values}


def _write(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(profiles._profile_bytes(values))


@pytest.mark.parametrize(
    ("system", "environment", "home", "expected"),
    [
        (
            "Windows",
            {"APPDATA": "C:/Users/a/AppData/Roaming"},
            Path("C:/Users/a"),
            Path("C:/Users/a/AppData/Roaming/Forge/profiles"),
        ),
        (
            "Linux",
            {"XDG_CONFIG_HOME": "/xdg"},
            Path("/home/a"),
            Path("/xdg/forge/profiles"),
        ),
        ("Linux", {}, Path("/home/a"), Path("/home/a/.config/forge/profiles")),
        (
            "Darwin",
            {},
            Path("/Users/a"),
            Path("/Users/a/Library/Application Support/Forge/profiles"),
        ),
    ],
)
def test_managed_profile_roots(
    system: str,
    environment: dict[str, str],
    home: Path,
    expected: Path,
) -> None:
    assert (
        profiles._managed_profile_root(system=system, environ=environment, home=home)
        == expected
    )


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b"])
def test_only_ruled_profile_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="profile name"):
        profiles._validate_profile_name(name)


@pytest.mark.parametrize("name", ["default", "team profile", "...", "x.toml"])
def test_other_profile_names_are_accepted(name: str) -> None:
    profiles._validate_profile_name(name)


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": 2},
    ],
)
def test_schema_version_is_exact_nonboolean_integer(
    document: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        profiles._parse_profile_document(document)


@pytest.mark.parametrize(
    "field",
    ["unknown", "profile", "config", "backend_api_key", "api_key"],
)
def test_unknown_selectors_and_credentials_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match="unknown profile fields"):
        profiles._parse_profile_document(
            _document(backend_url="http://host", **{field: "value"})
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("port", True),
        ("backend_port", 1.5),
        ("budget_tokens", False),
        ("max_retries", "3"),
        ("serialize", 0),
        ("backend_timeout", True),
        ("extra_flags", "--flag"),
        ("extra_flags", ["ok", 2]),
        ("verbose", "yes"),
    ],
)
def test_profile_types_are_exact(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="wrong TOML type"):
        profiles._parse_profile_document(
            _document(backend_url="http://host", **{field: value})
        )


@pytest.mark.parametrize("timeout", [4, 4.5])
def test_timeout_accepts_integer_or_float_and_normalizes_to_float(
    timeout: object,
) -> None:
    launch = profiles._parse_profile_document(
        _document(backend_url="http://host", backend_timeout=timeout)
    )
    assert launch.raw.backend_timeout == float(timeout)  # type: ignore[arg-type]


def test_profile_and_cli_share_normalization_with_inversions_and_tail() -> None:
    parser = proxy_cli._build_parser()
    args = parser.parse_args(
        [
            "--backend",
            "vllm",
            "--model-path",
            "literal/$MODEL",
            "--host",
            "0.0.0.0",
            "--port",
            "9010",
            "--no-serialize",
            "--no-rescue",
            "--verbose",
            "--extra-flags",
            "--dtype",
            "float16",
        ]
    )
    cli = _normalize_proxy_config(proxy_cli._raw_from_args(args))
    profile = profiles._parse_profile_document(
        _document(
            backend="vllm",
            model_path="literal/$MODEL",
            host="0.0.0.0",
            port=9010,
            serialize=False,
            no_rescue=True,
            verbose=True,
            extra_flags=["--dtype", "float16"],
        )
    )
    assert profile.normalized == cli
    assert profile.verbose is True
    assert profile.raw.model_path == "literal/$MODEL"


def test_unmanaged_profile_inherits_environment_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "external.toml"
    _write(path, backend_url="http://host")
    monkeypatch.setenv("FORGE_BACKEND_API_KEY", "environment-secret")
    launch = profiles._load_profile(path)
    assert launch.raw.backend_api_key == "environment-secret"
    assert b"secret" not in path.read_bytes()


@pytest.mark.parametrize(
    ("selector", "value"),
    [("--profile", "named"), ("--config", "external.toml")],
)
def test_profile_and_config_selectors_reject_mixed_configuration(
    selector: str,
    value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = proxy_cli._build_parser()
    argv = [selector, value, "--port", "8081"]
    args = parser.parse_args(argv)
    with pytest.raises(SystemExit) as exc:
        proxy_cli._selected_launch(parser, args, argv)
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "forge-proxy --profile NAME" in error
    assert "forge-proxy --backend-url URL" in error


def test_tail_selector_tokens_remain_backend_argv() -> None:
    argv = [
        "--backend",
        "vllm",
        "--model-path",
        "/m",
        "--extra-flags",
        "--profile",
        "tail",
        "--config=tail.toml",
        "--port",
        "8081",
    ]
    parser = proxy_cli._build_parser()
    args = parser.parse_args(argv)
    assert args.profile is None
    assert args.config is None
    assert supplied_proxy_options(argv) == {"backend", "model_path", "extra_flags"}
    assert args.extra_flags == [
        "--profile",
        "tail",
        "--config=tail.toml",
        "--port",
        "8081",
    ]


def test_default_discovery_and_external_config_use_profile_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "profiles"
    _write(root / "default.toml", backend_url="http://default", verbose=True)
    external = tmp_path / "external.toml"
    _write(external, backend_url="http://external")
    monkeypatch.setattr(profiles, "_managed_profile_root", lambda: root)

    parser = proxy_cli._build_parser()
    default = parser.parse_args([])
    raw, verbose, cli_only = proxy_cli._selected_launch(parser, default, [])
    assert raw.backend_url == "http://default"
    assert verbose is True
    assert cli_only is False

    argv = ["--config", str(external)]
    selected = parser.parse_args(argv)
    raw, _, cli_only = proxy_cli._selected_launch(parser, selected, argv)
    assert raw.backend_url == "http://external"
    assert cli_only is False


def test_missing_default_has_init_and_flag_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(profiles, "_managed_profile_root", lambda: tmp_path)
    parser = proxy_cli._build_parser()
    args = parser.parse_args([])
    with pytest.raises(SystemExit) as exc:
        proxy_cli._selected_launch(parser, args, [])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "forge-proxy init" in error
    assert "--backend-url URL" in error


def test_noninteractive_init_is_sparse_atomic_and_identical_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "profiles"
    monkeypatch.setattr(profiles, "_managed_profile_root", lambda: root)
    proxy_cli.main(
        [
            "init",
            "--non-interactive",
            "--backend-url",
            "http://host",
            "--host",
            "127.0.0.1",
            "--no-serialize",
        ]
    )
    path = root / "default.toml"
    content = path.read_bytes()
    assert b"schema_version = 1" in content
    assert b'backend_url = "http://host"' in content
    assert b'host = "127.0.0.1"' in content
    assert b"serialize = false" in content
    assert b"port" not in content
    timestamp = path.stat().st_mtime_ns
    with patch.object(profiles.os, "replace") as replace:
        proxy_cli.main(
            [
                "init",
                "--non-interactive",
                "--backend-url",
                "http://host",
                "--host",
                "127.0.0.1",
                "--no-serialize",
            ]
        )
    replace.assert_not_called()
    assert path.read_bytes() == content
    assert path.stat().st_mtime_ns == timestamp
    output = capsys.readouterr().out
    assert "Unchanged profile" in output
    assert output.count("Launch with: forge-proxy --profile default") == 2


def test_init_requires_force_for_different_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "profiles"
    monkeypatch.setattr(profiles, "_managed_profile_root", lambda: root)
    proxy_cli.main(["init", "--non-interactive", "--backend-url", "http://one"])
    with pytest.raises(SystemExit) as exc:
        proxy_cli.main(["init", "--non-interactive", "--backend-url", "http://two"])
    assert exc.value.code == 2
    assert b"http://one" in (root / "default.toml").read_bytes()
    proxy_cli.main(
        ["init", "--non-interactive", "--force", "--backend-url", "http://two"]
    )
    assert b"http://two" in (root / "default.toml").read_bytes()


def test_init_prints_usable_launch_command_for_spaced_profile_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "profiles"
    monkeypatch.setattr(profiles, "_managed_profile_root", lambda: root)
    proxy_cli.main(
        [
            "init",
            "--profile",
            "team profile",
            "--non-interactive",
            "--backend-url",
            "http://host",
        ]
    )

    expected = proxy_cli._profile_launch_command("team profile")
    assert expected != "forge-proxy --profile team profile"
    assert f"Launch with: {expected}" in capsys.readouterr().out


def test_interactive_enter_omits_defaults_and_typed_defaults_persist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "profiles"
    monkeypatch.setattr(profiles, "_managed_profile_root", lambda: root)
    answers = iter(["managed", "ollama", "tag", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    proxy_cli.main(["init", "--profile", "omitted"])
    omitted = (root / "omitted.toml").read_text(encoding="utf-8")
    assert "host" not in omitted
    assert "port" not in omitted

    answers = iter(["managed", "ollama", "tag", "127.0.0.1", "8081"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    proxy_cli.main(["init", "--profile", "explicit"])
    explicit = (root / "explicit.toml").read_text(encoding="utf-8")
    assert 'host = "127.0.0.1"' in explicit
    assert "port = 8081" in explicit


@pytest.mark.parametrize("backend", ["anthropic", "openai"])
def test_interactive_unmanaged_selector_prompts_for_missing_backend_url(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "profiles"
    monkeypatch.setattr(profiles, "_managed_profile_root", lambda: root)
    answers = iter(["https://backend.example", "", ""])
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", answer)
    proxy_cli.main(["init", "--profile", backend, "--backend", backend])

    content = (root / f"{backend}.toml").read_text(encoding="utf-8")
    assert f'backend = "{backend}"' in content
    assert 'backend_url = "https://backend.example"' in content
    assert prompts[0] == "Backend URL: "
    assert not any("ownership" in prompt for prompt in prompts)


def test_interactive_values_preserve_supplied_strings_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter(
        [
            "unmanaged",
            " https://backend.example/path ",
            " openai ",
            " 127.0.0.1 ",
            "",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    values = proxy_cli._interactive_values({})

    assert values["backend_url"] == " https://backend.example/path "
    assert values["backend"] == " openai "
    assert values["host"] == " 127.0.0.1 "


def test_init_validates_before_creating_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "profiles"
    monkeypatch.setattr(profiles, "_managed_profile_root", lambda: root)
    with pytest.raises(SystemExit):
        proxy_cli.main(["init", "--non-interactive", "--backend", "ollama"])
    assert not root.exists()


def test_check_rejects_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        proxy_cli.main(["check", "anything"])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("profile_count", [0, 1, 3])
def test_check_runs_one_health_check_for_any_profile_count(
    profile_count: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "profiles"
    for index in range(profile_count):
        if index == 0:
            _write(root / f"{index}.toml", backend_url="http://host")
        else:
            root.mkdir(parents=True, exist_ok=True)
            (root / f"{index}.toml").write_text(
                "schema_version = 1\nunknown = true\n", encoding="utf-8"
            )
    monkeypatch.setattr(profiles, "_managed_profile_root", lambda: root)
    monkeypatch.setattr(proxy_cli, "_managed_profile_root", lambda: root)
    monkeypatch.setattr(proxy_cli, "_runtime_check", lambda: None)
    health_check = AsyncMock()
    monkeypatch.setattr(proxy_cli, "_local_health_check", health_check)
    with patch.object(
        proxy_cli, "ProxyServer", side_effect=AssertionError("backend startup")
    ):
        if profile_count == 1:
            proxy_cli.main(["check"])
        else:
            with pytest.raises(SystemExit) as exc:
                proxy_cli.main(["check"])
            assert exc.value.code == 1
    health_check.assert_awaited_once_with()
    output = capsys.readouterr().out
    assert "OK local /forge/health" in output
    if profile_count == 0:
        assert "forge-proxy init" in output


def test_managed_profile_enumeration_is_deterministic(tmp_path: Path) -> None:
    for name in ("z.toml", "a.toml", "middle.toml"):
        _write(tmp_path / name, backend_url="http://host")
    assert [path.name for path in profiles._managed_profiles(root=tmp_path)] == [
        "a.toml",
        "middle.toml",
        "z.toml",
    ]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission contract")
def test_new_posix_managed_paths_have_private_permissions(tmp_path: Path) -> None:
    path = tmp_path / "profiles" / "default.toml"
    profiles._write_managed_profile(
        path, profiles._profile_bytes({"backend_url": "http://host"}), force=False
    )
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
