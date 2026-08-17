"""Focused tests for standalone target, build, and evidence behavior."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.standalone import build, evidence, smoke


def passing_evidence(form: str = "onedir") -> dict[str, object]:
    return {
        "target": "windows-x86_64",
        "form": form,
        "path": "forge-proxy.exe",
        "size_bytes": 1,
        "build_identity": {},
        "runtime_identity": {"version": "0.9.0"},
        "cold_start_seconds": 0.1,
        "shutdown_seconds": 0.1,
        "extraction": {
            "kind": "directory" if form == "onedir" else "temporary-onefile",
            "observed_path": "bundle",
            "cleanup": None if form == "onedir" else True,
        },
        "smoke": {
            "version": True,
            "help": True,
            "health": True,
            "openai": True,
            "anthropic": True,
            "graceful_shutdown": True,
            "listener_closed": True,
            "process_exited": True,
        },
        "dependency_evidence": {
            "required": {
                "forge.clients.anthropic": True,
                "forge_guardrails": True,
                "pydantic": True,
                "httpx": True,
                "anthropic": True,
                "tomli_w": True,
            },
            "excluded_present": [],
        },
        "glibc": {"verified": None, "max_version": None, "objects": []},
    }


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Windows", "AMD64", "windows-x86_64"),
        ("Linux", "x86_64", "linux-x86_64-gnu"),
        ("Darwin", "arm64", "macos-arm64"),
    ],
)
def test_native_target_selection(system: str, machine: str, expected: str) -> None:
    with (
        patch.object(build.platform, "system", return_value=system),
        patch.object(build.platform, "machine", return_value=machine),
    ):
        assert build.native_target() == expected


def test_non_native_target_is_rejected() -> None:
    with patch.object(build, "native_target", return_value="windows-x86_64"):
        with pytest.raises(ValueError, match="requires its native host"):
            build.require_native_target("macos-arm64")


def test_standalone_builder_requires_python_314(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build.sys, "version_info", (3, 13, 0))
    with pytest.raises(RuntimeError, match="require Python 3.14"):
        build.require_python_314()

    monkeypatch.setattr(build.sys, "version_info", (3, 14, 0))
    build.require_python_314()


def test_pyinstaller_forms_share_the_spec_and_inputs(tmp_path: Path) -> None:
    onedir = build.pyinstaller_args("windows-x86_64", "onedir", tmp_path)
    onefile = build.pyinstaller_args("windows-x86_64", "onefile", tmp_path)
    assert onedir[-1] == onefile[-1] == str(build.SPEC)
    assert "onedir" in onedir[onedir.index("--distpath") + 1]
    assert "onefile" in onefile[onefile.index("--distpath") + 1]


def test_onefile_requires_passing_onedir_evidence(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="completed passing onedir"):
        build.require_onedir_gate(tmp_path, "windows-x86_64")

    path = build.evidence_path(tmp_path, "windows-x86_64", "onedir")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(passing_evidence()), encoding="utf-8")
    build.require_onedir_gate(tmp_path, "windows-x86_64")


def test_passing_two_form_build_selects_onefile(tmp_path: Path) -> None:
    selected = build.artifact_path(tmp_path, "windows-x86_64", "onefile")
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"selected artifact")
    for form in ("onedir", "onefile"):
        path = build.evidence_path(tmp_path, "windows-x86_64", form)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = passing_evidence(form)
        if form == "onefile":
            record["path"] = str(selected)
        path.write_text(json.dumps(record), encoding="utf-8")
    output = build.write_selection(tmp_path, "windows-x86_64")
    selection = json.loads(output.read_text(encoding="utf-8"))
    assert selection["selected_form"] == "onefile"
    assert selection["size"] == len(b"selected artifact")
    assert len(selection["sha256"]) == 64


def test_evidence_policy_reports_missing_and_excluded_content() -> None:
    record = passing_evidence()
    required = record["dependency_evidence"]["required"]  # type: ignore[index]
    required["anthropic"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="anthropic"):
        evidence.validate_evidence(record)  # type: ignore[arg-type]

    record = passing_evidence()
    dependency = record["dependency_evidence"]  # type: ignore[assignment]
    dependency["excluded_present"] = ["pyarrow"]  # type: ignore[index]
    with pytest.raises(ValueError, match="pyarrow"):
        evidence.validate_evidence(record)  # type: ignore[arg-type]


def test_onefile_evidence_requires_extraction_cleanup() -> None:
    record = passing_evidence("onefile")
    record["extraction"]["cleanup"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="not cleaned up"):
        evidence.validate_evidence(record)  # type: ignore[arg-type]


def test_toc_dependency_inventory(tmp_path: Path) -> None:
    toc = tmp_path / "Analysis-00.toc"
    toc.write_text(
        repr(
            (
                ["pyarrow", "pytest", "mpmath"],
                [
                    (
                        "forge.clients.anthropic",
                        "forge/clients/anthropic.py",
                        "PYMODULE",
                    ),
                    ("forge.clients.vllm", "forge/clients/vllm.py", "PYMODULE"),
                    ("pydantic", "pydantic/__init__.py", "PYMODULE"),
                    ("httpx", "httpx/__init__.py", "PYMODULE"),
                    ("anthropic", "anthropic/__init__.py", "PYMODULE"),
                    ("tomli_w", "tomli_w/__init__.py", "PYMODULE"),
                    ("forge_guardrails-0.9.0.dist-info/METADATA", "metadata", "DATA"),
                ],
            )
        ),
        encoding="utf-8",
    )
    observed = evidence.dependency_observation(toc)
    assert all(observed["required"].values())
    assert observed["excluded_present"] == []


def test_backend_executable_fails_dependency_inventory(tmp_path: Path) -> None:
    toc = tmp_path / "Analysis-00.toc"
    toc.write_text(repr([]), encoding="utf-8")
    observed = evidence.dependency_observation(
        toc, ["forge-proxy/_internal/llama-server.exe"]
    )
    assert observed["excluded_present"] == ["llama-server.exe"]


def test_nested_elf_failure_is_not_hidden_by_passing_launcher(tmp_path: Path) -> None:
    launcher = tmp_path / "forge-proxy"
    nested = tmp_path / "_internal" / "pydantic_core.so"
    nested.parent.mkdir()
    launcher.write_bytes(b"\x7fELFlauncher")
    nested.write_bytes(b"\x7fELFnested")

    def fake_readelf(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        version = "GLIBC_2.35" if Path(args[-1]) == launcher else "GLIBC_2.36"
        return subprocess.CompletedProcess(args, 0, stdout=version, stderr="")

    with pytest.raises(ValueError, match="pydantic_core.*GLIBC_2.36"):
        evidence.inspect_glibc([launcher, nested], runner=fake_readelf)


def test_onefile_launcher_is_included_in_glibc_inspection(tmp_path: Path) -> None:
    launcher = tmp_path / "forge-proxy"
    nested = tmp_path / "_internal" / "pydantic_core.so"
    nested.parent.mkdir()
    launcher.write_bytes(b"\x7fELFlauncher")
    nested.write_bytes(b"\x7fELFnested")
    package_toc = tmp_path / "PKG-00.toc"
    package_toc.write_text(
        repr([("pydantic_core.so", str(nested), "BINARY")]),
        encoding="utf-8",
    )

    def fake_readelf(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        version = "GLIBC_2.36" if Path(args[-1]) == launcher else "GLIBC_2.35"
        return subprocess.CompletedProcess(args, 0, stdout=version, stderr="")

    paths = evidence.onefile_elf_inventory(package_toc, launcher)
    with pytest.raises(ValueError, match="forge-proxy.*GLIBC_2.36"):
        evidence.inspect_glibc(paths, runner=fake_readelf)


def test_cli_capture_uses_child_windows_locale_under_parent_utf8_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    executable = Path("forge-proxy.exe")
    completed = subprocess.CompletedProcess(
        [str(executable), "--version"],
        0,
        stdout="forge-proxy \u2014 standalone\n",
        stderr="",
    )

    with (
        patch.object(smoke.locale, "getencoding", return_value="cp1252"),
        patch.object(smoke.subprocess, "run", return_value=completed) as run,
    ):
        result = smoke.cli_check(executable, "--version", tmp_path)

    assert result.returncode == 0
    assert result.stdout == "forge-proxy \u2014 standalone\n"
    run.assert_called_once()
    command = run.call_args.args[0]
    kwargs = run.call_args.kwargs
    assert command == [str(executable), "--version"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["encoding"] == "cp1252"
    assert "PYTHONUTF8" not in kwargs["env"]
    assert "PYTHONIOENCODING" not in kwargs["env"]


def test_cli_capture_uses_python_310_locale_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = Path("forge-proxy")
    completed = subprocess.CompletedProcess(
        [str(executable), "--version"], 0, stdout="0.9.1\n", stderr=""
    )
    monkeypatch.delattr(smoke.locale, "getencoding")

    with (
        patch.object(
            smoke.locale, "getpreferredencoding", return_value="UTF-8"
        ) as fallback,
        patch.object(smoke.subprocess, "run", return_value=completed) as run,
    ):
        smoke.cli_check(executable, "--version", tmp_path)

    fallback.assert_called_once_with(False)
    assert run.call_args.kwargs["encoding"] == "UTF-8"


def test_windows_graceful_stop_uses_ctrl_break_and_requires_listener_close() -> None:
    process = MagicMock()
    process.returncode = 0
    ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", 1)
    with (
        patch.object(smoke.os, "name", "nt"),
        patch.object(smoke.signal, "CTRL_BREAK_EVENT", ctrl_break, create=True),
        patch.object(smoke, "port_closed", return_value=True),
    ):
        _, passed = smoke.graceful_stop(process, 8123)
    process.send_signal.assert_called_once_with(ctrl_break)
    process.communicate.assert_called_once_with(timeout=20)
    assert passed is True
