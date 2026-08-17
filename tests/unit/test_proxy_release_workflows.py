"""Structural safety and graph checks for Proxy release workflows."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def load(name: str) -> tuple[dict, str]:
    path = WORKFLOWS / name
    text = path.read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    assert isinstance(document, dict)
    return document, text


def test_candidate_matrix_is_read_only_and_preserves_same_linux_bytes() -> None:
    document, text = load("proxy-release-candidate.yml")
    trigger = document.get("on", document.get(True))
    assert set(trigger) == {"pull_request"}
    assert trigger["pull_request"]["paths"] == ["installer/proxy-stable.txt"]
    assert document["permissions"] == {"contents": "read"}
    assert set(document["jobs"]) == {"native"}
    assert all("permissions" not in job for job in document["jobs"].values())
    assert {
        row["target"]
        for row in document["jobs"]["native"]["strategy"]["matrix"]["include"]
    } == {"windows-x86_64", "linux-x86_64-gnu", "macos-arm64"}
    assert "ubuntu:22.04" in text
    assert "debian:12" in text
    assert "fedora:44" in text
    assert text.count("scripts.standalone.lifecycle_smoke") >= 2
    assert text.count("--target") >= 2
    assert "python3 ca-certificates curl" in text
    assert text.count("python-version: '3.14'") == 1
    assert "tar -czf" in text and "release verify" in text
    assert "tests/integration/bootstrap_contract" in text
    assert "project_version" in text and "installer/proxy-stable.txt" in text
    assert "linux-runtime-evidence" in text
    assert "real_backends" not in text
    assert "aggregate" not in document["jobs"]
    assert not (WORKFLOWS / "proxy-pointer.yml").exists()


def test_general_ci_has_only_three_always_on_python_suites() -> None:
    document, _text = load("tests.yml")
    trigger = document.get("on", document.get(True))
    assert set(trigger) == {"pull_request", "push"}
    assert set(document["jobs"]) == {"test"}
    assert document["jobs"]["test"]["strategy"]["matrix"]["python-version"] == [
        "3.12",
        "3.13",
        "3.14",
    ]


def test_exact_release_has_one_mutation_job_after_every_gate() -> None:
    document, text = load("proxy-release.yml")
    jobs = document["jobs"]
    writers = [
        name
        for name, job in jobs.items()
        if job.get("permissions", {}).get("contents") == "write"
    ]
    assert writers == ["publish"]
    assert jobs["publish"]["permissions"] == {
        "contents": "write",
        "id-token": "write",
        "attestations": "write",
    }
    assert set(jobs["staging"]["needs"]) == {"identity", "native", "linux_compat"}
    assert set(jobs["publish"]["needs"]) == {"identity", "staging"}
    assert set(jobs["exact_install"]["needs"]) == {"identity", "publish"}
    assert "environment: proxy-release" in text
    assert "actions/attest-build-provenance@v2" in text
    assert "manifest last" in text.lower()


def test_exact_identity_and_install_matrices_cover_ruled_targets() -> None:
    document, text = load("proxy-release.yml")
    jobs = document["jobs"]
    ruled = {"windows-x86_64", "linux-x86_64-gnu", "macos-arm64"}
    assert {
        row["target"] for row in jobs["native"]["strategy"]["matrix"]["include"]
    } == ruled
    assert {
        row["target"] for row in jobs["exact_install"]["strategy"]["matrix"]["include"]
    } == ruled
    assert "refs/tags/$TAG" in text
    assert 'git rev-parse "$TAG^{commit}"' in text
    assert "target_commitish (informational only)" in text
    assert "release verify-staging" in text
    assert "install.sh --version" in text and "install.ps1 -Version" in text
    assert text.count("--target") >= 2
    assert text.count("python-version: '3.14'") == 1


def test_windows_exact_install_propagates_native_init_and_check_failures() -> None:
    _document, text = load("proxy-release.yml")
    windows = text.split(
        "      - name: Exact install, initialize, check, and uninstall on Windows", 1
    )[1].split(
        "      - name: Exact install, initialize, check, and uninstall on POSIX", 1
    )[0]
    exit_check = "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
    assert (
        "& $proxy init --non-interactive --force --backend-url "
        "'http://127.0.0.1:1'\n          " + exit_check
    ) in windows
    assert "& $proxy check\n          " + exit_check in windows


def test_release_graph_has_no_forbidden_release_or_pointer_operations() -> None:
    _document, text = load("proxy-release.yml")
    lowered = text.lower()
    for forbidden in (
        "gh release create",
        "git tag ",
        "--clobber",
        "proxy-stable.txt",
        "cosign",
        "sigstore",
        "gpg --sign",
    ):
        assert forbidden not in lowered
    publish_text = text.split("  publish:", 1)[1].split("  exact_install:", 1)[0]
    assert "scripts.standalone.build" not in publish_text
