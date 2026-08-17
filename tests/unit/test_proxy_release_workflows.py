"""Structural contracts for Proxy candidate and publication workflows."""

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


def test_candidate_has_three_platform_jobs_and_one_aggregation_job() -> None:
    document, text = load("proxy-release-candidate.yml")
    trigger = document.get("on", document.get(True))
    assert set(trigger) == {"pull_request"}
    assert trigger["pull_request"]["paths"] == ["installer/proxy-stable.txt"]
    assert document["permissions"] == {"contents": "read"}
    assert set(document["jobs"]) == {"windows", "macos", "linux", "aggregate"}
    assert "needs" not in document["jobs"]["linux"]
    assert document["jobs"]["aggregate"]["needs"] == [
        "windows",
        "macos",
        "linux",
    ]
    linux_text = str(document["jobs"]["linux"])
    aggregate_text = str(document["jobs"]["aggregate"])
    assert "scripts.standalone.release assemble" not in linux_text
    assert "scripts.standalone.release record-candidate" not in linux_text
    assert "actions/download-artifact" not in linux_text
    assert "scripts.standalone.release assemble" in aggregate_text
    assert "scripts.standalone.release record-candidate" in aggregate_text
    assert "actions/download-artifact" in aggregate_text
    assert "proxy-release-candidate" in aggregate_text
    assert all("permissions" not in job for job in document["jobs"].values())

    for target in ("windows-x86_64", "linux-x86_64-gnu", "macos-arm64"):
        assert f"name: proxy-{target}" in text
    assert r".\scripts\standalone\build_windows.ps1" in text
    assert "sh ./scripts/standalone/build_macos.sh" in text
    assert (
        "docker build --file packaging/standalone/linux/Dockerfile "
        "--tag forge-proxy-linux-builder ."
    ) in text
    assert 'docker start --attach "$container_id"' in text
    assert (
        'docker cp "${container_id}:/forge/standalone-dist/." standalone-dist/'
        in text
    )
    assert "python -m scripts.standalone.build" not in text
    assert text.count("python-version: '3.14'") == 4
    assert text.count("tests/integration/bootstrap_contract") == 3
    assert text.count("scripts.standalone.lifecycle_smoke") >= 6
    assert "ubuntu:24.04" in text
    assert "debian:13" in text
    assert "fedora:43" in text
    assert "linux-runtime-evidence" in text
    assert "scripts.standalone.release assemble" in text
    assert "scripts.standalone.release verify-staging publication" in text
    assert "scripts.standalone.release record-candidate" in text
    assert "name: proxy-release-candidate" in text
    assert "proxy-publication.tgz" in text
    assert "real_backends" not in text
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


def test_linux_builder_uses_ubuntu_2404_and_python_314() -> None:
    dockerfile = (
        ROOT / "packaging" / "standalone" / "linux" / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert dockerfile.startswith("FROM ubuntu:24.04\n")
    assert "gpg-agent" in dockerfile
    assert "python3.14 python3.14-venv libpython3.14" in dockerfile
    assert "RUN python3.14 -m venv /build-env" in dockerfile
    assert "python3.12" not in dockerfile


def test_publication_only_verifies_and_uploads_a_successful_candidate() -> None:
    document, text = load("proxy-release.yml")
    trigger = document.get("on", document.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    assert set(trigger["workflow_dispatch"]["inputs"]) == {"tag", "candidate_run_id"}
    assert document["permissions"] == {"actions": "read", "contents": "read"}
    assert set(document["jobs"]) == {"identity", "publish"}
    assert document["jobs"]["publish"]["needs"] == "identity"

    writers = [
        name
        for name, job in document["jobs"].items()
        if job.get("permissions", {}).get("contents") == "write"
    ]
    assert writers == ["publish"]
    assert document["jobs"]["publish"]["permissions"] == {
        "actions": "read",
        "contents": "write",
        "id-token": "write",
        "attestations": "write",
    }
    assert "environment: proxy-release" in text
    assert "actions/attest-build-provenance@v2" in text
    assert text.count("python-version: '3.14'") == 2
    assert text.count("name: proxy-release-candidate") == 2
    assert text.count("run-id: ${{ inputs.candidate_run_id }}") == 2
    assert "scripts.standalone.build" not in text
    assert "scripts.standalone.lifecycle_smoke" not in text
    assert "install.sh" not in text and "install.ps1" not in text


def test_publication_binds_candidate_tree_to_the_exact_release_tag() -> None:
    _document, text = load("proxy-release.yml")
    assert 'git rev-parse "$TAG^{commit}"' in text
    assert "git rev-parse 'HEAD^{tree}'" in text
    assert "scripts.standalone.release verify-candidate" in text
    assert 'jq -r .conclusion)" = "success"' in text
    assert 'jq -r .event)" = "pull_request"' in text
    assert ".github/workflows/proxy-release-candidate.yml" in text
    assert "refs/tags/$TAG" not in text
    assert 'test "$DISPATCH_REF" = "refs/heads/main"' in text
    assert "DISPATCH_SHA" not in text
    assert "--expected-commit '${{ needs.identity.outputs.commit }}'" in text
    assert "--expected-commit '${{ github.sha }}'" not in text


def test_release_graph_has_no_build_tag_pointer_or_clobber_operations() -> None:
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
    assert "scripts.standalone.release verify-staging publication" in text
    assert "manifest last" in text.lower()
