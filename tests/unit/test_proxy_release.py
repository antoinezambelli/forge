"""Release assembly, pointer, and journaled publication contracts."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from scripts.standalone import release
from scripts.standalone.inputs import SUPPORTED_TARGETS


VERSION = release.project_version()
OTHER_VERSION = "9.9.9"
SOURCE_TREE = "a" * 40


def selections(tmp_path: Path) -> list[Path]:
    result = []
    for target in reversed(SUPPORTED_TARGETS):
        source = tmp_path / f"source-{target}"
        source.write_bytes(f"bytes-{target}".encode())
        output = tmp_path / f"selected-{target}"
        release.write_selection(source, target, output, version=VERSION, evidence={"passed": True})
        result.append(output)
    return result


def test_complete_assembly_is_canonical_and_revalidates(tmp_path: Path) -> None:
    output = release.assemble(selections(tmp_path), tmp_path / "publication", VERSION)
    manifest = release.validate_staging(output, VERSION)
    assert list(manifest["artifacts"]) == sorted(SUPPORTED_TARGETS)
    lines = (output / f"proxy-{VERSION}.sha256").read_text().splitlines()
    assert [line.split("  ")[1] for line in lines] == [
        release.artifact_name(target) for target in SUPPORTED_TARGETS
    ]


def test_candidate_identity_binds_version_and_source_tree(tmp_path: Path) -> None:
    path = tmp_path / "candidate-identity.json"
    assert release.write_candidate_identity(path, SOURCE_TREE, VERSION) == {
        "version": VERSION,
        "source_tree": SOURCE_TREE,
    }
    assert release.validate_candidate_identity(path, VERSION, SOURCE_TREE) == {
        "version": VERSION,
        "source_tree": SOURCE_TREE,
    }


@pytest.mark.parametrize(
    ("version", "source_tree"),
    [(OTHER_VERSION, SOURCE_TREE), (VERSION, "b" * 40)],
)
def test_candidate_identity_rejects_a_different_tag_tree(
    tmp_path: Path,
    version: str,
    source_tree: str,
) -> None:
    path = tmp_path / "candidate-identity.json"
    release.write_candidate_identity(path, SOURCE_TREE, VERSION)
    with pytest.raises(ValueError, match="does not match"):
        release.validate_candidate_identity(path, version, source_tree)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "version", "name", "size", "digest"])
def test_assembly_rejects_incomplete_or_changed_inputs(tmp_path: Path, failure: str) -> None:
    inputs = selections(tmp_path)
    if failure == "missing":
        inputs.pop()
    elif failure == "duplicate":
        inputs.append(inputs[0])
    else:
        record_path = inputs[0] / "selection.json"
        record = json.loads(record_path.read_text())
        if failure == "version":
            record["version"] = OTHER_VERSION
        elif failure == "name":
            record["name"] = "wrong"
        elif failure == "size":
            record["size"] += 1
        else:
            record["sha256"] = "f" * 64
        record_path.write_text(json.dumps(record))
    output = tmp_path / "publication"
    with pytest.raises(ValueError):
        release.assemble(inputs, output, VERSION)
    assert not output.exists()


def test_staging_detects_tamper_at_final_digest_boundary(tmp_path: Path) -> None:
    output = release.assemble(selections(tmp_path), tmp_path / "publication", VERSION)
    (output / release.artifact_name("linux-x86_64-gnu")).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match manifest"):
        release.validate_staging(output, VERSION)


def complete_manifest() -> bytes:
    return json.dumps({
        "version": VERSION,
        "artifacts": {
            target: {"name": release.artifact_name(target), "sha256": "a" * 64, "size": 1}
            for target in SUPPORTED_TARGETS
        },
    }).encode()


def test_pointer_absence_is_success_without_lookup(tmp_path: Path) -> None:
    calls = []
    assert release.validate_pointer(tmp_path / "missing", lambda version: calls.append(version)) is None
    assert calls == []


@pytest.mark.parametrize("content", [b"0.9.0", b"v0.9.0\n", b"0.9.0\nextra\n", b"01.2.3\n"])
def test_pointer_requires_one_bare_canonical_version_line(tmp_path: Path, content: bytes) -> None:
    pointer = tmp_path / "pointer"
    pointer.write_bytes(content)
    with pytest.raises(ValueError):
        release.validate_pointer(pointer, lambda _version: complete_manifest())
    assert pointer.read_bytes() == content


def test_pointer_resolves_exact_complete_manifest_and_is_read_only(tmp_path: Path) -> None:
    pointer = tmp_path / "pointer"
    pointer.write_bytes(f"{VERSION}\n".encode("ascii"))
    before = pointer.read_bytes()
    calls = []
    assert release.validate_pointer(pointer, lambda version: calls.append(version) or complete_manifest()) == VERSION
    assert calls == [VERSION]
    assert pointer.read_bytes() == before


def test_pointer_propagates_live_404(tmp_path: Path) -> None:
    pointer = tmp_path / "pointer"
    pointer.write_bytes(f"{VERSION}\n".encode("ascii"))
    def missing(_version: str) -> bytes:
        raise release.urllib.error.HTTPError("url", 404, "missing", {}, None)
    with pytest.raises(release.urllib.error.HTTPError):
        release.validate_pointer(pointer, missing)


class FakeClient:
    def __init__(
        self, *, fail_upload: int | None = None, fail_delete: bool = False,
        fail_assets_after: int | None = None,
    ) -> None:
        self.release_data = {"id": 7, "tag_name": f"v{VERSION}", "target_commitish": "main"}
        self.current = [{"id": 1, "name": "forge-wheel.whl"}]
        self.uploaded: list[str] = []
        self.deleted: list[int] = []
        self.fail_upload = fail_upload
        self.fail_delete = fail_delete
        self.fail_assets_after = fail_assets_after
        self.asset_calls = 0

    def release(self, _tag: str) -> dict[str, object]:
        return self.release_data

    def assets(self, _release_id: int) -> list[dict[str, object]]:
        if self.fail_assets_after is not None and self.asset_calls >= self.fail_assets_after:
            raise RuntimeError("asset verification failed")
        self.asset_calls += 1
        return list(self.current)

    def upload(self, _release_id: int, path: Path) -> int:
        position = len(self.uploaded)
        if self.fail_upload == position:
            raise RuntimeError("upload failed")
        asset_id = 100 + position
        self.uploaded.append(path.name)
        self.current.append({"id": asset_id, "name": path.name})
        return asset_id

    def delete(self, asset_id: int) -> None:
        if self.fail_delete:
            raise RuntimeError("cleanup failed")
        self.deleted.append(asset_id)
        self.current = [asset for asset in self.current if asset["id"] != asset_id]


def staged(tmp_path: Path) -> Path:
    return release.assemble(selections(tmp_path), tmp_path / "publication", VERSION)


def test_publication_ignores_branch_valued_target_commitish_and_uploads_manifest_last(tmp_path: Path) -> None:
    client = FakeClient()
    release.publish(client, f"v{VERSION}", "abc", "abc", staged(tmp_path))
    assert client.uploaded == release.proxy_asset_names(VERSION)
    assert client.uploaded[-1] == f"proxy-{VERSION}.json"
    assert client.current[0]["name"] == "forge-wheel.whl"


def test_github_upload_returns_the_created_asset_id_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "proxy artifact.bin"
    artifact.write_bytes(b"exact bytes")
    captured: dict[str, object] = {}

    def fake_urlopen(request: release.urllib.request.Request, timeout: int) -> io.BytesIO:
        captured["request"] = request
        captured["timeout"] = timeout
        return io.BytesIO(b'{"id": 321}')

    monkeypatch.setenv("GH_TOKEN", "release-token")
    monkeypatch.setattr(release.urllib.request, "urlopen", fake_urlopen)

    client = release.GhReleaseClient("owner/repo")
    assert client.upload(7, artifact) == 321
    request = captured["request"]
    assert isinstance(request, release.urllib.request.Request)
    assert request.full_url == (
        "https://uploads.github.com/repos/owner/repo/releases/7/assets?"
        "name=proxy%20artifact.bin"
    )
    assert request.data == b"exact bytes"
    assert request.get_header("Authorization") == "Bearer release-token"
    assert request.get_header("Content-type") == "application/octet-stream"
    assert captured["timeout"] == 120


@pytest.mark.parametrize("position", range(5))
def test_each_partial_upload_failure_removes_only_journaled_assets(tmp_path: Path, position: int) -> None:
    client = FakeClient(fail_upload=position)
    with pytest.raises(RuntimeError, match="upload failed"):
        release.publish(client, f"v{VERSION}", "abc", "abc", staged(tmp_path))
    assert client.current == [{"id": 1, "name": "forge-wheel.whl"}]
    assert len(client.deleted) == position


def test_cleanup_verification_failure_requires_manual_remediation(tmp_path: Path) -> None:
    client = FakeClient(fail_upload=1, fail_delete=True)
    with pytest.raises(RuntimeError, match="new version or perform manual remediation"):
        release.publish(client, f"v{VERSION}", "abc", "abc", staged(tmp_path))


def test_cleanup_asset_lookup_failure_requires_manual_remediation(tmp_path: Path) -> None:
    client = FakeClient(fail_upload=0, fail_assets_after=1)
    with pytest.raises(RuntimeError, match="new version or perform manual remediation"):
        release.publish(client, f"v{VERSION}", "abc", "abc", staged(tmp_path))


def test_publication_rejects_existing_expected_name_without_clobber(tmp_path: Path) -> None:
    client = FakeClient()
    client.current.append({"id": 2, "name": release.artifact_name(SUPPORTED_TARGETS[0])})
    with pytest.raises(ValueError, match="already exist"):
        release.publish(client, f"v{VERSION}", "abc", "abc", staged(tmp_path))
    assert client.uploaded == []


@pytest.mark.parametrize("failure", ["tag", "commit"])
def test_publication_binds_exact_release_tag_and_peeled_commit(tmp_path: Path, failure: str) -> None:
    client = FakeClient()
    if failure == "tag":
        client.release_data["tag_name"] = "v9.9.9"
    with pytest.raises(ValueError):
        release.publish(
            client, f"v{VERSION}", "wrong" if failure == "commit" else "abc",
            "abc", staged(tmp_path),
        )
    assert client.uploaded == []
