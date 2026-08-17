"""Assemble, validate, and publish exact-version Proxy release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from scripts.standalone.inputs import SUPPORTED_TARGETS


ROOT = Path(__file__).resolve().parents[2]
POINTER = ROOT / "installer" / "proxy-stable.txt"
VERSION_RE = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def project_version(pyproject: Path = ROOT / "pyproject.toml") -> str:
    import tomllib

    version = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    if not isinstance(version, str) or VERSION_RE.fullmatch(version) is None:
        raise ValueError("pyproject.toml project version must be X.Y.Z")
    return version


def exact_tag(version: str) -> str:
    if VERSION_RE.fullmatch(version) is None:
        raise ValueError("version must be a canonical X.Y.Z value")
    return f"v{version}"


def artifact_name(target: str) -> str:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported target: {target}")
    suffix = ".exe" if target == "windows-x86_64" else ""
    return f"forge-proxy-{target}{suffix}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_selection(
    artifact: Path,
    target: str,
    output: Path,
    *,
    version: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy one tested artifact to its release name and record its identity."""

    version = version or project_version()
    exact_tag(version)
    name = artifact_name(target)
    output.mkdir(parents=True, exist_ok=True)
    selected = output / name
    shutil.copy2(artifact, selected)
    record: dict[str, Any] = {
        "target": target,
        "name": name,
        "version": version,
        "size": selected.stat().st_size,
        "sha256": sha256(selected),
        "evidence": evidence or {},
    }
    (output / "selection.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def validate_selection(directory: Path, expected_version: str | None = None) -> dict[str, Any]:
    record = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    required = {"target", "name", "version", "size", "sha256", "evidence"}
    if set(record) != required:
        raise ValueError("selection record has unexpected or missing fields")
    target = record["target"]
    version = record["version"]
    exact_tag(version)
    if expected_version is not None and version != expected_version:
        raise ValueError("selection version does not match requested version")
    if record["name"] != artifact_name(target):
        raise ValueError("selection artifact name does not match target")
    if not isinstance(record["size"], int) or record["size"] < 1:
        raise ValueError("selection size must be a positive integer")
    if not isinstance(record["sha256"], str) or SHA256_RE.fullmatch(record["sha256"]) is None:
        raise ValueError("selection SHA-256 is invalid")
    artifact = directory / record["name"]
    if not artifact.is_file():
        raise ValueError(f"selected artifact is missing: {record['name']}")
    if artifact.stat().st_size != record["size"]:
        raise ValueError("selected artifact size does not match selection record")
    if sha256(artifact) != record["sha256"]:
        raise ValueError("selected artifact digest does not match selection record")
    return record


def validate_manifest(document: dict[str, Any], expected_version: str | None = None) -> dict[str, Any]:
    if set(document) != {"version", "artifacts"}:
        raise ValueError("manifest must contain only version and artifacts")
    version = document["version"]
    exact_tag(version)
    if expected_version is not None and version != expected_version:
        raise ValueError("manifest version does not match requested version")
    artifacts = document["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(SUPPORTED_TARGETS):
        raise ValueError("manifest must contain the complete ruled target set")
    for target in SUPPORTED_TARGETS:
        entry = artifacts[target]
        if not isinstance(entry, dict) or set(entry) != {"name", "sha256", "size"}:
            raise ValueError(f"invalid manifest entry for {target}")
        if entry["name"] != artifact_name(target):
            raise ValueError(f"manifest name does not match {target}")
        if not isinstance(entry["size"], int) or entry["size"] < 1:
            raise ValueError(f"invalid manifest size for {target}")
        if not isinstance(entry["sha256"], str) or SHA256_RE.fullmatch(entry["sha256"]) is None:
            raise ValueError(f"invalid manifest digest for {target}")
    return document


def assemble(inputs: Iterable[Path], output: Path, version: str | None = None) -> Path:
    """Atomically assemble exactly one verified input for every ruled target."""

    version = version or project_version()
    exact_tag(version)
    if output.exists():
        raise ValueError("publication directory must not already exist")
    records: dict[str, tuple[Path, dict[str, Any]]] = {}
    for directory in inputs:
        record = validate_selection(directory, version)
        target = record["target"]
        if target in records:
            raise ValueError(f"duplicate selected target: {target}")
        records[target] = (directory, record)
    missing = set(SUPPORTED_TARGETS) - set(records)
    if missing:
        raise ValueError(f"missing selected targets: {', '.join(sorted(missing))}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        for target in SUPPORTED_TARGETS:
            directory, record = records[target]
            destination = temporary / record["name"]
            shutil.copy2(directory / record["name"], destination)
            if destination.stat().st_size != record["size"] or sha256(destination) != record["sha256"]:
                raise ValueError(f"staged artifact identity changed for {target}")
            artifacts[target] = {
                "name": record["name"],
                "sha256": record["sha256"],
                "size": record["size"],
            }
        manifest = {"version": version, "artifacts": artifacts}
        manifest_name = f"proxy-{version}.json"
        (temporary / manifest_name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checksum_lines = [
            f"{artifacts[target]['sha256']}  {artifacts[target]['name']}"
            for target in SUPPORTED_TARGETS
        ]
        (temporary / f"proxy-{version}.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        validate_staging(temporary, version)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def validate_staging(directory: Path, version: str | None = None) -> dict[str, Any]:
    version = version or project_version()
    manifest_path = directory / f"proxy-{version}.json"
    manifest = validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), version)
    expected_files = {
        *(artifact_name(target) for target in SUPPORTED_TARGETS),
        f"proxy-{version}.json",
        f"proxy-{version}.sha256",
    }
    if {path.name for path in directory.iterdir() if path.is_file()} != expected_files:
        raise ValueError("staging directory does not contain the exact release file set")
    for target in SUPPORTED_TARGETS:
        entry = manifest["artifacts"][target]
        path = directory / entry["name"]
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"staged artifact does not match manifest for {target}")
    expected_checksums = "\n".join(
        f"{manifest['artifacts'][target]['sha256']}  {manifest['artifacts'][target]['name']}"
        for target in SUPPORTED_TARGETS
    ) + "\n"
    if (directory / f"proxy-{version}.sha256").read_text(encoding="utf-8") != expected_checksums:
        raise ValueError("checksum file does not match the canonical manifest order")
    return manifest


def http_manifest_resolver(version: str) -> bytes:
    url = (
        "https://github.com/antoinezambelli/forge/releases/download/"
        f"v{version}/proxy-{version}.json"
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def validate_pointer(
    pointer: Path = POINTER,
    resolver: Callable[[str], bytes] = http_manifest_resolver,
) -> str | None:
    """Validate an optional stable pointer without ever writing it."""

    if not pointer.exists():
        return None
    raw = pointer.read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("stable pointer must be one ASCII X.Y.Z line") from exc
    if not text.endswith("\n") or text.count("\n") != 1:
        raise ValueError("stable pointer must be one bare X.Y.Z line")
    version = text[:-1]
    exact_tag(version)
    payload = resolver(version)
    validate_manifest(json.loads(payload), version)
    return version


class ReleaseClient(Protocol):
    def release(self, tag: str) -> dict[str, Any]: ...
    def assets(self, release_id: int) -> list[dict[str, Any]]: ...
    def upload(self, tag: str, path: Path) -> int: ...
    def delete(self, asset_id: int) -> None: ...


def proxy_asset_names(version: str) -> list[str]:
    return [
        *(artifact_name(target) for target in SUPPORTED_TARGETS),
        f"proxy-{version}.sha256",
        f"proxy-{version}.json",
    ]


def publish(
    client: ReleaseClient,
    tag: str,
    peeled_commit: str,
    expected_commit: str,
    directory: Path,
) -> None:
    """Publish a complete namespace, rolling back only assets from this run."""

    version = project_version()
    if tag != exact_tag(version):
        raise ValueError("requested tag does not match pyproject.toml version")
    if peeled_commit != expected_commit:
        raise ValueError("peeled tag commit does not match checked-out commit")
    manifest = validate_staging(directory, version)
    release = client.release(tag)
    if release.get("tag_name") != tag:
        raise ValueError("existing GitHub Release tag_name does not match exact tag")
    release_id = int(release["id"])
    expected = set(proxy_asset_names(version))
    existing = {asset["name"] for asset in client.assets(release_id)}
    collision = expected & existing
    if collision:
        raise ValueError(f"Proxy release assets already exist: {', '.join(sorted(collision))}")

    journal: list[int] = []
    ordered = [
        *(manifest["artifacts"][target]["name"] for target in SUPPORTED_TARGETS),
        f"proxy-{version}.sha256",
        f"proxy-{version}.json",
    ]
    try:
        for name in ordered:
            path = directory / name
            if name in {entry["name"] for entry in manifest["artifacts"].values()}:
                entry = next(item for item in manifest["artifacts"].values() if item["name"] == name)
                if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
                    raise ValueError(f"artifact identity changed before upload: {name}")
            journal.append(client.upload(tag, path))
        final_names = {asset["name"] for asset in client.assets(release_id)}
        if final_names & expected != expected:
            raise RuntimeError("published Proxy namespace is incomplete")
    except BaseException as exc:
        cleanup_errors: list[str] = []
        for asset_id in reversed(journal):
            try:
                client.delete(asset_id)
            except BaseException as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
        try:
            remaining = {
                asset["name"] for asset in client.assets(release_id)
            } & expected
        except BaseException as verification_exc:
            cleanup_errors.append(str(verification_exc))
            remaining = expected
        if cleanup_errors or remaining:
            raise RuntimeError(
                "Proxy publication cleanup could not prove an empty namespace; "
                "use a new version or perform manual remediation"
            ) from exc
        raise


class GhReleaseClient:
    def __init__(self, repository: str) -> None:
        self.repository = repository

    def _json(self, *args: str) -> Any:
        result = subprocess.run(
            ["gh", *args], check=True, capture_output=True, text=True
        )
        return json.loads(result.stdout)

    def release(self, tag: str) -> dict[str, Any]:
        return self._json("api", f"repos/{self.repository}/releases/tags/{tag}")

    def assets(self, release_id: int) -> list[dict[str, Any]]:
        return self._json("api", f"repos/{self.repository}/releases/{release_id}/assets")

    def upload(self, tag: str, path: Path) -> int:
        subprocess.run(
            ["gh", "release", "upload", tag, str(path), "--repo", self.repository],
            check=True,
        )
        release_id = int(self.release(tag)["id"])
        return int(next(asset["id"] for asset in self.assets(release_id) if asset["name"] == path.name))

    def delete(self, asset_id: int) -> None:
        subprocess.run(
            ["gh", "api", "--method", "DELETE", f"repos/{self.repository}/releases/assets/{asset_id}"],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record")
    record.add_argument("--artifact", type=Path, required=True)
    record.add_argument("--target", choices=SUPPORTED_TARGETS, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--evidence", type=Path, action="append", default=[])
    verify = commands.add_parser("verify")
    verify.add_argument("directory", type=Path)
    assembly = commands.add_parser("assemble")
    assembly.add_argument("--input", type=Path, action="append", required=True)
    assembly.add_argument("--output", type=Path, required=True)
    staged = commands.add_parser("verify-staging")
    staged.add_argument("directory", type=Path)
    commands.add_parser("pointer")
    publication = commands.add_parser("publish")
    publication.add_argument("--repository", required=True)
    publication.add_argument("--tag", required=True)
    publication.add_argument("--peeled-commit", required=True)
    publication.add_argument("--expected-commit", required=True)
    publication.add_argument("directory", type=Path)
    args = parser.parse_args()

    if args.command == "record":
        portable = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in args.evidence
        }
        print(json.dumps(write_selection(args.artifact, args.target, args.output, evidence=portable), sort_keys=True))
    elif args.command == "verify":
        print(json.dumps(validate_selection(args.directory), sort_keys=True))
    elif args.command == "assemble":
        print(assemble(args.input, args.output))
    elif args.command == "verify-staging":
        print(json.dumps(validate_staging(args.directory), sort_keys=True))
    elif args.command == "pointer":
        print(validate_pointer() or "no stable Proxy release")
    else:
        publish(
            GhReleaseClient(args.repository), args.tag, args.peeled_commit,
            args.expected_commit, args.directory,
        )


if __name__ == "__main__":
    main()
