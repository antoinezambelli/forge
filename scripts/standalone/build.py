"""Build and verify a native standalone Forge Proxy artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from forge import __version__
from scripts.standalone.evidence import (
    dependency_observation,
    inspect_glibc,
    is_elf,
    onefile_elf_inventory,
    read_evidence,
    validate_evidence,
)
from scripts.standalone.inputs import SUPPORTED_TARGETS
from scripts.standalone.smoke import run_smoke


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "packaging" / "standalone" / "forge_proxy.spec"


def native_target() -> str | None:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows" and machine in {"amd64", "x86_64"}:
        return "windows-x86_64"
    if system == "linux" and machine in {"amd64", "x86_64"}:
        return "linux-x86_64-gnu"
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "macos-arm64"
    return None


def require_native_target(target: str) -> None:
    native = native_target()
    if target != native:
        raise ValueError(
            f"target {target!r} requires its native host; current host is {native!r}"
        )


def require_python_314() -> None:
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError(
            f"standalone builds require Python 3.14, got {platform.python_version()}"
        )


def pyinstaller_args(
    target: str,
    form: str,
    output_root: Path,
) -> list[str]:
    form_root = output_root / target / form
    return [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(form_root),
        "--workpath",
        str(form_root / "work"),
        str(SPEC),
    ]


def evidence_path(output_root: Path, target: str, form: str) -> Path:
    return output_root / target / form / "evidence.json"


def require_onedir_gate(output_root: Path, target: str) -> None:
    path = evidence_path(output_root, target, "onedir")
    if not path.is_file():
        raise RuntimeError("onefile requires a completed passing onedir evidence.json")
    read_evidence(path)


def artifact_path(output_root: Path, target: str, form: str) -> Path:
    suffix = ".exe" if target == "windows-x86_64" else ""
    root = output_root / target / form
    if form == "onedir":
        return root / "forge-proxy" / f"forge-proxy{suffix}"
    return root / f"forge-proxy{suffix}"


def recursive_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def artifact_files(artifact: Path, form: str) -> list[str]:
    if form == "onefile":
        return [artifact.name]
    return [
        str(path.relative_to(artifact.parent)).replace("\\", "/")
        for path in artifact.parent.rglob("*")
        if path.is_file()
    ]


def build_one(target: str, form: str, output_root: Path) -> Path:
    if form == "onefile":
        require_onedir_gate(output_root, target)

    form_root = output_root / target / form
    env = os.environ.copy()
    env["FORGE_STANDALONE_FORM"] = form
    subprocess.run(
        pyinstaller_args(target, form, output_root),
        cwd=ROOT,
        env=env,
        check=True,
    )

    artifact = artifact_path(output_root, target, form)
    if not artifact.is_file():
        raise RuntimeError(f"PyInstaller did not produce {artifact}")
    work = form_root / "work" / "forge_proxy"
    analysis_toc = work / "Analysis-00.toc"
    dependencies = dependency_observation(
        analysis_toc,
        artifact_files(artifact, form),
    )

    glibc: dict[str, Any]
    if target == "linux-x86_64-gnu":
        if form == "onedir":
            elf_paths = [path for path in artifact.parent.rglob("*") if is_elf(path)]
        else:
            elf_paths = onefile_elf_inventory(work / "PKG-00.toc", artifact)
        glibc = inspect_glibc(elf_paths)
    else:
        glibc = {"verified": None, "max_version": None, "objects": []}

    smoke = run_smoke(artifact, form, __version__)
    evidence: dict[str, Any] = {
        "target": target,
        "form": form,
        "path": str(artifact.resolve()),
        "size_bytes": recursive_size(
            artifact if form == "onefile" else artifact.parent
        ),
        "build_identity": {
            "python": platform.python_version(),
            "pyinstaller": importlib.metadata.version("pyinstaller"),
            "host_system": platform.system(),
            "host_machine": platform.machine(),
        },
        **smoke,
        "dependency_evidence": dependencies,
        "glibc": glibc,
    }
    validate_evidence(evidence)
    output = evidence_path(output_root, target, form)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return output


def write_selection(output_root: Path, target: str) -> Path:
    onedir = read_evidence(evidence_path(output_root, target, "onedir"))
    onefile = read_evidence(evidence_path(output_root, target, "onefile"))
    selection = {
        "target": target,
        "version": __version__,
        "name": (
            f"forge-proxy-{target}.exe"
            if target == "windows-x86_64"
            else f"forge-proxy-{target}"
        ),
        "size": Path(onefile["path"]).stat().st_size,
        "sha256": hashlib.sha256(Path(onefile["path"]).read_bytes()).hexdigest(),
        "selected_form": "onefile",
        "selected_path": onefile["path"],
        "reason": (
            "onefile passed the same supported smoke and graceful-shutdown "
            "checks as onedir, including extraction cleanup"
        ),
        "measurements": {
            "onedir": {
                "size_bytes": onedir["size_bytes"],
                "cold_start_seconds": onedir["cold_start_seconds"],
            },
            "onefile": {
                "size_bytes": onefile["size_bytes"],
                "cold_start_seconds": onefile["cold_start_seconds"],
            },
        },
    }
    output = output_root / target / "selection.json"
    output.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=SUPPORTED_TARGETS, required=True)
    parser.add_argument("--form", choices=("onedir", "onefile", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "standalone-dist")
    args = parser.parse_args()

    require_python_314()
    require_native_target(args.target)
    forms = ("onedir", "onefile") if args.form == "all" else (args.form,)
    for form in forms:
        build_one(args.target, form, args.output_root.resolve())
    if args.form == "all":
        write_selection(args.output_root.resolve(), args.target)


if __name__ == "__main__":
    main()
