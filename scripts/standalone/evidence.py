"""Artifact-derived inventory, evidence policy, and Linux ABI inspection."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from scripts.standalone.inputs import (
    EXCLUDED_ARTIFACT_NAMES,
    EXCLUDED_MODULES,
    REQUIRED_CONTENT,
)


_GLIBC = re.compile(r"GLIBC_(\d+)\.(\d+)")


def toc_inventory(path: Path) -> list[str]:
    """Return every string recorded in a completed PyInstaller TOC."""

    value = ast.literal_eval(path.read_text(encoding="utf-8"))
    found: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, str):
            found.append(item.replace("\\", "/"))
        elif isinstance(item, (tuple, list, set)):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)

    visit(value)
    return sorted(set(found))


def collected_toc_inventory(path: Path) -> list[str]:
    """Return names and sources from collected TOC entries, not build options."""

    value = ast.literal_eval(path.read_text(encoding="utf-8"))
    found: list[str] = []
    entry_types = {
        "BINARY", "DATA", "DEPENDENCY", "EXECUTABLE", "EXTENSION",
        "PYMODULE", "PYSOURCE", "PYZ",
    }

    def visit(item: object) -> None:
        if (
            isinstance(item, tuple)
            and len(item) >= 3
            and isinstance(item[2], str)
            and item[2] in entry_types
        ):
            for value in item[:2]:
                if isinstance(value, str):
                    found.append(value.replace("\\", "/"))
            return
        if isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return sorted(set(found))


def dependency_observation(
    analysis_toc: Path,
    artifact_files: Iterable[str] = (),
) -> dict[str, Any]:
    inventory = sorted(
        set(collected_toc_inventory(analysis_toc)) | set(artifact_files)
    )
    normalized_items = [item.lower().replace("-", "_") for item in inventory]
    normalized = "\n".join(normalized_items)
    required = {
        name: name.lower().replace("-", "_") in normalized
        for name in REQUIRED_CONTENT
    }
    excluded: list[str] = []
    for name in EXCLUDED_MODULES:
        token = name.lower().replace("-", "_")
        if any(
            item == token
            or item.startswith(f"{token}.")
            or f"/{token}/" in item
            for item in normalized_items
        ):
            excluded.append(name)
    artifact_basenames = {
        Path(item).name.lower().replace("-", "_") for item in normalized_items
    }
    for name in EXCLUDED_ARTIFACT_NAMES:
        if name.lower().replace("-", "_") in artifact_basenames:
            excluded.append(name)
    excluded.sort()
    return {
        "analysis_toc": str(analysis_toc.resolve()),
        "required": required,
        "excluded_present": excluded,
        "inventory": inventory,
    }


def validate_evidence(evidence: dict[str, Any]) -> None:
    """Enforce the release-gate fields and supported behavior."""

    required_fields = {
        "target", "form", "path", "size_bytes", "build_identity",
        "runtime_identity", "cold_start_seconds", "shutdown_seconds",
        "extraction", "smoke", "dependency_evidence", "glibc",
    }
    missing = sorted(required_fields - evidence.keys())
    if missing:
        raise ValueError(f"evidence missing required fields: {', '.join(missing)}")

    failed_smoke = sorted(
        name for name, passed in evidence["smoke"].items() if passed is not True
    )
    if failed_smoke:
        raise ValueError(f"smoke checks failed: {', '.join(failed_smoke)}")

    missing_content = sorted(
        name
        for name, present in evidence["dependency_evidence"]["required"].items()
        if present is not True
    )
    if missing_content:
        raise ValueError(
            f"artifact dependency content missing: {', '.join(missing_content)}"
        )
    excluded = evidence["dependency_evidence"]["excluded_present"]
    if excluded:
        raise ValueError(f"excluded artifact content present: {', '.join(excluded)}")

    if evidence["form"] == "onefile" and (
        evidence["extraction"].get("cleanup") is not True
    ):
        raise ValueError("onefile extraction directory was not cleaned up")
    if evidence["target"] == "linux-x86_64-gnu" and (
        evidence["glibc"].get("verified") is not True
    ):
        raise ValueError("Linux GLIBC inspection did not pass")


def read_evidence(path: Path) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    validate_evidence(evidence)
    return evidence


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(4) == b"\x7fELF"
    except OSError:
        return False


def onefile_elf_inventory(package_toc: Path, launcher: Path) -> list[Path]:
    """Return the onefile launcher and collected ELF source paths."""

    return sorted({
        Path(value)
        for value in [str(launcher), *toc_inventory(package_toc)]
        if Path(value).is_file() and is_elf(Path(value))
    })


def inspect_glibc(
    paths: Iterable[Path],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Inspect every supplied ELF object and enforce the Ubuntu 24.04 ceiling."""

    maximum = (0, 0)
    objects: list[str] = []
    for path in paths:
        if not is_elf(path):
            continue
        objects.append(str(path))
        result = runner(
            ["readelf", "--version-info", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        versions = [(int(a), int(b)) for a, b in _GLIBC.findall(result.stdout)]
        if versions:
            maximum = max(maximum, *versions)
        if any(version > (2, 39) for version in versions):
            rendered = max(versions)
            raise ValueError(
                f"{path} references GLIBC_{rendered[0]}.{rendered[1]} above 2.39"
            )
    return {
        "verified": True,
        "max_version": f"{maximum[0]}.{maximum[1]}",
        "objects": objects,
    }
