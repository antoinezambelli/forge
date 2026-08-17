"""Forge-owned Proxy profile locations, parsing, and managed writes."""

from __future__ import annotations

import os
import platform
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import tomli_w

from forge.proxy._config import (
    _NormalizedProxyConfig,
    _RawProxyConfig,
    _normalize_proxy_config,
)
from forge.proxy._options import option_defaults, profile_definitions


@dataclass(frozen=True)
class _ProfileLaunch:
    raw: _RawProxyConfig
    normalized: _NormalizedProxyConfig
    verbose: bool
    explicit_values: dict[str, object]


def _managed_profile_root(
    *,
    system: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    system = system or platform.system()
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else home
    if system == "Windows":
        return Path(environ["APPDATA"]) / "Forge" / "profiles"
    if system == "Darwin":
        return home / "Library" / "Application Support" / "Forge" / "profiles"
    config_root = Path(environ.get("XDG_CONFIG_HOME", home / ".config"))
    return config_root / "forge" / "profiles"


def _validate_profile_name(name: str) -> None:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(
            "profile name must be nonempty and may not be '.', '..', or contain '/' or '\\'"
        )


def _managed_profile_path(name: str, *, root: Path | None = None) -> Path:
    _validate_profile_name(name)
    return (root or _managed_profile_root()) / f"{name}.toml"


def _validate_profile_value(name: str, value: object, kind: str) -> object:
    if kind == "string":
        valid = isinstance(value, str)
    elif kind == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        if valid:
            value = float(value)
    elif kind == "boolean":
        valid = isinstance(value, bool)
    else:
        valid = (
            isinstance(value, list)
            and all(isinstance(item, str) for item in value)
        )
    if not valid:
        raise ValueError(f"profile field {name!r} has the wrong TOML type")
    return value


def _parse_profile_document(
    document: Mapping[str, object],
    *,
    backend_api_key: str | None = None,
) -> _ProfileLaunch:
    schema_version = document.get("schema_version")
    if not (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version == 1
    ):
        raise ValueError("profile requires integer schema_version = 1")

    definitions = profile_definitions()
    unknown = sorted(set(document) - {"schema_version", *definitions})
    if unknown:
        raise ValueError(f"unknown profile fields: {', '.join(unknown)}")

    explicit: dict[str, object] = {}
    for name, value in document.items():
        if name == "schema_version":
            continue
        definition = definitions[name]
        explicit[name] = _validate_profile_value(
            name, value, definition.profile_kind
        )

    values = option_defaults(backend_api_key_from_environment=False)
    values.update(explicit)
    values["backend_api_key"] = backend_api_key
    raw = _RawProxyConfig(
        backend_url=values["backend_url"],  # type: ignore[arg-type]
        backend=values["backend"],  # type: ignore[arg-type]
        model=values["model"],  # type: ignore[arg-type]
        gguf=values["gguf"],  # type: ignore[arg-type]
        model_path=values["model_path"],  # type: ignore[arg-type]
        backend_port=values["backend_port"],  # type: ignore[arg-type]
        budget_mode=values["budget_mode"],  # type: ignore[arg-type]
        budget_tokens=values["budget_tokens"],  # type: ignore[arg-type]
        extra_flags=values["extra_flags"],  # type: ignore[arg-type]
        host=values["host"],  # type: ignore[arg-type]
        port=values["port"],  # type: ignore[arg-type]
        serialize=values["serialize"],  # type: ignore[arg-type]
        max_retries=values["max_retries"],  # type: ignore[arg-type]
        max_tool_errors=values["max_tool_errors"],  # type: ignore[arg-type]
        rescue_enabled=not values["no_rescue"],
        backend_capability=values["backend_capability"],  # type: ignore[arg-type]
        inject_respond_tool=values["inject_respond_tool"],  # type: ignore[arg-type]
        backend_timeout=values["backend_timeout"],  # type: ignore[arg-type]
        reasoning_replay=values["reasoning_replay"],  # type: ignore[arg-type]
        backend_api_key=values["backend_api_key"],  # type: ignore[arg-type]
    )
    return _ProfileLaunch(
        raw=raw,
        normalized=_normalize_proxy_config(raw),
        verbose=bool(values["verbose"]),
        explicit_values=explicit,
    )


def _load_profile(path: Path) -> _ProfileLaunch:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    return _parse_profile_document(
        document,
        backend_api_key=os.environ.get("FORGE_BACKEND_API_KEY"),
    )


def _profile_bytes(explicit_values: Mapping[str, object]) -> bytes:
    definitions = profile_definitions()
    document: dict[str, object] = {"schema_version": 1}
    for name in definitions:
        if name in explicit_values:
            document[name] = explicit_values[name]
    return tomli_w.dumps(document).encode("utf-8")


def _write_managed_profile(path: Path, content: bytes, *, force: bool) -> bool:
    """Atomically write a managed profile; return False for identical content."""

    if path.exists():
        if path.read_bytes() == content:
            return False
        if not force:
            raise FileExistsError(f"profile already exists: {path}; use --force to replace it")

    missing_directories: list[Path] = []
    current = path.parent
    while not current.exists():
        missing_directories.append(current)
        current = current.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        for directory in reversed(missing_directories):
            directory.chmod(0o700)

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        if os.name != "nt":
            temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return True


def _managed_profiles(*, root: Path | None = None) -> list[Path]:
    directory = root or _managed_profile_root()
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.toml"), key=lambda path: path.name)
