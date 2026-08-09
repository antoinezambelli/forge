"""Pure, private normalization for the flat Proxy configuration surface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from forge._backend_profiles import (
    ArtifactIdentity,
    BackendProfile,
    ClientAdapter,
    LifecycleOwnership,
    ManagedBackendProfile,
    UnmanagedBackendProfile,
    find_managed_profile,
    proxy_backend_selectors,
    unmanaged_profile,
)
from forge._endpoint_layouts import ConnectionInputKind, replace_authority_port
from forge._resolved_backend import ResolvedBackend, resolve_backend
from forge.core.reasoning import ReasoningReplay, validate_reasoning_replay
from forge.server import BudgetMode


@dataclass(frozen=True)
class _RawProxyConfig:
    backend_url: str | None
    backend: str | None
    model: str | None
    gguf: str | Path | None
    model_path: str | Path | None
    backend_port: int | None
    budget_mode: BudgetMode | str | None
    budget_tokens: int | None
    extra_flags: list[str] | None
    host: str
    port: int
    serialize: bool | None
    max_retries: int
    max_tool_errors: int
    rescue_enabled: bool
    backend_capability: str
    inject_respond_tool: bool
    backend_timeout: float
    reasoning_replay: str
    backend_api_key: str | None


@dataclass(frozen=True)
class _NormalizedProxyConfig:
    backend_url: str | None
    backend: str | None
    profile: BackendProfile
    protocol: str
    resolved_backend: ResolvedBackend
    model: str | None
    gguf: str | Path | None
    model_path: str | Path | None
    backend_port: int | None
    budget_mode: BudgetMode | None
    budget_tokens: int | None
    extra_flags: tuple[str, ...]
    host: str
    port: int
    serialize: bool
    max_retries: int
    max_tool_errors: int
    rescue_enabled: bool
    backend_capability: str
    inject_respond_tool: bool
    backend_timeout: float
    reasoning_replay: ReasoningReplay
    backend_api_key: str | None


def _normalize_model(model: str | None) -> str | None:
    return model if model else None


def _normalize_api_key(api_key: str | None) -> str | None:
    return api_key if (api_key and api_key.strip()) else None


def _managed_identity_field(identity: ArtifactIdentity) -> str:
    if identity == ArtifactIdentity.MODEL_TAG:
        return "model"
    if identity == ArtifactIdentity.GGUF_PATH:
        return "gguf"
    if identity == ArtifactIdentity.MODEL_PATH:
        return "model_path"
    raise AssertionError(f"unexpected managed identity: {identity!r}")


def _validate_managed_identity(
    backend: str,
    profile: ManagedBackendProfile,
    *,
    model: str | None,
    gguf: str | Path | None,
    model_path: str | Path | None,
) -> None:
    required = _managed_identity_field(profile.required_identity)
    values = {"model": model, "gguf": gguf, "model_path": model_path}
    if not values[required]:
        raise ValueError(f"backend={backend!r} requires {required}")
    for field, value in values.items():
        if field != required and value is not None:
            raise ValueError(
                f"backend={backend!r} does not accept {field} (use {required})"
            )


def _normalize_budget_mode(value: BudgetMode | str | None) -> BudgetMode | None:
    if value is None:
        return None
    try:
        return BudgetMode(value)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in BudgetMode)
        raise ValueError(f"budget_mode must be one of: {choices}") from exc


def _validate_extra_flags(
    profile: BackendProfile,
    extra_flags: tuple[str, ...],
) -> None:
    if not extra_flags:
        return
    if isinstance(profile, UnmanagedBackendProfile):
        raise ValueError("extra_flags are not supported for unmanaged backends")

    if profile.lifecycle != LifecycleOwnership.SPAWNED:
        raise ValueError("extra_flags are not supported for managed Ollama")

    for token in extra_flags:
        conflict = token in profile.proxy_owned_flags or any(
            owned.startswith("--") and token.startswith(f"{owned}=")
            for owned in profile.proxy_owned_flags
        )
        if conflict:
            raise ValueError(
                f"extra_flags token {token!r} conflicts with a Forge-owned option"
            )


def _normalize_proxy_config(raw: _RawProxyConfig) -> _NormalizedProxyConfig:
    """Validate and resolve all Proxy configuration without performing I/O."""

    if raw.backend is not None and raw.backend not in proxy_backend_selectors():
        raise ValueError(f"unsupported backend: {raw.backend!r}")

    model = _normalize_model(raw.model)
    backend_api_key = _normalize_api_key(raw.backend_api_key)
    if raw.backend_url is None:
        if raw.backend is None:
            raise ValueError("Provide either backend_url (unmanaged) or backend (managed)")
        profile = find_managed_profile(raw.backend)
        if profile is None:
            raise ValueError(
                f"backend={raw.backend!r} requires backend_url (unmanaged mode)"
            )
        _validate_managed_identity(
            raw.backend,
            profile,
            model=model,
            gguf=raw.gguf,
            model_path=raw.model_path,
        )
    else:
        profile = unmanaged_profile(raw.backend)
        if raw.gguf is not None:
            raise ValueError("unmanaged backends do not accept gguf")
        if raw.model_path is not None:
            raise ValueError("unmanaged backends do not accept model_path")

    budget_mode = _normalize_budget_mode(raw.budget_mode)
    if isinstance(profile, ManagedBackendProfile):
        budget_mode = budget_mode or BudgetMode.BACKEND
        if budget_mode == BudgetMode.MANUAL:
            if raw.budget_tokens is None or raw.budget_tokens <= 0:
                raise ValueError("manual budget_mode requires positive budget_tokens")
        elif raw.budget_tokens is not None:
            raise ValueError("budget_tokens are only accepted with manual budget_mode")
    else:
        if budget_mode is not None:
            raise ValueError("unmanaged backends do not accept budget_mode")
        if raw.budget_tokens is not None and raw.budget_tokens <= 0:
            raise ValueError("unmanaged budget_tokens must be positive")

    if raw.max_retries < 0:
        raise ValueError("max_retries must be nonnegative")
    if raw.max_tool_errors < 0:
        raise ValueError("max_tool_errors must be nonnegative")
    if not math.isfinite(raw.backend_timeout) or raw.backend_timeout <= 0:
        raise ValueError("backend_timeout must be a finite value greater than 0")
    if raw.backend_capability not in profile.family_profile.tool_capabilities:
        if raw.backend_capability == "prompt":
            raise ValueError(
                "backend_capability='prompt' is only supported for "
                "llama-shaped backends"
            )
        supported = ", ".join(sorted(profile.family_profile.tool_capabilities))
        raise ValueError(f"backend_capability must be one of: {supported}")

    extra_flags = tuple(raw.extra_flags or ())
    _validate_extra_flags(profile, extra_flags)

    if isinstance(profile, ManagedBackendProfile):
        backend_port = (
            raw.backend_port if raw.backend_port is not None else profile.default_port
        )
        default_root = f"http://localhost:{profile.default_port}"
        root = (
            replace_authority_port(default_root, raw.backend_port)
            if raw.backend_port is not None
            else default_root
        )
        input_kind = (
            ConnectionInputKind.OLLAMA_DAEMON_ROOT
            if profile.family_profile.client_adapter == ClientAdapter.OLLAMA
            else ConnectionInputKind.PROXY_MOUNT_ROOT
        )
        backend_url = None
    else:
        assert raw.backend_url is not None
        root = (
            replace_authority_port(raw.backend_url, raw.backend_port)
            if raw.backend_port is not None
            else raw.backend_url
        )
        input_kind = (
            ConnectionInputKind.ANTHROPIC_SDK_SERVICE_ROOT
            if profile.family_profile.client_adapter == ClientAdapter.ANTHROPIC
            else ConnectionInputKind.PROXY_MOUNT_ROOT
        )
        backend_url = root
        backend_port = raw.backend_port

    resolved_backend = resolve_backend(profile, root, input_kind)
    return _NormalizedProxyConfig(
        backend_url=backend_url,
        backend=raw.backend,
        profile=profile,
        protocol=profile.family_profile.protocol.value,
        resolved_backend=resolved_backend,
        model=model,
        gguf=raw.gguf,
        model_path=raw.model_path,
        backend_port=backend_port,
        budget_mode=budget_mode,
        budget_tokens=raw.budget_tokens,
        extra_flags=extra_flags,
        host=raw.host,
        port=raw.port,
        serialize=(
            isinstance(profile, ManagedBackendProfile)
            if raw.serialize is None else raw.serialize
        ),
        max_retries=raw.max_retries,
        max_tool_errors=raw.max_tool_errors,
        rescue_enabled=raw.rescue_enabled,
        backend_capability=raw.backend_capability,
        inject_respond_tool=raw.inject_respond_tool,
        backend_timeout=raw.backend_timeout,
        reasoning_replay=validate_reasoning_replay(raw.reasoning_replay),
        backend_api_key=backend_api_key,
    )
