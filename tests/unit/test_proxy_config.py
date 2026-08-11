"""Focused coverage for pure Proxy configuration normalization."""

from __future__ import annotations

import builtins
import socket
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from forge._backend_profiles import (
    ClientAdapter,
    ManagedBackendProfile,
    UnmanagedBackendProfile,
    proxy_backend_selectors,
)
from forge.proxy.proxy import ProxyServer
from forge.server import BudgetMode


MANAGED_IDENTITIES = {
    "llamaserver": {"gguf": "m.gguf"},
    "llamafile": {"gguf": "m.gguf"},
    "ollama": {"model": "tag"},
    "vllm": {"model_path": "/models/m"},
}


@pytest.mark.parametrize(("backend", "identity"), MANAGED_IDENTITIES.items())
def test_managed_selectors_normalize_with_profile_defaults(
    backend: str, identity: dict[str, str],
) -> None:
    proxy = ProxyServer(backend=backend, **identity)
    assert isinstance(proxy._profile, ManagedBackendProfile)
    assert proxy._backend_url is None
    assert proxy._backend_port == (11434 if backend == "ollama" else 8080)
    assert proxy._budget_mode == BudgetMode.BACKEND
    assert proxy._serialize is True
    assert proxy._backend_protocol == proxy._profile.family_profile.protocol.value


@pytest.mark.parametrize(
    ("backend", "selector", "adapter"),
    [
        (None, "openai", ClientAdapter.LLAMAFILE),
        ("openai", "openai", ClientAdapter.LLAMAFILE),
        ("anthropic", "anthropic", ClientAdapter.ANTHROPIC),
        ("llamaserver", "llamaserver", ClientAdapter.LLAMAFILE),
        ("llamafile", "llamafile", ClientAdapter.LLAMAFILE),
        ("ollama", "ollama", ClientAdapter.LLAMAFILE),
        ("vllm", "vllm", ClientAdapter.VLLM),
    ],
)
def test_unmanaged_selector_matrix(
    backend: str | None,
    selector: str,
    adapter: ClientAdapter,
) -> None:
    proxy = ProxyServer(backend_url="https://gateway.example/root/v1", backend=backend)
    assert proxy._backend_url == "https://gateway.example/root/v1"
    assert isinstance(proxy._profile, UnmanagedBackendProfile)
    assert proxy._profile.selector == selector
    assert proxy._profile.family_profile.client_adapter == adapter
    assert proxy._serialize is False


def test_explicit_whitespace_model_remains_opaque() -> None:
    proxy = ProxyServer(
        backend_url="http://host",
        backend="vllm",
        model="   ",
    )

    assert proxy._model == "   "


def test_proxy_selector_accessor_is_complete_and_closed() -> None:
    assert proxy_backend_selectors() == (
        "llamaserver", "llamafile", "ollama", "vllm", "openai", "anthropic",
    )


def test_managed_identity_matrix_rejects_irrelevant_fields() -> None:
    cases = [
        ("llamaserver", {"gguf": "m.gguf"}, {"model": "tag"}),
        ("llamaserver", {"gguf": "m.gguf"}, {"model_path": "/m"}),
        ("llamafile", {"gguf": "m.gguf"}, {"model": "tag"}),
        ("ollama", {"model": "tag"}, {"gguf": "m.gguf"}),
        ("ollama", {"model": "tag"}, {"model_path": "/m"}),
        ("vllm", {"model_path": "/m"}, {"model": "tag"}),
        ("vllm", {"model_path": "/m"}, {"gguf": "m.gguf"}),
    ]

    for backend, identity, forbidden in cases:
        case = f"{backend}:{next(iter(forbidden))}"
        try:
            ProxyServer(backend=backend, **identity, **forbidden)
        except ValueError as exc:
            assert "does not accept" in str(exc), case
        else:
            pytest.fail(f"{case} was accepted")


@pytest.mark.parametrize("field", [{"gguf": "m.gguf"}, {"model_path": "/m"}])
def test_unmanaged_rejects_artifact_identity(field: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="unmanaged backends do not accept"):
        ProxyServer(backend_url="http://host", **field)


@pytest.mark.parametrize("budget_mode", list(BudgetMode))
def test_all_managed_context_modes_are_accepted(budget_mode: BudgetMode) -> None:
    kwargs = {"budget_tokens": 4096} if budget_mode == BudgetMode.MANUAL else {}
    proxy = ProxyServer(
        backend="llamaserver", gguf="m.gguf", budget_mode=budget_mode, **kwargs,
    )
    assert proxy._budget_mode == budget_mode


@pytest.mark.parametrize("budget_tokens", [None, 0, -1])
def test_managed_manual_requires_positive_tokens(budget_tokens: int | None) -> None:
    with pytest.raises(ValueError, match="requires positive"):
        ProxyServer(
            backend="llamaserver",
            gguf="m.gguf",
            budget_mode=BudgetMode.MANUAL,
            budget_tokens=budget_tokens,
        )


def test_managed_nonmanual_rejects_budget_tokens() -> None:
    with pytest.raises(ValueError, match="only accepted with manual"):
        ProxyServer(backend="ollama", model="tag", budget_tokens=4096)


def test_unmanaged_rejects_explicit_budget_mode_even_backend() -> None:
    with pytest.raises(ValueError, match="do not accept budget_mode"):
        ProxyServer(backend_url="http://host", budget_mode=BudgetMode.BACKEND)


@pytest.mark.parametrize("budget_tokens", [0, -1])
def test_unmanaged_reporting_denominator_must_be_positive(budget_tokens: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ProxyServer(backend_url="http://host", budget_tokens=budget_tokens)


def test_unmanaged_omitted_port_preserves_url_authority() -> None:
    proxy = ProxyServer(backend_url="https://gateway.example/team/backend/v1")
    assert proxy._backend_url == "https://gateway.example/team/backend/v1"
    assert proxy._resolved_backend.adapter_base_url == (
        "https://gateway.example/team/backend/v1"
    )


def test_unmanaged_explicit_port_preserves_prefix() -> None:
    proxy = ProxyServer(
        backend_url="https://gateway.example/team/backend/v1", backend_port=9443,
    )
    assert proxy._backend_url == "https://gateway.example:9443/team/backend/v1"
    assert proxy._resolved_backend.adapter_base_url == (
        "https://gateway.example:9443/team/backend/v1"
    )


@pytest.mark.parametrize("serialize", [None, True, False])
def test_serialization_override_resolves_after_mode(serialize: bool | None) -> None:
    managed = ProxyServer(backend="ollama", model="tag", serialize=serialize)
    unmanaged = ProxyServer(backend_url="http://host", serialize=serialize)
    assert managed._serialize is (True if serialize is None else serialize)
    assert unmanaged._serialize is (False if serialize is None else serialize)


@pytest.mark.parametrize("field", ["max_retries", "max_tool_errors"])
def test_nonnegative_retry_controls(field: str) -> None:
    assert getattr(ProxyServer(backend_url="http://host", **{field: 0}), f"_{field}") == 0
    with pytest.raises(ValueError, match="nonnegative"):
        ProxyServer(backend_url="http://host", **{field: -1})


def test_profile_owned_extra_flag_conflicts() -> None:
    cases = [
        (backend, {"gguf": "m.gguf"}, token)
        for backend in ("llamaserver", "llamafile")
        for token in (
            "--host", "--host=0.0.0.0", "--port", "--port=9000",
            "-m", "--model", "--model=other.gguf",
            "-c", "--ctx-size", "--ctx-size=8192",
        )
    ] + [
        ("vllm", {"model_path": "/m"}, token)
        for token in (
            "--host", "--host=0.0.0.0", "--port", "--port=9000",
            "--max-model-len", "--max-model-len=8192",
        )
    ]

    for backend, identity, token in cases:
        case = f"{backend}:{token}"
        try:
            ProxyServer(backend=backend, extra_flags=[token], **identity)
        except ValueError as exc:
            assert "Forge-owned" in str(exc), case
        else:
            pytest.fail(f"{case} was accepted")


def test_short_assignment_and_backend_tuning_flags_are_not_overparsed() -> None:
    flags = ["-m=alternate", "-c=4096", "--cache-type-k", "q8_0", "bare"]
    proxy = ProxyServer(backend="llamaserver", gguf="m.gguf", extra_flags=flags)
    assert proxy._extra_flags == flags


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "ollama", "model": "tag", "extra_flags": ["--verbose"]},
        {"backend_url": "http://host", "extra_flags": ["--verbose"]},
    ],
)
def test_nonspawned_extra_flags_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="extra_flags are not supported"):
        ProxyServer(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "ollama", "model": "tag", "extra_flags": []},
        {"backend_url": "http://host", "extra_flags": []},
    ],
)
def test_explicit_empty_extra_flags_are_absent(kwargs: dict[str, object]) -> None:
    assert ProxyServer(**kwargs)._extra_flags is None


def test_invalid_construction_precedes_runtime_and_environment_side_effects() -> None:
    def fired(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("side effect fired")

    with (
        patch.object(threading, "Thread", side_effect=fired),
        patch.object(subprocess, "Popen", side_effect=fired),
        patch.object(socket, "socket", side_effect=fired),
        patch.object(httpx, "Client", side_effect=fired),
        patch.object(httpx, "AsyncClient", side_effect=fired),
        patch.object(Path, "exists", side_effect=fired),
        patch.object(builtins, "open", side_effect=fired),
        patch("forge.proxy.proxy.HTTPServer", side_effect=fired),
        patch("forge.proxy.proxy.LlamafileClient", side_effect=fired),
        patch("forge.proxy.proxy.OllamaClient", side_effect=fired),
        patch("forge.proxy.proxy.VLLMClient", side_effect=fired),
        pytest.raises(ValueError, match="requires gguf"),
    ):
        ProxyServer(backend="llamaserver")
