"""Focused coverage for private backend profiles and endpoint resolution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from forge._backend_profiles import (
    BackendFamilyProfile,
    ClientAdapter,
    LifecycleOwnership,
    ManagedBackendProfile,
    MetadataFormat,
    ModelCatalogEntry,
    parse_vllm_model_catalog,
    UnmanagedBackendProfile,
    all_profiles,
    managed_profile,
    unmanaged_profile,
)
from forge._endpoint_layouts import (
    BackendOperation,
    ConnectionInputKind,
    EndpointLayout,
    append_mount_path,
    layout_operations,
    normalize_connection,
    normalize_proxy_mount_root,
    replace_authority_port,
    resolve_endpoint,
)
from forge._resolved_backend import resolve_backend


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://host:8080", "http://host:8080"),
        ("http://host:8080/", "http://host:8080"),
        ("http://host:8080/v1", "http://host:8080"),
        ("http://host:8080/v1/", "http://host:8080"),
        (
            "https://gateway.example/team/model/v1?token=x#frag",
            "https://gateway.example/team/model?token=x#frag",
        ),
        ("https://gateway.example/team/v11", "https://gateway.example/team/v11"),
    ],
)
def test_proxy_mount_normalization(url: str, expected: str) -> None:
    assert normalize_proxy_mount_root(url) == expected


@pytest.mark.parametrize(
    ("layout", "operation", "expected_path"),
    [
        (EndpointLayout.LLAMA_OPENAI, BackendOperation.INFERENCE, "/edge/a/v1/chat/completions"),
        (EndpointLayout.LLAMA_OPENAI, BackendOperation.MODEL_CATALOG, "/edge/a/v1/models"),
        (EndpointLayout.LLAMA_OPENAI, BackendOperation.PROPERTIES, "/edge/a/props"),
        (EndpointLayout.LLAMA_OPENAI, BackendOperation.HEALTH, "/edge/a/health"),
        (EndpointLayout.LLAMA_OPENAI, BackendOperation.VERSIONED_HEALTH, "/edge/a/v1/health"),
        (EndpointLayout.LLAMA_OPENAI, BackendOperation.STARTUP_READINESS, "/edge/a/props"),
        (EndpointLayout.VLLM_OPENAI, BackendOperation.INFERENCE, "/edge/a/v1/chat/completions"),
        (EndpointLayout.VLLM_OPENAI, BackendOperation.MODEL_CATALOG, "/edge/a/v1/models"),
        (EndpointLayout.VLLM_OPENAI, BackendOperation.HEALTH, "/edge/a/health"),
        (EndpointLayout.VLLM_OPENAI, BackendOperation.VERSIONED_HEALTH, "/edge/a/v1/health"),
        (EndpointLayout.VLLM_OPENAI, BackendOperation.STARTUP_READINESS, "/edge/a/v1/models"),
        (EndpointLayout.OPENAI_COMPAT, BackendOperation.INFERENCE, "/edge/a/v1/chat/completions"),
        (EndpointLayout.OPENAI_COMPAT, BackendOperation.MODEL_CATALOG, "/edge/a/v1/models"),
        (EndpointLayout.OPENAI_COMPAT, BackendOperation.HEALTH, "/edge/a/health"),
        (EndpointLayout.OPENAI_COMPAT, BackendOperation.VERSIONED_HEALTH, "/edge/a/v1/health"),
        (EndpointLayout.OLLAMA_NATIVE, BackendOperation.INFERENCE, "/edge/a/api/chat"),
        (EndpointLayout.OLLAMA_NATIVE, BackendOperation.HEALTH, "/edge/a/health"),
        (EndpointLayout.OLLAMA_NATIVE, BackendOperation.VERSIONED_HEALTH, "/edge/a/v1/health"),
        (EndpointLayout.ANTHROPIC_MESSAGES, BackendOperation.INFERENCE, "/edge/a/v1/messages"),
        (EndpointLayout.ANTHROPIC_MESSAGES, BackendOperation.HEALTH, "/edge/a/health"),
        (EndpointLayout.ANTHROPIC_MESSAGES, BackendOperation.VERSIONED_HEALTH, "/edge/a/v1/health"),
    ],
)
def test_every_known_proxy_operation_preserves_nested_prefix(
    layout: EndpointLayout,
    operation: BackendOperation,
    expected_path: str,
) -> None:
    connection = normalize_connection(
        "https://gateway.example/edge/a/v1",
        ConnectionInputKind.PROXY_MOUNT_ROOT,
    )
    assert resolve_endpoint(layout, operation, connection) == (
        f"https://gateway.example{expected_path}"
    )


def test_direct_openai_api_base_stays_literal_for_inference() -> None:
    connection = normalize_connection(
        "https://host.example/deploy/custom-api",
        ConnectionInputKind.OPENAI_API_BASE,
    )
    assert resolve_endpoint(
        EndpointLayout.LLAMA_OPENAI, BackendOperation.INFERENCE, connection,
    ) == "https://host.example/deploy/custom-api/chat/completions"
    assert resolve_endpoint(
        EndpointLayout.LLAMA_OPENAI, BackendOperation.PROPERTIES, connection,
    ) == "https://host.example/deploy/custom-api/props"


def test_direct_llama_v1_base_only_drops_v1_for_properties() -> None:
    connection = normalize_connection(
        "https://host.example/deploy/v1/",
        ConnectionInputKind.OPENAI_API_BASE,
    )
    assert resolve_endpoint(
        EndpointLayout.LLAMA_OPENAI, BackendOperation.INFERENCE, connection,
    ) == "https://host.example/deploy/v1/chat/completions"
    assert resolve_endpoint(
        EndpointLayout.LLAMA_OPENAI, BackendOperation.PROPERTIES, connection,
    ) == "https://host.example/deploy/props"


def test_direct_ollama_and_anthropic_roots_keep_their_meaning() -> None:
    ollama = normalize_connection(
        "http://daemon:11434/root/", ConnectionInputKind.OLLAMA_DAEMON_ROOT,
    )
    anthropic = normalize_connection(
        "https://anthropic.example/service/",
        ConnectionInputKind.ANTHROPIC_SDK_SERVICE_ROOT,
    )
    assert resolve_endpoint(
        EndpointLayout.OLLAMA_NATIVE, BackendOperation.INFERENCE, ollama,
    ) == "http://daemon:11434/root/api/chat"
    assert resolve_endpoint(
        EndpointLayout.ANTHROPIC_MESSAGES, BackendOperation.INFERENCE, anthropic,
    ) == "https://anthropic.example/service/v1/messages"


def test_vllm_catalog_preserves_opaque_model_identities_verbatim() -> None:
    catalog = parse_vllm_model_catalog({
        "data": [
            {"id": "  routed model  ", "max_model_len": 32768},
            {"id": "   ", "max_model_len": 8192},
            {"id": "", "max_model_len": 4096},
        ],
    })

    assert catalog.first_served_id == "  routed model  "
    assert catalog.entries == (
        ModelCatalogEntry("  routed model  ", 32768),
        ModelCatalogEntry("   ", 8192),
    )
    assert catalog.context_length_for("  routed model  ") == 32768


@pytest.mark.parametrize(
    ("url", "port", "expected"),
    [
        (
            "https://user:pw@gateway.example:8443/team/v1?q=1#f",
            9443,
            "https://user:pw@gateway.example:9443/team/v1?q=1#f",
        ),
        (
            "http://gateway.example/team?x=1",
            8080,
            "http://gateway.example:8080/team?x=1",
        ),
        ("http://[::1]:8000/prefix", 9000, "http://[::1]:9000/prefix"),
    ],
)
def test_replace_authority_port_preserves_other_components(
    url: str, port: int, expected: str,
) -> None:
    assert replace_authority_port(url, port) == expected


def test_profile_types_match_the_closed_runtime_shape() -> None:
    assert {field.name for field in fields(BackendFamilyProfile)} == {
        "family", "protocol", "endpoint_layout", "client_adapter",
        "metadata_format", "operations", "tool_capabilities",
    }
    assert {field.name for field in fields(ManagedBackendProfile)} == {
        "selector", "family_profile", "lifecycle", "required_identity",
        "default_port", "proxy_owned_flags",
    }
    assert {field.name for field in fields(UnmanagedBackendProfile)} == {
        "selector", "family_profile", "identity_discovery",
    }


def test_profile_registry_covers_all_current_selectors_and_layouts() -> None:
    profiles = all_profiles()
    assert sum(isinstance(profile, ManagedBackendProfile) for profile in profiles) == 4
    assert sum(isinstance(profile, UnmanagedBackendProfile) for profile in profiles) == 6
    assert {profile.selector for profile in profiles} == {
        "openai", "anthropic", "llamaserver", "llamafile", "ollama", "vllm",
    }
    for profile in profiles:
        assert profile.family_profile.operations <= layout_operations(
            profile.family_profile.endpoint_layout,
        )


@pytest.mark.parametrize("selector", ["llamaserver", "llamafile", "vllm"])
def test_identical_managed_and_unmanaged_surfaces_share_one_family(
    selector: str,
) -> None:
    assert managed_profile(selector).family_profile is unmanaged_profile(selector).family_profile


def test_managed_and_unmanaged_ollama_are_intentionally_distinct() -> None:
    managed = managed_profile("ollama")
    unmanaged = unmanaged_profile("ollama")
    assert managed.family_profile is not unmanaged.family_profile
    assert managed.family_profile.client_adapter == ClientAdapter.OLLAMA
    assert managed.lifecycle == LifecycleOwnership.ATTACHED_DAEMON
    assert managed.default_port == 11434
    assert unmanaged.family_profile.client_adapter == ClientAdapter.LLAMAFILE
    assert unmanaged.selector == "ollama"
    assert unmanaged.identity_discovery is False


def test_compatibility_adapters_do_not_imply_intrinsic_metadata() -> None:
    generic = unmanaged_profile(None)
    ollama_compat = unmanaged_profile("ollama")
    anthropic = unmanaged_profile("anthropic")
    for profile in (generic, ollama_compat, anthropic):
        assert profile.family_profile.metadata_format == MetadataFormat.NONE
        assert BackendOperation.PROPERTIES not in profile.family_profile.operations


def test_concrete_unmanaged_profiles_keep_contractual_metadata() -> None:
    llama = unmanaged_profile("llamaserver")
    vllm = unmanaged_profile("vllm")
    assert llama.family_profile.metadata_format == MetadataFormat.LLAMA_PROPERTIES
    assert BackendOperation.PROPERTIES in llama.family_profile.operations
    assert llama.identity_discovery is False
    assert vllm.family_profile.metadata_format == MetadataFormat.VLLM_MODELS
    assert BackendOperation.MODEL_CATALOG in vllm.family_profile.operations
    assert vllm.identity_discovery is True


def test_anthropic_profile_selects_its_protocol_adapter() -> None:
    profile = unmanaged_profile("anthropic")
    assert profile.selector == "anthropic"
    assert profile.family_profile.client_adapter == ClientAdapter.ANTHROPIC


def test_resolved_backend_is_immutable_and_gates_unsupported_operations() -> None:
    resolved = resolve_backend(
        unmanaged_profile("vllm"),
        "https://gateway.example/deploy/v1",
        ConnectionInputKind.PROXY_MOUNT_ROOT,
    )
    assert resolved.connection.mount_root == "https://gateway.example/deploy"
    assert resolved.adapter_base_url == "https://gateway.example/deploy/v1"
    assert resolved.address(BackendOperation.INFERENCE).endswith(
        "/deploy/v1/chat/completions"
    )
    with pytest.raises(ValueError, match="does not support"):
        resolved.address(BackendOperation.PROPERTIES)
    with pytest.raises(FrozenInstanceError):
        resolved.adapter_base_url = "http://mutated"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("backend", "readiness"),
    [("llamaserver", "/deploy/props"), ("vllm", "/deploy/v1/models")],
)
def test_resolved_health_is_distinct_from_startup_readiness(
    backend: str, readiness: str,
) -> None:
    resolved = resolve_backend(
        managed_profile(backend),
        "https://gateway.example/deploy",
        ConnectionInputKind.PROXY_MOUNT_ROOT,
    )
    assert resolved.address(BackendOperation.HEALTH) == (
        "https://gateway.example/deploy/health"
    )
    assert resolved.address(BackendOperation.VERSIONED_HEALTH) == (
        "https://gateway.example/deploy/v1/health"
    )
    assert resolved.address(BackendOperation.STARTUP_READINESS) == (
        f"https://gateway.example{readiness}"
    )


def test_generic_profile_gates_llama_fallback_metadata() -> None:
    resolved = resolve_backend(
        unmanaged_profile(None),
        "https://gateway.example/deploy",
        ConnectionInputKind.PROXY_MOUNT_ROOT,
    )
    with pytest.raises(ValueError, match="does not support"):
        resolved.address(BackendOperation.PROPERTIES)


def test_passthrough_path_append_is_independent_of_semantic_capabilities() -> None:
    assert append_mount_path(
        "https://gateway.example/deploy", "/approved/future/path",
    ) == "https://gateway.example/deploy/approved/future/path"
