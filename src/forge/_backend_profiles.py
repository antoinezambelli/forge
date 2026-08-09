"""Private, data-only backend family and connection profiles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from forge._endpoint_layouts import BackendOperation, EndpointLayout


class BackendFamily(str, Enum):
    LLAMA_SERVER = "llamaserver"
    LLAMAFILE = "llamafile"
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI_COMPAT = "openai-compatible"
    ANTHROPIC_COMPAT = "anthropic-compatible"


class WireProtocol(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


class ClientAdapter(str, Enum):
    LLAMAFILE = "llamafile"
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI_COMPAT = "openai-compatible"
    ANTHROPIC = "anthropic"


class MetadataFormat(str, Enum):
    LLAMA_PROPERTIES = "llama-properties"
    VLLM_MODELS = "vllm-models"
    ANTHROPIC_MODELS = "anthropic-models"
    NONE = "none"


class LifecycleOwnership(str, Enum):
    SPAWNED = "spawned"
    ATTACHED_DAEMON = "attached-daemon"


class ArtifactIdentity(str, Enum):
    GGUF_PATH = "gguf-path"
    MODEL_PATH = "model-path"
    MODEL_TAG = "model-tag"


@dataclass(frozen=True)
class BackendFamilyProfile:
    family: BackendFamily
    protocol: WireProtocol
    endpoint_layout: EndpointLayout
    client_adapter: ClientAdapter
    metadata_format: MetadataFormat
    operations: frozenset[BackendOperation]
    tool_capabilities: frozenset[str]


@dataclass(frozen=True)
class ManagedBackendProfile:
    selector: str
    family_profile: BackendFamilyProfile
    lifecycle: LifecycleOwnership
    required_identity: ArtifactIdentity
    default_port: int
    proxy_owned_flags: frozenset[str]


@dataclass(frozen=True)
class UnmanagedBackendProfile:
    selector: str
    family_profile: BackendFamilyProfile
    identity_discovery: bool


BackendProfile = ManagedBackendProfile | UnmanagedBackendProfile


@dataclass(frozen=True)
class ModelCatalogEntry:
    """One usable backend model identity and its optional reporting window."""

    served_id: str
    max_model_len: int | None


@dataclass(frozen=True)
class ModelCatalog:
    """Immutable model-catalog facts with exact-identity window lookup."""

    entries: tuple[ModelCatalogEntry, ...]
    first_served_id: str | None = None

    def context_length_for(self, served_id: str) -> int | None:
        """Return a trustworthy positive window for an exact served ID."""

        for entry in self.entries:
            if entry.served_id == served_id:
                return entry.max_model_len
        return None


def parse_vllm_model_catalog(payload: Any) -> ModelCatalog:
    """Parse vLLM facts without mutating routing or context policy."""

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("vLLM /v1/models returned a malformed envelope")
    rows = payload["data"]
    first = rows[0] if rows else None
    first_id = first.get("id") if isinstance(first, dict) else None
    selected_id = first_id if isinstance(first_id, str) and first_id else None

    entries: list[ModelCatalogEntry] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_id = row.get("id")
        if not isinstance(raw_id, str) or not raw_id:
            continue
        raw_window = row.get("max_model_len")
        window = (
            raw_window
            if isinstance(raw_window, int)
            and not isinstance(raw_window, bool)
            and raw_window > 0
            else None
        )
        entries.append(ModelCatalogEntry(raw_id, window))
    return ModelCatalog(tuple(entries), first_served_id=selected_id)


_NATIVE_AND_PROMPT = frozenset({"native", "prompt"})
_NATIVE_ONLY = frozenset({"native"})
_LLAMA_OPERATIONS = frozenset({
    BackendOperation.INFERENCE,
    BackendOperation.MODEL_CATALOG,
    BackendOperation.PROPERTIES,
    BackendOperation.HEALTH,
    BackendOperation.VERSIONED_HEALTH,
    BackendOperation.STARTUP_READINESS,
})
_VLLM_OPERATIONS = frozenset({
    BackendOperation.INFERENCE,
    BackendOperation.MODEL_CATALOG,
    BackendOperation.HEALTH,
    BackendOperation.VERSIONED_HEALTH,
    BackendOperation.STARTUP_READINESS,
})
_GENERIC_OPENAI_OPERATIONS = frozenset({
    BackendOperation.INFERENCE,
    BackendOperation.MODEL_CATALOG,
})


def _family(
    family: BackendFamily,
    protocol: WireProtocol,
    layout: EndpointLayout,
    adapter: ClientAdapter,
    metadata: MetadataFormat,
    operations: frozenset[BackendOperation],
    tool_capabilities: frozenset[str],
) -> BackendFamilyProfile:
    return BackendFamilyProfile(
        family=family,
        protocol=protocol,
        endpoint_layout=layout,
        client_adapter=adapter,
        metadata_format=metadata,
        operations=operations,
        tool_capabilities=tool_capabilities,
    )


_LLAMA_SERVER_FAMILY = _family(
    BackendFamily.LLAMA_SERVER, WireProtocol.OPENAI,
    EndpointLayout.LLAMA_OPENAI, ClientAdapter.LLAMAFILE,
    MetadataFormat.LLAMA_PROPERTIES, _LLAMA_OPERATIONS, _NATIVE_AND_PROMPT,
)
_LLAMAFILE_FAMILY = _family(
    BackendFamily.LLAMAFILE, WireProtocol.OPENAI,
    EndpointLayout.LLAMA_OPENAI, ClientAdapter.LLAMAFILE,
    MetadataFormat.LLAMA_PROPERTIES, _LLAMA_OPERATIONS, _NATIVE_AND_PROMPT,
)
_VLLM_FAMILY = _family(
    BackendFamily.VLLM, WireProtocol.OPENAI,
    EndpointLayout.VLLM_OPENAI, ClientAdapter.VLLM,
    MetadataFormat.VLLM_MODELS, _VLLM_OPERATIONS, _NATIVE_ONLY,
)
_OLLAMA_NATIVE_FAMILY = _family(
    BackendFamily.OLLAMA, WireProtocol.OLLAMA,
    EndpointLayout.OLLAMA_NATIVE, ClientAdapter.OLLAMA,
    MetadataFormat.NONE, frozenset({BackendOperation.INFERENCE}), _NATIVE_ONLY,
)
_OLLAMA_OPENAI_COMPAT_FAMILY = _family(
    BackendFamily.OLLAMA, WireProtocol.OPENAI,
    EndpointLayout.LLAMA_OPENAI, ClientAdapter.LLAMAFILE,
    MetadataFormat.NONE, _GENERIC_OPENAI_OPERATIONS, _NATIVE_ONLY,
)
_OPENAI_COMPAT_FAMILY = _family(
    BackendFamily.OPENAI_COMPAT, WireProtocol.OPENAI,
    EndpointLayout.LLAMA_OPENAI, ClientAdapter.LLAMAFILE,
    MetadataFormat.NONE, _GENERIC_OPENAI_OPERATIONS, _NATIVE_AND_PROMPT,
)
_ANTHROPIC_COMPAT_FAMILY = _family(
    BackendFamily.ANTHROPIC_COMPAT, WireProtocol.ANTHROPIC,
    EndpointLayout.ANTHROPIC_MESSAGES, ClientAdapter.ANTHROPIC,
    MetadataFormat.NONE, frozenset({BackendOperation.INFERENCE}), _NATIVE_ONLY,
)


_MANAGED: dict[str, ManagedBackendProfile] = {
    "llamaserver": ManagedBackendProfile(
        selector="llamaserver",
        family_profile=_LLAMA_SERVER_FAMILY,
        lifecycle=LifecycleOwnership.SPAWNED,
        required_identity=ArtifactIdentity.GGUF_PATH,
        default_port=8080,
        proxy_owned_flags=frozenset({
            "--host", "--port", "-m", "--model", "-c", "--ctx-size",
        }),
    ),
    "llamafile": ManagedBackendProfile(
        selector="llamafile",
        family_profile=_LLAMAFILE_FAMILY,
        lifecycle=LifecycleOwnership.SPAWNED,
        required_identity=ArtifactIdentity.GGUF_PATH,
        default_port=8080,
        proxy_owned_flags=frozenset({
            "--host", "--port", "-m", "--model", "-c", "--ctx-size",
        }),
    ),
    "ollama": ManagedBackendProfile(
        selector="ollama",
        family_profile=_OLLAMA_NATIVE_FAMILY,
        lifecycle=LifecycleOwnership.ATTACHED_DAEMON,
        required_identity=ArtifactIdentity.MODEL_TAG,
        default_port=11434,
        proxy_owned_flags=frozenset(),
    ),
    "vllm": ManagedBackendProfile(
        selector="vllm",
        family_profile=_VLLM_FAMILY,
        lifecycle=LifecycleOwnership.SPAWNED,
        required_identity=ArtifactIdentity.MODEL_PATH,
        default_port=8080,
        proxy_owned_flags=frozenset({"--host", "--port", "--max-model-len"}),
    ),
}


_UNMANAGED: dict[str, UnmanagedBackendProfile] = {
    "openai": UnmanagedBackendProfile(
        selector="openai",
        family_profile=_OPENAI_COMPAT_FAMILY,
        identity_discovery=False,
    ),
    "llamaserver": UnmanagedBackendProfile(
        selector="llamaserver",
        family_profile=_LLAMA_SERVER_FAMILY,
        identity_discovery=False,
    ),
    "llamafile": UnmanagedBackendProfile(
        selector="llamafile",
        family_profile=_LLAMAFILE_FAMILY,
        identity_discovery=False,
    ),
    # External backend="ollama" selects the legacy OpenAI-compatible Ollama
    # surface; managed Ollama uses the native daemon profile above.
    "ollama": UnmanagedBackendProfile(
        selector="ollama",
        family_profile=_OLLAMA_OPENAI_COMPAT_FAMILY,
        identity_discovery=False,
    ),
    "vllm": UnmanagedBackendProfile(
        selector="vllm",
        family_profile=_VLLM_FAMILY,
        identity_discovery=True,
    ),
    "anthropic": UnmanagedBackendProfile(
        selector="anthropic",
        family_profile=_ANTHROPIC_COMPAT_FAMILY,
        identity_discovery=False,
    ),
}


def managed_profile(backend: str) -> ManagedBackendProfile:
    """Return the managed profile for a validated backend name."""

    try:
        return _MANAGED[backend]
    except KeyError as exc:
        raise ValueError(f"unsupported backend: {backend!r}") from exc


def find_managed_profile(backend: str | None) -> ManagedBackendProfile | None:
    """Return a known managed profile without changing validation timing."""

    return _MANAGED.get(backend or "")


def unmanaged_profile(backend: str | None) -> UnmanagedBackendProfile:
    """Return the unmanaged profile for a validated Proxy selector."""

    selector = backend or "openai"
    try:
        return _UNMANAGED[selector]
    except KeyError as exc:
        raise ValueError(f"unsupported backend: {backend!r}") from exc


def proxy_backend_selectors() -> tuple[str, ...]:
    """Return the complete closed selector set for the flat Proxy surface."""

    return tuple(_MANAGED) + tuple(
        selector for selector in _UNMANAGED if selector not in _MANAGED
    )


def all_profiles() -> tuple[BackendProfile, ...]:
    """Return every managed and unmanaged registry profile for tests."""

    return tuple(_MANAGED.values()) + tuple(_UNMANAGED.values())
