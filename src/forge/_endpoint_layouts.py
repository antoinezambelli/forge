"""Private backend endpoint address book.

This module owns URL topology only.  It deliberately knows nothing about
backend capabilities, payloads, response parsing, lifecycle, or validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from urllib.parse import SplitResult, urlsplit, urlunsplit


class EndpointLayout(str, Enum):
    """Known backend endpoint topologies."""

    LLAMA_OPENAI = "llama-openai"
    VLLM_OPENAI = "vllm-openai"
    OPENAI_COMPAT = "openai-compatible"
    OLLAMA_NATIVE = "ollama-native"
    ANTHROPIC_MESSAGES = "anthropic-messages"


class BackendOperation(str, Enum):
    """Semantic operations whose addresses forge may resolve internally."""

    INFERENCE = "inference"
    MODEL_CATALOG = "model-catalog"
    PROPERTIES = "properties"
    HEALTH = "health"
    VERSIONED_HEALTH = "versioned-health"
    STARTUP_READINESS = "startup-readiness"


class ConnectionInputKind(str, Enum):
    """Meaning of the URL supplied to endpoint resolution."""

    PROXY_MOUNT_ROOT = "proxy-mount-root"
    OPENAI_API_BASE = "openai-api-base"
    OLLAMA_DAEMON_ROOT = "ollama-daemon-root"
    ANTHROPIC_SDK_SERVICE_ROOT = "anthropic-sdk-service-root"


@dataclass(frozen=True)
class NormalizedConnection:
    """A parsed connection with both caller input and normalized mount root."""

    input_url: str
    input_kind: ConnectionInputKind
    mount_root: str


_HEALTH_PATHS = {
    BackendOperation.HEALTH: "health",
    BackendOperation.VERSIONED_HEALTH: "v1/health",
}

_LAYOUT_PATHS: dict[EndpointLayout, dict[BackendOperation, str]] = {
    EndpointLayout.LLAMA_OPENAI: {
        BackendOperation.INFERENCE: "v1/chat/completions",
        BackendOperation.MODEL_CATALOG: "v1/models",
        BackendOperation.PROPERTIES: "props",
        BackendOperation.STARTUP_READINESS: "props",
        **_HEALTH_PATHS,
    },
    EndpointLayout.VLLM_OPENAI: {
        BackendOperation.INFERENCE: "v1/chat/completions",
        BackendOperation.MODEL_CATALOG: "v1/models",
        BackendOperation.STARTUP_READINESS: "v1/models",
        **_HEALTH_PATHS,
    },
    EndpointLayout.OPENAI_COMPAT: {
        BackendOperation.INFERENCE: "v1/chat/completions",
        BackendOperation.MODEL_CATALOG: "v1/models",
        **_HEALTH_PATHS,
    },
    EndpointLayout.OLLAMA_NATIVE: {
        BackendOperation.INFERENCE: "api/chat",
        **_HEALTH_PATHS,
    },
    EndpointLayout.ANTHROPIC_MESSAGES: {
        BackendOperation.INFERENCE: "v1/messages",
        **_HEALTH_PATHS,
    },
}

_OPENAI_LAYOUTS = frozenset({
    EndpointLayout.LLAMA_OPENAI,
    EndpointLayout.VLLM_OPENAI,
    EndpointLayout.OPENAI_COMPAT,
})


def _replace_path(parts: SplitResult, path: str) -> str:
    return urlunsplit(parts._replace(path=path))


def _trim_terminal_slash(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    return _replace_path(parts, path)


def normalize_proxy_mount_root(url: str) -> str:
    """Normalize a Proxy mount, removing only a terminal ``/v1`` segment.

    Scheme, authority, deployment prefix, query, fragment, and an omitted port
    are preserved.  A terminal slash is treated as a path separator so roots,
    ``/v1``, and ``/v1/`` resolve consistently.
    """

    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    if path == "/v1":
        path = ""
    elif path.endswith("/v1"):
        path = path[:-3]
    return _replace_path(parts, path)


def replace_authority_port(url: str, port: int) -> str:
    """Replace or add only the URL authority port."""

    parts = urlsplit(url)
    # Accessing ``port`` validates an existing explicit port while leaving the
    # raw user-info and host spelling available for exact reconstruction.
    existing_port = parts.port
    userinfo, separator, host_port = parts.netloc.rpartition("@")
    prefix = f"{userinfo}@" if separator else ""
    if host_port.startswith("["):
        close = host_port.find("]")
        host = host_port[: close + 1]
    elif existing_port is not None:
        host = host_port.rsplit(":", 1)[0]
    else:
        host = host_port
    return urlunsplit(parts._replace(netloc=f"{prefix}{host}:{port}"))


def normalize_connection(url: str, input_kind: ConnectionInputKind) -> NormalizedConnection:
    """Normalize the mount root appropriate to a typed connection input."""

    if input_kind in (
        ConnectionInputKind.PROXY_MOUNT_ROOT,
        ConnectionInputKind.OPENAI_API_BASE,
    ):
        mount_root = normalize_proxy_mount_root(url)
    else:
        mount_root = _trim_terminal_slash(url)
    return NormalizedConnection(url, input_kind, mount_root)


def append_mount_path(mount_root: str, path: str) -> str:
    """Append a relative path without allowing it to discard a prefix."""

    parts = urlsplit(mount_root)
    base_path = parts.path.rstrip("/")
    suffix = path.strip("/")
    joined = f"{base_path}/{suffix}" if suffix else base_path
    if not joined.startswith("/"):
        joined = f"/{joined}"
    return _replace_path(parts, joined)


def client_base_url(
    layout: EndpointLayout,
    connection: NormalizedConnection,
) -> str:
    """Return the constructor-compatible base for a layout's client adapter."""

    if connection.input_kind == ConnectionInputKind.OPENAI_API_BASE:
        return _trim_terminal_slash(connection.input_url)
    if connection.input_kind == ConnectionInputKind.ANTHROPIC_SDK_SERVICE_ROOT:
        # The Anthropic SDK owns service-root semantics.  Preserve it verbatim.
        return connection.input_url
    if layout in _OPENAI_LAYOUTS:
        return append_mount_path(connection.mount_root, "v1")
    return connection.mount_root


def resolve_endpoint(
    layout: EndpointLayout,
    operation: BackendOperation,
    connection: NormalizedConnection,
) -> str:
    """Resolve a known operation from a typed root and endpoint layout."""

    try:
        path = _LAYOUT_PATHS[layout][operation]
    except KeyError as exc:
        raise KeyError(
            f"layout {layout.value!r} has no {operation.value!r} operation"
        ) from exc

    if (
        connection.input_kind == ConnectionInputKind.OPENAI_API_BASE
        and layout in _OPENAI_LAYOUTS
        and path.startswith("v1/")
    ):
        path = path[3:]
        base = _trim_terminal_slash(connection.input_url)
    elif connection.input_kind == ConnectionInputKind.ANTHROPIC_SDK_SERVICE_ROOT:
        base = _trim_terminal_slash(connection.input_url)
    else:
        base = connection.mount_root
    return append_mount_path(base, path)


def layout_operations(layout: EndpointLayout) -> frozenset[BackendOperation]:
    """Return the operations for which a layout defines an address."""

    return frozenset(_LAYOUT_PATHS[layout])
