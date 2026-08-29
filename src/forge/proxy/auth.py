"""Inbound credential handling for the proxy.

forge forwards at most one credential identity. A matching Authorization +
X-Api-Key pair is one identity represented in two slots: same-protocol requests
preserve the pair, while cross-protocol requests render it once in the target's
canonical slot. Zero credentials is valid for an ungated backend. Distinct
credentials anywhere remain a hard error (Design Principle #1: fail loud).

Only resolved auth headers are forwarded to the backend; the rest of the
inbound header set is NOT forwarded (httpx recomputes transport headers for the
re-serialized body, so there is nothing to strip).
"""

from __future__ import annotations

from collections.abc import Mapping

from forge.clients.base import (
    BEARER_PREFIX,
    _auth_credential_entries,
    _equivalent_dual_auth_token,
    count_auth_credentials,
)
from forge.errors import MultipleCredentialsError

# Marker the proxy's header reader injects when a single auth header NAME
# appears more than once on the inbound request. A plain header dict collapses
# duplicates to last-wins, which would silently pick a credential winner — the
# marker forces the same fail-loud refusal as two distinct auth headers.
DUPLICATE_AUTH_MARKER = "x-forge-duplicate-auth"


def extract_inbound_credentials(
    headers: Mapping[str, str] | None,
) -> dict[str, str]:
    """Return the zero, one, or equivalent-two inbound auth headers.

    Names are lowercased. Empty, whitespace-only, and scheme-only values are
    absent. A populated Authorization + X-Api-Key pair is retained only when
    both slots carry the same effective token. Distinct credentials and
    repeated same-name headers raise ``MultipleCredentialsError``.
    """
    headers = headers or {}
    if headers.get(DUPLICATE_AUTH_MARKER):
        raise MultipleCredentialsError(
            "inbound request carries the same auth header more than once"
        )
    entries = _auth_credential_entries(headers)
    if count_auth_credentials(headers) > 1:
        slots = ", ".join(sorted(slot for slot, _, _ in entries))
        raise MultipleCredentialsError(f"inbound request carries auth headers: {slots}")
    return {slot: value for slot, value, _ in entries}


def relocate_credential(
    slot: str,
    value: str,
    source_protocol: str,
    target_protocol: str,
) -> dict[str, str]:
    """Place the one credential in the target protocol's canonical auth slot.

    forge never inspects the credential's secret value; it only decides which
    slot it belongs in for the target, and (cross-protocol) normalizes the
    scheme so the token lands correctly.

    - Same protocol both ends: forwarded verbatim (the canonical slot already
      matches; preserves non-Bearer schemes).
    - Cross-protocol: normalize to the raw token (strip a leading ``Bearer ``
      from an ``Authorization`` value; ``x-api-key`` is already raw), then
      write the target's canonical slot — Anthropic ``x-api-key``, OpenAI-wire
      ``Authorization: Bearer <token>``.

    The one documented limitation (design §4): an Anthropic OAuth token (which
    must ride ``Authorization: Bearer``) pushed through the OpenAI endpoint to
    an Anthropic backend is relocated to ``x-api-key`` and rejected by
    Anthropic. Coherent setups never hit this — OAuth callers use the Anthropic
    endpoint (same-protocol, verbatim).
    """
    if source_protocol == target_protocol:
        return {slot: value}

    if slot == "authorization" and value[: len(BEARER_PREFIX)].lower() == BEARER_PREFIX:
        token = value[len(BEARER_PREFIX):].strip()
    else:
        token = value

    if target_protocol == "anthropic":
        return {"x-api-key": token}
    return {"Authorization": f"Bearer {token}"}


def relocate_credentials(
    auth_headers: Mapping[str, str],
    source_protocol: str,
    target_protocol: str,
) -> dict[str, str]:
    """Preserve one identity on the same protocol or map it across protocols."""
    if source_protocol == target_protocol:
        return dict(auth_headers)

    if len(auth_headers) == 1:
        slot, value = next(iter(auth_headers.items()))
        return relocate_credential(slot, value, source_protocol, target_protocol)

    token = _equivalent_dual_auth_token(auth_headers)
    if token is None:
        raise MultipleCredentialsError("inbound request carries distinct auth headers")
    if target_protocol == "anthropic":
        return {"x-api-key": token}
    return {"Authorization": f"Bearer {token}"}


def resolve_inbound_credential(
    headers: Mapping[str, str] | None,
    source_protocol: str,
    target_protocol: str,
    backend_api_key_present: bool,
) -> dict[str, str] | None:
    """Resolve the per-call credential header to forward, or None.

    Extracts one inbound identity, enforces the static-source conflict, then
    preserves an equivalent pair on the same protocol or maps the identity to
    the target protocol. Returns None when no inbound identity is present.
    """
    auth_headers = extract_inbound_credentials(headers)
    if not auth_headers:
        return None
    if backend_api_key_present:
        raise MultipleCredentialsError(
            "inbound auth header + --backend-api-key (static backend credential)"
        )
    return relocate_credentials(auth_headers, source_protocol, target_protocol)


def _resolve_metadata_credential(
    headers: Mapping[str, str] | None,
    target_protocol: str,
    backend_api_key: str | None,
    source_protocol: str | None = None,
) -> dict[str, str] | None:
    """Resolve the one credential for a protocol-neutral metadata request.

    A known source protocol uses the inference preserve/map matrix. Raw
    forwarded metadata GETs are protocol-neutral and preserve an equivalent
    pair; their existing single-header relocation remains unchanged. Static
    keys are placed directly in the target's canonical slot.
    """
    auth_headers = extract_inbound_credentials(headers)
    if auth_headers and backend_api_key is not None:
        raise MultipleCredentialsError(
            "inbound auth header + --backend-api-key (static backend credential)"
        )
    if not auth_headers:
        if backend_api_key is None:
            return None
        if target_protocol == "anthropic":
            return {"x-api-key": backend_api_key}
        return {"Authorization": f"Bearer {backend_api_key}"}

    if source_protocol is not None:
        return relocate_credentials(auth_headers, source_protocol, target_protocol)
    if len(auth_headers) > 1:
        return dict(auth_headers)

    slot, value = next(iter(auth_headers.items()))
    if target_protocol not in {"openai", "anthropic", "ollama"}:
        return {slot: value}
    source_protocol = "openai" if slot == "authorization" else "anthropic"
    return relocate_credential(slot, value, source_protocol, target_protocol)
