"""Inbound credential handling for the proxy (forge v0.8.0).

forge forwards exactly ONE credential to the backend, in the backend's native
auth header. The proxy either relocates a single inbound auth header into the
target protocol's canonical slot, or (when ``--backend-api-key`` is configured)
uses that static credential — never both. Two credentials anywhere is a hard
error (Design Principle #1: fail loud, no silent merge).

Only the single relocated credential is forwarded to the backend; the rest of
the inbound header set is NOT forwarded (httpx recomputes transport headers for
the re-serialized body, so there is nothing to strip).
"""

from __future__ import annotations

from collections.abc import Mapping

from forge.clients.base import AUTH_HEADER_NAMES, BEARER_PREFIX, auth_credential_token
from forge.errors import MultipleCredentialsError

# Marker the proxy's header reader injects when a single auth header NAME
# appears more than once on the inbound request. A plain header dict collapses
# duplicates to last-wins, which would silently pick a credential winner — the
# marker forces the same fail-loud refusal as two distinct auth headers.
DUPLICATE_AUTH_MARKER = "x-forge-duplicate-auth"


def extract_inbound_credential(
    headers: Mapping[str, str] | None,
) -> tuple[str | None, str | None]:
    """Return ``(slot, value)`` for the single inbound auth header.

    ``slot`` is the lowercased header name (``authorization`` or ``x-api-key``).
    Returns ``(None, None)`` when no auth header carries a credential — an empty,
    whitespace-only, or scheme-only value (e.g. ``Bearer `` with no token) is
    treated as absent, so it fails loud downstream rather than forwarding an
    empty credential. Raises ``MultipleCredentialsError`` if the request carries
    two distinct auth headers, or the same auth header name more than once —
    forge never picks a winner.
    """
    headers = headers or {}
    if headers.get(DUPLICATE_AUTH_MARKER):
        raise MultipleCredentialsError(
            "inbound request carries the same auth header more than once"
        )
    found: list[tuple[str, str]] = []
    for name, value in headers.items():
        slot = name.lower()
        if slot in AUTH_HEADER_NAMES and value and auth_credential_token(slot, value):
            found.append((slot, value))
    if len(found) > 1:
        slots = ", ".join(sorted(s for s, _ in found))
        raise MultipleCredentialsError(f"inbound request carries auth headers: {slots}")
    if found:
        return found[0]
    return None, None


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


def resolve_inbound_credential(
    headers: Mapping[str, str] | None,
    source_protocol: str,
    target_protocol: str,
    backend_api_key_present: bool,
) -> dict[str, str] | None:
    """Resolve the per-call credential header to forward, or None.

    Extracts the single inbound auth header (raising on two), enforces the
    one-credential rule against a configured static ``--backend-api-key``, and
    relocates the credential to the backend's canonical slot. Returns None when
    the request carries no inbound credential (the static key, if any, is
    already baked into the client at construction).
    """
    slot, value = extract_inbound_credential(headers)
    if slot is None:
        return None
    if backend_api_key_present:
        raise MultipleCredentialsError(
            "inbound auth header + --backend-api-key (static backend credential)"
        )
    return relocate_credential(slot, value, source_protocol, target_protocol)
