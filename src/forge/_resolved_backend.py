"""Private composition of backend profile facts and endpoint addresses."""

from __future__ import annotations

from dataclasses import dataclass

from forge._backend_profiles import BackendProfile
from forge._endpoint_layouts import (
    BackendOperation,
    ConnectionInputKind,
    NormalizedConnection,
    client_base_url,
    normalize_connection,
    resolve_endpoint,
)


@dataclass(frozen=True)
class ResolvedBackend:
    """Immutable selected profile, normalized connection, and operation URLs."""

    profile: BackendProfile
    connection: NormalizedConnection
    adapter_base_url: str
    _addresses: tuple[tuple[BackendOperation, str], ...]

    def address(self, operation: BackendOperation) -> str:
        """Return a capability-approved internal semantic operation address."""

        if operation not in self.profile.family_profile.operations:
            raise ValueError(
                f"backend profile {self.profile.selector!r} does not support "
                f"{operation.value!r}"
            )
        return dict(self._addresses)[operation]


def resolve_backend(
    profile: BackendProfile,
    url: str,
    input_kind: ConnectionInputKind,
) -> ResolvedBackend:
    """Join a selected backend profile with typed connection facts."""

    connection = normalize_connection(url, input_kind)
    addresses = tuple(
        (operation, resolve_endpoint(
            profile.family_profile.endpoint_layout, operation, connection,
        ))
        for operation in sorted(profile.family_profile.operations, key=lambda item: item.value)
    )
    return ResolvedBackend(
        profile=profile,
        connection=connection,
        adapter_base_url=client_base_url(profile.family_profile.endpoint_layout, connection),
        _addresses=addresses,
    )
