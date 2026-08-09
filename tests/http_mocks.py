"""Shared lightweight HTTPX construction helpers for unit tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import ParamSpec, TypeVar
from unittest.mock import AsyncMock, MagicMock, patch

import httpx


P = ParamSpec("P")
T = TypeVar("T")


def _mock_async_client(*args: object, **kwargs: object) -> AsyncMock:
    """Return the HTTP surface client tests use without opening a real pool."""
    del args
    client = AsyncMock()
    client.stream = MagicMock()
    headers = kwargs.get("headers")
    client.headers = httpx.Headers(headers) if headers is not None else httpx.Headers()
    timeout = kwargs.get("timeout")
    client.timeout = (
        timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)
    )
    return client


@contextmanager
def patch_httpx_client_constructor() -> Iterator[MagicMock]:
    """Replace ``AsyncClient`` construction while preserving call inspection."""
    constructor = MagicMock(side_effect=_mock_async_client)
    with patch.object(httpx, "AsyncClient", constructor):
        yield constructor


def build_with_mock_http(
    factory: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    """Construct one production client without creating a real HTTPX pool."""
    with patch_httpx_client_constructor():
        return factory(*args, **kwargs)
