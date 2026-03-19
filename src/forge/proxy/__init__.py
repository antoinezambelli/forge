"""OpenAI-compatible proxy server with forge guardrails.

Sits between any OpenAI-compatible client and a model server,
applying guardrails transparently. See ADR-012 for design.
"""

from forge.proxy.proxy import ProxyServer

__all__ = ["ProxyServer"]
