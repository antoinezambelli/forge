"""The shared PyInstaller inputs and artifact dependency policy."""

from __future__ import annotations

SUPPORTED_TARGETS = (
    "windows-x86_64",
    "linux-x86_64-gnu",
    "macos-arm64",
)

COLLECT_PACKAGES = (
    "forge",
    "pydantic",
    "httpx",
    "anthropic",
    "tomli_w",
)

REQUIRED_CONTENT = (
    "forge.clients.anthropic",
    "forge_guardrails",
    "pydantic",
    "httpx",
    "anthropic",
    "tomli_w",
)

EXCLUDED_MODULES = (
    "pyarrow",
    "pytest",
    "_pytest",
    "mpmath",
    "datasets",
    "torch",
    "tensorflow",
    "jax",
    "vllm",
)

EXCLUDED_ARTIFACT_NAMES = (
    "llama-server",
    "llama-server.exe",
    "llamafile",
    "llamafile.exe",
    "ollama.exe",
    "ollama",
    "vllm.exe",
    "vllm",
)
