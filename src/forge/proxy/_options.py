"""Shared definitions for the Proxy CLI and TOML profile surface."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from forge._backend_profiles import proxy_backend_selectors
from forge.core.reasoning import DEFAULT_REASONING_REPLAY, REASONING_REPLAY_CHOICES


ProfileKind = Literal["string", "integer", "number", "boolean", "string_list"]


@dataclass(frozen=True)
class _OptionDefinition:
    name: str
    flags: tuple[str, ...]
    profile_kind: ProfileKind
    default: object = None
    argparse_kwargs: dict[str, Any] = field(default_factory=dict)
    group: str | None = None
    credential: bool = False


_OPTIONS = (
    _OptionDefinition(
        "backend_url", ("--backend-url",), "string",
        argparse_kwargs={"help": "URL of an externally managed backend (external mode)"},
    ),
    _OptionDefinition(
        "backend", ("--backend",), "string",
        argparse_kwargs={
            "choices": proxy_backend_selectors(),
            "help": "Managed backend or unmanaged wire/profile selector.",
        },
    ),
    _OptionDefinition(
        "model", ("--model",), "string",
        argparse_kwargs={
            "help": "Model name (required for managed ollama). External generic "
            "OpenAI/llama profiles use it as a fallback when the request omits "
            "model; external vLLM and Anthropic profiles use it as a wire-model "
            "pin. It does not provide or suppress context-window reporting metadata.",
        },
    ),
    _OptionDefinition(
        "gguf", ("--gguf",), "string",
        argparse_kwargs={"help": "Path to GGUF file (llamaserver/llamafile)"},
    ),
    _OptionDefinition(
        "model_path", ("--model-path",), "string",
        argparse_kwargs={"help": "Model directory or HF repo id (vllm, managed mode)"},
    ),
    _OptionDefinition(
        "backend_port", ("--backend-port",), "integer",
        argparse_kwargs={"type": int, "help": "Backend target port"},
    ),
    _OptionDefinition(
        "budget_mode", ("--budget-mode",), "string",
        argparse_kwargs={
            "choices": ("backend", "manual", "forge-full", "forge-fast"),
            "help": "Managed context budget mode (default: backend)",
        },
    ),
    _OptionDefinition(
        "budget_tokens", ("--budget-tokens",), "integer",
        argparse_kwargs={
            "type": int,
            "help": "Positive managed manual allocation with --budget-mode manual; "
            "in unmanaged mode, reporting denominator only (never compacts or "
            "enforces caller history)",
        },
    ),
    _OptionDefinition(
        "extra_flags", ("--extra-flags",), "string_list",
        argparse_kwargs={
            "nargs": argparse.REMAINDER,
            "help": "Terminal argv tail for a Forge-spawned llama-server, llamafile, "
            "or vLLM backend; rejected for Ollama and unmanaged mode; all Forge "
            "options must precede it.",
        },
    ),
    _OptionDefinition(
        "host", ("--host",), "string", "127.0.0.1",
        argparse_kwargs={"help": "Proxy listen host (default: 127.0.0.1)"},
    ),
    _OptionDefinition(
        "port", ("--port",), "integer", 8081,
        argparse_kwargs={"type": int, "help": "Proxy listen port (default: 8081)"},
    ),
    _OptionDefinition(
        "serialize", ("--serialize",), "boolean", None,
        argparse_kwargs={
            "dest": "serialize", "action": "store_true",
            "help": "Force request serialization",
        },
        group="serialization",
    ),
    _OptionDefinition(
        "serialize", ("--no-serialize",), "boolean", None,
        argparse_kwargs={
            "dest": "serialize", "action": "store_false",
            "help": "Disable request serialization",
        },
        group="serialization",
    ),
    _OptionDefinition(
        "max_retries", ("--max-retries",), "integer", 3,
        argparse_kwargs={"type": int, "help": "Max retries per request (default: 3)"},
    ),
    _OptionDefinition(
        "max_tool_errors", ("--max-tool-errors",), "integer", 2,
        argparse_kwargs={
            "type": int,
            "help": "Max consecutive tool-call errors per request (default: 2)",
        },
    ),
    _OptionDefinition(
        "backend_timeout", ("--backend-timeout",), "number", 300.0,
        argparse_kwargs={
            "type": float,
            "help": "Backend response timeout in seconds (default: 300)",
        },
    ),
    _OptionDefinition(
        "no_rescue", ("--no-rescue",), "boolean", False,
        argparse_kwargs={"action": "store_true", "help": "Disable rescue parsing"},
    ),
    _OptionDefinition(
        "backend_api_key", ("--backend-api-key",), "string", None,
        argparse_kwargs={
            "help": "Static credential forge sends to the backend in its native auth "
            "header (LM Studio, hosted providers, service accounts). forge relocates "
            "it to the backend's protocol slot. When set, an inbound auth header is "
            "refused as a second credential (at most one credential per request). "
            "This is backend authentication, not caller authorization; Proxy does "
            "not authenticate callers. Defaults to the FORGE_BACKEND_API_KEY env var.",
        },
        credential=True,
    ),
    _OptionDefinition(
        "backend_capability", ("--backend-capability",), "string", "native",
        argparse_kwargs={
            "choices": ("native", "prompt"),
            "help": "Tool-calling protocol for the backend (default: native). 'native' "
            "uses the selected adapter's structured-tool path; compatible OpenAI-shaped "
            "clean paths preserve raw tool fields, while other adapters convert or "
            "rebuild them. 'prompt' opts into prompt-injection for non-FC llama.cpp/"
            "llamafile backends (strips tools into the prompt, parses the JSON call "
            "back). Frozen at startup — never probed or switched mid-stream.",
        },
    ),
    _OptionDefinition(
        "inject_respond_tool", ("--inject-respond-tool",), "boolean", False,
        argparse_kwargs={
            "action": "store_true",
            "help": "Inject forge's synthetic respond() tool when the client sends tools. Default off.",
        },
    ),
    _OptionDefinition(
        "reasoning_replay", ("--reasoning-replay",), "string",
        DEFAULT_REASONING_REPLAY,
        argparse_kwargs={
            "choices": REASONING_REPLAY_CHOICES,
            "help": "How much captured reasoning to replay to the backend (default: none).",
        },
    ),
    _OptionDefinition(
        "verbose", ("--verbose", "-v"), "boolean", False,
        argparse_kwargs={"action": "store_true", "help": "Verbose logging"},
    ),
)


def option_definitions(*, include_credentials: bool = True) -> tuple[_OptionDefinition, ...]:
    return tuple(
        option for option in _OPTIONS
        if include_credentials or not option.credential
    )


def profile_definitions() -> dict[str, _OptionDefinition]:
    return {
        option.name: option
        for option in option_definitions(include_credentials=False)
    }


def option_defaults(*, backend_api_key_from_environment: bool) -> dict[str, object]:
    defaults = {
        option.name: option.default
        for option in option_definitions()
    }
    if backend_api_key_from_environment:
        defaults["backend_api_key"] = os.environ.get("FORGE_BACKEND_API_KEY")
    return defaults


def add_proxy_options(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
    include_credentials: bool = True,
) -> None:
    groups = {"serialization": parser.add_mutually_exclusive_group()}
    seen_defaults: set[str] = set()
    for option in option_definitions(include_credentials=include_credentials):
        target = groups.get(option.group, parser)
        kwargs = dict(option.argparse_kwargs)
        if suppress_defaults:
            kwargs["default"] = argparse.SUPPRESS
        elif option.name not in seen_defaults:
            kwargs["default"] = (
                os.environ.get("FORGE_BACKEND_API_KEY")
                if option.credential else option.default
            )
        target.add_argument(*option.flags, **kwargs)
        seen_defaults.add(option.name)


def supplied_proxy_options(argv: list[str]) -> set[str]:
    """Return Proxy values explicitly supplied before the terminal backend tail."""

    head = argv[: argv.index("--extra-flags") + 1] if "--extra-flags" in argv else argv
    by_flag = {
        flag: option.name
        for option in option_definitions()
        for flag in option.flags
    }
    supplied: set[str] = set()
    for token in head:
        flag = token.split("=", 1)[0]
        if flag in by_flag:
            supplied.add(by_flag[flag])
    return supplied
