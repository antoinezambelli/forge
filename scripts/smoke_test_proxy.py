"""Smoke test: forge proxy end-to-end.

Starts llama-server + proxy via ProxyServer (managed mode), runs tests
via the openai SDK, then stops everything. No manual setup needed.

Usage:
  python scripts/smoke_test_proxy.py --gguf path/to/model.gguf
"""

from __future__ import annotations

import argparse
import json
import sys

from openai import OpenAI

from forge.proxy import ProxyServer


def run_tests(proxy_url: str, model: str) -> tuple[int, int]:
    client = OpenAI(base_url=f"{proxy_url}/v1", api_key="dummy")
    tests = [
        ("Model list", lambda: test_model_list(client, model)),
        ("Simple chat (batch, no tools)", lambda: test_simple_chat(client, model)),
        ("Streaming chat (no tools)", lambda: test_streaming_chat(client, model)),
        ("Tool call (streaming)", lambda: test_tool_call(client, model)),
        ("Tool call (batch)", lambda: test_tool_call_batch(client, model)),
    ]

    passed = 0
    for i, (name, test) in enumerate(tests, 1):
        print(f"{i}. {name}...", end=" ")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")

    return passed, len(tests)


def test_model_list(client: OpenAI, model: str):
    models = client.models.list()
    names = [m.id for m in models.data]
    assert model in names, f"Expected {model} in {names}"
    print(f"OK ({len(names)} models)")


def test_simple_chat(client: OpenAI, model: str):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        max_tokens=20,
        stream=False,
    )
    text = resp.choices[0].message.content
    assert text, "Empty response"
    print(f"OK -> {text[:50]!r}")


def test_streaming_chat(client: OpenAI, model: str):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count from 1 to 3."}],
        max_tokens=50,
        stream=True,
    )
    chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            chunks.append(delta.content)
    text = "".join(chunks)
    assert text, "Empty stream"
    print(f"OK -> {text[:50]!r}")


def test_tool_call(client: OpenAI, model: str):
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }]

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools,
        max_tokens=200,
        stream=True,
    )

    tool_calls = {}
    text_parts = []
    for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            text_parts.append(delta.content)
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.id:
                    tool_calls[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments

    if tool_calls:
        for idx, tc in sorted(tool_calls.items()):
            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            print(f"OK -> {tc['name']}({args})")
    elif text_parts:
        text = "".join(text_parts)
        print(f"GOT TEXT (no tool call) -> {text[:80]!r}")
    else:
        print("EMPTY RESPONSE")


def test_tool_call_batch(client: OpenAI, model: str):
    tools = [{
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    }]

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Search for 'forge framework'"}],
        tools=tools,
        max_tokens=200,
        stream=False,
    )

    msg = resp.choices[0].message
    if msg.tool_calls:
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"OK -> {tc.function.name}({args})")
    elif msg.content:
        print(f"GOT TEXT -> {msg.content[:80]!r}")
    else:
        print("EMPTY RESPONSE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forge proxy end-to-end smoke test")
    parser.add_argument(
        "--gguf",
        required=True,
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Proxy port (default: 8081)",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8080,
        help="Backend port (default: 8080)",
    )
    args = parser.parse_args()

    print(f"\nForge Proxy Smoke Test (end-to-end)")
    print(f"GGUF: {args.gguf}\n")

    # Start everything: llama-server + proxy
    print("Starting proxy (managed mode)...")
    proxy = ProxyServer(
        backend="llamaserver",
        gguf=args.gguf,
        port=args.port,
        backend_port=args.backend_port,
    )
    proxy.start()
    print(f"Proxy ready at {proxy.url}")

    # Discover model name
    client = OpenAI(base_url=f"{proxy.url}/v1", api_key="dummy")
    models = client.models.list()
    model = models.data[0].id
    print(f"Model: {model}\n")

    try:
        passed, total = run_tests(proxy.url, model)
        print(f"\n{passed}/{total} passed")
    finally:
        print("Stopping proxy + backend...")
        proxy.stop()
        print("Done.")

    sys.exit(0 if passed == total else 1)
