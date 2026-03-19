"""Smoke test: forge proxy with openai SDK + real llama-server.

Exercises the full chain: openai SDK -> forge proxy -> llama-server.
Tests streaming, tool calling, and (implicitly) guardrail intervention.

Prerequisites:
  - llama-server running on port 8080
  - forge proxy running on port 8081:
      python -m forge.proxy --backend-url http://localhost:8080 --port 8081 --verbose
"""

from __future__ import annotations

import json
import sys

from openai import OpenAI

PROXY_URL = "http://localhost:8081/v1"
MODEL = "Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf"

client = OpenAI(base_url=PROXY_URL, api_key="dummy")


def test_model_list():
    """Verify model discovery through the proxy."""
    print("1. Model list...", end=" ")
    models = client.models.list()
    names = [m.id for m in models.data]
    assert MODEL in names, f"Expected {MODEL} in {names}"
    print(f"OK ({len(names)} models)")


def test_simple_chat():
    """Non-streaming chat completion, no tools."""
    print("2. Simple chat (batch, no tools)...", end=" ")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        max_tokens=20,
        stream=False,
    )
    text = resp.choices[0].message.content
    assert text, "Empty response"
    print(f"OK -> {text[:50]!r}")


def test_streaming_chat():
    """Streaming chat completion, no tools."""
    print("3. Streaming chat (no tools)...", end=" ")
    stream = client.chat.completions.create(
        model=MODEL,
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


def test_tool_call():
    """Streaming chat with tools -- the core proxy use case."""
    print("4. Tool call (streaming)...", end=" ")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools,
        max_tokens=200,
        stream=True,
    )

    # Collect the streamed response
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
        print("   (Model didn't produce a tool call -- check proxy logs for rescue/retry)")
    else:
        print("EMPTY RESPONSE")


def test_tool_call_batch():
    """Batch (non-streaming) chat with tools."""
    print("5. Tool call (batch)...", end=" ")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    resp = client.chat.completions.create(
        model=MODEL,
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
    print(f"\nForge Proxy Smoke Test")
    print(f"Proxy: {PROXY_URL}")
    print(f"Model: {MODEL}\n")

    tests = [
        test_model_list,
        test_simple_chat,
        test_streaming_chat,
        test_tool_call,
        test_tool_call_batch,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")

    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
