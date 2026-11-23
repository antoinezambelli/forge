from forge.core.messages import RichMessage, ReasoningPart
from forge.core.codecs import argument_value
from tests.canonical import send, send_stream, calls as reply_calls, visible, reasoning

"""Tests for forge.clients.openai_compat — OpenAICompatClient with mocked HTTP."""
import json
import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock
from forge.clients.openai_compat import OpenAICompatClient
from forge.clients.base import ChunkType
from forge.core.workflow import ToolDefinition
from forge.errors import BackendError, MultipleCredentialsError

pytestmark = pytest.mark.usefixtures("mock_httpx_client_constructor")
PART_SCHEMA = {
    "properties": {
        "part": {"description": "Part number", "title": "Part", "type": "string"}
    },
    "required": ["part"],
    "title": "PartParams",
    "type": "object",
}


def _make_spec(name: str = "get_pricing") -> ToolDefinition:
    return ToolDefinition(
        name=name, description=f"Get {name}", input_schema=PART_SCHEMA
    )


def _make_client(model: str = "test-model", api_key: str = "tok") -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="https://api.example.com/v1", model=model, api_key=api_key
    )


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


class _MockStreamResponse:
    """Mock for httpx streaming response with aiter_lines / aread."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return "".join(self._lines).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class TestSend:
    @pytest.mark.asyncio
    async def test_returns_tool_call(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": '{"part": "X123"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert len(reply_calls(result)) == 1
        assert reply_calls(result)[0].name == "get_pricing"
        assert argument_value(reply_calls(result)[0]) == {"part": "X123"}

    @pytest.mark.asyncio
    async def test_returns_text_response(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "I need more info"}}
                ]
            }
        )
        result = await send(client, [{"role": "user", "content": "test"}])
        assert isinstance(result, RichMessage)
        assert visible(result) == "I need more info"

    @pytest.mark.asyncio
    async def test_missing_choices_raises_backend_error(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response({"object": "error"})
        with pytest.raises(BackendError, match="response has no choices"):
            await send(client, [{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_null_content_returns_empty_text(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {"choices": [{"message": {"role": "assistant", "content": None}}]}
        )
        result = await send(client, [{"role": "user", "content": "test"}])
        assert isinstance(result, RichMessage)
        assert visible(result) == ""

    @pytest.mark.asyncio
    async def test_malformed_tool_args_kept_as_raw_args(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": "{not json",
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        )
        result = await send(client, [{"role": "user", "content": "test"}])
        assert isinstance(result, RichMessage)
        assert reply_calls(result)[0].name == "get_pricing"
        assert argument_value(reply_calls(result)[0]) == "{not json"

    @pytest.mark.asyncio
    async def test_malformed_tool_args_kept_even_with_assistant_text(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Let me look that up.",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": "{not json",
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        )
        result = await send(client, [{"role": "user", "content": "test"}])
        assert isinstance(result, RichMessage)
        assert argument_value(reply_calls(result)[0]) == "{not json"

    @pytest.mark.asyncio
    async def test_malformed_among_several_kept_per_call(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": '{"part": "A"}',
                                    }
                                },
                                {
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": "{broken",
                                    }
                                },
                            ],
                        }
                    }
                ]
            }
        )
        result = await send(client, [{"role": "user", "content": "test"}])
        assert isinstance(result, RichMessage)
        assert argument_value(reply_calls(result)[0]) == {"part": "A"}
        assert argument_value(reply_calls(result)[1]) == "{broken"

    @pytest.mark.asyncio
    async def test_dict_tool_args_accepted(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": {"part": "X123"},
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        )
        result = await send(client, [{"role": "user", "content": "test"}])
        assert isinstance(result, RichMessage)
        assert argument_value(reply_calls(result)[0]) == {"part": "X123"}

    @pytest.mark.asyncio
    async def test_formats_tools_in_request(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
        await send(client, [{"role": "user", "content": "test"}], tools=[_make_spec()])
        body = client._http.post.call_args.kwargs["json"]
        assert "tools" in body
        tool = body["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_pricing"
        assert "parameters" in tool["function"]

    @pytest.mark.asyncio
    async def test_request_body_structure(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
        await send(client, [{"role": "user", "content": "hi"}])
        body = client._http.post.call_args.kwargs["json"]
        assert body["model"] == "test-model"
        assert body["stream"] is False
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert "temperature" not in body

    @pytest.mark.asyncio
    async def test_posts_to_chat_completions(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
        await send(client, [{"role": "user", "content": "hi"}])
        url = client._http.post.call_args.args[0]
        assert url == "https://api.example.com/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_sampling_override(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
        await send(
            client, [{"role": "user", "content": "hi"}], sampling={"temperature": 0.2}
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_http_error_raises_backend_error(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {"error": "bad"}, status_code=401
        )
        with pytest.raises(BackendError) as exc:
            await send(client, [{"role": "user", "content": "test"}])
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "field", ["reasoning_content", "reasoning", "reasoning_text"]
    )
    async def test_structured_reasoning_field_captured(self, field) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            field: "I should check the price first.",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": '{"part": "X"}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        )
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "I should check the price first."

    @pytest.mark.asyncio
    async def test_extracts_think_tags_from_content_with_tool_call(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>plan</think>",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": '{"part": "X"}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        )
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "plan"

    @pytest.mark.asyncio
    async def test_reasoning_field_preferred_over_content_tags(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>inline</think>",
                            "reasoning": "structured",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": '{"part": "X"}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        )
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert reasoning(result) == "structured"

    @pytest.mark.asyncio
    async def test_content_preamble_not_treated_as_reasoning(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Let me look that up.",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": '{"part": "X"}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        )
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reasoning(result) == ""

    @pytest.mark.asyncio
    async def test_think_tags_stripped_from_text_response(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>x</think>The answer is 42.",
                        }
                    }
                ]
            }
        )
        result = await send(client, [{"role": "user", "content": "test"}])
        assert isinstance(result, RichMessage)
        assert visible(result) == "The answer is 42."

    @pytest.mark.asyncio
    async def test_plain_preamble_text_response_survives(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Let me look that up.",
                        }
                    }
                ]
            }
        )
        result = await send(client, [{"role": "user", "content": "test"}])
        assert isinstance(result, RichMessage)
        assert visible(result) == "Let me look that up."

    @pytest.mark.asyncio
    async def test_reasoning_attached_to_first_tool_call_only(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning": "because",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": '{"part": "A"}',
                                    }
                                },
                                {
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": '{"part": "B"}',
                                    }
                                },
                            ],
                        }
                    }
                ]
            }
        )
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "because"
        assert len([p for p in result.parts if isinstance(p, ReasoningPart)]) == 1

    @pytest.mark.asyncio
    async def test_empty_string_structured_field_falls_through_to_tags(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning": "",
                            "content": "<think>plan</think>",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": '{"part": "X"}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        )
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "plan"

    @pytest.mark.asyncio
    async def test_non_string_reasoning_field_raises(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning": [{"type": "text", "text": "block"}],
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_pricing",
                                        "arguments": '{"part": "X"}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        )
        with pytest.raises(BackendError, match="not a string"):
            await send(
                client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
            )

    @pytest.mark.asyncio
    async def test_records_usage(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )
        await send(client, [{"role": "user", "content": "test"}])
        assert client.last_usage[0].prompt_tokens == 10
        assert client.last_usage[0].completion_tokens == 5
        assert client.last_usage[0].total_tokens == 15


class TestAuth:
    def test_bearer_header_set_when_key_provided(self) -> None:
        client = OpenAICompatClient(
            base_url="https://x/v1", model="m", api_key="secret"
        )
        assert client._http.headers["Authorization"] == "Bearer secret"

    def test_no_auth_header_when_no_key(self) -> None:
        client = OpenAICompatClient(base_url="https://x/v1", model="m")
        assert "Authorization" not in client._http.headers

    def test_base_url_trailing_slash_stripped(self) -> None:
        client = OpenAICompatClient(base_url="https://x/v1/", model="m")
        assert client.base_url == "https://x/v1"


class TestSendStream:
    @pytest.mark.asyncio
    async def test_read_timeout_during_stream_raises_backend_error(self) -> None:
        class _TimeoutStreamResponse(_MockStreamResponse):
            async def aiter_lines(self):
                yield (
                    "data: "
                    + json.dumps({"choices": [{"delta": {"content": "partial"}}]})
                )
                raise httpx.ReadTimeout("timed out")

        client = _make_client()
        client._http.stream.return_value = _TimeoutStreamResponse([])
        chunks = []
        with pytest.raises(BackendError) as exc_info:
            async for chunk in send_stream(
                client, [{"role": "user", "content": "test"}]
            ):
                chunks.append(chunk)
        assert [chunk.content for chunk in chunks] == ["partial"]
        assert exc_info.value.status_code == 408
        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_yields_text_deltas_and_final(self) -> None:
        client = _make_client()
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": " world"}}]}),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = []
        async for chunk in send_stream(client, [{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
        text_deltas = [c for c in chunks if c.type == ChunkType.TEXT_DELTA]
        assert [c.content for c in text_deltas] == ["Hello", " world"]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, RichMessage)
        assert visible(finals[0].response) == "Hello world"

    @pytest.mark.asyncio
    async def test_yields_final_with_tool_call(self) -> None:
        client = _make_client()
        lines = [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "function": {
                                            "name": "get_pricing",
                                            "arguments": "",
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '{"part": '}}
                                ]
                            }
                        }
                    ]
                }
            ),
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '"X"}'}}
                                ]
                            }
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = []
        async for chunk in send_stream(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, RichMessage)
        assert reply_calls(finals[0].response)[0].name == "get_pricing"
        assert argument_value(reply_calls(finals[0].response)[0]) == {"part": "X"}

    @pytest.mark.asyncio
    async def test_stream_malformed_tool_args_kept_as_raw_args(self) -> None:
        client = _make_client()
        lines = [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "get_pricing",
                                            "arguments": "{not",
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": " json"}}
                                ]
                            }
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = [
            c
            async for c in send_stream(
                client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
            )
        ]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        result = finals[0].response
        assert isinstance(result, RichMessage)
        assert reply_calls(result)[0].name == "get_pricing"
        assert argument_value(reply_calls(result)[0]) == "{not json"

    @pytest.mark.asyncio
    async def test_stream_non_string_arg_fragment_not_dropped(self) -> None:
        client = _make_client()
        lines = [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "get_pricing",
                                            "arguments": {"part": "X9"},
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = [
            c
            async for c in send_stream(
                client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
            )
        ]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, RichMessage)
        assert argument_value(reply_calls(finals[0].response)[0]) == {"part": "X9"}

    @pytest.mark.asyncio
    async def test_accumulates_reasoning_across_deltas(self) -> None:
        client = _make_client()
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"reasoning": "Let me "}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"reasoning": "think... "}}]}),
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "get_pricing",
                                            "arguments": '{"part": "X"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = [
            c
            async for c in send_stream(
                client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
            )
        ]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        result = finals[0].response
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "Let me think... "

    @pytest.mark.asyncio
    async def test_stream_extracts_think_tags_from_content_with_tool_call(self) -> None:
        client = _make_client()
        lines = [
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "<think>inline "}}]}),
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "plan</think>"}}]}),
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "get_pricing",
                                            "arguments": '{"part": "X"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = [
            c
            async for c in send_stream(
                client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
            )
        ]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        result = finals[0].response
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "inline plan"

    @pytest.mark.asyncio
    async def test_stream_content_preamble_not_treated_as_reasoning(self) -> None:
        client = _make_client()
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Let me "}}]}),
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "look that up."}}]}),
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "get_pricing",
                                            "arguments": '{"part": "X"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = [
            c
            async for c in send_stream(
                client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
            )
        ]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        result = finals[0].response
        assert isinstance(result, RichMessage)
        assert reasoning(result) == ""

    @pytest.mark.asyncio
    async def test_stream_strips_think_tags_from_text_response(self) -> None:
        client = _make_client()
        lines = [
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "<think>x</think>"}}]}),
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "The answer is 42."}}]}),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = [
            c async for c in send_stream(client, [{"role": "user", "content": "test"}])
        ]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, RichMessage)
        assert visible(finals[0].response) == "The answer is 42."

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", ["reasoning_content", "reasoning_text"])
    async def test_stream_other_canonical_fields_accumulate(self, field) -> None:
        client = _make_client()
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {field: "step one "}}]}),
            "data: " + json.dumps({"choices": [{"delta": {field: "step two"}}]}),
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "get_pricing",
                                            "arguments": '{"part": "X"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = [
            c
            async for c in send_stream(
                client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
            )
        ]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert isinstance(finals[0].response, RichMessage)
        assert reasoning(finals[0].response) == "step one step two"

    @pytest.mark.asyncio
    async def test_stream_structured_preferred_over_content_tags(self) -> None:
        client = _make_client()
        lines = [
            "data: "
            + json.dumps({"choices": [{"delta": {"reasoning": "structured"}}]}),
            "data: "
            + json.dumps(
                {"choices": [{"delta": {"content": "<think>inline</think>"}}]}
            ),
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "get_pricing",
                                            "arguments": '{"part": "X"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = [
            c
            async for c in send_stream(
                client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
            )
        ]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert isinstance(finals[0].response, RichMessage)
        assert reasoning(finals[0].response) == "structured"

    @pytest.mark.asyncio
    async def test_stream_reasoning_first_tool_call_only(self) -> None:
        client = _make_client()
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"reasoning": "because"}}]}),
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "get_pricing",
                                            "arguments": '{"part": "A"}',
                                        },
                                    },
                                    {
                                        "index": 1,
                                        "function": {
                                            "name": "get_pricing",
                                            "arguments": '{"part": "B"}',
                                        },
                                    },
                                ]
                            }
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = [
            c
            async for c in send_stream(
                client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
            )
        ]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert isinstance(finals[0].response, RichMessage)
        assert reasoning(finals[0].response) == "because"
        assert (
            len([p for p in finals[0].response.parts if isinstance(p, ReasoningPart)])
            <= 1
        )

    @pytest.mark.asyncio
    async def test_stream_non_string_reasoning_raises(self) -> None:
        client = _make_client()
        lines = [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {"delta": {"reasoning": [{"type": "text", "text": "block"}]}}
                    ]
                }
            ),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        with pytest.raises(BackendError, match="not a string"):
            async for _ in send_stream(client, [{"role": "user", "content": "test"}]):
                pass

    @pytest.mark.asyncio
    async def test_stream_http_error_raises(self) -> None:
        client = _make_client()
        client._http.stream.return_value = _MockStreamResponse(
            ['{"error": "nope"}'], status_code=500
        )
        with pytest.raises(BackendError) as exc:
            async for _ in send_stream(client, [{"role": "user", "content": "x"}]):
                pass
        assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_ignores_non_data_lines(self) -> None:
        client = _make_client()
        lines = [
            "",
            ": keep-alive comment",
            "data: " + json.dumps({"choices": [{"delta": {"content": "hi"}}]}),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = [
            c async for c in send_stream(client, [{"role": "user", "content": "x"}])
        ]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert visible(finals[0].response) == "hi"


class TestContextLength:
    @pytest.mark.asyncio
    async def test_returns_none(self) -> None:
        client = _make_client()
        assert await client.get_context_length() is None


class TestConstructor:
    def test_extra_headers_merged_alongside_bearer(self) -> None:
        client = OpenAICompatClient(
            model="test-model",
            base_url="https://x/v1",
            api_key="tok",
            extra_headers={"HTTP-Referer": "https://example.com", "X-Title": "MyApp"},
        )
        assert client._http.headers["authorization"] == "Bearer tok"
        assert client._http.headers["http-referer"] == "https://example.com"
        assert client._http.headers["x-title"] == "MyApp"

    def test_api_key_plus_auth_header_raises(self) -> None:
        with pytest.raises(MultipleCredentialsError):
            OpenAICompatClient(
                model="test-model",
                base_url="https://x/v1",
                api_key="ignored",
                extra_headers={"Authorization": "ApiKey custom-scheme"},
            )

    def test_extra_headers_alone_set_custom_scheme(self) -> None:
        client = OpenAICompatClient(
            model="test-model",
            base_url="https://x/v1",
            extra_headers={"Authorization": "ApiKey custom-scheme"},
        )
        assert client._http.headers["authorization"] == "ApiKey custom-scheme"

    def test_sampling_kwargs_stored_as_instance_fields(self) -> None:
        client = OpenAICompatClient(
            model="test-model",
            base_url="https://x/v1",
            top_k=40,
            min_p=0.05,
            repeat_penalty=1.1,
            presence_penalty=0.2,
            chat_template_kwargs={"enable_thinking": True},
        )
        assert client.top_k == 40
        assert client.min_p == 0.05
        assert client.repeat_penalty == 1.1
        assert client.presence_penalty == 0.2
        assert client.chat_template_kwargs == {"enable_thinking": True}

    def test_recommended_sampling_off_is_silent_for_unknown_model(self) -> None:
        client = OpenAICompatClient(
            model="definitely-not-in-registry-zzz", base_url="https://x/v1"
        )
        assert client.temperature is None
        assert client.top_p is None
        assert client.top_k is None

    def test_recommended_sampling_on_raises_for_unknown_model(self) -> None:
        from forge.errors import UnsupportedModelError

        with pytest.raises(UnsupportedModelError):
            OpenAICompatClient(
                model="definitely-not-in-registry-zzz",
                base_url="https://x/v1",
                recommended_sampling=True,
            )


class TestSendInstanceSampling:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "field,value",
        [
            ("top_k", 40),
            ("min_p", 0.05),
            ("repeat_penalty", 1.1),
            ("presence_penalty", 0.2),
            ("chat_template_kwargs", {"enable_thinking": True}),
        ],
    )
    async def test_instance_sampling_flows_into_body(self, field, value) -> None:
        client = OpenAICompatClient(
            model="test-model", base_url="https://x/v1", api_key="tok", **{field: value}
        )
        mock_http = AsyncMock()
        mock_http.headers = {}
        client._http = mock_http
        mock_http.post.return_value = _mock_response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
        await send(client, [{"role": "user", "content": "hi"}])
        body = mock_http.post.call_args.kwargs["json"]
        assert body[field] == value


class TestPassthrough:
    @pytest.mark.asyncio
    async def test_passthrough_fields_appear_in_body(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
        await send(
            client,
            [{"role": "user", "content": "hi"}],
            passthrough={"max_tokens": 512, "stop": ["END"], "tool_choice": "auto"},
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["max_tokens"] == 512
        assert body["stop"] == ["END"]
        assert body["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_forge_owned_fields_override_passthrough(self) -> None:
        client = _make_client(model="forge-model")
        client._http.post.return_value = _mock_response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
        await send(
            client,
            [{"role": "user", "content": "hi"}],
            passthrough={
                "model": "evil",
                "messages": [{"role": "system", "content": "evil"}],
                "stream": True,
            },
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["model"] == "forge-model"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["stream"] is False

    @pytest.mark.asyncio
    async def test_inbound_anthropic_body_accepted_and_ignored(self) -> None:
        client = _make_client()
        client._http.post.return_value = _mock_response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
        await send(client, [{"role": "user", "content": "hi"}])
        import inspect

        assert list(inspect.signature(client.send).parameters) == ["request"]
        body = client._http.post.call_args.kwargs["json"]
        assert "some_anthropic_field" not in body
        assert "shape" not in body

    @pytest.mark.asyncio
    async def test_send_stream_accepts_passthrough_and_inbound(self) -> None:
        client = _make_client()
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}),
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockStreamResponse(lines)
        chunks = [
            c
            async for c in send_stream(
                client,
                [{"role": "user", "content": "x"}],
                passthrough={"max_tokens": 64},
            )
        ]
        assert any((c.type == ChunkType.FINAL for c in chunks))


class TestAclose:
    @pytest.mark.asyncio
    async def test_aclose_closes_http_pool(self) -> None:
        client = _make_client()
        await client.aclose()
        client._http.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["openai", "llamafile", "vllm"])
@pytest.mark.parametrize("field", ["reasoning_content", "reasoning", "reasoning_text"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("with_calls", [False, True])
async def test_nullable_reasoning_preserves_text_and_calls(
    backend, field, stream, with_calls
):
    from forge.clients.llamafile import LlamafileClient
    from forge.clients.vllm import VLLMClient
    from forge.core.messages import ValidArguments

    if backend == "openai":
        client = _make_client()
    elif backend == "llamafile":
        client = LlamafileClient("model.gguf", base_url="http://test/v1")
    else:
        client = VLLMClient("model", base_url="http://test/v1")
    message = {"content": "visible answer", field: None}
    if with_calls:
        message["tool_calls"] = [
            {"index": 0, "function": {"name": "noop", "arguments": "{}"}}
        ]
    if stream:
        client._http.stream.return_value = _MockStreamResponse(
            [
                "data: " + json.dumps({"choices": [{"delta": message}]}),
                "data: [DONE]",
            ]
        )
        chunks = [
            chunk
            async for chunk in send_stream(client, [{"role": "user", "content": "hi"}])
        ]
        assert chunks[-1].type == ChunkType.FINAL
        assert sum(chunk.type == ChunkType.FINAL for chunk in chunks) == 1
        reply = chunks[-1].response
    else:
        client._http.post.return_value = _mock_response(
            {"choices": [{"message": message}]}
        )
        reply = await send(client, [{"role": "user", "content": "hi"}])
    assert visible(reply) == "visible answer"
    assert reasoning(reply) == ""
    calls = reply_calls(reply)
    assert len(calls) == int(with_calls)
    if with_calls:
        assert calls[0].name == "noop" and calls[0].arguments == ValidArguments({})
