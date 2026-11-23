from forge.core.messages import RichMessage, ReasoningPart
from forge.core.codecs import argument_value
from tests.canonical import send, send_stream, calls as reply_calls, visible, reasoning

"""Tests for forge.clients.llamafile — LlamafileClient with mocked HTTP."""
import json
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx
from forge.clients.llamafile import LlamafileClient, _merge_consecutive
from forge.core.workflow import ToolDefinition
from forge.errors import BackendError
from forge.clients.base import ChunkType

pytestmark = pytest.mark.usefixtures("mock_httpx_client_constructor")
PART_SCHEMA = {
    "properties": {
        "part": {"description": "Part number", "title": "Part", "type": "string"}
    },
    "required": ["part"],
    "title": "PartParams",
    "type": "object",
}
WRITE_SCHEMA = {
    "properties": {
        "file_path": {
            "description": "Target path",
            "title": "File Path",
            "type": "string",
        },
        "content": {"description": "File body", "title": "Content", "type": "string"},
    },
    "required": ["file_path", "content"],
    "title": "WriteParams",
    "type": "object",
}
NO_REQUIRED_SCHEMA = {
    "properties": {
        "note": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "description": "Optional note",
            "title": "Note",
        }
    },
    "title": "_NoRequiredParams",
    "type": "object",
}


def _make_write_spec() -> ToolDefinition:
    return ToolDefinition(
        name="write", description="Write a file", input_schema=WRITE_SCHEMA
    )


def _malformed_500_body(raw_input: str) -> str:
    """A llama.cpp tool-call parse rejection body embedding ``raw_input``."""
    return json.dumps(
        {
            "error": {
                "code": 500,
                "message": f"Failed to parse input at pos 84: {raw_input}",
                "type": "server_error",
            }
        }
    )


def _renamed_500_body(raw_input: str) -> str:
    """A 500 from a hypothetical llama.cpp build that RENAMED the parse-error
    phrase but still embeds the raw generation — deliberately contains NO
    "Failed to parse input". Proves the rescue gate is structural, not pinned
    to any error string."""
    return json.dumps(
        {
            "error": {
                "code": 500,
                "message": f"The model produced output that does not match the expected format: {raw_input}",
                "type": "server_error",
            }
        }
    )


def _b9656_generic_parse_500_body() -> str:
    """The post-581e8eca8 (b9656+) parse-rejection body: a generic format
    complaint with the rejected generation moved to server logs — no tool-call
    XML tokens in the body. Shape confirmed against a live b9656+ build."""
    return json.dumps(
        {
            "error": {
                "code": 500,
                "message": "Failed to parse tool call: generated output does not match the expected format",
                "type": "server_error",
            }
        }
    )


_SKELETON_BLOCK = "<tool_call>\n<function=write>\n<parameter=file_path>\ntests/test_x.py\n</parameter>\n</function>\n</tool_call>"
_COMPLETE_BLOCK = "<tool_call>\n<function=write>\n<parameter=file_path>\ntests/test_x.py\n</parameter>\n<parameter=content>\nimport unittest\n\n\nclass T(unittest.TestCase):\n    pass\n</parameter>\n</function>\n</tool_call>"


def _make_spec(name: str = "get_pricing") -> ToolDefinition:
    return ToolDefinition(
        name=name, description=f"Get {name}", input_schema=PART_SCHEMA
    )


def _make_client(mode: str = "native", think: bool | None = None) -> LlamafileClient:
    """Create a LlamafileClient with a mocked HTTP client."""
    return LlamafileClient(
        base_url="http://test:8080/v1", gguf_path="test-model", mode=mode, think=think
    )


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx Response."""
    resp = MagicMock()
    resp.json.return_value = data
    resp.status_code = status_code
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _openai_tool_call_response(
    tool_name: str = "get_pricing", args: str = '{"part": "X"}'
) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": args},
                        }
                    ],
                }
            }
        ]
    }


def _openai_text_response(content: str = "Hello") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class TestLlamafileNativeSend:
    @pytest.mark.asyncio
    async def test_custom_direct_api_base_is_literal_on_wire(self) -> None:
        client = LlamafileClient(
            base_url="https://gateway.example/deploy/custom-api",
            gguf_path="model.gguf",
            mode="native",
        )
        client._http = AsyncMock()
        client._http.headers = {}
        client._http.post.return_value = _mock_response(_openai_text_response("ok"))
        await send(client, [{"role": "user", "content": "test"}])
        assert (
            client._http.post.await_args.args[0]
            == "https://gateway.example/deploy/custom-api/chat/completions"
        )

    @pytest.mark.asyncio
    async def test_returns_tool_call(self) -> None:
        client = _make_client("native")
        client._http.post.return_value = _mock_response(
            _openai_tool_call_response("get_pricing", '{"part": "X123"}')
        )
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert (
            client._http.post.await_args.args[0]
            == "http://test:8080/v1/chat/completions"
        )
        assert isinstance(result, RichMessage)
        assert reply_calls(result)[0].name == "get_pricing"
        assert argument_value(reply_calls(result)[0]) == {"part": "X123"}

    @pytest.mark.asyncio
    async def test_returns_text_response(self) -> None:
        client = _make_client("native")
        client._http.post.return_value = _mock_response(
            _openai_text_response("I need more info")
        )
        result = await send(client, [{"role": "user", "content": "test"}])
        assert isinstance(result, RichMessage)
        assert visible(result) == "I need more info"

    @pytest.mark.asyncio
    async def test_malformed_tool_call_500_retries_with_clean_nudge(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        resp.text = '{"error":{"code":500,"message":"Failed to parse input at pos 84: <tool_call>\\n<function=write>\\n<parameter=file_path>\\nx.py\\n</parameter>\\n</function>\\n</tool_call>","type":"server_error"}}'
        client._http.post.return_value = resp
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert "malformed" in visible(result).lower()
        assert "Failed to parse input" not in visible(result)

    @pytest.mark.asyncio
    async def test_malformed_500_stutter_returns_only_complete_call(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        resp.text = _malformed_500_body(f"{_SKELETON_BLOCK}\n{_COMPLETE_BLOCK}")
        client._http.post.return_value = resp
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_write_spec()]
        )
        assert isinstance(result, RichMessage)
        assert len(reply_calls(result)) == 1
        assert reply_calls(result)[0].name == "write"
        assert argument_value(reply_calls(result)[0])["file_path"] == "tests/test_x.py"
        assert argument_value(reply_calls(result)[0])["content"].startswith(
            "import unittest"
        )
        assert argument_value(reply_calls(result)[0])["content"].endswith("    pass")
        assert client.rescued_tool_calls == 1

    @pytest.mark.asyncio
    async def test_malformed_500_rescue_dedupes_repeated_blocks(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        resp.text = _malformed_500_body(
            f"{_SKELETON_BLOCK}\n{_SKELETON_BLOCK}\n{_COMPLETE_BLOCK}\n{_COMPLETE_BLOCK}"
        )
        client._http.post.return_value = resp
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_write_spec()]
        )
        assert isinstance(result, RichMessage)
        assert len(reply_calls(result)) == 1
        assert argument_value(reply_calls(result)[0])["content"].startswith(
            "import unittest"
        )
        assert client.rescued_tool_calls == 1

    @pytest.mark.asyncio
    async def test_malformed_500_skeleton_only_returns_nudge(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        resp.text = _malformed_500_body(_SKELETON_BLOCK)
        client._http.post.return_value = resp
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_write_spec()]
        )
        assert isinstance(result, RichMessage)
        assert "ONE complete" in visible(result)
        assert "Failed to parse input" not in visible(result)
        assert client.rescued_tool_calls == 0

    @pytest.mark.asyncio
    async def test_malformed_500_no_required_params_partial_survives(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        no_req_spec = ToolDefinition(
            name="ping", description="ping", input_schema=NO_REQUIRED_SCHEMA
        )
        resp.text = _malformed_500_body(
            "<tool_call>\n<function=ping>\n</function>\n</tool_call>"
        )
        client._http.post.return_value = resp
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[no_req_spec]
        )
        assert isinstance(result, RichMessage)
        assert len(reply_calls(result)) == 1
        assert reply_calls(result)[0].name == "ping"
        assert argument_value(reply_calls(result)[0]) == {}
        assert client.rescued_tool_calls == 1

    @pytest.mark.asyncio
    async def test_malformed_500_two_distinct_complete_calls_both_survive(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        second_complete = _COMPLETE_BLOCK.replace("tests/test_x.py", "tests/test_y.py")
        resp.text = _malformed_500_body(f"{_COMPLETE_BLOCK}\n{second_complete}")
        client._http.post.return_value = resp
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_write_spec()]
        )
        assert isinstance(result, RichMessage)
        assert len(reply_calls(result)) == 2
        assert {argument_value(c)["file_path"] for c in reply_calls(result)} == {
            "tests/test_x.py",
            "tests/test_y.py",
        }
        assert client.rescued_tool_calls == 2

    @pytest.mark.asyncio
    async def test_malformed_500_without_blocks_keeps_generic_nudge(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        resp.text = _malformed_500_body("<function=write> garbled fragment")
        client._http.post.return_value = resp
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_write_spec()]
        )
        assert isinstance(result, RichMessage)
        assert "Re-emitting a single, complete, well-formed tool call" in visible(
            result
        )
        assert "ONE complete" not in visible(result)

    @pytest.mark.asyncio
    async def test_malformed_500_unknown_tool_not_fabricated(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        resp.text = _malformed_500_body(
            "<tool_call>\n<function=rm_rf>\n<parameter=path>\n/\n</parameter>\n</function>\n</tool_call>"
        )
        client._http.post.return_value = resp
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_write_spec()]
        )
        assert isinstance(result, RichMessage)
        assert client.rescued_tool_calls == 0

    @pytest.mark.asyncio
    async def test_arbitrary_500_cascades_as_backend_error(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        resp.text = '{"error":{"message":"CUDA out of memory","type":"server_error"}}'
        client._http.post.return_value = resp
        with pytest.raises(BackendError) as excinfo:
            await send(
                client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
            )
        assert "CUDA out of memory" not in str(excinfo.value)
        assert "CUDA out of memory" in excinfo.value.body

    @pytest.mark.asyncio
    async def test_malformed_500_rescued_without_parse_phrase(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        resp.text = _renamed_500_body(_COMPLETE_BLOCK)
        assert "Failed to parse input" not in resp.text
        client._http.post.return_value = resp
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_write_spec()]
        )
        assert isinstance(result, RichMessage)
        assert len(reply_calls(result)) == 1
        assert argument_value(reply_calls(result)[0])["content"].startswith(
            "import unittest"
        )
        assert client.rescued_tool_calls == 1

    @pytest.mark.asyncio
    async def test_malformed_500_no_phrase_skeleton_only_returns_nudge(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        resp.text = _renamed_500_body(_SKELETON_BLOCK)
        client._http.post.return_value = resp
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_write_spec()]
        )
        assert isinstance(result, RichMessage)
        assert "ONE complete" in visible(result)
        assert client.rescued_tool_calls == 0

    @pytest.mark.asyncio
    async def test_tools_schema_dump_500_cascades(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        resp.text = json.dumps(
            {
                "error": {
                    "message": 'Failed to parse tools: bad schema; tools = [{"type":"function","function":{"name":"write"}}]',
                    "type": "server_error",
                }
            }
        )
        client._http.post.return_value = resp
        with pytest.raises(BackendError):
            await send(
                client,
                [{"role": "user", "content": "test"}],
                tools=[_make_write_spec()],
            )

    @pytest.mark.asyncio
    async def test_truncated_open_tag_500_returns_generic_nudge(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        resp.text = json.dumps(
            {
                "error": {
                    "code": 500,
                    "message": "Failed to parse input at pos 84: <tool_call",
                    "type": "server_error",
                }
            }
        )
        client._http.post.return_value = resp
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_write_spec()]
        )
        assert isinstance(result, RichMessage)
        assert "ONE complete" not in visible(result)
        assert client.rescued_tool_calls == 0

    @pytest.mark.asyncio
    async def test_b9656_generic_parse_500_cascades(self) -> None:
        client = _make_client("native")
        resp = MagicMock()
        resp.status_code = 500
        resp.text = _b9656_generic_parse_500_body()
        client._http.post.return_value = resp
        with pytest.raises(BackendError):
            await send(
                client,
                [{"role": "user", "content": "test"}],
                tools=[_make_write_spec()],
            )
        assert client.rescued_tool_calls == 0

    @pytest.mark.asyncio
    async def test_missing_choices_raises_backend_error(self) -> None:
        client = _make_client("native")
        client._http.post.return_value = _mock_response({"object": "error"})
        with pytest.raises(BackendError, match="response has no choices"):
            await send(client, [{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_arguments_parsed_from_string(self) -> None:
        """OpenAI format sends arguments as JSON string, not dict."""
        client = _make_client("native")
        client._http.post.return_value = _mock_response(
            _openai_tool_call_response("get_pricing", '{"part": "X", "qty": 100}')
        )
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert argument_value(reply_calls(result)[0]) == {"part": "X", "qty": 100}

    @pytest.mark.asyncio
    async def test_malformed_args_kept_as_raw(self) -> None:
        """Malformed argument JSON is NOT coerced to {} or collapsed to a
        TextResponse: the raw (non-dict) string rides through on the ToolCall
        so ResponseValidator routes it to the tool-error channel."""
        client = _make_client("native")
        client._http.post.return_value = _mock_response(
            _openai_tool_call_response("get_pricing", "{not json")
        )
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reply_calls(result)[0].name == "get_pricing"
        assert argument_value(reply_calls(result)[0]) == "{not json"

    @pytest.mark.asyncio
    async def test_captures_reasoning_with_tool_call(self) -> None:
        """When content accompanies tool_calls, it is captured as reasoning."""
        client = _make_client("native")
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I should check the pricing.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_pricing",
                                    "arguments": '{"part": "X"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert visible(result) == "I should check the pricing."

    @pytest.mark.asyncio
    async def test_reasoning_content_preferred_over_content(self) -> None:
        """reasoning_content field (reasoning model) is preferred over content."""
        client = _make_client("native")
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Final answer",
                        "reasoning_content": "I need to think about this carefully...",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_pricing",
                                    "arguments": '{"part": "X"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "I need to think about this carefully..."

    @pytest.mark.asyncio
    async def test_content_used_when_no_reasoning_content(self) -> None:
        """Without reasoning_content, content is used as reasoning (instruct model)."""
        client = _make_client("native")
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me check.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_pricing",
                                    "arguments": '{"part": "X"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert visible(result) == "Let me check."

    @pytest.mark.asyncio
    async def test_null_content_gives_no_reasoning(self) -> None:
        """None/null content alongside tool_calls → reasoning is None."""
        client = _make_client("native")
        client._http.post.return_value = _mock_response(_openai_tool_call_response())
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reasoning(result) == ""

    @pytest.mark.asyncio
    async def test_empty_content_gives_no_reasoning(self) -> None:
        """Empty string content alongside tool_calls → reasoning is None."""
        client = _make_client("native")
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_pricing",
                                    "arguments": '{"part": "X"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reasoning(result) == ""

    @pytest.mark.asyncio
    async def test_native_mode_preserves_tool_role(self) -> None:
        """Native mode passes role='tool' and structured tool_calls through
        unchanged — llama-server with --jinja supports them natively."""
        client = _make_client("native")
        client._http.post.return_value = _mock_response(_openai_text_response("ok"))
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "fetch", "arguments": {}}}],
            },
            {"role": "tool", "content": "fetch_result"},
        ]
        await send(client, messages)
        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        sent_messages = body["messages"]
        assert sent_messages[3]["role"] == "tool"
        assert sent_messages[3]["content"] == "fetch_result"
        assert "tool_calls" in sent_messages[2]
        assert sent_messages[2]["tool_calls"][0]["function"]["name"] == "fetch"


class TestLlamafilePromptSend:
    @pytest.mark.asyncio
    async def test_extracts_tool_call_from_text(self) -> None:
        client = _make_client("prompt")
        client._http.post.return_value = _mock_response(
            _openai_text_response('{"tool": "get_pricing", "args": {"part": "X"}}')
        )
        result = await send(
            client,
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "test"}],
            tools=[_make_spec()],
        )
        assert isinstance(result, RichMessage)
        assert reply_calls(result)[0].name == "get_pricing"

    @pytest.mark.asyncio
    async def test_returns_text_when_no_json(self) -> None:
        client = _make_client("prompt")
        client._http.post.return_value = _mock_response(
            _openai_text_response("I don't know which tool to call")
        )
        result = await send(
            client,
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "test"}],
            tools=[_make_spec()],
        )
        assert isinstance(result, RichMessage)

    @pytest.mark.asyncio
    async def test_injects_tool_prompt_into_system_message(self) -> None:
        client = _make_client("prompt")
        client._http.post.return_value = _mock_response(
            _openai_text_response("no tool call")
        )
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "test"},
        ]
        await send(client, messages, tools=[_make_spec()])
        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        sent_system = body["messages"][0]["content"]
        assert "get_pricing" in sent_system
        assert "You are helpful." in sent_system
        assert messages[0]["content"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_no_tools_parameter_in_request(self) -> None:
        client = _make_client("prompt")
        client._http.post.return_value = _mock_response(_openai_text_response("ok"))
        await send(
            client,
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "test"}],
            tools=[_make_spec()],
        )
        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "tools" not in body

    @pytest.mark.asyncio
    async def test_tool_role_downgraded_in_prompt_mode(self) -> None:
        """Prompt mode also downgrades role='tool' to role='user'."""
        client = _make_client("prompt")
        client._http.post.return_value = _mock_response(
            _openai_text_response("no tool call")
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "fetch", "arguments": {}}}],
            },
            {"role": "tool", "content": "fetch_result"},
        ]
        await send(client, messages, tools=[_make_spec()])
        call_args = client._http.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        sent_messages = body["messages"]
        assert sent_messages[3]["role"] == "user"
        assert sent_messages[3]["content"] == "fetch_result"


class TestLlamafileGetContextLength:
    @pytest.mark.asyncio
    async def test_parses_n_ctx(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response(
            {"default_generation_settings": {"n_ctx": 8192}}
        )
        result = await client.get_context_length()
        assert result == 8192

    @pytest.mark.asyncio
    async def test_strips_v1_from_url(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response(
            {"default_generation_settings": {"n_ctx": 4096}}
        )
        await client.get_context_length()
        call_args = client._http.get.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert url == "http://test:8080/props"

    @pytest.mark.asyncio
    async def test_returns_none_on_missing_data(self) -> None:
        client = _make_client()
        client._http.get.return_value = _mock_response({})
        result = await client.get_context_length()
        assert result is None

    @pytest.mark.asyncio
    async def test_propagates_error(self) -> None:
        client = _make_client()
        client._http.get.side_effect = Exception("connection error")
        with pytest.raises(Exception, match="connection error"):
            await client.get_context_length()


class _MockSSEStreamResponse:
    """Mock for httpx streaming response with SSE format."""

    status_code = 200

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def aread(self):
        return "".join([line async for line in self.aiter_lines()]).encode()


class _MockSSE500Response:
    """Mock httpx streaming response that returns an error status body.

    Defaults to 500. Yields the body as a SINGLE line — llama.cpp's 500 is
    compact single-line JSON (embedded newlines are JSON-escaped), matching how
    send_stream reassembles error_body by concatenation.
    """

    def __init__(self, body: str, split: bool = False, status_code: int = 500) -> None:
        self.status_code = status_code
        self._body = body
        self._split = split

    async def aiter_lines(self):
        if self._split:
            parts = self._body.split(",")
            for i, part in enumerate(parts):
                yield (part + ("," if i < len(parts) - 1 else ""))
        else:
            yield self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def aread(self):
        return "".join([line async for line in self.aiter_lines()]).encode()


class TestLlamafileSendStream:
    @pytest.mark.asyncio
    async def test_yields_text_deltas_and_final(self) -> None:
        client = _make_client("native")
        sse_lines = [
            f"data: {json.dumps({'choices': [{'delta': {'content': 'Hello'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': ' world'}}]})}",
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)
        chunks = []
        async for chunk in send_stream(client, [{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
        assert client._http.stream.call_args.args[:2] == (
            "POST",
            "http://test:8080/v1/chat/completions",
        )
        text_deltas = [c for c in chunks if c.type == ChunkType.TEXT_DELTA]
        assert len(text_deltas) == 2
        assert text_deltas[0].content == "Hello"
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, RichMessage)
        assert visible(finals[0].response) == "Hello world"

    @pytest.mark.asyncio
    async def test_yields_tool_call_from_stream(self) -> None:
        client = _make_client("native")
        sse_lines = [
            f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'get_pricing', 'arguments': ''}}]}}]})}",
            f"""data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"part":'}}]}}]})}""",
            f"""data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "X"}'}}]}}]})}""",
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)
        chunks = []
        async for chunk in send_stream(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)
        tc_deltas = [c for c in chunks if c.type == ChunkType.TOOL_CALL_DELTA]
        assert len(tc_deltas) >= 1
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, RichMessage)
        assert reply_calls(finals[0].response)[0].name == "get_pricing"

    @pytest.mark.asyncio
    async def test_streaming_captures_reasoning_with_tool_call(self) -> None:
        """Content streamed before tool_calls is captured as reasoning."""
        client = _make_client("native")
        sse_lines = [
            f"data: {json.dumps({'choices': [{'delta': {'content': 'Let me '}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': 'check...'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'get_pricing', 'arguments': ''}}]}}]})}",
            f"""data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"part": "X"}'}}]}}]})}""",
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)
        chunks = []
        async for chunk in send_stream(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)
        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, RichMessage)
        assert visible(final.response) == "Let me check..."

    @pytest.mark.asyncio
    async def test_streaming_reasoning_content_preferred(self) -> None:
        """Streamed reasoning_content is preferred over content for reasoning."""
        client = _make_client("native")
        sse_lines = [
            f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': 'Let me '}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': 'reason...'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': 'Final.'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'get_pricing', 'arguments': ''}}]}}]})}",
            f"""data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"part": "X"}'}}]}}]})}""",
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)
        chunks = []
        async for chunk in send_stream(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)
        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, RichMessage)
        assert reasoning(final.response) == "Let me reason..."

    @pytest.mark.asyncio
    async def test_streaming_no_reasoning_when_no_content(self) -> None:
        """Tool call stream with no content deltas → reasoning is None."""
        client = _make_client("native")
        sse_lines = [
            f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'get_pricing', 'arguments': ''}}]}}]})}",
            f"""data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"part": "X"}'}}]}}]})}""",
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)
        chunks = []
        async for chunk in send_stream(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)
        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, RichMessage)
        assert reasoning(final.response) == ""

    @pytest.mark.asyncio
    async def test_stream_malformed_500_stutter_returns_only_complete(self) -> None:
        client = _make_client("native")
        client._http.stream.return_value = _MockSSE500Response(
            _malformed_500_body(f"{_SKELETON_BLOCK}\n{_COMPLETE_BLOCK}")
        )
        chunks = [
            c
            async for c in send_stream(
                client,
                [{"role": "user", "content": "test"}],
                tools=[_make_write_spec()],
            )
        ]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, RichMessage)
        assert len(reply_calls(finals[0].response)) == 1
        assert reply_calls(finals[0].response)[0].name == "write"
        assert argument_value(reply_calls(finals[0].response)[0])["content"].startswith(
            "import unittest"
        )
        assert client.rescued_tool_calls == 1

    @pytest.mark.asyncio
    async def test_stream_malformed_500_skeleton_only_returns_nudge(self) -> None:
        client = _make_client("native")
        client._http.stream.return_value = _MockSSE500Response(
            _malformed_500_body(_SKELETON_BLOCK)
        )
        chunks = [
            c
            async for c in send_stream(
                client,
                [{"role": "user", "content": "test"}],
                tools=[_make_write_spec()],
            )
        ]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, RichMessage)
        assert "ONE complete" in visible(finals[0].response)
        assert "Failed to parse input" not in visible(finals[0].response)
        assert client.rescued_tool_calls == 0

    @pytest.mark.asyncio
    async def test_stream_malformed_500_reassembles_multiline_body(self) -> None:
        client = _make_client("native")
        client._http.stream.return_value = _MockSSE500Response(
            _malformed_500_body(f"{_SKELETON_BLOCK}\n{_COMPLETE_BLOCK}"), split=True
        )
        chunks = [
            c
            async for c in send_stream(
                client,
                [{"role": "user", "content": "test"}],
                tools=[_make_write_spec()],
            )
        ]
        finals = [c for c in chunks if c.type == ChunkType.FINAL]
        assert len(finals) == 1
        assert isinstance(finals[0].response, RichMessage)
        assert len(reply_calls(finals[0].response)) == 1
        assert argument_value(reply_calls(finals[0].response)[0])["content"].startswith(
            "import unittest"
        )
        assert client.rescued_tool_calls == 1

    @pytest.mark.asyncio
    async def test_stream_arbitrary_500_cascades(self) -> None:
        client = _make_client("native")
        client._http.stream.return_value = _MockSSE500Response(
            '{"error":{"message":"CUDA out of memory","type":"server_error"}}'
        )
        with pytest.raises(BackendError) as excinfo:
            async for _ in send_stream(
                client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
            ):
                pass
        assert "CUDA out of memory" not in str(excinfo.value)
        assert "CUDA out of memory" in excinfo.value.body


class TestLlamafileNon200:
    """Both chat-completion paths classify every non-200 as ``BackendError``.

    A non-200 stream body carries no ``data:`` lines, so without an explicit
    classification the SSE loop falls through and yields an empty FINAL as if
    the turn had succeeded. That hits ``WorkflowRunner(stream=True)`` and any
    direct ``send_stream()`` caller: an empty assistant turn instead of a
    visible dead backend.
    """

    @pytest.mark.asyncio
    async def test_send_stream_503_raises_backend_error(self) -> None:
        client = _make_client("native")
        client._http.stream.return_value = _MockSSE500Response(
            "<html>backend restarting</html>", status_code=503
        )
        chunks = []
        with pytest.raises(BackendError) as excinfo:
            async for chunk in send_stream(
                client, [{"role": "user", "content": "test"}]
            ):
                chunks.append(chunk)
        assert excinfo.value.status_code == 503
        assert excinfo.value.body == "<html>backend restarting</html>"
        assert [c for c in chunks if c.type == ChunkType.FINAL] == []

    @pytest.mark.asyncio
    async def test_send_stream_429_raises_backend_error(self) -> None:
        client = _make_client("native")
        client._http.stream.return_value = _MockSSE500Response(
            '{"error":{"message":"rate limited"}}', status_code=429
        )
        chunks = []
        with pytest.raises(BackendError) as excinfo:
            async for chunk in send_stream(
                client, [{"role": "user", "content": "test"}]
            ):
                chunks.append(chunk)
        assert excinfo.value.status_code == 429
        assert excinfo.value.body == '{"error":{"message":"rate limited"}}'
        assert [c for c in chunks if c.type == ChunkType.FINAL] == []

    @pytest.mark.asyncio
    async def test_send_stream_non_malformed_500_still_cascades(self) -> None:
        client = _make_client("native")
        client._http.stream.return_value = _MockSSE500Response(
            '{"error":{"message":"slot unavailable","type":"server_error"}}'
        )
        with pytest.raises(BackendError) as excinfo:
            async for _ in send_stream(
                client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
            ):
                pass
        assert excinfo.value.status_code == 500
        assert "slot unavailable" in excinfo.value.body

    @pytest.mark.asyncio
    async def test_prompt_mode_send_503_raises_backend_error_not_httpx(self) -> None:
        client = _make_client("prompt")
        resp = _mock_response({}, status_code=503)
        resp.text = "<html>backend restarting</html>"
        client._http.post.return_value = resp
        with pytest.raises(BackendError) as excinfo:
            await send(
                client,
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "test"},
                ],
                tools=[_make_spec()],
            )
        assert not isinstance(excinfo.value, httpx.HTTPStatusError)
        assert excinfo.value.status_code == 503
        assert excinfo.value.body == "<html>backend restarting</html>"


class TestMode:
    def test_auto_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode must be 'native' or 'prompt'"):
            LlamafileClient(gguf_path="test", mode="auto")


class TestApplySampling:
    """Tests that sampling kwargs land in llama-server request bodies."""

    @pytest.mark.parametrize(
        "mode_kwargs", [{}, {"mode": "prompt"}], ids=["native-default", "prompt"]
    )
    @pytest.mark.asyncio
    async def test_instance_sampling_on_wire(self, mode_kwargs: dict[str, str]) -> None:
        client = LlamafileClient(
            base_url="http://test:8080/v1",
            gguf_path="test-model",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            repeat_penalty=1.05,
            presence_penalty=1.5,
            **mode_kwargs,
        )
        client._http.post.return_value = _mock_response(_openai_text_response("hi"))
        await send(client, [{"role": "user", "content": "hi"}])
        body = client._http.post.await_args.kwargs["json"]
        assert {
            key: body[key]
            for key in (
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repeat_penalty",
                "presence_penalty",
            )
        } == {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.05,
            "presence_penalty": 1.5,
        }

    @pytest.mark.parametrize(
        "mode_kwargs", [{}, {"mode": "prompt"}], ids=["native-default", "prompt"]
    )
    @pytest.mark.asyncio
    async def test_default_sampling_absent_on_wire(
        self, mode_kwargs: dict[str, str]
    ) -> None:
        client = LlamafileClient(
            base_url="http://test:8080/v1", gguf_path="test-model", **mode_kwargs
        )
        client._http.post.return_value = _mock_response(_openai_text_response("hi"))
        await send(client, [{"role": "user", "content": "hi"}])
        body = client._http.post.await_args.kwargs["json"]
        assert (
            not {
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repeat_penalty",
                "presence_penalty",
            }
            & body.keys()
        )


class TestMergeConsecutive:
    def test_merges_consecutive_user_messages(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "retry nudge"},
        ]
        result = _merge_consecutive(messages)
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "hello\n\nretry nudge"

    def test_merges_consecutive_assistant_messages(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "I should check pricing."},
            {"role": "assistant", "content": "[tool_call] get_pricing({})"},
        ]
        result = _merge_consecutive(messages)
        assert len(result) == 3
        assert result[2]["role"] == "assistant"
        assert "I should check pricing." in result[2]["content"]
        assert "[tool_call]" in result[2]["content"]

    def test_does_not_merge_across_tool_calls(self) -> None:
        messages = [
            {"role": "assistant", "content": "thinking..."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "fetch", "arguments": {}}}],
            },
        ]
        result = _merge_consecutive(messages)
        assert len(result) == 2
        assert "tool_calls" in result[1]

    def test_does_not_merge_tool_role(self) -> None:
        messages = [
            {"role": "tool", "content": "result1"},
            {"role": "tool", "content": "result2"},
        ]
        assert len(_merge_consecutive(messages)) == 2

    def test_preserves_correct_alternation(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
        ]
        assert len(_merge_consecutive(messages)) == 4

    def test_empty_messages(self) -> None:
        assert _merge_consecutive([]) == []

    def test_does_not_merge_into_tool_calls(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "a", "arguments": {}}}],
            },
            {"role": "assistant", "content": "reasoning"},
        ]
        assert len(_merge_consecutive(messages)) == 2

    def test_merges_users_separated_by_invisible_messages(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "fetch", "arguments": {}}}],
            },
            {"role": "tool", "content": "result"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "report", "arguments": {}}}],
            },
            {"role": "user", "content": "step nudge"},
        ]
        result = _merge_consecutive(messages)
        user_msgs = [m for m in result if m["role"] == "user" and "tool_calls" not in m]
        assert len(user_msgs) == 1
        assert "go" in user_msgs[0]["content"]
        assert "step nudge" in user_msgs[0]["content"]

    def test_merges_list_content_into_string_message(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "kept reasoning"}],
            },
            {"role": "assistant", "content": "final answer"},
        ]
        result = _merge_consecutive(messages)
        assert len(result) == 1
        assert result[0]["content"] == [
            {"type": "text", "text": "kept reasoning"},
            {"type": "text", "text": "\n\n"},
            {"type": "text", "text": "final answer"},
        ]

    def test_merges_string_message_into_list_content(self) -> None:
        messages = [
            {"role": "user", "content": "question"},
            {"role": "user", "content": [{"type": "text", "text": "follow-up"}]},
        ]
        result = _merge_consecutive(messages)
        assert len(result) == 1
        assert result[0]["content"] == [
            {"type": "text", "text": "question"},
            {"type": "text", "text": "\n\n"},
            {"type": "text", "text": "follow-up"},
        ]

    def test_merges_two_list_contents(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "part three"}]},
        ]
        result = _merge_consecutive(messages)
        assert len(result) == 1
        assert result[0]["content"] == [
            {"type": "text", "text": "part one"},
            {"type": "text", "text": "part two"},
            {"type": "text", "text": "\n\n"},
            {"type": "text", "text": "part three"},
        ]


class TestThinkFlagSync:
    @pytest.mark.asyncio
    async def test_think_true_captures_think_tags_as_reasoning(self) -> None:
        """think=True: [THINK] content parsed into ToolCall.reasoning."""
        client = _make_client("native", think=True)
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "[THINK]I should check pricing.[/THINK]",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_pricing",
                                    "arguments": '{"part": "X"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "I should check pricing."

    @pytest.mark.asyncio
    async def test_think_false_discards_reasoning_content(self) -> None:
        """think=False: reasoning_content field discarded."""
        client = _make_client("native", think=False)
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Deep reasoning here...",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_pricing",
                                    "arguments": '{"part": "X"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reasoning(result) == ""

    @pytest.mark.asyncio
    async def test_think_false_discards_think_tags(self) -> None:
        """think=False: [THINK] tags in content discarded."""
        client = _make_client("native", think=False)
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "[THINK]Reasoning here[/THINK]",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_pricing",
                                    "arguments": '{"part": "X"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reasoning(result) == ""

    @pytest.mark.asyncio
    async def test_think_true_reasoning_content_preferred(self) -> None:
        """think=True: reasoning_content field preferred over [THINK] in content."""
        client = _make_client("native", think=True)
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "[THINK]Content reasoning[/THINK]",
                        "reasoning_content": "Server-parsed reasoning",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_pricing",
                                    "arguments": '{"part": "X"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "Server-parsed reasoning"

    @pytest.mark.asyncio
    async def test_think_default_is_true(self) -> None:
        """think=None (auto) defaults to capturing reasoning."""
        client = _make_client("native")
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "[THINK]Auto-captured reasoning[/THINK]",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_pricing",
                                    "arguments": '{"part": "X"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert reasoning(result) == "Auto-captured reasoning"

    @pytest.mark.asyncio
    async def test_think_tags_stripped_from_text_response(self) -> None:
        """[THINK] tags are always stripped from TextResponse content."""
        client = _make_client("native", think=True)
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "[THINK]Some reasoning[/THINK]I don't know which tool to call.",
                    }
                }
            ]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await send(client, [{"role": "user", "content": "test"}])
        assert isinstance(result, RichMessage)
        assert visible(result) == "I don't know which tool to call."
        assert "[THINK]" not in visible(result)

    @pytest.mark.asyncio
    async def test_content_fallback_when_no_think_tags(self) -> None:
        """think=True, no tags, no reasoning_content: content used as reasoning."""
        client = _make_client("native", think=True)
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me check the pricing.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_pricing",
                                    "arguments": '{"part": "X"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        client._http.post.return_value = _mock_response(response_data)
        result = await send(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        )
        assert isinstance(result, RichMessage)
        assert visible(result) == "Let me check the pricing."


class TestThinkFlagStream:
    @pytest.mark.asyncio
    async def test_stream_think_true_captures_think_tags(self) -> None:
        """Streaming: [THINK] tags in accumulated content → reasoning."""
        client = _make_client("native", think=True)
        sse_lines = [
            f"data: {json.dumps({'choices': [{'delta': {'content': '[THINK]I need to '}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': 'check.[/THINK]'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'get_pricing', 'arguments': ''}}]}}]})}",
            f"""data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"part": "X"}'}}]}}]})}""",
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)
        chunks = []
        async for chunk in send_stream(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)
        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, RichMessage)
        assert reasoning(final.response) == "I need to check."

    @pytest.mark.asyncio
    async def test_stream_think_false_discards_reasoning(self) -> None:
        """Streaming: think=False discards reasoning_content."""
        client = _make_client("native", think=False)
        sse_lines = [
            f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': 'Thinking...'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'get_pricing', 'arguments': ''}}]}}]})}",
            f"""data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"part": "X"}'}}]}}]})}""",
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)
        chunks = []
        async for chunk in send_stream(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)
        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, RichMessage)
        assert reasoning(final.response) == ""

    @pytest.mark.asyncio
    async def test_stream_think_false_discards_think_tags(self) -> None:
        """Streaming: think=False discards [THINK] tags from content."""
        client = _make_client("native", think=False)
        sse_lines = [
            f"data: {json.dumps({'choices': [{'delta': {'content': '[THINK]reasoning[/THINK]'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'get_pricing', 'arguments': ''}}]}}]})}",
            f"""data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"part": "X"}'}}]}}]})}""",
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)
        chunks = []
        async for chunk in send_stream(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)
        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, RichMessage)
        assert reasoning(final.response) == ""

    @pytest.mark.asyncio
    async def test_stream_reasoning_content_preferred_over_tags(self) -> None:
        """Streaming: reasoning_content preferred over [THINK] in content."""
        client = _make_client("native", think=True)
        sse_lines = [
            f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': 'Server reasoning'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': '[THINK]Content reasoning[/THINK]'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'get_pricing', 'arguments': ''}}]}}]})}",
            f"""data: {json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"part": "X"}'}}]}}]})}""",
            "data: [DONE]",
        ]
        client._http.stream.return_value = _MockSSEStreamResponse(sse_lines)
        chunks = []
        async for chunk in send_stream(
            client, [{"role": "user", "content": "test"}], tools=[_make_spec()]
        ):
            chunks.append(chunk)
        final = [c for c in chunks if c.type == ChunkType.FINAL][0]
        assert isinstance(final.response, RichMessage)
        assert reasoning(final.response) == "Server reasoning"


class TestIdSlot:
    """id_slot injection into request bodies."""

    @pytest.mark.parametrize("id_slot", [None, 1])
    @pytest.mark.asyncio
    async def test_native_send_id_slot_on_wire(self, id_slot: int | None) -> None:
        client = LlamafileClient(
            base_url="http://test:8080/v1",
            gguf_path="test",
            mode="native",
            id_slot=id_slot,
        )
        mock_http = AsyncMock()
        mock_http.headers = {}
        client._http = mock_http
        tool_call_data = {
            "choices": [
                {
                    "message": {"content": "hello", "tool_calls": None},
                    "finish_reason": "stop",
                }
            ]
        }
        mock_http.post.return_value = _mock_response(tool_call_data)
        await send(client, [{"role": "user", "content": "test"}], tools=None)
        call_kwargs = mock_http.post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        if id_slot is None:
            assert "id_slot" not in body
        else:
            assert body["id_slot"] == id_slot


class TestRecommendedSampling:
    """Issue B: recommended_sampling flag on LlamafileClient."""

    def test_strict_known_model_applies_map_values(self) -> None:
        """recommended_sampling=True + known model: map values populate fields."""
        client = LlamafileClient(
            gguf_path="qwen3:8b-q4_K_M", mode="native", recommended_sampling=True
        )
        assert client.temperature == 0.6
        assert client.top_p == 0.95
        assert client.top_k == 20
        assert client.min_p == 0.0

    def test_inkling_shard_applies_sampling_and_effort(self) -> None:
        """The production Inkling shard name resolves to its complete row."""
        client = LlamafileClient(
            gguf_path="Inkling-Small-UD-IQ4_XS-00001-of-00004.gguf",
            mode="native",
            recommended_sampling=True,
        )
        assert client.model == "Inkling-Small-UD-IQ4_XS"
        assert client.chat_template_kwargs == {"reasoning_effort": "max"}
        assert client.temperature == 1.0
        assert client.top_p == 1.0
        assert client.top_k is None
        assert client.min_p == 0.0

    def test_strict_unknown_model_raises(self) -> None:
        """recommended_sampling=True + unknown model: raises UnsupportedModelError."""
        from forge.errors import UnsupportedModelError

        with pytest.raises(UnsupportedModelError):
            LlamafileClient(
                gguf_path="nonexistent-model:1b",
                mode="native",
                recommended_sampling=True,
            )

    def test_explicit_kwarg_wins_over_map(self) -> None:
        """Caller's explicit kwarg overrides the map entry field-by-field."""
        client = LlamafileClient(
            gguf_path="qwen3:8b-q4_K_M",
            mode="native",
            recommended_sampling=True,
            temperature=0.99,
        )
        assert client.temperature == 0.99
        assert client.top_p == 0.95
        assert client.top_k == 20

    def test_default_no_opt_in_no_map_values(self) -> None:
        """recommended_sampling=False (default) + known model: map values not applied."""
        client = LlamafileClient(gguf_path="qwen3:8b-q4_K_M", mode="native")
        assert client.temperature is None
        assert client.top_p is None
        assert client.top_k is None


class TestPerCallSampling:
    """Issue A: per-call sampling overrides on send."""

    @pytest.mark.asyncio
    async def test_per_call_sampling_overrides_instance(self) -> None:
        """sampling=... on send() overrides instance fields for this call only."""
        client = LlamafileClient(
            base_url="http://test:8080/v1",
            gguf_path="test-model",
            mode="native",
            temperature=0.7,
            top_p=0.9,
        )
        mock_http = AsyncMock()
        mock_http.headers = {}
        mock_http.stream = MagicMock()
        client._http = mock_http
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {"content": "ok", "tool_calls": None},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        await send(
            client,
            [{"role": "user", "content": "hi"}],
            tools=None,
            sampling={"temperature": 0.0, "seed": 42},
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["temperature"] == 0.0
        assert body["seed"] == 42
        assert body["top_p"] == 0.9
        assert client.temperature == 0.7
        assert client.top_p == 0.9

    @pytest.mark.asyncio
    async def test_per_call_sampling_none_uses_instance(self) -> None:
        """sampling=None: only instance fields go on the wire."""
        client = LlamafileClient(
            base_url="http://test:8080/v1",
            gguf_path="test-model",
            mode="native",
            temperature=0.5,
        )
        mock_http = AsyncMock()
        mock_http.headers = {}
        mock_http.stream = MagicMock()
        client._http = mock_http
        client._http.post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {"content": "ok", "tool_calls": None},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        await send(
            client, [{"role": "user", "content": "hi"}], tools=None, sampling=None
        )
        body = client._http.post.call_args.kwargs["json"]
        assert body["temperature"] == 0.5
        assert "seed" not in body


class TestModelIdentityInvariant:
    """client.model == client.sampling_key is an invariant."""

    def test_model_equals_sampling_key_for_every_path_shape(self) -> None:
        cases = [
            (Path("mimo-v2.5.gguf"), "mimo-v2.5"),
            (Path("Model.Q4_K_M.llamafile"), "Model.Q4_K_M"),
            (Path("model-00001-of-00003.gguf"), "model"),
            (Path("qwen3:8b-q4_K_M.gguf"), "qwen3:8b-q4_K_M"),
            (Path("custom-model"), "custom-model"),
            (Path("mimo-v2.5-00001-of-00003.gguf"), "mimo-v2.5"),
            ("mimo-v2.5", "mimo-v2.5"),
            (
                Path("Mistral-Nemo-Instruct-2407.Q4_K_M.llamafile"),
                "Mistral-Nemo-Instruct-2407.Q4_K_M",
            ),
        ]
        for model_path, expected in cases:
            client = LlamafileClient(gguf_path=model_path)
            assert client.model == client.sampling_key == expected, model_path


def test_native_merge_preserves_fields_first_message_extras_and_history():
    from copy import deepcopy
    from forge.core.codecs import openai_state
    from forge.clients.base import RequestMetadata

    client = _make_client()
    state = openai_state(
        [
            {
                "role": "user",
                "name": "first",
                "first": True,
                "content": [
                    {
                        "type": "text",
                        "text": "a",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c", "function": {"name": "get_pricing", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "c"},
            {
                "role": "user",
                "name": "later",
                "later": True,
                "content": [{"type": "text", "text": "b", "x": 2}],
            },
        ]
    )
    before = deepcopy(state)
    request = client.render_request(state, None, RequestMetadata())
    merged = request.payload["messages"][0]
    assert merged["content"] == [
        {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "\n\n"},
        {"type": "text", "text": "b", "x": 2},
    ]
    assert (
        merged["name"] == "first" and merged["first"] is True and "later" not in merged
    )
    assert state == before


def test_native_merge_null_content_retains_existing_empty_text_compatibility():
    assert _merge_consecutive(
        [{"role": "assistant", "content": None}, {"role": "assistant", "content": "x"}]
    ) == [{"role": "assistant", "content": "\n\nx"}]
