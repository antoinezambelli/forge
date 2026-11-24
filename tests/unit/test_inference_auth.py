"""Per-call credentials belong to metadata and the rendered request on every attempt."""

import pytest
from forge.clients.base import RequestMetadata
from forge.core.codecs import openai_state
from forge.core.inference import run_inference
from forge.core.messages import TextPart
from forge.context.manager import ContextManager
from forge.context.strategies import NoCompact
from forge.guardrails import ResponseValidator, ErrorTracker
from tests.conftest import MockClient
from tests.canonical import assistant, call


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("headers", [None, {"Authorization": "Bearer rotating"}])
async def test_metadata_credentials_reach_every_rendered_attempt(stream, headers):
    client = MockClient([assistant(TextPart("retry")), assistant(call())])
    await run_inference(
        openai_state([]),
        client,
        ContextManager(NoCompact(), None),
        ResponseValidator(["search"]),
        ErrorTracker(),
        [],
        RequestMetadata(extra_headers=headers),
        stream=stream,
    )
    requests = client.send_stream_calls if stream else client.send_calls
    assert len(requests) == 2
    assert all(request.headers == headers for request in requests)
