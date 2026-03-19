"""SSE stream buffering and replay.

Consumes an SSE stream from the backend, buffers all chunks, and
provides the raw chunks for replay to the client.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx


@dataclass
class BufferedStream:
    """A fully buffered SSE stream from the backend.

    Attributes:
        chunks: Raw SSE lines (including "data: " prefix) in order.
        complete: True if the stream ended with "data: [DONE]".
    """

    chunks: list[str] = field(default_factory=list)
    complete: bool = False


async def buffer_sse_stream(response: httpx.Response) -> BufferedStream:
    """Consume an SSE stream from the backend and buffer all chunks.

    Args:
        response: An httpx streaming response (content-type: text/event-stream).

    Returns:
        BufferedStream with all SSE data lines captured.
    """
    buf = BufferedStream()
    async for line in response.aiter_lines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            buf.chunks.append(line)
            if line == "data: [DONE]":
                buf.complete = True
    return buf


def iter_sse_bytes(buf: BufferedStream) -> list[bytes]:
    """Convert buffered SSE chunks to bytes for replay to the client.

    Each chunk becomes "data: ...\n\n" as per the SSE spec.
    """
    return [(chunk + "\n\n").encode() for chunk in buf.chunks]
