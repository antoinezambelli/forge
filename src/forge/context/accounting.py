"""Read-only legacy content-character estimate; no argument/schema accounting."""

from forge.core.messages import (
    ConversationState,
    TextPart,
    ReasoningPart,
    ToolResultPart,
)


def estimate_content_tokens(state: ConversationState) -> int:
    chars = 0
    for message in state.messages:
        texts = [p.text for p in message.parts if isinstance(p, TextPart)]
        chars += len("\n".join(texts))
        chars += sum(len(p.text) for p in message.parts if isinstance(p, ReasoningPart))
        for part in message.parts:
            if isinstance(part, ToolResultPart):
                chars += len("\n".join(p.text for p in part.content))
    return chars // 4
