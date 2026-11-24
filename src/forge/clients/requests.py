"""Request-field translation; model input is always rendered from state."""

from copy import deepcopy
from forge.clients.base import RequestMetadata


def request_extras(metadata: RequestMetadata, target: str) -> dict:
    body = {
        k: deepcopy(v)
        for k, v in metadata.extra_fields.items()
        if k not in {"messages", "tools", "system", "stream", "stream_options"}
    }
    if metadata.model is not None:
        body["model"] = metadata.model
    if metadata.source_protocol == "anthropic" and target == "openai":
        body = {
            k: v
            for k, v in body.items()
            if k
            in {
                "model",
                "litellm_session_id",
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "stop_sequences",
                "tool_choice",
            }
        }
        if "stop_sequences" in body:
            body["stop"] = body.pop("stop_sequences")
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            kind = choice.get("type")
            body["tool_choice"] = {
                "auto": "auto",
                "any": "required",
                "none": "none",
            }.get(kind, choice)
            if kind == "tool":
                body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": choice.get("name", "")},
                }
    elif metadata.source_protocol == "openai" and target == "anthropic":
        if "stop" in body:
            stop = body.pop("stop")
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        if "max_completion_tokens" in body:
            body["max_tokens"] = body.pop("max_completion_tokens")
        choice = body.get("tool_choice")
        if isinstance(choice, str):
            body["tool_choice"] = {"type": {"required": "any"}.get(choice, choice)}
        elif isinstance(choice, dict) and choice.get("type") == "function":
            body["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
    return body
