# Migrating to canonical messages and tool definitions

The canonical-primitives update changes Forge's Python message, response, and
tool-definition types. **Proxy users: no setup changes are required.** Your
existing supported HTTP integration and Proxy configuration remain valid;
review the [behavior notes](#behavior-notes-including-proxy-users) below.

## Do I need to change anything?

| How you use Forge | Migration |
|---|---|
| Standalone or Python-hosted Proxy, using its HTTP endpoints | No CLI/configuration or client-code migration for ordinary supported requests. Review the behavior notes below. |
| WorkflowRunner | Update tool registration. Also update callbacks or saved-history code if it accesses the old message types. |
| Guardrails in your own loop | Supply RichMessage replies and use the new tool-call accessors. Your loop still owns execution and history. |
| Direct clients, custom client adapters, shared inference, or custom compaction | Update the low-level Python contracts described below. |

There are no aliases or automatic converters for removed Python types. Update
code that imports or constructs those types before upgrading. This is a Python
API break even though much of the normal workflow and Proxy usage is unchanged.

## 1. WorkflowRunner: separate tool definitions from execution

`ToolDefinition` replaces `ToolSpec`. `ToolDef`, which combined a definition,
handler, and prerequisites, is removed. Register the same tool names in three
places as applicable:

- `Workflow.tools`: model-visible definitions.
- `Workflow.handlers`: Python callables, keyed by tool name.
- `Workflow.prerequisites`: dependencies, keyed by tool name; omit if unused.

For an existing Pydantic-authored tool, you can keep your application model.
Pass its generated JSON Schema explicitly; Forge no longer stores the model
class as its tool-definition representation.

Before:

```python
from pydantic import BaseModel
from forge import ToolSpec, ToolDef, Workflow

class LookupParams(BaseModel):
    key: str

def lookup(key: str) -> str:
    return f"record:{key}"

workflow = Workflow(
    name="lookup", description="Look up a record",
    tools={
        "lookup": ToolDef(
            spec=ToolSpec(
                name="lookup", description="Find a record by key",
                parameters=LookupParams,
            ),
            callable=lookup,
        ),
    },
    required_steps=[], terminal_tool="lookup",
    system_prompt_template="Use lookup to answer the request.",
)
```

After, with the same application model and handler:

```python
from forge import ToolDefinition, Workflow

workflow = Workflow(
    name="lookup", description="Look up a record",
    tools={
        "lookup": ToolDefinition(
            name="lookup", description="Find a record by key",
            input_schema=LookupParams.model_json_schema(),
        ),
    },
    handlers={"lookup": lookup},
    required_steps=[], terminal_tool="lookup",
    system_prompt_template="Use lookup to answer the request.",
)
```

If you already have a JSON Schema dictionary, pass it directly as input_schema.
`ToolSpec.from_json_schema(...)` is no longer necessary or available.

Move a former `ToolDef.prerequisites` value into
`Workflow.prerequisites[tool_name]`; its dependency syntax is unchanged.
Every definition needs a matching handler, and registry names are checked at
construction. The tools dictionary still determines model-visible tool order.

Ordinary execution is unchanged:

```python
value = await runner.run(workflow, "Look up key 42")
```

Handlers still receive arguments as keyword arguments. The runner still returns
the successful terminal handler's actual Python value. Required steps,
prerequisites, terminal tools, and retry/error budgets remain workflow controls.

Related replacements:

| Before | After |
|---|---|
| `workflow.get_tool_specs()` | `workflow.get_tool_definitions()` |
| `workflow.get_callable(name)` | `workflow.handlers[name]` |
| `tool_spec.get_json_schema()` | `tool_definition.input_schema` |
| `respond_spec()` | `respond_definition()` |
| `respond_tool()` in Workflow.tools | `respond_definition()` in tools, and `respond` in handlers |

The built-in respond helpers are exported from both forge and forge.tools.
If respond previously had workflow prerequisites, move those into the separate
prerequisites mapping too.

## 2. History and callbacks: one reply can contain several kinds of content

`RichMessage` replaces `Message`. Its ordered parts can contain text, reasoning,
tool calls, or tool results. A single assistant reply can contain visible text,
multiple reasoning blocks, and multiple calls together; it is not emitted as
separate artificial messages for each category.

| Before | After |
|---|---|
| `Message(role, content, MessageMeta(...))` | `RichMessage(role, [TextPart(content)])` for a plain-text message |
| `message.content` | Read the appropriate typed parts; there is no universal content string |
| `message.metadata.step_index` | `message.step_index` |
| `MessageType.REASONING` classification | `ReasoningPart` values in message.parts |
| Retry/step/prerequisite MessageType | `message.forge_purpose`: retry, step, prerequisite; other correction purposes are unknown_tool and tool_arg_validation |
| `ToolCallInfo` / response `ToolCall` | `ToolCallPart` |
| `call.tool`, `call.args` on response calls | `call.name`, `call.arguments.value` after validation |
| `message.tool_call_id` on a result | `ToolResultPart.call_id` |
| Result string in message.content | TextPart values in ToolResultPart.content, with outcome success/error |

For example, update a display callback to inspect parts instead of assuming
every message is a single text value:

```python
from forge import TextPart, ReasoningPart, ToolCallPart, ToolResultPart

def display(message):
    for part in message.parts:
        if isinstance(part, TextPart):
            print(part.text)
        elif isinstance(part, ReasoningPart):
            print("reasoning:", part.text)
        elif isinstance(part, ToolCallPart):
            print("tool:", part.name, part.arguments)
        elif isinstance(part, ToolResultPart):
            print("result:", part.call_id, part.outcome)
```

`message.tool_calls` is always a list of actual invocations, derived from parts.
It is empty when there are no calls. Do not replace it or append to the returned
list expecting to modify the message; the authoritative collection is parts.

`on_message` receives new rich messages, including rejected attempts and
corrections. Do not assume legacy callback counts or one callback per content
category. Avoid splitting/reconstructing the received assistant reply for reuse.

For multi-turn history, use `list[RichMessage]` for initial_messages. Include the
system prompt and new user turn in that list yourself. Supplied seeds are not
re-emitted to on_message and do not get another automatic system/user pair.

```python
from forge import RichMessage, MessageRole, TextPart

seed = [*conversation,
        RichMessage(MessageRole.USER, [TextPart("Look up key 43")])]
value = await runner.run(workflow, "Look up key 43", initial_messages=seed)
```

For a callback-maintained conversation, store the supplied new user turn once
yourself, and append new callback messages once. Context-warning callbacks are
transient events: exclude `forge_purpose == "context_warning"` from persistent
history. Callback history is event delivery, not a guaranteed snapshot after
configured compaction. For cancellation handlers, exception.messages likewise
contains rich messages.

`StreamChunk` and its event categories remain, but FINAL.response is now a
RichMessage, not a list of ToolCall objects or a TextResponse. FINAL delivery
occurs before Runner reply callbacks and execution; it is not a second history
entry to append alongside on_message.

For direct provider-stream callbacks, a mixed response can deliver text deltas
before tool-call fragments. Do not infer tool-execution order from that
interleaving; the Runner acts on the complete FINAL response.

## 3. Middleware: keep your loop, update its reply boundary

`Guardrails.check()` and `ResponseValidator.validate()` now accept an assistant
RichMessage. Forge's clients return this type directly. If you parse another
client's output yourself, construct typed parts at that boundary rather than
passing a raw dictionary, a legacy call list, or TextResponse.

```python
from forge import RichMessage, MessageRole, TextPart, ToolCallPart, ValidArguments

plain_reply = RichMessage(MessageRole.ASSISTANT, [TextPart("Let me check.")])
tool_reply = RichMessage(MessageRole.ASSISTANT, [
    ToolCallPart("lookup", ValidArguments({"key": "42"})),
])
result = guardrails.check(tool_reply)
```

On execute, result.tool_calls contains ToolCallPart values: dispatch by name
and read arguments.value. Keep handling retry, tool_error, step_blocked, and
fatal actions as appropriate to your existing loop; record successful execution
through guardrails.record as before. Forge's facade still does not execute
your tools or own your conversation.

ValidArguments means the arguments have dictionary/object shape; it does not
mean Forge has validated every field against input_schema. InvalidArguments
retains malformed input for rejection/history; it is not executable arguments.
This refactor does not add automatic Pydantic or full JSON-Schema validation.

## 4. Low-level integrations only

Most Proxy users and ordinary WorkflowRunner callers do not need to construct
these request objects.

Direct clients now use an explicit render/send boundary:

```python
from forge import ConversationState, RequestMetadata

state = ConversationState(seed)
request = client.render_request(
    state, workflow.get_tool_definitions(),
    RequestMetadata(reasoning_replay="none"),
)
reply = await client.send(request)
```

For send_stream, render with stream=True, then pass that request to
client.send_stream(request). Custom client adapters must implement render_request
and return RichMessage replies / rich FINAL chunks, rather than accepting the
old messages/tools/raw-sidecar send arguments. Context discovery and close
methods remain part of LLMClient. InferenceRequest is a disposable rendered
request, not a conversation store; it can include credentials in headers.

Additional public contracts:

| Integration | Change |
|---|---|
| `run_inference` | Supply state=ConversationState(...), tool_definitions=..., and per-request metadata=RequestMetadata(...); result.response is RichMessage |
| Direct ContextManager calls | estimate_tokens, check_thresholds, and maybe_compact take ConversationState rather than a plain message list |
| Custom CompactStrategy | compact receives ConversationState and returns (ConversationState, phase), operating on rich parts |
| Prompt extraction/rescue helpers | Return ToolCallPart values; build_tool_prompt consumes ToolDefinition values |
| Old serialization helpers | fold_and_serialize and prepare_backend_messages are removed; use the client's renderer instead |

Removed top-level legacy types are Message, MessageMeta, MessageType,
ToolCallInfo, ToolCall, TextResponse, LLMResponse, ToolSpec, and ToolDef.
There is no legacy-input adapter or fallback object model.

## Behavior notes, including Proxy users

The aim is to preserve supported conversation content through validation and
retries, not change the guardrail rules. Relevant observable changes are:

- Supported text and reasoning blocks can coexist with tool calls without the
  old flatten/reconstruct path. Plain-text retry history is retained correctly.
- Complete function-tool schemas and applicable extras are retained instead of
  passing through a lossy Pydantic reconstruction. This can expose schema detail
  that the previous representation discarded.
- Malformed/unsupported function-tool definitions and duplicate names fail
  clearly instead of being silently skipped or deduplicated. Function tools are
  the supported category; this is not support for arbitrary vendor tool kinds.
- Ollama no longer guesses reasoning settings from model names. With think
  unset, Forge leaves generation to the backend default and retains supported
  reasoning returned by it. Review setups that depended on the old guessing.
- Replay controls the rendered request, not reasoning capture in canonical
  history. Its default remains none. keep-last retains the complete last
  reasoning-bearing assistant turn; full retains all eligible reasoning.
  Native Anthropic signatures are retained when native thinking is replayed.
  OpenAI-origin reasoning is omitted from Anthropic-bound requests, not
  synthesized as unsigned thinking or converted into assistant text.

The Proxy still accepts the same supported OpenAI/Anthropic-shaped HTTP routes,
owns no tool execution, and does not compact caller history or enforce workflow
steps. No new CLI flag, configuration format, or auth policy is introduced by
this refactor. Do not read richer internal messages as a new multimodal feature.
