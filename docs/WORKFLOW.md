# Workflow

Visual guide to the forge agentic tool-calling loop.

---

## Quick Reference

**Entry Point:** `WorkflowRunner.run()` in `src/forge/core/runner.py`

**Critical Files:**
- `src/forge/core/inference.py` - run_inference() — shared "front half" (compact, render, validate, record, retry)
- `src/forge/core/runner.py` - Agentic loop "back half" (step enforcement, tool execution, terminal check)
- `src/forge/core/workflow.py` - Workflow, ToolDefinition, separate handlers/prerequisites
- `src/forge/core/messages.py` - ConversationState, RichMessage, typed parts
- `src/forge/core/steps.py` - StepTracker (used internally by StepEnforcer)
- `src/forge/guardrails/` - Composable middleware (ResponseValidator, StepEnforcer, ErrorTracker, Nudge)
- `src/forge/context/manager.py` - ContextManager, CompactEvent
- `src/forge/context/strategies.py` - TieredCompact (3-phase compaction)
- `src/forge/clients/base.py` - LLMClient protocol
- `src/forge/server.py` - ServerManager, BudgetMode, setup_backend()
- `src/forge/tools/respond.py` - Synthetic respond tool (respond_definition(), respond())
- `src/forge/prompts/nudges.py` - Retry, unknown-tool, and step nudges
- `src/forge/prompts/templates.py` - Prompt-injected tool prompt, extract/rescue

---

## Agentic Loop

Runner and Proxy use the same canonical inference lifecycle. Runner owns
configured context management, step/prerequisite enforcement and actual handler
execution. Proxy uses NoCompact, no warnings, no step enforcement and leaves
execution to its client.

```mermaid
flowchart TD
    SEED["Fresh rich system/user seeds or silent list of RichMessage"] --> STATE["One ConversationState"]
    STATE --> CANCEL{"Cancelled?"}
    CANCEL -- yes --> STOP["WorkflowCancelledError with rich history"]
    CANCEL -- no --> INFER["Shared inference: compact, render, send, validate"]
    INFER --> RECORD["Record assistant RichMessage once; assign call IDs"]
    RECORD --> VALID{"Accepted?"}
    VALID -- no --> RETRY["Correlated tool errors or user retry correction"]
    RETRY --> BUDGET{"Attempts remain?"}
    BUDGET -- yes --> INFER
    BUDGET -- no --> EXHAUST["Keep final rejected reply; preserve exception or iteration exit"]
    VALID -- yes --> GROUP["Group same reply at completed iteration; notify new messages"]
    GROUP --> STEP{"Required steps and prerequisites satisfied?"}
    STEP -- no --> CORRECTION["Purpose-marked correlated error results or enforcement exhaustion"]
    CORRECTION --> CANCEL
    STEP -- yes --> EXEC["Execute every call in original order, including after terminal"]
    EXEC --> RESULTS["Append typed results; record successful steps; update error budget"]
    RESULTS --> TERMINAL{"Terminal succeeded?"}
    TERMINAL -- yes --> RETURN["Return actual Python terminal value"]
    TERMINAL -- no --> CANCEL
```

Formatting retries consume the iteration budget. Malformed arguments use the
shared tool-error budget and never reach handlers. Callable failures retain the
`[ToolError]` channel and recovery behavior. `ToolResolutionError` produces
correlated error history without recording step completion or consuming the
callable-error budget. Step/prerequisite checks block the whole batch against
pre-batch state. A terminal handler does not bypass later handlers or post-batch
error bookkeeping.

## Message Lifecycle

```mermaid
sequenceDiagram
    participant R as WorkflowRunner
    participant I as Shared inference
    participant C as Selected client renderer/transport
    participant E as on_chunk / on_message
    R->>I: state, definitions, RequestMetadata, remaining attempts
    I->>I: configured compaction and transient warnings
    I->>C: render_request(state, definitions, metadata)
    C-->>I: complete disposable InferenceRequest
    I->>C: send(request) or send_stream(request)
    C-->>I: assistant RichMessage / FINAL
    I->>E: await on_chunk(FINAL), if streaming
    I->>I: validate/rescue, assign IDs, append same reply once
    I-->>R: InferenceResult with existing new_messages references
    R->>R: accepted reply.step_index = completed iteration
    R->>E: on_message for those existing references
    R->>R: enforcement and execution
    R->>E: append and emit correlated ToolResultPart messages
```

On attempt-limit or validation exhaustion, inference flushes the existing
per-call messages to Runner callbacks, including the final rejected reply.
No unused correction or extra attempt is added. Transport exception behavior
is unchanged. Context warnings are callback events and outgoing-view additions,
never persistent canonical history.

### Role, parts and purpose

| View | Canonical representation | Owner |
|---|---|---|
| System/user seed | Role plus ordered `TextPart` values | Runner or supplied seed |
| Assistant reply | One message with ordered text, reasoning and calls | Shared inference |
| Accepted call | `ToolCallPart(name, ValidArguments(value), call_id)` | Validator/inference |
| Invalid call | `ToolCallPart` with `InvalidArguments(received)` | Validator; never executed |
| Execution result | `ToolResultPart(call_id, content, outcome)` | Runner |
| Corrections | `forge_purpose`: `retry`, `unknown_tool`, `tool_arg_validation`, `step`, `prerequisite` | Inference or Runner |
| Warning | `forge_purpose == "context_warning"` | Transient context event |

`step_index` and `forge_purpose` are direct Forge control fields. There is no
MessageType classifier or universal text projection. Display iterates ordered
parts; metrics count individual calls/names and reasoning parts, using purpose
and outcome/prefix to distinguish corrections from callable failures. Actual
terminal Python values stay local; successful non-string model-visible results
remain JSON text. Result `name` remains an applicable message extra.

---

## Compaction Phases

TieredCompact applies three escalating phases. Each phase fires only if the previous didn't reduce tokens below the threshold. All phases are deterministic text manipulation — no LLM calls.

```mermaid
flowchart TD
    TRIGGER["Token estimate >= budget × 0.75<br/>(default first-phase trigger)"]

    subgraph P1["Phase 1: Light"]
        P1A["Remove eligible Forge corrections<br/>by purpose"]
        P1B["Truncate each old ToolResultPart<br/>to 200 text chars plus notice"]
        P1A --> P1B
    end

    subgraph P2["Phase 2: Moderate"]
        P2A["Remove eligible Forge corrections"]
        P2B["Drop old ToolResultPart values"]
        P2C["Assistant text, reasoning<br/>and tool-call parts PRESERVED"]
        P2A --> P2B --> P2C
    end

    subgraph P3["Phase 3: Emergency"]
        P3A["Remove eligible Forge corrections"]
        P3B["Drop old ToolResultPart values"]
        P3C["Drop old assistant ReasoningPart values"]
        P3D["Drop old assistant TextPart values"]
        P3E["Retain tool-call parts<br/>from eligible assistant replies"]
        P3A --> P3B --> P3C --> P3D --> P3E
    end

    CHECK1{"Under<br/>threshold?"}
    CHECK2{"Under<br/>threshold?"}

    TRIGGER --> P1
    P1 --> CHECK1
    CHECK1 -- "yes" --> DONE1["Return (phase 1)"]
    CHECK1 -- "no" --> P2
    P2 --> CHECK2
    CHECK2 -- "yes" --> DONE2["Return (phase 2)"]
    CHECK2 -- "no" --> P3
    P3 --> DONE3["Return (phase 3)"]

    style P1 fill:#e8f5e9
    style P2 fill:#fff3e0
    style P3 fill:#ffebee
```

**Protected window:** The first two messages and the `keep_recent` most recent loop iterations (default 2) are never compacted, regardless of phase. Only older messages in the eligible window are affected.

---

## Client Adapter Flow

The `LLMClient` protocol abstracts backend differences. The runner never sees
raw HTTP — it gets an assistant `RichMessage`. Built-in adapters cover
generic OpenAI-compatible endpoints, Ollama, llama-server/llamafile, vLLM, and
Anthropic. The protocol also requires an honest `get_context_length()` result
for budget discovery (`None` when unavailable).

```mermaid
flowchart LR
    subgraph Runner["WorkflowRunner"]
        SEND_CALL["send() or send_stream()"]
    end

    subgraph Clients["LLMClient Implementations"]
        direction TB

        subgraph OL["OllamaClient"]
            OL_API["api_format = 'ollama'"]
            OL_NATIVE["Native FC via /api/chat<br/>+ tools parameter"]
            OL_CTX["set_num_ctx() on every request"]
        end

        subgraph LF["LlamafileClient"]
            LF_API["api_format = 'openai'"]
            LF_MODE{"mode?"}
            LF_NATIVE["Native FC<br/>(--jinja required)"]
            LF_PROMPT["Prompt-injected<br/>JSON extraction"]
            LF_DOWN["_downgrade_messages()<br/>_merge_consecutive()"]
            LF_MODE -- "native" --> LF_NATIVE
            LF_MODE -- "prompt" --> LF_PROMPT
            LF_PROMPT --> LF_DOWN
        end

        subgraph OAI["OpenAICompatClient"]
            OAI_API["api_format = 'openai'"]
            OAI_NATIVE["Generic OpenAI<br/>chat completions"]
        end

        subgraph VL["VLLMClient"]
            VL_API["api_format = 'openai'"]
            VL_NATIVE["Native FC + server-side<br/>tool/reasoning parsing"]
            VL_ID["Served identity + context<br/>via /v1/models"]
        end

        subgraph AN["AnthropicClient"]
            AN_API["api_format = 'openai'"]
            AN_CONV["Render canonical state → Anthropic<br/>Messages before each call"]
        end
    end

    subgraph Backends["Backends"]
        OLLAMA_SVC["Ollama Service<br/>localhost:11434"]
        LLAMA_SRV["llama-server / llamafile<br/>localhost:8080"]
        OPENAI_SVC["Generic OpenAI-compatible<br/>endpoint"]
        VLLM_SVC["vLLM<br/>localhost:8000"]
        ANTHROPIC["Anthropic API<br/>api.anthropic.com"]
    end

    SEND_CALL --> OL --> OLLAMA_SVC
    SEND_CALL --> LF --> LLAMA_SRV
    SEND_CALL --> OAI --> OPENAI_SVC
    SEND_CALL --> VL --> VLLM_SVC
    SEND_CALL --> AN --> ANTHROPIC
```

### Streaming Flow

```mermaid
sequenceDiagram
    participant R as WorkflowRunner
    participant C as LLMClient
    participant API as Backend (SSE)
    participant UI as on_chunk callback

    R->>C: render_request(state, tools, metadata, stream=True)
    R->>C: send_stream(request)
    C->>API: HTTP request (stream=true)

    loop SSE chunks
        API-->>C: data: {...}
        C->>C: Parse chunk type
        C-->>R: StreamChunk(TEXT_DELTA, "partial...")
        R->>UI: await on_chunk(TEXT_DELTA)
    end

    C->>C: Assemble final response
    C-->>R: StreamChunk(FINAL, response=RichMessage)

```

The diagram shows a successful stream. A stream ending without FINAL raises
`StreamError`; built-in clients do not automatically resend that request.
`RETRY` remains available for custom adapters that implement retry behavior,
and consumers should discard failed-attempt partial output when it is emitted.
The eval harness can retry a whole run separately; that is not a built-in
client transport retry.

---

## Budget Resolution

`ServerManager` resolves context budgets before the agentic loop starts. The budget flows into `ContextManager`, which uses it as the compaction threshold.

This diagram describes native `setup_backend()`/`WorkflowRunner` composition.
Proxy keeps the same managed allocation modes but always composes `NoCompact`;
it never mutates caller history. In unmanaged Proxy mode, `budget_tokens` and
metadata are reporting-only inputs for `/forge/usage`, and the backend
operator owns allocation, overflow, model swaps, and failures.

```mermaid
flowchart TD
    subgraph Entry["setup_backend()"]
        SETUP["Wire ServerManager +<br/>ContextManager together"]
    end

    MODE{"BudgetMode?"}

    subgraph Backend_Path["BACKEND"]
        B_TRUST["Trust backend default<br/>No override sent"]
    end

    subgraph Manual_Path["MANUAL"]
        M_USER["Use user-specified<br/>token count"]
    end

    subgraph Full_Path["FORGE_FULL"]
        direction TB
        WHICH{"Backend?"}
        OL_VRAM["Ollama: VRAM tier lookup<br/>&lt;24GB → 4,096<br/>24-48GB → 32,768<br/>≥48GB → 262,144"]
        LS_PROPS["llama-server/llamafile:<br/>/props endpoint → n_ctx (auto-tuned)"]
        WHICH -- "ollama" --> OL_VRAM
        WHICH -- "llamaserver/llamafile" --> LS_PROPS
    end

    subgraph Fast_Path["FORGE_FAST"]
        F_HALF["FORGE_FULL ÷ 2<br/>Trades context for speed"]
        F_NOTE["llama-server: discover max via /props,<br/>restart with -c = max ÷ 2"]
    end

    subgraph Wire["Wiring"]
        SET_CTX["OllamaClient.set_num_ctx(budget)"]
        CTX_MGR["ContextManager(budget_tokens=budget,<br/>strategy=TieredCompact())"]
    end

    SETUP --> MODE
    MODE -- "backend" --> Backend_Path
    MODE -- "manual" --> Manual_Path
    MODE -- "forge-full" --> Full_Path
    MODE -- "forge-fast" --> Fast_Path

    Backend_Path --> Wire
    Manual_Path --> Wire
    Full_Path --> Wire
    Fast_Path --> Wire
```

---

## Module Structure

```mermaid
flowchart TB
    subgraph downstream["Downstream Project"]
        APP["Defines tools, prompts,<br/>workflows, routing"]
    end

    subgraph forge["src/forge/"]
        subgraph core["core/"]
            RUNNER["runner.py<br/>WorkflowRunner"]
            WORKFLOW["workflow.py<br/>Workflow, ToolDefinition,<br/>handlers, prerequisites"]
            MESSAGES["messages.py<br/>ConversationState, RichMessage"]
            STEPTRACK["steps.py<br/>StepTracker"]
        end

        subgraph context["context/"]
            CTX_MGR["manager.py<br/>ContextManager"]
            STRATS["strategies.py<br/>TieredCompact"]
            HW["hardware.py<br/>HardwareProfile"]
        end

        subgraph clients["clients/"]
            BASE["base.py<br/>LLMClient protocol"]
            OLLAMA["ollama.py<br/>OllamaClient"]
            LLAMAFILE["llamafile.py<br/>LlamafileClient"]
            OPENAI_COMPAT["openai_compat.py<br/>OpenAICompatClient"]
            VLLM["vllm.py<br/>VLLMClient"]
            ANTHROPIC["anthropic.py<br/>AnthropicClient"]
        end

        subgraph prompts["prompts/"]
            TEMPLATES["templates.py<br/>build_tool_prompt<br/>extract/rescue"]
            NUDGES["nudges.py<br/>retry, unknown_tool,<br/>step nudges"]
        end

        SERVER["server.py<br/>ServerManager<br/>BudgetMode"]
        ERRORS["errors.py<br/>ForgeError hierarchy"]
    end

    subgraph eval["tests/eval/"]
        SCENARIOS["scenarios/<br/>plumbing, model_quality,<br/>compaction, stateful"]
        EVAL_RUN["eval_runner.py<br/>CLI + runner"]
        METRICS["metrics.py<br/>Aggregate stats"]
        BATCH["batch_eval.py<br/>Multi-config runner"]
        REPORT["report.py<br/>ASCII tables + HTML"]
        ABLATION["ablation.py<br/>Guardrail presets"]
    end

    APP --> RUNNER
    RUNNER --> WORKFLOW
    RUNNER --> MESSAGES
    RUNNER --> STEPTRACK
    RUNNER --> CTX_MGR
    RUNNER --> BASE
    RUNNER --> NUDGES
    RUNNER --> TEMPLATES
    CTX_MGR --> STRATS
    SERVER --> HW
    SERVER --> CTX_MGR
    BASE --> OLLAMA
    BASE --> LLAMAFILE
    BASE --> ANTHROPIC
    EVAL_RUN --> SCENARIOS
    EVAL_RUN --> METRICS
    BATCH --> EVAL_RUN
    REPORT --> BATCH
    ABLATION --> EVAL_RUN
```

---

## Data Types

### Core Types (`src/forge/core/workflow.py`)

```
Workflow
├── name: str
├── description: str
├── tools: dict[str, ToolDefinition]   # ordered model-visible definitions
├── handlers: dict[str, Callable]      # execution keyed by tool name
├── prerequisites: dict[str, list]     # conditional dependencies keyed by name
├── required_steps: list[str]          # must be called before terminal
├── terminal_tool: str | list[str]     # tool(s) that end the workflow
├── terminal_tools: frozenset[str]     # normalized (init=False)
└── system_prompt_template: str        # may contain {placeholders}

ToolDefinition
├── name: str
├── description: str                   # defaults to empty string
├── input_schema: dict                 # complete schema
└── extra_fields: dict                 # applicable function-tool extras
```

### Canonical History (`src/forge/core/messages.py`)

```text
ConversationState(messages: list[RichMessage])
RichMessage(role, parts, step_index=None, forge_purpose=None, extra_fields={})
TextPart(text, extra_fields={})
ReasoningPart(text, extra_fields={}, field_name=None)
ToolCallPart(name, arguments: ValidArguments | InvalidArguments, call_id="")
ToolResultPart(call_id, content: list[TextPart], outcome="success")
```

Backend replies, callbacks and supplied seeds all use this surface. See the
[Python migration examples](USER_GUIDE.md#multi-turn-conversations) for concrete
old-to-new authoring and callback operations.

### Streaming Types (`src/forge/clients/base.py`)

```
StreamChunk
├── type: ChunkType                    # TEXT_DELTA, TOOL_CALL_DELTA, FINAL, RETRY
├── content: str                       # partial text for deltas
└── response: RichMessage | None       # only set when type == FINAL
```

---

## Command Reference

```bash
# Run unit tests (no backend needed)
python -m pytest tests/ -v --tb=short

# Run with coverage
python -m pytest tests/ --cov=forge --cov-report=term-missing

# Eval: single model, all scenarios
python -m tests.eval.eval_runner --backend ollama --model "ministral-3:8b-instruct-2512-q4_K_M" --runs 10 --stream --verbose

# Eval: specific scenarios
python -m tests.eval.eval_runner --backend ollama --model "ministral-3:8b-instruct-2512-q4_K_M" --runs 10 --scenario basic_2step sequential_3step

# Eval: llama-server (start server in separate terminal first)
python -m tests.eval.eval_runner --backend llamafile --llamafile-mode native --gguf path/to/Ministral-3-14B-Instruct-2512-Q4_K_M.gguf --runs 10 --stream

# Eval: Anthropic baseline
python -m tests.eval.eval_runner --backend anthropic --model claude-haiku-4-5-20251001 --runs 5 --stream

# Batch eval (multi-model, auto-resume)
python -m tests.eval.batch_eval --config ollama --runs 50

# Check batch progress
python -m tests.eval.report eval_results.jsonl --progress

# Probe context budget (no eval run)
python -m tests.eval.eval_runner --backend ollama --model "ministral-3:8b-instruct-2512-q4_K_M" --probe
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design document.
