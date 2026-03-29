# Workflow

Visual guide to the forge agentic tool-calling loop.

---

## Quick Reference

**Entry Point:** `WorkflowRunner.run()` in `src/forge/core/runner.py`

**Critical Files:**
- `src/forge/core/inference.py` - run_inference() — shared "front half" (compact, fold, validate, retry)
- `src/forge/core/runner.py` - Agentic loop "back half" (step enforcement, tool execution, terminal check)
- `src/forge/core/workflow.py` - Workflow, ToolSpec, ToolCall, TextResponse
- `src/forge/core/messages.py` - Message, MessageRole, MessageType, MessageMeta
- `src/forge/core/steps.py` - StepTracker (used internally by StepEnforcer)
- `src/forge/guardrails/` - Composable middleware (ResponseValidator, StepEnforcer, ErrorTracker, Nudge)
- `src/forge/context/manager.py` - ContextManager, CompactEvent
- `src/forge/context/strategies.py` - TieredCompact (3-phase compaction)
- `src/forge/clients/base.py` - LLMClient protocol
- `src/forge/server.py` - ServerManager, BudgetMode, setup_backend()
- `src/forge/tools/respond.py` - Synthetic respond tool (respond_tool(), respond_spec())
- `src/forge/prompts/nudges.py` - Retry, unknown-tool, and step nudges
- `src/forge/prompts/templates.py` - Prompt-injected tool prompt, extract/rescue

---

## Agentic Loop

The core of forge. The runner delegates inference to `run_inference()` (the shared "front half" — compaction, reasoning folding, serialization, sending, validation, and retry), then handles step enforcement, tool execution, and terminal checks (the "back half"). The proxy also consumes `run_inference()` directly, sharing the same validation logic.

```mermaid
flowchart TD
    subgraph Init["Initialization"]
        BUILD["Build messages:<br/>system prompt + user input<br/>(or seed from initial_messages)"]
        GUARDS["Initialize guardrail middleware:<br/>ResponseValidator, StepEnforcer,<br/>ErrorTracker"]
        BUILD --> GUARDS
    end

    subgraph Loop["Main Loop (up to max_iterations)"]
        direction TB
        CANCEL{"cancel_event<br/>set?"}
        CANCELLED["Raise WorkflowCancelledError<br/>(messages, completed_steps, iteration)"]
        COMPACT["3a. ContextManager.maybe_compact()"]
        SEND["3b. Fold REASONING into next TOOL_CALL content,<br/>serialize & send to LLM (stream or batch)"]
        CHECK{"list[ToolCall] or<br/>TextResponse?"}

        subgraph TextPath["ValidationResult.needs_retry Path"]
            VALIDATE["ResponseValidator.validate()<br/>rescue + retry + unknown tool"]
            RESCUED{"needs_retry<br/>= False?"}
            RETRY_COUNT{"ErrorTracker<br/>retries_exhausted?"}
            NUDGE["Emit assistant content + nudge<br/>→ next iteration"]
            FAIL_RETRY["Raise ToolCallError"]
        end

        subgraph ToolPath["ToolCall Batch Path"]
            KNOWN{"ResponseValidator:<br/>all tools known?"}
            UNKNOWN_NUDGE["Emit TOOL_CALL + unknown_tool_nudge<br/>→ next iteration"]
            TERMINAL{"StepEnforcer.check():<br/>premature terminal?"}
            STEP_NUDGE["Emit TOOL_CALL + escalating step_nudge<br/>(tier 1/2/3)"]
            STEP_FAIL["Raise StepEnforcementError<br/>(premature_exhausted)"]
            PREREQ{"StepEnforcer.<br/>check_prerequisites()?"}
            PREREQ_NUDGE["Emit TOOL_CALL + prerequisite_nudge<br/>→ next iteration"]
            PREREQ_FAIL["Raise PrerequisiteError<br/>(prereq_exhausted)"]
            EMIT["Emit REASONING (if present)<br/>+ TOOL_CALL message"]
            EXEC_BATCH["Execute ALL tools in batch"]
            TOOL_ERROR{"Exception type?"}
            RESOLUTION_ERR["ToolResolutionError:<br/>feed back, don't count error,<br/>don't record step"]
            EXEC_ERR["Other exception:<br/>feed back as [ToolError],<br/>count toward consecutive errors"]
            TOOL_FAIL["Raise ToolExecutionError<br/>(ErrorTracker.tool_errors_exhausted)"]
            BATCH_DONE{"Terminal tool<br/>succeeded?"}
            RETURN["Return terminal result"]
            RECORD["StepEnforcer.record() per success<br/>reset_premature(), reset_errors()<br/>→ next iteration"]
        end

        CANCEL -- "yes" --> CANCELLED
        CANCEL -- "no" --> COMPACT
        COMPACT --> SEND --> CHECK
        CHECK -- "any response" --> VALIDATE
        VALIDATE --> RESCUED

        RESCUED -- "yes (tool_calls)" --> TERMINAL
        RESCUED -- "no (needs_retry)" --> RETRY_COUNT
        RETRY_COUNT -- "no" --> NUDGE
        RETRY_COUNT -- "yes" --> FAIL_RETRY

        TERMINAL -- "needs_nudge" --> STEP_NUDGE
        TERMINAL -- "no premature" --> PREREQ
        STEP_NUDGE -- "premature_exhausted" --> STEP_FAIL
        PREREQ -- "needs_nudge" --> PREREQ_NUDGE
        PREREQ -- "satisfied" --> EMIT
        PREREQ_NUDGE -- "prereq_exhausted" --> PREREQ_FAIL

        EMIT --> EXEC_BATCH
        EXEC_BATCH --> TOOL_ERROR
        TOOL_ERROR -- "ToolResolutionError" --> RESOLUTION_ERR
        TOOL_ERROR -- "other Exception" --> EXEC_ERR
        EXEC_ERR -- "tool_errors_exhausted" --> TOOL_FAIL
        TOOL_ERROR -- "no error" --> BATCH_DONE
        BATCH_DONE -- "yes" --> RETURN
        BATCH_DONE -- "no" --> RECORD
    end

    Init --> CANCEL
```

---

## Message Lifecycle

Every message flows through three stages: creation (with metadata), API serialization (metadata stripped, reasoning folded), and compaction eligibility (prioritized by type).

```mermaid
sequenceDiagram
    participant R as WorkflowRunner
    participant M as Message
    participant C as LLMClient
    participant API as LLM Backend
    participant CM as ContextManager

    R->>M: Create Message(role, content, metadata)
    Note over M: metadata: type, step_index,<br/>token_estimate

    R->>R: Reasoning folding: merge standalone<br/>REASONING msg into next TOOL_CALL's<br/>content field for wire format

    R->>M: to_api_dict(format)
    Note over M: Strips metadata.<br/>format="ollama" or "openai"

    M->>C: {"role": ..., "content": ...}
    C->>API: HTTP request

    API-->>C: Response (list[ToolCall] or TextResponse)
    C-->>R: Parsed LLMResponse

    R->>R: Append new Messages to history

    R->>CM: maybe_compact(messages)
    Note over CM: Check token estimate<br/>vs budget × 0.75

    alt Over threshold
        CM->>CM: CompactStrategy.compact()
        CM->>R: Compacted messages + CompactEvent
    else Under threshold
        CM->>R: Messages unchanged
    end
```

### Message Types and Compaction Priority

Messages are tagged with `MessageType` metadata that determines their compaction priority:

| MessageType | Role | Created By | Cut Order |
|-------------|------|-----------|-----------|
| `system_prompt` | system | Runner init | Never cut |
| `user_input` | user | Runner init | Never cut |
| `tool_call` | assistant | After LLM response | Never cut (all phases) |
| `tool_result` | tool | After tool execution | Truncated P1, dropped P2 |
| `reasoning` | assistant | Thinking models | Preserved through P2, dropped P3 |
| `text_response` | assistant | Failed tool call attempt | Preserved through P2, dropped P3 |
| `step_nudge` | user | Runner step enforcement | Dropped P1 |
| `prerequisite_nudge` | user | Runner prereq enforcement | Dropped P1 |
| `retry_nudge` | user | Runner retry logic | Dropped P1 |
| `summary` | system | Compaction output | Never cut |

---

## Compaction Phases

TieredCompact applies three escalating phases. Each phase fires only if the previous didn't reduce tokens below the threshold. All phases are deterministic text manipulation — no LLM calls.

```mermaid
flowchart TD
    TRIGGER["Token estimate > budget × 0.75"]

    subgraph P1["Phase 1: Light"]
        P1A["Drop all step_nudge, prerequisite_nudge, retry_nudge"]
        P1B["Truncate old tool_results<br/>to ~200 chars"]
        P1A --> P1B
    end

    subgraph P2["Phase 2: Moderate"]
        P2A["Drop step_nudge, prerequisite_nudge, retry_nudge"]
        P2B["Drop old tool_results entirely"]
        P2C["Reasoning + text_response PRESERVED"]
        P2A --> P2B --> P2C
    end

    subgraph P3["Phase 3: Emergency"]
        P3A["Drop step_nudge, prerequisite_nudge, retry_nudge"]
        P3B["Drop old tool_results"]
        P3C["Drop reasoning"]
        P3D["Drop text_response"]
        P3E["Only tool_call messages remain<br/>in eligible window"]
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

**Protected window:** The `keep_recent` most recent loop iterations (default 2) are never compacted, regardless of phase. Only older messages in the eligible window are affected.

---

## Client Adapter Flow

The `LLMClient` protocol abstracts backend differences. The runner never sees raw HTTP — it gets `list[ToolCall] | TextResponse`. All clients also expose `get_context_length()` for budget discovery.

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

        subgraph AN["AnthropicClient"]
            AN_API["api_format = 'openai'"]
            AN_CONV["Convert OpenAI → Anthropic<br/>format before each call"]
        end
    end

    subgraph Backends["Backends"]
        OLLAMA_SVC["Ollama Service<br/>localhost:11434"]
        LLAMA_SRV["llama-server<br/>localhost:8080"]
        ANTHROPIC["Anthropic API<br/>api.anthropic.com"]
    end

    SEND_CALL --> OL --> OLLAMA_SVC
    SEND_CALL --> LF --> LLAMA_SRV
    SEND_CALL --> AN --> ANTHROPIC
```

### Streaming Flow

```mermaid
sequenceDiagram
    participant R as WorkflowRunner
    participant C as LLMClient
    participant API as Backend (SSE)
    participant UI as on_chunk callback

    R->>C: send_stream(messages, tools)
    C->>API: HTTP request (stream=true)

    loop SSE chunks
        API-->>C: data: {...}
        C->>C: Parse chunk type
        C-->>R: StreamChunk(TEXT_DELTA, "partial...")
        R->>UI: await on_chunk(TEXT_DELTA)
    end

    C->>C: Assemble final response
    C-->>R: StreamChunk(FINAL, response=list[ToolCall]|TextResponse)

    alt Malformed stream
        C->>C: Internal retry (stream_retries++)
        C-->>R: StreamChunk(RETRY)
        Note over R,UI: UI discards partial output
        C->>API: Re-send request
    end
```

---

## Budget Resolution

`ServerManager` resolves context budgets before the agentic loop starts. The budget flows into `ContextManager`, which uses it as the compaction threshold.

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
            WORKFLOW["workflow.py<br/>Workflow, ToolSpec,<br/>ToolCall, TextResponse"]
            MESSAGES["messages.py<br/>Message, MessageMeta"]
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
        SCENARIOS["scenarios/<br/>20 scenarios across<br/>plumbing, model_quality,<br/>compaction, stateful"]
        EVAL_RUN["eval_runner.py<br/>CLI + runner"]
        METRICS["metrics.py<br/>Aggregate stats"]
        BATCH["batch_eval.py<br/>Multi-config runner"]
        REPORT["report.py<br/>ASCII tables + HTML"]
        ABLATION["ablation.py<br/>Guardrail presets"]
        BFCL["bfcl/<br/>BFCL v4 benchmark"]
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
├── tools: dict[str, ToolDef]          # keyed by tool name
├── required_steps: list[str]          # must be called before terminal
├── terminal_tool: str | list[str]     # tool(s) that end the workflow
├── terminal_tools: frozenset[str]     # normalized (init=False)
└── system_prompt_template: str        # may contain {placeholders}

ToolDef
├── spec: ToolSpec
├── callable: Callable[..., Any]
└── prerequisites: list[str | dict]   # conditional dependencies (default [])

ToolSpec
├── name: str
├── description: str
└── parameters: type[BaseModel]        # dynamic Pydantic model (from_json_schema)

ToolCall
├── tool: str
├── args: dict[str, Any]
└── reasoning: str | None              # chain-of-thought (thinking models)

TextResponse
└── content: str                       # non-tool-call output
```

### Message Types (`src/forge/core/messages.py`)

```
ToolCallInfo (frozen)
├── name: str
├── args: dict[str, Any]
└── call_id: str

Message
├── role: MessageRole                  # system, user, assistant, tool
├── content: str
├── metadata: MessageMeta
│   ├── type: MessageType              # compaction priority tag
│   ├── step_index: int | None
│   ├── original_type: MessageType | None
│   └── token_estimate: int | None
├── tool_name: str | None              # for role="tool" results
├── tool_call_id: str | None           # OpenAI-format correlation
└── tool_calls: list[ToolCallInfo] | None  # for assistant tool calls (1+ entries)
```

### Streaming Types (`src/forge/clients/base.py`)

```
StreamChunk
├── type: ChunkType                    # TEXT_DELTA, TOOL_CALL_DELTA, FINAL, RETRY
├── content: str                       # partial text for deltas
└── response: LLMResponse | None       # only set when type == FINAL
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
python -m tests.eval.eval_runner --backend llamafile --llamafile-mode native --model ministral-14b-instruct-q4_k_m --runs 10 --stream

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
