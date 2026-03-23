"""forge — a reusable framework for self-hosted LLM tool-calling and multi-step agentic workflows."""

from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from forge.core.workflow import (
    LLMResponse,
    TextResponse,
    ToolCall,
    ToolDef,
    ToolSpec,
    Workflow,
)
from forge.core.steps import StepTracker
from forge.core.inference import InferenceResult, fold_and_serialize, run_inference
from forge.core.runner import WorkflowRunner
from forge.clients.base import ChunkType, LLMClient, StreamChunk
from forge.clients.llamafile import LlamafileClient
from forge.clients.ollama import OllamaClient
from forge.context import (
    CompactEvent,
    CompactStrategy,
    ContextManager,
    HardwareProfile,
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
    detect_hardware,
)
from forge.server import BudgetMode, ServerManager, setup_backend
from forge.tools import RESPOND_TOOL_NAME, respond_spec, respond_tool
from forge.prompts import build_tool_prompt, extract_tool_call, rescue_tool_call, retry_nudge, step_nudge
from forge.guardrails import (
    CheckResult,
    ErrorTracker,
    Guardrails,
    Nudge,
    ResponseValidator,
    StepCheck,
    StepEnforcer,
    ValidationResult,
)
from forge.errors import (
    BudgetResolutionError,
    ContextBudgetExceeded,
    ContextDiscoveryError,
    ForgeError,
    HardwareDetectionError,
    MaxIterationsError,
    StepEnforcementError,
    StreamError,
    ThinkingNotSupportedError,
    ToolCallError,
    ToolExecutionError,
    ToolResolutionError,
)

__all__ = [
    # Messages
    "Message",
    "MessageMeta",
    "MessageRole",
    "MessageType",
    "ToolCallInfo",
    # Tools & Workflow
    "LLMResponse",
    "TextResponse",
    "ToolCall",
    "ToolDef",
    "ToolParam",
    "ToolSpec",
    "Workflow",
    # Steps
    "StepTracker",
    # Inference (front half — shared by runner and proxy)
    "InferenceResult",
    "fold_and_serialize",
    "run_inference",
    # Runner
    "WorkflowRunner",
    # Client
    "ChunkType",
    "LLMClient",
    "LlamafileClient",
    "OllamaClient",
    "StreamChunk",
    # Context
    "CompactEvent",
    "CompactStrategy",
    "ContextManager",
    "HardwareProfile",
    "NoCompact",
    "SlidingWindowCompact",
    "TieredCompact",
    "detect_hardware",
    # Prompts
    "build_tool_prompt",
    "extract_tool_call",
    "rescue_tool_call",
    "retry_nudge",
    "step_nudge",
    # Server
    "BudgetMode",
    "ServerManager",
    "setup_backend",
    # Built-in tools
    "RESPOND_TOOL_NAME",
    "respond_spec",
    "respond_tool",
    # Guardrails
    "CheckResult",
    "Guardrails",
    # Guardrails (granular middleware)
    "ErrorTracker",
    "Nudge",
    "ResponseValidator",
    "StepCheck",
    "StepEnforcer",
    "ValidationResult",
    # Errors
    "BudgetResolutionError",
    "ContextBudgetExceeded",
    "ContextDiscoveryError",
    "ForgeError",
    "HardwareDetectionError",
    "MaxIterationsError",
    "StepEnforcementError",
    "StreamError",
    "ThinkingNotSupportedError",
    "ToolCallError",
    "ToolExecutionError",
    "ToolResolutionError",
]
