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
from forge.prompts import build_tool_prompt, extract_tool_call, rescue_tool_call, retry_nudge, step_nudge
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
    "ToolSpec",
    "Workflow",
    # Steps
    "StepTracker",
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
