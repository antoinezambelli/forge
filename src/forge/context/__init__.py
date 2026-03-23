"""Context management for the forge library.

Provides compaction strategies, context budget management, and
hardware detection for VRAM-based budget estimation.
"""

from forge.context.hardware import (
    HardwareProfile,
    detect_hardware,
)
from forge.context.manager import CompactEvent, ContextManager, default_context_warning
from forge.context.strategies import (
    CompactStrategy,
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
)

__all__ = [
    "CompactEvent",
    "CompactStrategy",
    "ContextManager",
    "default_context_warning",
    "HardwareProfile",
    "NoCompact",
    "SlidingWindowCompact",
    "TieredCompact",
    "detect_hardware",
]
