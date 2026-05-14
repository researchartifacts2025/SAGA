"""SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters.

Top-level convenience imports for the most common entry types. Importing
``saga`` is safe and side-effect free.
"""

from saga.__version__ import __version__
from saga.core.aeg import AEGEdge, AEGNode, AgentExecutionGraph
from saga.core.types import (
    KVCacheEntry,
    Session,
    SessionState,
    Task,
    Tenant,
    ToolCall,
    ToolType,
    Worker,
)
from saga.native import build_info as native_build_info
from saga.native import is_native_available


__all__ = [
    "AEGEdge",
    "AEGNode",
    "AgentExecutionGraph",
    "KVCacheEntry",
    "Session",
    "SessionState",
    "Task",
    "Tenant",
    "ToolCall",
    "ToolType",
    "Worker",
    "__version__",
    "is_native_available",
    "native_build_info",
]
