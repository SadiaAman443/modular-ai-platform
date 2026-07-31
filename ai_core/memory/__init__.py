"""Conversational memory framework for AI Core.

This package defines the domain-agnostic interfaces and data models for managing
working, episodic, semantic, and custom memories without coupling to specific
database or vector store backends.
"""

from ai_core.memory.base import BaseMemoryProvider
from ai_core.memory.exceptions import (
    MemoryException,
    MemoryNotFoundError,
    MemoryProviderError,
    MemoryValidationError,
)
from ai_core.memory.manager import MemoryManager
from ai_core.memory.models import (
    MemoryContext,
    MemoryRecord,
    MemoryType,
)

__all__ = [
    "BaseMemoryProvider",
    "MemoryContext",
    "MemoryException",
    "MemoryManager",
    "MemoryNotFoundError",
    "MemoryProviderError",
    "MemoryRecord",
    "MemoryType",
    "MemoryValidationError",
]
