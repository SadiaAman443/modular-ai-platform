"""Memory framework exception hierarchy.

This module defines domain-agnostic exceptions for memory operations,
provider failures, and validation errors in AI Core.
"""


class MemoryException(Exception):
    """Base exception for all memory framework errors."""


class MemoryProviderError(MemoryException):
    """Raised when a memory provider fails to perform a storage or search operation."""


class MemoryNotFoundError(MemoryException):
    """Raised when a requested memory record or context is not found."""


class MemoryValidationError(MemoryException):
    """Raised when a memory record, search query, or context is invalid."""
