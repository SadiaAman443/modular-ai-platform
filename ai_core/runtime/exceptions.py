"""Custom exception hierarchy for the AI Core Conversation Runtime.

This module defines structured error types for runtime lifecycle violations,
event processing failures, and event dispatch errors.
"""

from typing import Any, Optional


class RuntimeException(Exception):
    """Base exception class for all Conversation Runtime errors.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary containing extra error context.
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """Initializes the RuntimeException.

        Args:
            message: Explanation of the error.
            details: Optional metadata or debugging context.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Returns a formatted error message including details if present."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class RuntimeStateError(RuntimeException):
    """Raised when an illegal runtime lifecycle operation is attempted."""

    pass


class RuntimeEventError(RuntimeException):
    """Raised when an incoming event is invalid or malformed."""

    pass


class RuntimeExecutionError(RuntimeException):
    """Raised when message processing or event execution fails."""

    pass
