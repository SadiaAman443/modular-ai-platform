"""Custom exception classes for the AI Core Conversation module.

This module defines a structured exception hierarchy for handling errors
relating to conversation state transitions, lifecycle violations, and message flow.
"""

from typing import Any, Optional


class ConversationException(Exception):
    """Base exception class for all errors in the Conversation module.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary containing extra error context.
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """Initializes the ConversationException.

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


class ConversationStateError(ConversationException):
    """Raised when an illegal conversation state transition is attempted."""

    pass


class ConversationLifecycleError(ConversationException):
    """Raised when an operation is performed in an invalid lifecycle phase."""

    pass
