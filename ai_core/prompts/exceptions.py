"""Custom exception classes for the AI Core Prompt module.

This module defines a structured exception hierarchy for handling errors
during prompt template loading, validation, and rendering.
"""

from typing import Any, Optional


class PromptException(Exception):
    """Base exception class for all errors in the Prompt module.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary containing extra error context.
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """Initializes the PromptException.

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


class PromptTemplateError(PromptException):
    """Raised when a prompt template syntax or structure is malformed."""

    pass


class PromptRenderError(PromptException):
    """Raised when rendering a prompt template fails (e.g., missing variables)."""

    pass


class PromptLoadError(PromptException):
    """Raised when loading a prompt template from an external source fails."""

    pass
