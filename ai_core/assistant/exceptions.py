"""Assistant framework exception hierarchy.

This module defines domain-agnostic exceptions for assistant configuration,
dependency injection validation, and turn coordination errors in AI Core.
"""


class AssistantException(Exception):
    """Base exception for all assistant framework errors."""


class AssistantConfigurationError(AssistantException):
    """Raised when an assistant is built or initialized with invalid dependencies."""


class AssistantExecutionError(AssistantException):
    """Raised when an assistant turn coordination or module execution fails."""
