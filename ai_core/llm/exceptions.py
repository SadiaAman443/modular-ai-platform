"""Custom exception classes for the AI Core LLM layer.

This module defines a structured exception hierarchy for handling errors
across all LLM providers in a uniform, domain-agnostic manner.
"""

from typing import Any, Optional


class LLMException(Exception):
    """Base exception class for all errors originating in the LLM layer.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary containing extra debugging context.
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """Initializes the LLMException.

        Args:
            message: Explanation of the error.
            details: Optional metadata or provider-specific error payloads.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Returns a formatted error message including details if present."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class LLMConfigurationError(LLMException):
    """Raised when an LLM provider or generation configuration is invalid."""

    pass


class LLMProviderError(LLMException):
    """Raised when an external LLM provider API call fails.

    Attributes:
        provider: Name of the LLM provider (e.g., 'gemini').
        status_code: Optional HTTP or RPC status code returned by the provider.
    """

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        status_code: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initializes the LLMProviderError.

        Args:
            message: Explanation of the provider error.
            provider: Identifier of the provider where the error occurred.
            status_code: Optional status code from the upstream service.
            details: Optional dictionary of additional error context.
        """
        super().__init__(message, details=details)
        self.provider = provider
        self.status_code = status_code

    def __str__(self) -> str:
        """Returns string representation with provider and status code."""
        base = f"[{self.provider.upper()}] {self.message}"
        if self.status_code:
            base += f" (Status Code: {self.status_code})"
        if self.details:
            base += f" | Details: {self.details}"
        return base


class LLMRateLimitError(LLMProviderError):
    """Raised when an upstream LLM provider rate limit is exceeded."""

    pass


class LLMAuthenticationError(LLMProviderError):
    """Raised when authentication or authorization with an LLM provider fails."""

    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when a request to an LLM provider times out."""

    pass


class LLMUnsupportedProviderError(LLMException):
    """Raised when an unsupported or unregistered LLM provider is requested."""

    pass
