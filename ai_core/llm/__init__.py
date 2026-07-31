"""LLM module for the AI Core Platform.

This package provides domain-agnostic, replaceable LLM provider adapters,
interfaces, and factories for enterprise AI applications.
"""

from ai_core.llm.base import BaseLLM, LLMResponse, UsageMetadata
from ai_core.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMException,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnsupportedProviderError,
)
from ai_core.llm.factory import LLMFactory
from ai_core.llm.gemini_adapter import GeminiAdapter

__all__ = [
    "BaseLLM",
    "GeminiAdapter",
    "LLMAuthenticationError",
    "LLMConfigurationError",
    "LLMException",
    "LLMFactory",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMUnsupportedProviderError",
    "UsageMetadata",
]
