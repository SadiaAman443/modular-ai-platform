"""AI Core Platform Foundation.

A domain-agnostic, provider-independent AI Core foundation designed to power
multiple enterprise AI applications.
"""

from ai_core.config import (
    ConfigLoader,
    GenerationConfig,
    LLMConfig,
    LLMProviderConfig,
)
from ai_core.llm import (
    BaseLLM,
    GeminiAdapter,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMException,
    LLMFactory,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnsupportedProviderError,
    UsageMetadata,
)

__version__ = "0.1.0"

__all__ = [
    "BaseLLM",
    "ConfigLoader",
    "GeminiAdapter",
    "GenerationConfig",
    "LLMAuthenticationError",
    "LLMConfig",
    "LLMConfigurationError",
    "LLMException",
    "LLMFactory",
    "LLMProviderConfig",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMUnsupportedProviderError",
    "UsageMetadata",
    "__version__",
]
