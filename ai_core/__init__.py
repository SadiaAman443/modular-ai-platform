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
from ai_core.conversation import (
    ConversationContext,
    ConversationEngine,
    ConversationException,
    ConversationLifecycleError,
    ConversationState,
    ConversationStateError,
    ConversationStateManager,
    Message,
    MessageRole,
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
from ai_core.prompts import (
    PromptEngine,
    PromptException,
    PromptLoadError,
    PromptLoader,
    PromptRenderError,
    PromptTemplate,
    PromptTemplateError,
    PromptVariable,
    RenderedPrompt,
)

__version__ = "0.2.0"

__all__ = [
    "BaseLLM",
    "ConfigLoader",
    "ConversationContext",
    "ConversationEngine",
    "ConversationException",
    "ConversationLifecycleError",
    "ConversationState",
    "ConversationStateError",
    "ConversationStateManager",
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
    "Message",
    "MessageRole",
    "PromptEngine",
    "PromptException",
    "PromptLoadError",
    "PromptLoader",
    "PromptRenderError",
    "PromptTemplate",
    "PromptTemplateError",
    "PromptVariable",
    "RenderedPrompt",
    "UsageMetadata",
    "__version__",
]
