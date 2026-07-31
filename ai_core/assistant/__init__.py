"""Primary assistant orchestration framework for AI Core.

This package defines `AIAssistant` and `AssistantBuilder`, the top-level orchestration
entry point for combining language models, conversation engines, memory managers,
and workflow components via clean Dependency Injection.
"""

from ai_core.assistant.assistant import AIAssistant
from ai_core.assistant.builder import AssistantBuilder
from ai_core.assistant.exceptions import (
    AssistantConfigurationError,
    AssistantException,
    AssistantExecutionError,
)
from ai_core.assistant.models import (
    AssistantConfig,
    AssistantTurnResult,
)

__all__ = [
    "AIAssistant",
    "AssistantBuilder",
    "AssistantConfig",
    "AssistantConfigurationError",
    "AssistantException",
    "AssistantExecutionError",
    "AssistantTurnResult",
]
