"""Conversation foundation module for the AI Core Platform.

This package provides domain-agnostic conversation lifecycle management, state
tracking, and message flow orchestration without coupling to any specific LLM provider.
"""

from ai_core.conversation.engine import ConversationEngine
from ai_core.conversation.exceptions import (
    ConversationException,
    ConversationLifecycleError,
    ConversationStateError,
)
from ai_core.conversation.models import (
    ConversationContext,
    ConversationState,
    Message,
    MessageRole,
)
from ai_core.conversation.state import ConversationStateManager

__all__ = [
    "ConversationContext",
    "ConversationEngine",
    "ConversationException",
    "ConversationLifecycleError",
    "ConversationState",
    "ConversationStateError",
    "ConversationStateManager",
    "Message",
    "MessageRole",
]
