"""Conversation Runtime package for the AI Core Platform.

This package provides a reusable, provider-agnostic execution layer between external
conversation transports (voice/chat providers) and the AI Core foundations.
"""

from ai_core.runtime.dispatcher import EventCallback, EventDispatcher
from ai_core.runtime.engine import ConversationRuntime
from ai_core.runtime.events import Event, EventType
from ai_core.runtime.exceptions import (
    RuntimeEventError,
    RuntimeException,
    RuntimeExecutionError,
    RuntimeStateError,
)
from ai_core.runtime.models import RuntimeContext, RuntimeState

__all__ = [
    "ConversationRuntime",
    "Event",
    "EventCallback",
    "EventDispatcher",
    "EventType",
    "RuntimeContext",
    "RuntimeEventError",
    "RuntimeException",
    "RuntimeExecutionError",
    "RuntimeState",
    "RuntimeStateError",
]
