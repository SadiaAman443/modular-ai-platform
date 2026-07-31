"""Data models for conversation state, lifecycle, and message history.

This module defines domain-agnostic structures for tracking conversation flow,
role attribution, and session metadata.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MessageRole(str, Enum):
    """Enumeration of standard message participant roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationState(str, Enum):
    """Enumeration of valid conversation lifecycle states."""

    INITIALIZED = "initialized"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"


@dataclass
class Message:
    """A single message turn within a conversation.

    Attributes:
        role: The participant role responsible for the message.
        content: The text content of the message.
        timestamp: Unix timestamp when the message was created.
        metadata: Optional dictionary for domain-agnostic usage metadata.
    """

    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the Message instance to a dictionary.

        Returns:
            A dictionary representation of the message.
        """
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Constructs a Message instance from a dictionary.

        Args:
            data: Mapping containing message properties.

        Returns:
            A populated Message instance.
        """
        role_raw = data.get("role", MessageRole.USER.value)
        try:
            role = MessageRole(role_raw)
        except ValueError:
            role = MessageRole.USER

        return cls(
            role=role,
            content=str(data.get("content", "")),
            timestamp=float(data.get("timestamp", time.time())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ConversationContext:
    """Container for conversation session state, history, and metadata.

    Attributes:
        conversation_id: Unique identifier for the conversation session.
        state: Current lifecycle state of the conversation.
        history: Chronological list of Message turns.
        metadata: Session-level metadata mapping.
        created_at: Unix timestamp when the session was created.
        updated_at: Unix timestamp when the session was last modified.
    """

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: ConversationState = ConversationState.INITIALIZED
    history: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Updates the `updated_at` timestamp to the current Unix time."""
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serializes the ConversationContext to a dictionary.

        Returns:
            A dictionary representation of the conversation context.
        """
        return {
            "conversation_id": self.conversation_id,
            "state": self.state.value,
            "history": [msg.to_dict() for msg in self.history],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationContext":
        """Constructs a ConversationContext from a dictionary.

        Args:
            data: Mapping containing context properties.

        Returns:
            A populated ConversationContext instance.
        """
        state_raw = data.get("state", ConversationState.INITIALIZED.value)
        try:
            state = ConversationState(state_raw)
        except ValueError:
            state = ConversationState.INITIALIZED

        history_raw = data.get("history", [])
        history = [Message.from_dict(m) for m in history_raw if isinstance(m, dict)]

        return cls(
            conversation_id=str(data.get("conversation_id", str(uuid.uuid4()))),
            state=state,
            history=history,
            metadata=dict(data.get("metadata", {})),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )
