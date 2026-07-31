"""Event definitions and models for the Conversation Runtime.

This module provides a provider-agnostic event model and enumeration representing
all supported conversation lifecycle and message interactions.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    """Enumeration of supported runtime event types."""

    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    USER_MESSAGE = "user_message"
    ASSISTANT_RESPONSE = "assistant_response"
    INTERRUPTION = "interruption"
    ERROR = "error"


@dataclass
class Event:
    """Represents a domain-agnostic conversation runtime event.

    Attributes:
        event_type: The category of the event.
        session_id: Unique identifier for the conversation session.
        payload: Event-specific data mapping (e.g., message text, usage data).
        timestamp: Unix timestamp when the event was created.
        metadata: Optional dictionary for tracking or routing metadata.
    """

    event_type: EventType
    session_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the Event instance to a dictionary.

        Returns:
            A dictionary representation of the event.
        """
        return {
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Constructs an Event instance from a dictionary.

        Args:
            data: Mapping containing event properties.

        Returns:
            A populated Event instance.
        """
        raw_type = data.get("event_type", EventType.USER_MESSAGE.value)
        try:
            event_type = EventType(raw_type)
        except ValueError:
            event_type = EventType.USER_MESSAGE

        return cls(
            event_type=event_type,
            session_id=str(data.get("session_id", "")),
            payload=dict(data.get("payload", {})),
            timestamp=float(data.get("timestamp", time.time())),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def create_session_started(
        cls,
        session_id: str,
        payload: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Event":
        """Factory method to create a SESSION_STARTED event."""
        return cls(
            event_type=EventType.SESSION_STARTED,
            session_id=session_id,
            payload=payload or {},
            metadata=metadata or {},
        )

    @classmethod
    def create_session_ended(
        cls,
        session_id: str,
        payload: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Event":
        """Factory method to create a SESSION_ENDED event."""
        return cls(
            event_type=EventType.SESSION_ENDED,
            session_id=session_id,
            payload=payload or {},
            metadata=metadata or {},
        )

    @classmethod
    def create_user_message(
        cls,
        session_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Event":
        """Factory method to create a USER_MESSAGE event."""
        return cls(
            event_type=EventType.USER_MESSAGE,
            session_id=session_id,
            payload={"content": content},
            metadata=metadata or {},
        )

    @classmethod
    def create_assistant_response(
        cls,
        session_id: str,
        content: str,
        usage: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Event":
        """Factory method to create an ASSISTANT_RESPONSE event."""
        payload = {"content": content}
        if usage:
            payload["usage"] = usage
        return cls(
            event_type=EventType.ASSISTANT_RESPONSE,
            session_id=session_id,
            payload=payload,
            metadata=metadata or {},
        )

    @classmethod
    def create_interruption(
        cls,
        session_id: str,
        payload: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Event":
        """Factory method to create an INTERRUPTION event."""
        return cls(
            event_type=EventType.INTERRUPTION,
            session_id=session_id,
            payload=payload or {},
            metadata=metadata or {},
        )

    @classmethod
    def create_error(
        cls,
        session_id: str,
        error_message: str,
        details: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "Event":
        """Factory method to create an ERROR event."""
        payload = {"message": error_message, "details": details or {}}
        return cls(
            event_type=EventType.ERROR,
            session_id=session_id,
            payload=payload,
            metadata=metadata or {},
        )
