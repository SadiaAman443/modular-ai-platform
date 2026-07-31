"""Data models for assistant orchestration in AI Core.

This module defines `AssistantConfig` and `AssistantTurnResult`, providing
type-safe containers for assistant configuration and turn execution outputs.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AssistantConfig:
    """Configuration metadata for an AIAssistant instance.

    Attributes:
        name: A human-readable identifier for the assistant instance.
        system_prompt: Optional system prompt or instruction text.
        session_id: Optional active conversation session identifier.
        metadata: Optional dictionary of extra configuration attributes.
    """

    name: str = "AIAssistant"
    system_prompt: Optional[str] = None
    session_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes AssistantConfig to a dictionary.

        Returns:
            A dictionary representation of the assistant configuration.
        """
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "session_id": self.session_id,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssistantConfig":
        """Deserializes a dictionary into an AssistantConfig instance.

        Args:
            data: Dictionary containing configuration fields.

        Returns:
            An initialized AssistantConfig instance.
        """
        return cls(
            name=str(data.get("name", "AIAssistant")),
            system_prompt=data.get("system_prompt"),
            session_id=data.get("session_id"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class AssistantTurnResult:
    """Represents the output of a coordinated conversational turn.

    Attributes:
        user_message: The input user message text.
        assistant_response: The main text response produced by the assistant.
        session_id: Optional session ID associated with the turn.
        memory_records: List of memory records stored or retrieved during the turn.
        events: List of runtime lifecycle events emitted during execution.
        metadata: Additional turn metadata (e.g., tokens, latency, tool calls).
    """

    user_message: str
    assistant_response: str
    session_id: Optional[str] = None
    memory_records: list[Any] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes AssistantTurnResult to a dictionary.

        Returns:
            A dictionary representation of the turn result.
        """
        return {
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
            "session_id": self.session_id,
            "memory_records": [
                r.to_dict() if hasattr(r, "to_dict") else str(r)
                for r in self.memory_records
            ],
            "events": [
                e.to_dict() if hasattr(e, "to_dict") else str(e) for e in self.events
            ],
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssistantTurnResult":
        """Deserializes a dictionary into an AssistantTurnResult instance.

        Args:
            data: Dictionary containing turn result fields.

        Returns:
            An initialized AssistantTurnResult instance.
        """
        return cls(
            user_message=str(data.get("user_message", "")),
            assistant_response=str(data.get("assistant_response", "")),
            session_id=data.get("session_id"),
            memory_records=list(data.get("memory_records", [])),
            events=list(data.get("events", [])),
            metadata=dict(data.get("metadata", {})),
        )
