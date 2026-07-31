"""Data models for conversational memory in AI Core.

This module defines `MemoryType`, `MemoryRecord`, and `MemoryContext`, providing
type-safe containers for working, episodic, semantic, and custom conversational memories.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Optional


class MemoryType(str, Enum):
    """Categorization of conversational memory entries.

    Attributes:
        WORKING: Short-term working memory for the active session or turn.
        EPISODIC: Episodic memory summarizing past conversations or events.
        SEMANTIC: Semantic facts, learned preferences, or entity attributes.
        CUSTOM: Domain-specific or extended memory classification.
    """

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    CUSTOM = "custom"


@dataclass
class MemoryRecord:
    """Represents a single conversational memory entry.

    Attributes:
        record_id: Unique identifier for this memory record.
        memory_type: The category of memory (`WORKING`, `EPISODIC`, etc.).
        content: The text content, observation, or summary of the memory.
        session_id: Optional associated conversation session identifier.
        entity_id: Optional associated entity or user identifier.
        timestamp: Unix timestamp (in seconds) when the record was created/updated.
        metadata: Optional dictionary of domain-agnostic metadata.
    """

    record_id: str
    memory_type: MemoryType
    content: str
    session_id: Optional[str] = None
    entity_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the MemoryRecord to a dictionary.

        Returns:
            A dictionary representation of the memory record.
        """
        return {
            "record_id": self.record_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "session_id": self.session_id,
            "entity_id": self.entity_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        """Deserializes a dictionary into a MemoryRecord instance.

        Args:
            data: Dictionary containing memory record fields.

        Returns:
            An initialized MemoryRecord instance.

        Raises:
            KeyError: If required fields (`record_id`, `memory_type`, `content`) are missing.
            ValueError: If `memory_type` is not a valid MemoryType value.
        """
        return cls(
            record_id=str(data["record_id"]),
            memory_type=MemoryType(data["memory_type"]),
            content=str(data["content"]),
            session_id=data.get("session_id"),
            entity_id=data.get("entity_id"),
            timestamp=float(data.get("timestamp", time.time())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class MemoryContext:
    """Container of retrieved or assembled memory records for prompt context injection.

    Attributes:
        session_id: Optional associated session identifier.
        entity_id: Optional associated entity or user identifier.
        records: List of relevant memory records retrieved for context.
        metadata: Optional metadata about retrieval or summarization.
    """

    session_id: Optional[str] = None
    entity_id: Optional[str] = None
    records: list[MemoryRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_record(self, record: MemoryRecord) -> None:
        """Appends a MemoryRecord to the context container.

        Args:
            record: The memory record to add.
        """
        self.records.append(record)

    def get_records_by_type(self, memory_type: MemoryType) -> list[MemoryRecord]:
        """Returns all memory records matching a specific MemoryType.

        Args:
            memory_type: The target MemoryType to filter by.

        Returns:
            List of matching MemoryRecord instances.
        """
        return [r for r in self.records if r.memory_type == memory_type]

    def to_string(self, separator: str = "\n") -> str:
        """Formats the contents of all records into a single string.

        Useful for injecting memories directly into system instructions or prompt templates.

        Args:
            separator: Delimiter string between record contents.

        Returns:
            A single formatted text block of memory contents.
        """
        return separator.join(r.content for r in self.records)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the MemoryContext to a dictionary.

        Returns:
            A dictionary representation of the memory context.
        """
        return {
            "session_id": self.session_id,
            "entity_id": self.entity_id,
            "records": [r.to_dict() for r in self.records],
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryContext":
        """Deserializes a dictionary into a MemoryContext instance.

        Args:
            data: Dictionary containing memory context fields.

        Returns:
            An initialized MemoryContext instance.
        """
        records_data = data.get("records", [])
        records = [MemoryRecord.from_dict(r) for r in records_data]
        return cls(
            session_id=data.get("session_id"),
            entity_id=data.get("entity_id"),
            records=records,
            metadata=dict(data.get("metadata", {})),
        )
