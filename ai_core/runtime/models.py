"""Data models for runtime session state and execution context.

This module provides structures for tracking runtime lifecycle status and
session-level configuration without coupling to any transport provider.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class RuntimeState(str, Enum):
    """Enumeration of valid runtime lifecycle states."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class RuntimeContext:
    """Container for runtime execution session state and metadata.

    Attributes:
        session_id: Unique identifier for the runtime session.
        state: Current lifecycle state of the runtime.
        system_prompt: Optional active system prompt string for this session.
        metadata: Domain-agnostic runtime metadata mapping.
        created_at: Unix timestamp when the runtime context was initialized.
        updated_at: Unix timestamp when the runtime context was last updated.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: RuntimeState = RuntimeState.IDLE
    system_prompt: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Updates `updated_at` timestamp to current Unix time."""
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serializes the RuntimeContext to a dictionary.

        Returns:
            A dictionary representation of the context.
        """
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeContext":
        """Constructs a RuntimeContext from a dictionary.

        Args:
            data: Mapping containing context properties.

        Returns:
            A populated RuntimeContext instance.
        """
        state_raw = data.get("state", RuntimeState.IDLE.value)
        try:
            state = RuntimeState(state_raw)
        except ValueError:
            state = RuntimeState.IDLE

        return cls(
            session_id=str(data.get("session_id", str(uuid.uuid4()))),
            state=state,
            system_prompt=data.get("system_prompt"),
            metadata=dict(data.get("metadata", {})),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )
