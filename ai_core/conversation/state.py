"""State and lifecycle manager for conversation sessions.

This module enforces valid lifecycle transitions and manages message history
within a ConversationContext.
"""

from typing import Any, Optional

from ai_core.conversation.exceptions import (
    ConversationLifecycleError,
    ConversationStateError,
)
from ai_core.conversation.models import (
    ConversationContext,
    ConversationState,
    Message,
    MessageRole,
)


class ConversationStateManager:
    """Manages lifecycle transitions and message history for a conversation.

    Attributes:
        context: The underlying ConversationContext data container.
    """

    def __init__(self, context: Optional[ConversationContext] = None) -> None:
        """Initializes the ConversationStateManager.

        Args:
            context: Optional existing ConversationContext. If None, creates a new one.
        """
        self.context = context or ConversationContext()

    @property
    def state(self) -> ConversationState:
        """Returns the current conversation lifecycle state."""
        return self.context.state

    @property
    def conversation_id(self) -> str:
        """Returns the unique conversation identifier."""
        return self.context.conversation_id

    def start(self) -> None:
        """Transitions the conversation from INITIALIZED or PAUSED to ACTIVE.

        Raises:
            ConversationStateError: If transition is attempted from ENDED or ERROR.
        """
        if self.state in (ConversationState.ACTIVE, ConversationState.INITIALIZED, ConversationState.PAUSED):
            self.context.state = ConversationState.ACTIVE
            self.context.touch()
        else:
            raise ConversationStateError(
                f"Cannot start conversation from terminal state '{self.state.value}'."
            )

    def pause(self) -> None:
        """Transitions the conversation from ACTIVE to PAUSED.

        Raises:
            ConversationStateError: If the conversation is not currently ACTIVE.
        """
        if self.state == ConversationState.ACTIVE:
            self.context.state = ConversationState.PAUSED
            self.context.touch()
        else:
            raise ConversationStateError(
                f"Cannot pause conversation from state '{self.state.value}'."
            )

    def resume(self) -> None:
        """Transitions the conversation from PAUSED to ACTIVE.

        Raises:
            ConversationStateError: If the conversation is not currently PAUSED.
        """
        if self.state == ConversationState.PAUSED:
            self.context.state = ConversationState.ACTIVE
            self.context.touch()
        else:
            raise ConversationStateError(
                f"Cannot resume conversation from state '{self.state.value}'."
            )

    def end(self) -> None:
        """Transitions the conversation to ENDED.

        Once ended, no further messages can be appended to the conversation.
        """
        self.context.state = ConversationState.ENDED
        self.context.touch()

    def set_error(self, error_details: Optional[dict[str, Any]] = None) -> None:
        """Transitions the conversation to ERROR state and records details.

        Args:
            error_details: Optional dictionary of debugging information.
        """
        self.context.state = ConversationState.ERROR
        if error_details:
            self.context.metadata.setdefault("errors", []).append(error_details)
        self.context.touch()

    def _ensure_can_append_message(self) -> None:
        """Verifies that messages can be added in the current lifecycle state.

        Raises:
            ConversationLifecycleError: If state is ENDED or ERROR.
        """
        if self.state in (ConversationState.ENDED, ConversationState.ERROR):
            raise ConversationLifecycleError(
                f"Cannot append messages to conversation in state '{self.state.value}'."
            )

    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Message:
        """Appends a new message to the conversation history.

        Args:
            role: The role of the message participant.
            content: Text content of the message.
            metadata: Optional domain-agnostic metadata dictionary.

        Returns:
            The created Message instance.

        Raises:
            ConversationLifecycleError: If the conversation is ENDED or ERROR.
        """
        self._ensure_can_append_message()
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.context.history.append(msg)
        self.context.touch()
        return msg

    def add_user_message(self, content: str, metadata: Optional[dict[str, Any]] = None) -> Message:
        """Convenience method to append a USER message."""
        return self.add_message(MessageRole.USER, content, metadata=metadata)

    def add_assistant_message(self, content: str, metadata: Optional[dict[str, Any]] = None) -> Message:
        """Convenience method to append an ASSISTANT message."""
        return self.add_message(MessageRole.ASSISTANT, content, metadata=metadata)

    def add_system_message(self, content: str, metadata: Optional[dict[str, Any]] = None) -> Message:
        """Convenience method to append a SYSTEM message."""
        return self.add_message(MessageRole.SYSTEM, content, metadata=metadata)

    def get_history(self) -> list[Message]:
        """Returns a copy of the chronological message history."""
        return list(self.context.history)

    def get_last_message(self) -> Optional[Message]:
        """Returns the most recent message in history, or None if empty."""
        if not self.context.history:
            return None
        return self.context.history[-1]

    def get_messages_by_role(self, role: MessageRole) -> list[Message]:
        """Returns all messages matching the specified role."""
        return [msg for msg in self.context.history if msg.role == role]
