"""PDA AI Assistant manager integrating AI Core ConversationRuntime.

This module provides the `PDAAssistant` class, which orchestrates PDA Engineering
College student support conversations by combining business-specific prompt
generation (`projects.pda_ai.prompts`) with the provider-agnostic `ai_core.runtime`
execution engine.
"""

from typing import Any, Optional

from ai_core.conversation import Message, MessageRole
from ai_core.llm import BaseLLM
from ai_core.runtime import (
    ConversationRuntime,
    Event,
    EventCallback,
    EventType,
    RuntimeState,
)
from projects.pda_ai.prompts import build_system_prompt, get_time_based_greeting


class PDAAssistant:
    """Manages an AI Student Support conversation session using AI Core.

    The `PDAAssistant` owns a `ConversationRuntime` and coordinates session
    lifecycle, prompt template rendering, and event subscriptions for a specific
    student and attendance campaign interaction.

    Attributes:
        llm: The injected domain-agnostic LLM adapter.
        runtime: The underlying provider-agnostic ConversationRuntime.
        student: Dictionary of student details.
        campaign: Optional campaign metadata dictionary.
    """

    def __init__(
        self,
        llm: BaseLLM,
        student: dict[str, Any],
        campaign: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Initializes the PDAAssistant.

        Args:
            llm: An instance implementing BaseLLM.
            student: Dictionary containing `parent_name`, `student_name`,
                `attendance_percentage`.
            campaign: Optional dictionary containing campaign `type`.
            session_id: Optional custom session identifier.
        """
        self.llm = llm
        self.student = student
        self.campaign = campaign
        self._system_prompt = build_system_prompt(student, campaign)

        self.runtime = ConversationRuntime(
            llm=self.llm,
            session_id=session_id,
            system_prompt=self._system_prompt,
        )

    @property
    def session_id(self) -> str:
        """Returns the unique session identifier."""
        return self.runtime.session_id

    @property
    def state(self) -> RuntimeState:
        """Returns the current runtime lifecycle state."""
        return self.runtime.state

    @property
    def system_prompt(self) -> str:
        """Returns the rendered PDA system prompt string for this session."""
        return self._system_prompt

    def get_greeting(self) -> str:
        """Returns the time-based greeting for the current session."""
        return get_time_based_greeting()

    def start(self, custom_system_prompt: Optional[str] = None) -> list[Event]:
        """Starts the conversation session in the underlying runtime.

        Args:
            custom_system_prompt: Optional override for the rendered system prompt.

        Returns:
            List of Events produced during session startup.
        """
        prompt = custom_system_prompt or self._system_prompt
        return self.runtime.start_session(system_prompt=prompt)

    def stop(self) -> list[Event]:
        """Terminates the conversation session in the underlying runtime.

        Returns:
            List of Events produced during session termination.
        """
        return self.runtime.stop_session()

    def send_user_message(
        self, content: str, metadata: Optional[dict[str, Any]] = None
    ) -> list[Event]:
        """Routes a user message or voice transcript through the runtime.

        Args:
            content: Text message or transcribed speech from the caller.
            metadata: Optional transport or routing metadata.

        Returns:
            List of outgoing Events produced by the runtime.
        """
        return self.runtime.send_user_message(content, metadata=metadata)

    def record_external_response(
        self,
        content: str,
        usage: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[Event]:
        """Records an external assistant turn (e.g. from voice stream) in runtime.

        Args:
            content: The text transcript of the assistant's response.
            usage: Optional token usage metadata.
            metadata: Optional additional metadata.

        Returns:
            List of Events dispatched by the runtime.
        """
        self.runtime.conversation_engine.state_manager.add_message(
            MessageRole.ASSISTANT,
            content=content,
            metadata=metadata or {},
        )
        event = Event.create_assistant_response(
            self.session_id,
            content=content,
            usage=usage,
            metadata=metadata,
        )
        return self.runtime.process_event(event)

    def record_interruption(
        self, metadata: Optional[dict[str, Any]] = None
    ) -> list[Event]:
        """Records a caller interruption event in the runtime.

        Args:
            metadata: Optional details about the interruption.

        Returns:
            List of Events dispatched by the runtime.
        """
        event = Event.create_interruption(self.session_id, metadata=metadata)
        return self.runtime.process_event(event)

    def subscribe(self, event_type: EventType, callback: EventCallback) -> None:
        """Subscribes a listener callback to a specific EventType."""
        self.runtime.subscribe(event_type, callback)

    def subscribe_all(self, callback: EventCallback) -> None:
        """Subscribes a listener callback to all runtime events."""
        self.runtime.subscribe_all(callback)

    def get_history(self) -> list[Message]:
        """Returns the chronological message history of the session."""
        return self.runtime.conversation_engine.get_history()
