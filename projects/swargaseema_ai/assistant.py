"""Swargaseema AI inbound receptionist assistant manager.

This module defines `SwargaseemaAssistant`, orchestrating Swargaseema Sandalwood
Farms inbound calls by integrating domain prompt templates (`projects.swargaseema_ai.prompts`)
with the provider-agnostic `ai_core.runtime` execution engine.
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
from projects.swargaseema_ai.config import SwargaseemaSettings
from projects.swargaseema_ai.prompts import build_system_prompt, get_time_based_greeting


class SwargaseemaAssistant:
    """Manages an inbound AI receptionist conversation session for Swargaseema.

    The `SwargaseemaAssistant` owns an `ai_core.runtime.ConversationRuntime` and
    coordinates session lifecycle, prompt template rendering, and event pub-sub
    subscriptions for a specific caller interaction.

    Attributes:
        llm: The injected domain-agnostic LLM adapter.
        runtime: The underlying provider-agnostic ConversationRuntime.
        customer: Dictionary containing caller metadata.
        project: Dictionary containing project metadata.
        settings: Project configuration settings.
    """

    def __init__(
        self,
        llm: BaseLLM,
        customer: Optional[dict[str, Any]] = None,
        project: Optional[dict[str, Any]] = None,
        settings: Optional[SwargaseemaSettings] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Initializes the SwargaseemaAssistant.

        Args:
            llm: An instance implementing BaseLLM.
            customer: Optional dictionary containing `customer_name` and
                `preferred_language`.
            project: Optional dictionary containing `project_name` and
                `project_location`.
            settings: Optional project settings override.
            session_id: Optional custom session identifier.
        """
        self.llm = llm
        self.customer = customer or {}
        self.project = project or {}
        self.settings = settings or SwargaseemaSettings()

        self._system_prompt = build_system_prompt(self.customer, self.project)
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
        """Returns the rendered Swargaseema system prompt string."""
        return self._system_prompt

    def get_greeting(self, current_hour: Optional[int] = None) -> str:
        """Returns the time-based greeting for the current session.

        Args:
            current_hour: Optional hour override for testing.

        Returns:
            A greeting string such as 'Good Morning'.
        """
        return get_time_based_greeting(current_hour=current_hour)

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
        """Routes a caller text message or speech transcript through runtime.

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
