"""Provider-agnostic integration bridge for Swargaseema AI.

This module provides `SwargaseemaBridge`, which exposes generic telephony and
(e.g. SIP, WebRTC, HTTP audio, or any telephony provider) can call without
importing provider SDKs.
"""

from typing import Any, Optional

from ai_core.conversation import Message
from ai_core.runtime import Event
from projects.swargaseema_ai.assistant import SwargaseemaAssistant
from projects.swargaseema_ai.config import SwargaseemaSettings
from projects.swargaseema_ai.prompts import (
    build_system_prompt,
    get_time_based_greeting,
)


class SwargaseemaBridge:
    """Provider-agnostic adapter bridge for Swargaseema Sandalwood Farms.

    `SwargaseemaBridge` decouples external voice/telephony providers from the
    underlying `SwargaseemaAssistant` and AI Core runtime. It provides generic
    methods for session configuration, opening greetings, message turns,
    interruption logging, and session termination.

    Attributes:
        assistant: Optional SwargaseemaAssistant instance wrapping ConversationRuntime.
        settings: Project configuration settings.
    """

    def __init__(
        self,
        assistant: Optional[SwargaseemaAssistant] = None,
        settings: Optional[SwargaseemaSettings] = None,
    ) -> None:
        """Initializes the SwargaseemaBridge.

        Args:
            assistant: Optional SwargaseemaAssistant instance.
            settings: Optional project configuration settings override.
        """
        self.assistant = assistant
        self.settings = settings or (
            assistant.settings if assistant else SwargaseemaSettings()
        )

    def get_session_config(
        self,
        customer: Optional[dict[str, Any]] = None,
        project: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Returns a provider-agnostic session configuration dictionary.

        Any external voice or streaming adapter can use this dictionary to
        configure its model, system prompt, voice, and language settings.

        Args:
            customer: Optional customer details dictionary.
            project: Optional project details dictionary.

        Returns:
            A dictionary containing `system_prompt`, `voice_name`, `model_name`,
            `preferred_language`, `project_name`, and optional `session_id`.
        """
        if self.assistant:
            sys_prompt = self.assistant.system_prompt
            session_id = self.assistant.session_id
        else:
            sys_prompt = build_system_prompt(customer, project)
            session_id = None

        customer = customer or (self.assistant.customer if self.assistant else {})
        project = project or (self.assistant.project if self.assistant else {})

        config = {
            "system_prompt": sys_prompt,
            "voice_name": self.settings.default_voice_name,
            "model_name": self.settings.default_model_name,
            "preferred_language": customer.get(
                "preferred_language", self.settings.default_preferred_language
            ),
            "project_name": project.get(
                "project_name", self.settings.default_project_name
            ),
        }
        if session_id:
            config["session_id"] = session_id
        return config

    def get_time_based_greeting(self, current_hour: Optional[int] = None) -> str:
        """Returns a time-based greeting such as 'Good Morning'.

        Args:
            current_hour: Optional hour override for testing.

        Returns:
            Greeting string.
        """
        if self.assistant:
            return self.assistant.get_greeting(current_hour=current_hour)
        return get_time_based_greeting(current_hour=current_hour)

    def get_initial_greeting_message(
        self,
        current_hour: Optional[int] = None,
        project_name: Optional[str] = None,
    ) -> str:
        """Returns the full introductory greeting message for inbound callers.

        Args:
            current_hour: Optional hour override.
            project_name: Optional project title override.

        Returns:
            A formatted opening string suitable for speech synthesis.
        """
        greeting = self.get_time_based_greeting(current_hour=current_hour)
        name = project_name or self.settings.default_project_name
        return (
            f"{greeting}! Thank you for calling {name}. "
            "How may I assist you with our sandalwood farmland opportunities today?"
        )

    def start_session(
        self, custom_system_prompt: Optional[str] = None
    ) -> list[Event]:
        """Starts the conversation session in the underlying assistant runtime.

        Args:
            custom_system_prompt: Optional override for the rendered system prompt.

        Returns:
            List of Events emitted during session start.
        """
        if not self.assistant:
            return []
        return self.assistant.start(custom_system_prompt=custom_system_prompt)

    def process_inbound_text(
        self, text: str, metadata: Optional[dict[str, Any]] = None
    ) -> list[Event]:
        """Routes transcribed caller speech through the assistant runtime.

        Args:
            text: Transcribed text from the caller.
            metadata: Optional transport or routing metadata.

        Returns:
            List of Events emitted by the runtime.
        """
        if not self.assistant:
            return []
        return self.assistant.send_user_message(text, metadata=metadata)

    def process_outbound_text(
        self,
        text: str,
        usage: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[Event]:
        """Records an external assistant response turn in the assistant runtime.

        Args:
            text: Transcript of the assistant's response.
            usage: Optional token usage dictionary.
            metadata: Optional extra metadata.

        Returns:
            List of Events emitted by the runtime.
        """
        if not self.assistant:
            return []
        return self.assistant.record_external_response(
            text, usage=usage, metadata=metadata
        )

    def handle_interruption(
        self, metadata: Optional[dict[str, Any]] = None
    ) -> list[Event]:
        """Records a caller interruption event in the assistant runtime.

        Args:
            metadata: Optional details about the interruption.

        Returns:
            List of Events emitted by the runtime.
        """
        if not self.assistant:
            return []
        return self.assistant.record_interruption(metadata=metadata)

    def end_session(self) -> list[Event]:
        """Terminates the conversation session in the assistant runtime.

        Returns:
            List of Events emitted during termination.
        """
        if not self.assistant:
            return []
        return self.assistant.stop()

    def get_history(self) -> list[Message]:
        """Returns the chronological message history from the assistant.

        Returns:
            List of Message instances.
        """
        if not self.assistant:
            return []
        return self.assistant.get_history()
