"""Unit tests for Swargaseema AI project layer integration."""

import sys
from typing import AsyncIterator, Iterator, Optional

import pytest

from ai_core.config.models import GenerationConfig, LLMProviderConfig
from ai_core.llm.base import BaseLLM, LLMResponse, UsageMetadata
from ai_core.runtime import EventType, RuntimeState
from projects.swargaseema_ai import (
    SwargaseemaAssistant,
    SwargaseemaBridge,
    SwargaseemaSettings,
    build_system_prompt,
    get_time_based_greeting,
)


class MockSwargaseemaLLM(BaseLLM):
    """Mock BaseLLM for testing SwargaseemaAssistant."""

    def __init__(self) -> None:
        cfg = LLMProviderConfig(provider_name="mock-swarga", model_name="mock-model")
        super().__init__(cfg)
        self.calls: list[str] = []

    @property
    def provider_name(self) -> str:
        return "mock-swarga"

    @property
    def model_name(self) -> str:
        return "mock-model"

    def generate_content(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        self.calls.append(user_message)
        return LLMResponse(
            content=f"Swargaseema receptionist response to: {user_message}",
            model_name=self.model_name,
            provider_name=self.provider_name,
            finish_reason="STOP",
            usage=UsageMetadata(prompt_tokens=15, completion_tokens=10, total_tokens=25),
        )

    async def agenerate_content(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        return self.generate_content(
            user_message,
            system_prompt=system_prompt,
            generation_config=generation_config,
        )

    def generate_stream(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        yield f"Swargaseema stream: {user_message}"

    async def agenerate_stream(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        yield f"Swargaseema async stream: {user_message}"


def test_swargaseema_settings(monkeypatch):
    settings = SwargaseemaSettings()
    assert settings.default_project_name == "Swargaseema Sandalwood Farms"
    assert settings.default_voice_name == "Aura"

    data = settings.to_dict()
    assert data["default_project_location"] == "Hyderabad, Telangana"

    custom = SwargaseemaSettings.from_dict({"default_voice_name": "Puck"})
    assert custom.default_voice_name == "Puck"

    monkeypatch.setenv("SWARGASEEMA_PROJECT_NAME", "Swargaseema Phase 3")
    env_settings = SwargaseemaSettings.from_env()
    assert env_settings.default_project_name == "Swargaseema Phase 3"


def test_swargaseema_prompts():
    prompt = build_system_prompt(
        customer={"customer_name": "Lakshmi N", "preferred_language": "Telugu"},
        project={"project_name": "Swargaseema Phase 5", "project_location": "Shadnagar"},
    )
    assert "Lakshmi N" in prompt
    assert "Telugu" in prompt
    assert "Swargaseema Phase 5" in prompt
    assert "Shadnagar" in prompt
    assert "Guaranteed fixed financial returns" not in prompt

    assert get_time_based_greeting(9) == "Good Morning"
    assert get_time_based_greeting(14) == "Good Afternoon"
    assert get_time_based_greeting(20) == "Good Evening"


def test_swargaseema_assistant_lifecycle_and_runtime():
    llm = MockSwargaseemaLLM()
    assistant = SwargaseemaAssistant(
        llm=llm,
        customer={"customer_name": "Vijay"},
        session_id="swarga-call-101",
    )

    assert assistant.session_id == "swarga-call-101"
    assert assistant.state == RuntimeState.IDLE
    assert "Vijay" in assistant.system_prompt

    events = assistant.start()
    assert assistant.state == RuntimeState.RUNNING
    assert any(e.event_type == EventType.SESSION_STARTED for e in events)

    msg_events = assistant.send_user_message("Tell me about plot sizes.")
    assert any(e.event_type == EventType.ASSISTANT_RESPONSE for e in msg_events)
    assert llm.calls == ["Tell me about plot sizes."]

    ext_events = assistant.record_external_response("We offer 100 sq yard plots.")
    assert any(e.event_type == EventType.ASSISTANT_RESPONSE for e in ext_events)

    int_events = assistant.record_interruption({"reason": "caller_spoke"})
    assert any(e.event_type == EventType.INTERRUPTION for e in int_events)

    history = assistant.get_history()
    assert len(history) == 3
    assert history[0].role.value == "user"
    assert history[1].role.value == "assistant"
    assert history[2].role.value == "assistant"
    assert history[2].content == "We offer 100 sq yard plots."

    assistant.stop()
    assert assistant.state == RuntimeState.STOPPED


def test_swargaseema_bridge_provider_agnostic_methods():
    llm = MockSwargaseemaLLM()
    assistant = SwargaseemaAssistant(llm=llm, customer={"preferred_language": "Hindi"})
    bridge = SwargaseemaBridge(assistant=assistant)

    # Check session config
    config = bridge.get_session_config()
    assert config["system_prompt"] == assistant.system_prompt
    assert config["voice_name"] == "Aura"
    assert config["preferred_language"] == "Hindi"

    # Check greeting messages
    greeting_msg = bridge.get_initial_greeting_message(current_hour=10)
    assert "Good Morning" in greeting_msg
    assert "Swargaseema Sandalwood Farms" in greeting_msg
    assert "sandalwood farmland opportunities" in greeting_msg

    # Check session lifecycle routing
    start_events = bridge.start_session()
    assert len(start_events) > 0

    inbound_events = bridge.process_inbound_text("Can I book a site visit?")
    assert len(inbound_events) > 0

    outbound_events = bridge.process_outbound_text("Certainly, our advisor will call.")
    assert len(outbound_events) > 0

    history = bridge.get_history()
    assert len(history) == 3

    stop_events = bridge.end_session()
    assert len(stop_events) > 0


def test_swargaseema_bridge_no_forbidden_imports():
    import projects.swargaseema_ai.bridge as bridge_mod

    module_names = set(sys.modules.keys())
    # Ensure bridge module doesn't import forbidden packages
    bridge_source = bridge_mod.__file__
    with open(bridge_source, "r", encoding="utf-8") as f:
        content = f.read()
    assert "twilio" not in content.lower()
    assert "bolna" not in content.lower()
    assert "fastapi" not in content.lower()
    assert "websockets" not in content.lower()
