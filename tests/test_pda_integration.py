"""Unit tests for PDA integration with AI Core and ConversationRuntime."""

import audioop
import base64
import json
import os
from typing import AsyncIterator, Iterator, Optional

import pytest

from ai_core.config.models import GenerationConfig, LLMProviderConfig
from ai_core.conversation import ConversationState
from ai_core.llm.base import BaseLLM, LLMResponse, UsageMetadata
from ai_core.runtime import EventType, RuntimeState
from projects.pda_ai import (
    PDAAssistant,
    PDAGeminiServiceBridge,
    PDATwilioBridge,
    build_system_prompt,
)


class MockPDALLM(BaseLLM):
    """Mock BaseLLM for testing PDAAssistant integration."""

    def __init__(self) -> None:
        cfg = LLMProviderConfig(provider_name="mock-pda", model_name="mock-model")
        super().__init__(cfg)
        self.calls: list[str] = []

    @property
    def provider_name(self) -> str:
        return "mock-pda"

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
            content=f"PDA Assistant answer to: {user_message}",
            model_name=self.model_name,
            provider_name=self.provider_name,
            finish_reason="STOP",
            usage=UsageMetadata(prompt_tokens=12, completion_tokens=8, total_tokens=20),
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
        yield f"PDA stream: {user_message}"

    async def agenerate_stream(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        yield f"PDA async stream: {user_message}"


@pytest.fixture
def sample_student():
    return {
        "parent_name": "Ramesh Rao",
        "student_name": "Aarav Rao",
        "attendance_percentage": "68%",
    }


@pytest.fixture
def sample_campaign():
    return {"type": "attendance_shortage"}


def test_pda_assistant_lifecycle_and_runtime_routing(sample_student, sample_campaign):
    llm = MockPDALLM()
    assistant = PDAAssistant(
        llm=llm,
        student=sample_student,
        campaign=sample_campaign,
        session_id="pda-session-101",
    )

    assert assistant.session_id == "pda-session-101"
    assert assistant.state == RuntimeState.IDLE
    assert "Aarav Rao" in assistant.system_prompt
    assert "68%" in assistant.system_prompt

    # Start session in runtime
    events = assistant.start()
    assert assistant.state == RuntimeState.RUNNING
    assert any(e.event_type == EventType.SESSION_STARTED for e in events)

    # Route user message
    msg_events = assistant.send_user_message("Is Aarav attending classes?")
    assert any(e.event_type == EventType.ASSISTANT_RESPONSE for e in msg_events)
    assert llm.calls == ["Is Aarav attending classes?"]

    # Check conversation history
    history = assistant.get_history()
    assert len(history) == 2
    assert history[0].role.value == "user"
    assert history[1].role.value == "assistant"

    # Record interruption
    int_events = assistant.record_interruption({"reason": "caller_spoke_over"})
    assert any(e.event_type == EventType.INTERRUPTION for e in int_events)

    # Stop session
    stop_events = assistant.stop()
    assert assistant.state == RuntimeState.STOPPED
    assert any(e.event_type == EventType.SESSION_ENDED for e in stop_events)


def test_pda_twilio_bridge_prompt_and_setup_generation(sample_student, sample_campaign):
    llm = MockPDALLM()
    assistant = PDAAssistant(llm=llm, student=sample_student, campaign=sample_campaign)
    bridge = PDATwilioBridge(assistant=assistant, api_key="TEST_GEMINI_KEY")

    # 1. Check WebSocket connection URL
    url = bridge.get_connection_url()
    assert url.startswith("wss://generativelanguage.googleapis.com/ws/")
    assert "TEST_GEMINI_KEY" in url

    # 2. Check greeting
    greeting = bridge.get_time_based_greeting()
    assert greeting in ("Good Morning", "Good Afternoon", "Good Evening")

    # 3. Check setup JSON and automatic runtime start
    setup_json = bridge.build_setup_message(sample_student, sample_campaign)
    data = json.loads(setup_json)
    assert data["setup"]["model"] == "models/gemini-2.5-flash-native-audio-latest"
    assert "AUDIO" in data["setup"]["generationConfig"]["responseModalities"]
    assert data["setup"]["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Puck"

    sys_text = data["setup"]["systemInstruction"]["parts"][0]["text"]
    assert "Aarav Rao" in sys_text
    assert assistant.state == RuntimeState.RUNNING


def test_pda_twilio_bridge_client_content_and_runtime_routing(sample_student):
    llm = MockPDALLM()
    assistant = PDAAssistant(llm=llm, student=sample_student)
    bridge = PDATwilioBridge(assistant=assistant)

    client_json = bridge.build_client_content_message("Hello caller")
    data = json.loads(client_json)
    assert data["clientContent"]["turns"][0]["role"] == "user"
    assert data["clientContent"]["turns"][0]["parts"][0]["text"] == "Hello caller"
    assert data["clientContent"]["turnComplete"] is True

    # Check that message was routed to ConversationRuntime
    history = assistant.get_history()
    assert len(history) == 2
    assert history[0].content == "Hello caller"


def test_pda_twilio_bridge_audioop_resampling():
    bridge = PDATwilioBridge()

    # Create dummy 8kHz u-law audio bytes (100 samples of u-law silence/data)
    dummy_ulaw = bytes([127] * 100)
    b64_ulaw = base64.b64encode(dummy_ulaw).decode("utf-8")

    # 1. Process Twilio audio -> Gemini PCM 16kHz payload
    realtime_json = bridge.process_twilio_audio(b64_ulaw)
    data = json.loads(realtime_json)
    chunk = data["realtimeInput"]["mediaChunks"][0]
    assert chunk["mimeType"] == "audio/pcm;rate=16000"

    pcm_b64 = chunk["data"]
    pcm_bytes = base64.b64decode(pcm_b64)
    # Since we resampled 100 samples of 8kHz to 16kHz (200 samples of 16-bit linear = 400 bytes)
    assert len(pcm_bytes) > 0

    # 2. Process Gemini server response -> Twilio 8kHz u-law audio & transcript
    dummy_pcm_24k = bytes([0] * 480)  # 240 samples of 16-bit PCM (10ms at 24kHz)
    pcm_24k_b64 = base64.b64encode(dummy_pcm_24k).decode("utf-8")
    gemini_server_msg = {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {"text": "Aarav has 68% attendance."},
                    {
                        "inlineData": {
                            "mimeType": "audio/pcm;rate=24000",
                            "data": pcm_24k_b64,
                        }
                    },
                ]
            }
        }
    }

    twilio_audio_b64, transcript = bridge.process_gemini_response(gemini_server_msg)
    assert transcript == "Aarav has 68% attendance."
    assert twilio_audio_b64 is not None
    out_ulaw_bytes = base64.b64decode(twilio_audio_b64)
    assert len(out_ulaw_bytes) > 0


def test_pda_twilio_bridge_auto_records_transcript_in_runtime(sample_student):
    llm = MockPDALLM()
    assistant = PDAAssistant(llm=llm, student=sample_student)
    bridge = PDATwilioBridge(assistant=assistant)
    assistant.start()

    gemini_server_msg = {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {"text": "Namaskara! This is PDA College AI assistant."},
                ]
            }
        }
    }

    _, transcript = bridge.process_gemini_response(gemini_server_msg)
    assert transcript == "Namaskara! This is PDA College AI assistant."

    # Verify assistant runtime recorded the external turn
    history = assistant.get_history()
    assert len(history) == 1
    assert history[0].role.value == "assistant"
    assert history[0].content == "Namaskara! This is PDA College AI assistant."


def test_backward_compatibility_alias():
    assert PDAGeminiServiceBridge is PDATwilioBridge
