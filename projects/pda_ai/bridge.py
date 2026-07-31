"""PDA Twilio integration bridge for the AI Core Platform.

This module provides `PDATwilioBridge` (aliased as `PDAGeminiServiceBridge`), which
replaces the legacy monolithic `gemini_service.py` by:
1. Delegating all prompt generation to `projects.pda_ai.prompts`.
2. Routing conversation execution and lifecycle events through `PDAAssistant` and
   the provider-agnostic `ai_core.runtime.ConversationRuntime`.
3. Preserving existing Twilio transport framing and audioop PCM-mulaw resampling
   for the Gemini Live Audio API.
"""

import audioop
import base64
import json
import os
from typing import Any, Optional

from projects.pda_ai.assistant import PDAAssistant
from projects.pda_ai.prompts import build_system_prompt, get_time_based_greeting


class PDATwilioBridge:
    """Bridge connecting existing Twilio audio transport to AI Core ConversationRuntime.

    This adapter maintains full backward compatibility with the legacy Twilio
    WebSocket transport and audio streaming format while cleanly separating
    prompt generation (`projects.pda_ai.prompts`) and conversation execution
    (`ai_core.runtime.ConversationRuntime`).

    Attributes:
        assistant: Optional PDAAssistant instance wrapping ConversationRuntime.
        api_key: Optional Gemini API key string for Live WebSocket authentication.
    """

    def __init__(
        self,
        assistant: Optional[PDAAssistant] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initializes the PDATwilioBridge.

        Args:
            assistant: Optional PDAAssistant instance for runtime event tracking.
            api_key: Optional API key override (defaults to GEMINI_API_KEY env var).
        """
        self.assistant = assistant
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    def get_connection_url(self) -> str:
        """Returns the WebSocket connection URL for Gemini Live Audio API.

        Returns:
            Formatted wss:// URL string including the authentication key.
        """
        return (
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"
            f"?key={self.api_key}"
        )

    def get_time_based_greeting(self) -> str:
        """Returns the appropriate time-based greeting for the call opening.

        Delegates to `projects.pda_ai.prompts.get_time_based_greeting()`.

        Returns:
            A greeting string such as 'Good Morning'.
        """
        if self.assistant:
            return self.assistant.get_greeting()
        return get_time_based_greeting()

    def generate_system_instruction(
        self,
        student: dict[str, Any],
        campaign: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generates the domain-specific PDA Engineering College system instruction.

        Delegates to `projects.pda_ai.prompts.build_system_prompt()`.

        Args:
            student: Dictionary containing student details.
            campaign: Optional dictionary containing campaign details.

        Returns:
            The rendered plain system prompt string.
        """
        return build_system_prompt(student, campaign)

    def build_setup_message(
        self,
        student: dict[str, Any],
        campaign: Optional[dict[str, Any]] = None,
    ) -> str:
        """Builds the initial setup configuration JSON payload for Gemini Live.

        Also notifies the underlying `ConversationRuntime` that a session has started.

        Args:
            student: Dictionary containing student details.
            campaign: Optional dictionary containing campaign details.

        Returns:
            JSON-serialized string of the Gemini setup payload.
        """
        system_instruction_text = self.generate_system_instruction(student, campaign)

        if self.assistant:
            self.assistant.start(custom_system_prompt=system_instruction_text)

        setup_payload = {
            "setup": {
                "model": "models/gemini-2.5-flash-native-audio-latest",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": "Puck"
                            }
                        }
                    },
                },
                "systemInstruction": {
                    "parts": [{"text": system_instruction_text}]
                },
            }
        }
        return json.dumps(setup_payload)

    def build_client_content_message(self, text: str) -> str:
        """Builds a clientContent JSON payload and routes text through runtime.

        Args:
            text: Text content to send (e.g., call trigger greeting or user speech).

        Returns:
            JSON-serialized string of the clientContent turn payload.
        """
        if self.assistant:
            self.assistant.send_user_message(text)

        payload = {
            "clientContent": {
                "turns": [
                    {
                        "role": "user",
                        "parts": [{"text": text}],
                    }
                ],
                "turnComplete": True,
            }
        }
        return json.dumps(payload)

    def process_twilio_audio(self, base64_payload: str) -> str:
        """Converts Twilio base64 audio/x-mulaw (8kHz) to Gemini 16kHz linear PCM.

        Args:
            base64_payload: Base64-encoded u-law audio payload from Twilio.

        Returns:
            JSON-serialized string containing the Gemini realtimeInput mediaChunk.
        """
        mulaw_bytes = base64.b64decode(base64_payload)
        pcm_8k_bytes = audioop.ulaw2lin(mulaw_bytes, 2)
        pcm_16k_bytes, _ = audioop.ratecv(pcm_8k_bytes, 2, 1, 8000, 16000, None)
        pcm_16k_b64 = base64.b64encode(pcm_16k_bytes).decode("utf-8")

        payload = {
            "realtimeInput": {
                "mediaChunks": [
                    {
                        "mimeType": "audio/pcm;rate=16000",
                        "data": pcm_16k_b64,
                    }
                ]
            }
        }
        return json.dumps(payload)

    def process_gemini_response(
        self, gemini_msg: dict[str, Any]
    ) -> tuple[Optional[str], str]:
        """Extracts audio and transcript from Gemini serverContent response.

        Converts 24kHz linear PCM audio back to 8kHz u-law for Twilio playback,
        and routes any recognized text transcript through ConversationRuntime.

        Args:
            gemini_msg: Dictionary parsed from Gemini server content JSON.

        Returns:
            A tuple of `(twilio_base64_audio, ai_text_transcript)`.
        """
        audio_b64: Optional[str] = None
        transcript = ""

        server_content = gemini_msg.get("serverContent", {})
        model_turn = server_content.get("modelTurn", {})
        parts = model_turn.get("parts", [])

        for part in parts:
            if "text" in part and part["text"]:
                transcript += str(part["text"])

            if "inlineData" in part:
                mime_type = str(part["inlineData"].get("mimeType", ""))
                if "audio/pcm" in mime_type:
                    pcm_data_b64 = str(part["inlineData"].get("data", ""))
                    if pcm_data_b64:
                        pcm_24k_bytes = base64.b64decode(pcm_data_b64)
                        pcm_8k_bytes, _ = audioop.ratecv(
                            pcm_24k_bytes, 2, 1, 24000, 8000, None
                        )
                        mulaw_bytes = audioop.lin2ulaw(pcm_8k_bytes, 2)
                        audio_b64 = base64.b64encode(mulaw_bytes).decode("utf-8")

        if transcript and self.assistant:
            self.assistant.record_external_response(content=transcript)

        return audio_b64, transcript


# Backward compatibility alias for legacy integrations
PDAGeminiServiceBridge = PDATwilioBridge
