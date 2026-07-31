"""Configuration settings for Swargaseema AI inbound receptionist project.

This module defines `SwargaseemaSettings`, encapsulating project-specific defaults
and voice/LLM configuration parameters without coupling to external frameworks.
"""

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class SwargaseemaSettings:
    """Project settings and defaults for Swargaseema AI Sandalwood Farms.

    Attributes:
        default_project_name: Official title of the farmland project.
        default_project_location: Geographic location of the project.
        default_preferred_language: Default spoken language for callers.
        default_voice_name: Default prebuilt voice name for TTS/audio models.
        default_model_name: Recommended model identifier for inbound receptionist.
        max_turn_history: Maximum number of conversation turns to retain in history.
    """

    default_project_name: str = "Swargaseema Sandalwood Farms"
    default_project_location: str = "Hyderabad, Telangana"
    default_preferred_language: str = "English"
    default_voice_name: str = "Aura"
    default_model_name: str = "gemini-2.5-flash-native-audio-latest"
    max_turn_history: int = 20

    def to_dict(self) -> dict[str, Any]:
        """Serializes the settings to a dictionary."""
        return {
            "default_project_name": self.default_project_name,
            "default_project_location": self.default_project_location,
            "default_preferred_language": self.default_preferred_language,
            "default_voice_name": self.default_voice_name,
            "default_model_name": self.default_model_name,
            "max_turn_history": self.max_turn_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SwargaseemaSettings":
        """Constructs a SwargaseemaSettings instance from a dictionary."""
        return cls(
            default_project_name=str(
                data.get("default_project_name", cls.default_project_name)
            ),
            default_project_location=str(
                data.get("default_project_location", cls.default_project_location)
            ),
            default_preferred_language=str(
                data.get("default_preferred_language", cls.default_preferred_language)
            ),
            default_voice_name=str(
                data.get("default_voice_name", cls.default_voice_name)
            ),
            default_model_name=str(
                data.get("default_model_name", cls.default_model_name)
            ),
            max_turn_history=int(data.get("max_turn_history", cls.max_turn_history)),
        )

    @classmethod
    def from_env(cls) -> "SwargaseemaSettings":
        """Constructs a SwargaseemaSettings instance from environment variables."""
        return cls(
            default_project_name=os.getenv(
                "SWARGASEEMA_PROJECT_NAME", cls.default_project_name
            ),
            default_project_location=os.getenv(
                "SWARGASEEMA_PROJECT_LOCATION", cls.default_project_location
            ),
            default_preferred_language=os.getenv(
                "SWARGASEEMA_DEFAULT_LANGUAGE", cls.default_preferred_language
            ),
            default_voice_name=os.getenv(
                "SWARGASEEMA_VOICE_NAME", cls.default_voice_name
            ),
            default_model_name=os.getenv(
                "SWARGASEEMA_MODEL_NAME", cls.default_model_name
            ),
            max_turn_history=int(
                os.getenv("SWARGASEEMA_MAX_TURN_HISTORY", str(cls.max_turn_history))
            ),
        )
