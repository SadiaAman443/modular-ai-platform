"""Configuration loader utilities for the AI Core Platform.

This module provides functions and classes to load LLM configurations
from JSON files, YAML files, dictionaries, and environment variables.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

from ai_core.config.models import GenerationConfig, LLMConfig, LLMProviderConfig
from ai_core.llm.exceptions import LLMConfigurationError


class ConfigLoader:
    """Loads and validates LLMConfig from various external sources.

    This class provides static and class methods to instantiate configuration
    models from dictionaries, filesystem files (JSON/YAML), and environment
    variables without coupling to application-specific paths.
    """

    @classmethod
    def load_from_dict(cls, data: dict[str, Any]) -> LLMConfig:
        """Loads configuration from a Python dictionary.

        Args:
            data: Raw configuration dictionary.

        Returns:
            A validated LLMConfig instance.

        Raises:
            LLMConfigurationError: If the dictionary structure is malformed.
        """
        try:
            return LLMConfig.from_dict(data)
        except Exception as exc:
            raise LLMConfigurationError(f"Failed to load LLMConfig from dictionary: {exc}") from exc

    @classmethod
    def load_from_json_file(cls, file_path: Union[str, Path]) -> LLMConfig:
        """Loads configuration from a JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            A validated LLMConfig instance.

        Raises:
            LLMConfigurationError: If the file is missing or contains invalid JSON.
        """
        path = Path(file_path)
        if not path.exists():
            raise LLMConfigurationError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.load_from_dict(data)
        except json.JSONDecodeError as exc:
            raise LLMConfigurationError(f"Invalid JSON in configuration file '{path}': {exc}") from exc
        except Exception as exc:
            raise LLMConfigurationError(f"Error reading configuration file '{path}': {exc}") from exc

    @classmethod
    def load_from_yaml_file(cls, file_path: Union[str, Path]) -> LLMConfig:
        """Loads configuration from a YAML file.

        Args:
            file_path: Path to the YAML configuration file.

        Returns:
            A validated LLMConfig instance.

        Raises:
            LLMConfigurationError: If PyYAML is missing, the file is missing, or YAML is invalid.
        """
        try:
            import yaml
        except ImportError as exc:
            raise LLMConfigurationError(
                "PyYAML is required to load YAML configuration files. Please install 'pyyaml'."
            ) from exc

        path = Path(file_path)
        if not path.exists():
            raise LLMConfigurationError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise LLMConfigurationError("Top-level YAML content must be a mapping (dictionary).")
            return cls.load_from_dict(data)
        except Exception as exc:
            raise LLMConfigurationError(f"Error reading YAML configuration file '{path}': {exc}") from exc

    @classmethod
    def load_from_env(cls, default_provider: str = "gemini") -> LLMConfig:
        """Loads configuration from standard environment variables.

        Supported environment variables:
            - AI_CORE_DEFAULT_PROVIDER (defaults to 'default_provider' param)
            - GEMINI_API_KEY
            - GEMINI_MODEL (defaults to 'gemini-2.5-pro')
            - GEMINI_TIMEOUT_SECONDS (defaults to '60.0')

        Args:
            default_provider: Default provider name if not set in environment.

        Returns:
            A populated LLMConfig instance derived from environment variables.
        """
        selected_provider = os.getenv("AI_CORE_DEFAULT_PROVIDER", default_provider)

        gemini_config = LLMProviderConfig(
            provider_name="gemini",
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
            api_key=os.getenv("GEMINI_API_KEY"),
            timeout_seconds=float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60.0")),
            default_generation_config=GenerationConfig(
                temperature=cls._get_optional_float_env("GEMINI_TEMPERATURE"),
                max_output_tokens=cls._get_optional_int_env("GEMINI_MAX_OUTPUT_TOKENS"),
            ),
        )

        return LLMConfig(
            default_provider=selected_provider,
            providers={"gemini": gemini_config},
        )

    @classmethod
    def load(cls, source: Optional[Union[str, Path, dict[str, Any]]] = None) -> LLMConfig:
        """Universal loader method that resolves config from dictionary, file path, or environment.

        Args:
            source: Can be a dictionary, a file path (JSON or YAML), or None (uses environment).

        Returns:
            A validated LLMConfig instance.

        Raises:
            LLMConfigurationError: If the configuration cannot be loaded or is invalid.
        """
        if source is None:
            return cls.load_from_env()

        if isinstance(source, dict):
            return cls.load_from_dict(source)

        if isinstance(source, (str, Path)):
            path_str = str(source).lower()
            if path_str.endswith((".yaml", ".yml")):
                return cls.load_from_yaml_file(source)
            if path_str.endswith(".json"):
                return cls.load_from_json_file(source)
            raise LLMConfigurationError(
                f"Unsupported configuration file format for '{source}'. Use .json or .yaml."
            )

        raise LLMConfigurationError(
            f"Invalid source type '{type(source).__name__}'. Expected dict, str, Path, or None."
        )

    @staticmethod
    def _get_optional_float_env(key: str) -> Optional[float]:
        val = os.getenv(key)
        if val is None:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    @staticmethod
    def _get_optional_int_env(key: str) -> Optional[int]:
        val = os.getenv(key)
        if val is None:
            return None
        try:
            return int(val)
        except ValueError:
            return None
