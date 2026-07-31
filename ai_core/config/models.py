"""Configuration data models for the AI Core Platform.

This module defines domain-agnostic, strongly-typed configuration structures
for LLM generation parameters and provider settings.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GenerationConfig:
    """Provider-agnostic text generation settings for LLM calls.

    Attributes:
        temperature: Controls randomness in generation (e.g., 0.0 for deterministic, 1.0 for creative).
        max_output_tokens: Maximum number of tokens to generate in the response.
        top_p: Nucleus sampling cumulative probability threshold.
        top_k: Top-k vocabulary sampling limit.
        stop_sequences: List of strings that will stop output generation when encountered.
        response_mime_type: Expected MIME type for the output (e.g., 'text/plain', 'application/json').
        extra_params: Provider-specific generation parameters not covered by standard fields.
    """

    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    response_mime_type: Optional[str] = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts non-None configuration fields to a dictionary.

        Returns:
            A dictionary containing only configured generation parameters.
        """
        data: dict[str, Any] = {}
        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            data["max_output_tokens"] = self.max_output_tokens
        if self.top_p is not None:
            data["top_p"] = self.top_p
        if self.top_k is not None:
            data["top_k"] = self.top_k
        if self.stop_sequences is not None:
            data["stop_sequences"] = self.stop_sequences
        if self.response_mime_type is not None:
            data["response_mime_type"] = self.response_mime_type
        if self.extra_params:
            data.update(self.extra_params)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationConfig":
        """Creates a GenerationConfig instance from a dictionary.

        Args:
            data: Raw configuration dictionary.

        Returns:
            A populated GenerationConfig instance.
        """
        copy_data = dict(data)
        return cls(
            temperature=copy_data.pop("temperature", None),
            max_output_tokens=copy_data.pop("max_output_tokens", None),
            top_p=copy_data.pop("top_p", None),
            top_k=copy_data.pop("top_k", None),
            stop_sequences=copy_data.pop("stop_sequences", None),
            response_mime_type=copy_data.pop("response_mime_type", None),
            extra_params=copy_data,
        )


@dataclass
class LLMProviderConfig:
    """Configuration settings for an individual LLM provider.

    Attributes:
        provider_name: Unique identifier for the provider (e.g., 'gemini', 'openai').
        model_name: Target model identifier (e.g., 'gemini-2.5-pro').
        api_key: Optional API key. If None, adapters should resolve via environment variables.
        timeout_seconds: Request timeout duration in seconds.
        max_retries: Number of retry attempts on transient network or provider errors.
        default_generation_config: Baseline generation configuration for this provider.
        extra_settings: Provider-specific configuration options (e.g., endpoint URL, version).
    """

    provider_name: str
    model_name: str
    api_key: Optional[str] = None
    timeout_seconds: float = 60.0
    max_retries: int = 3
    default_generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    extra_settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMProviderConfig":
        """Creates an LLMProviderConfig instance from a dictionary.

        Args:
            data: Raw provider configuration dictionary.

        Returns:
            A populated LLMProviderConfig instance.
        """
        copy_data = dict(data)
        gen_config_raw = copy_data.pop("default_generation_config", {})
        gen_config = (
            GenerationConfig.from_dict(gen_config_raw)
            if isinstance(gen_config_raw, dict)
            else gen_config_raw
        )
        return cls(
            provider_name=copy_data.pop("provider_name", "unknown"),
            model_name=copy_data.pop("model_name", "default-model"),
            api_key=copy_data.pop("api_key", None),
            timeout_seconds=float(copy_data.pop("timeout_seconds", 60.0)),
            max_retries=int(copy_data.pop("max_retries", 3)),
            default_generation_config=gen_config,
            extra_settings=copy_data,
        )


@dataclass
class LLMConfig:
    """Top-level LLM layer configuration container.

    Attributes:
        default_provider: Name of the default provider to use when none is explicitly specified.
        providers: Mapping of provider names to their respective configurations.
    """

    default_provider: str = "gemini"
    providers: dict[str, LLMProviderConfig] = field(default_factory=dict)

    def get_provider_config(self, provider_name: Optional[str] = None) -> LLMProviderConfig:
        """Retrieves configuration for the requested provider or the default provider.

        Args:
            provider_name: Optional name of the provider. If None, uses default_provider.

        Returns:
            The matching LLMProviderConfig.

        Raises:
            KeyError: If the requested provider configuration is not found.
        """
        target = provider_name or self.default_provider
        if target not in self.providers:
            raise KeyError(f"Configuration for LLM provider '{target}' not found.")
        return self.providers[target]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMConfig":
        """Creates an LLMConfig instance from a nested dictionary.

        Args:
            data: Raw configuration dictionary.

        Returns:
            A populated LLMConfig instance.
        """
        default_provider = data.get("default_provider", "gemini")
        providers_raw = data.get("providers", {})
        providers: dict[str, LLMProviderConfig] = {}

        for name, cfg in providers_raw.items():
            if isinstance(cfg, dict):
                cfg_copy = dict(cfg)
                cfg_copy.setdefault("provider_name", name)
                providers[name] = LLMProviderConfig.from_dict(cfg_copy)
            elif isinstance(cfg, LLMProviderConfig):
                providers[name] = cfg

        return cls(default_provider=default_provider, providers=providers)
