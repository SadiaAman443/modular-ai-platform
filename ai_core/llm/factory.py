"""LLM adapter factory for the AI Core Platform.

This module implements the Factory and Registry patterns for dynamically
instantiating provider-agnostic BaseLLM adapters from configurations.
"""

from typing import Any, Callable, Optional, Type

from ai_core.config.models import LLMConfig, LLMProviderConfig
from ai_core.llm.base import BaseLLM
from ai_core.llm.exceptions import LLMConfigurationError, LLMUnsupportedProviderError
from ai_core.llm.gemini_adapter import GeminiAdapter


class LLMFactory:
    """Factory and registry for instantiating LLM provider adapters.

    The factory decouples application code from concrete LLM SDK implementations.
    New provider adapters can be dynamically registered via `register_provider`
    without modifying core platform code.
    """

    _registry: dict[str, Type[BaseLLM]] = {
        "gemini": GeminiAdapter,
    }

    @classmethod
    def register_provider(cls, name: str, adapter_cls: Type[BaseLLM]) -> None:
        """Registers a new LLM provider adapter class in the factory registry.

        Args:
            name: Unique case-insensitive provider name (e.g., 'openai').
            adapter_cls: A class inheriting from BaseLLM.

        Raises:
            ValueError: If adapter_cls does not inherit from BaseLLM.
        """
        if not issubclass(adapter_cls, BaseLLM):
            raise ValueError(
                f"Adapter class '{adapter_cls.__name__}' must inherit from BaseLLM."
            )
        cls._registry[name.lower()] = adapter_cls

    @classmethod
    def get_registered_providers(cls) -> list[str]:
        """Returns a sorted list of all currently registered provider names."""
        return sorted(cls._registry.keys())

    @classmethod
    def create(
        cls,
        provider_name: str,
        config: Optional[LLMProviderConfig] = None,
        *,
        client: Optional[Any] = None,
    ) -> BaseLLM:
        """Instantiates an LLM adapter by provider name.

        Args:
            provider_name: Name of the registered provider (e.g., 'gemini').
            config: Optional configuration for the provider. If None, default
                settings are created.
            client: Optional pre-configured client instance for DI or mocking.

        Returns:
            An instantiated adapter implementing BaseLLM.

        Raises:
            LLMUnsupportedProviderError: If provider_name is not registered.
        """
        key = provider_name.lower()
        adapter_cls = cls._registry.get(key)
        if not adapter_cls:
            registered = ", ".join(cls.get_registered_providers())
            raise LLMUnsupportedProviderError(
                f"Unsupported LLM provider '{provider_name}'. "
                f"Registered providers: [{registered}]"
            )

        if config is None:
            config = LLMProviderConfig(provider_name=key, model_name="default")

        return adapter_cls(config, client=client)

    @classmethod
    def create_from_config(
        cls,
        config: LLMConfig,
        provider_name: Optional[str] = None,
        *,
        client: Optional[Any] = None,
    ) -> BaseLLM:
        """Instantiates an LLM adapter using a top-level LLMConfig container.

        Args:
            config: The top-level LLM configuration instance.
            provider_name: Optional provider name to select from config.
                If None, uses `config.default_provider`.
            client: Optional pre-configured client instance for DI or mocking.

        Returns:
            An instantiated adapter implementing BaseLLM.

        Raises:
            LLMConfigurationError: If configuration resolution fails.
            LLMUnsupportedProviderError: If the provider is not registered.
        """
        target_name = (provider_name or config.default_provider).lower()
        try:
            provider_config = config.get_provider_config(target_name)
        except KeyError as exc:
            raise LLMConfigurationError(str(exc)) from exc

        return cls.create(target_name, provider_config, client=client)
