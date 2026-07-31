"""Tests for the LLMFactory and provider registration."""

import pytest

from ai_core.config.models import LLMConfig, LLMProviderConfig
from ai_core.llm.base import BaseLLM, LLMResponse
from ai_core.llm.exceptions import LLMUnsupportedProviderError
from ai_core.llm.factory import LLMFactory
from ai_core.llm.gemini_adapter import GeminiAdapter


class DummyProviderAdapter(BaseLLM):
    @property
    def provider_name(self) -> str:
        return "dummy"

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def generate_content(self, user_message: str, **kwargs) -> LLMResponse:
        return LLMResponse(
            content=f"Dummy Response to: {user_message}",
            model_name=self.model_name,
            provider_name=self.provider_name,
        )

    async def agenerate_content(self, user_message: str, **kwargs) -> LLMResponse:
        return self.generate_content(user_message, **kwargs)

    def generate_stream(self, user_message: str, **kwargs):
        yield "Dummy Stream"

    async def agenerate_stream(self, user_message: str, **kwargs):
        yield "Dummy Async Stream"


def test_factory_default_registry():
    providers = LLMFactory.get_registered_providers()
    assert "gemini" in providers


def test_factory_create_gemini():
    cfg = LLMProviderConfig(provider_name="gemini", model_name="gemini-2.5-pro")
    adapter = LLMFactory.create("gemini", config=cfg)
    assert isinstance(adapter, GeminiAdapter)
    assert adapter.model_name == "gemini-2.5-pro"


def test_factory_register_and_create_custom_provider():
    LLMFactory.register_provider("dummy", DummyProviderAdapter)
    assert "dummy" in LLMFactory.get_registered_providers()

    adapter = LLMFactory.create("dummy")
    assert isinstance(adapter, DummyProviderAdapter)
    assert adapter.provider_name == "dummy"

    res = adapter.generate_content("Hello World")
    assert res.content == "Dummy Response to: Hello World"


def test_factory_unsupported_provider():
    with pytest.raises(LLMUnsupportedProviderError):
        LLMFactory.create("non-existent-provider-999")


def test_factory_create_from_config():
    config = LLMConfig(
        default_provider="gemini",
        providers={
            "gemini": LLMProviderConfig(
                provider_name="gemini",
                model_name="gemini-2.5-pro",
                api_key="test",
            )
        },
    )
    adapter = LLMFactory.create_from_config(config)
    assert isinstance(adapter, GeminiAdapter)
    assert adapter.model_name == "gemini-2.5-pro"
