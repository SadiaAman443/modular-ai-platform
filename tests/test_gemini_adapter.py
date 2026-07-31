"""Tests for the GeminiAdapter implementation using a mock SDK client."""

import asyncio
import pytest

from ai_core.config.models import GenerationConfig, LLMProviderConfig
from ai_core.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ai_core.llm.gemini_adapter import GeminiAdapter


class MockUsageMetadata:
    def __init__(self, prompt: int = 10, completion: int = 20, total: int = 30):
        self.prompt_token_count = prompt
        self.candidates_token_count = completion
        self.total_token_count = total


class MockCandidate:
    def __init__(self, finish_reason: str = "STOP"):
        self.finish_reason = finish_reason


class MockResponse:
    def __init__(self, text: str = "Mocked LLM Response"):
        self.text = text
        self.candidates = [MockCandidate("STOP")]
        self.usage_metadata = MockUsageMetadata()

    def __iter__(self):
        for word in self.text.split(" "):
            yield MockResponse(word + " ")


class MockAsyncResponse:
    def __init__(self, text: str = "Mocked Async LLM Response"):
        self.text = text
        self.candidates = [MockCandidate("STOP")]
        self.usage_metadata = MockUsageMetadata()

    def __aiter__(self):
        return self._async_generator()

    async def _async_generator(self):
        for word in self.text.split(" "):
            yield MockResponse(word + " ")


class MockGenerativeModel:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def generate_content(self, prompt: str, stream: bool = False):
        if stream:
            return MockResponse("Streaming Mocked LLM Response")
        return MockResponse("Synchronous Mocked LLM Response")

    async def generate_content_async(self, prompt: str, stream: bool = False):
        if stream:
            return MockAsyncResponse("Streaming Mocked Async LLM Response")
        return MockResponse("Asynchronous Mocked LLM Response")


class MockGenAISDK:
    def __init__(self):
        self.GenerativeModel = MockGenerativeModel
        self.configured_api_key = None

    def configure(self, api_key: str):
        self.configured_api_key = api_key


@pytest.fixture
def mock_config():
    return LLMProviderConfig(
        provider_name="gemini",
        model_name="gemini-2.5-pro",
        api_key="test-api-key",
    )


def test_gemini_adapter_properties(mock_config):
    mock_sdk = MockGenAISDK()
    adapter = GeminiAdapter(mock_config, client=mock_sdk)
    assert adapter.provider_name == "gemini"
    assert adapter.model_name == "gemini-2.5-pro"


def test_gemini_adapter_sync_generation(mock_config):
    mock_sdk = MockGenAISDK()
    adapter = GeminiAdapter(mock_config, client=mock_sdk)

    res = adapter.generate_content(
        "Hello",
        system_prompt="Be polite",
        generation_config=GenerationConfig(temperature=0.3),
    )

    assert res.content == "Synchronous Mocked LLM Response"
    assert res.model_name == "gemini-2.5-pro"
    assert res.provider_name == "gemini"
    assert res.finish_reason == "STOP"
    assert res.usage.prompt_tokens == 10
    assert res.usage.completion_tokens == 20
    assert res.usage.total_tokens == 30


def test_gemini_adapter_async_generation(mock_config):
    mock_sdk = MockGenAISDK()
    adapter = GeminiAdapter(mock_config, client=mock_sdk)

    res = asyncio.run(
        adapter.agenerate_content(
            "Hello Async",
            system_prompt="Be concise",
        )
    )

    assert res.content == "Asynchronous Mocked LLM Response"
    assert res.finish_reason == "STOP"
    assert res.usage.total_tokens == 30


def test_gemini_adapter_streaming_generation(mock_config):
    mock_sdk = MockGenAISDK()
    adapter = GeminiAdapter(mock_config, client=mock_sdk)

    chunks = list(adapter.generate_stream("Hello Stream"))
    full_text = "".join(chunks)
    assert "Streaming Mocked LLM Response" in full_text


def test_gemini_adapter_exception_translation(mock_config):
    mock_sdk = MockGenAISDK()
    adapter = GeminiAdapter(mock_config, client=mock_sdk)

    rate_limit_err = adapter._convert_exception(Exception("429 Rate limit exceeded"))
    assert isinstance(rate_limit_err, LLMRateLimitError)

    auth_err = adapter._convert_exception(Exception("401 Unauthorized API key"))
    assert isinstance(auth_err, LLMAuthenticationError)

    timeout_err = adapter._convert_exception(Exception("Request deadline timeout"))
    assert isinstance(timeout_err, LLMTimeoutError)

    generic_err = adapter._convert_exception(Exception("Some unknown upstream error"))
    assert isinstance(generic_err, LLMProviderError)
