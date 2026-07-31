"""Google Gemini LLM provider adapter for the AI Core Platform.

This module implements the BaseLLM interface for Google's Gemini models
using the official Google GenAI Python SDK (`google-generativeai`). It is
completely domain-agnostic and contains no application-specific prompts or
business logic.
"""

import asyncio
import logging
from typing import Any, AsyncIterator, Iterator, Optional

from ai_core.config.models import GenerationConfig, LLMProviderConfig
from ai_core.llm.base import BaseLLM, LLMResponse, UsageMetadata
from ai_core.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseLLM):
    """Adapter implementation for Google Gemini models.

    This class adapts the Google GenAI SDK to the BaseLLM interface. It supports
    dependency injection by allowing an optional pre-configured client or SDK
    wrapper to be passed into the constructor.

    Attributes:
        config: Provider configuration settings including API key and model name.
    """

    def __init__(
        self,
        config: LLMProviderConfig,
        client: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the GeminiAdapter.

        Args:
            config: Configuration settings for the Gemini provider.
            client: Optional injected SDK module or client instance for testing or DI.
            **kwargs: Additional keyword arguments.

        Raises:
            LLMConfigurationError: If no API key is provided or found in environment.
        """
        super().__init__(config, client=client, **kwargs)
        self._client = client
        self._sdk_configured = False

    @property
    def provider_name(self) -> str:
        """Returns the provider name ('gemini')."""
        return "gemini"

    @property
    def model_name(self) -> str:
        """Returns the configured Gemini model name."""
        return self.config.model_name

    def _get_sdk(self) -> Any:
        """Lazily imports and configures the google.generativeai SDK.

        Returns:
            The configured google.generativeai module or injected client.

        Raises:
            LLMConfigurationError: If the SDK is not installed or API key is missing.
        """
        if self._client is not None:
            return self._client

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise LLMConfigurationError(
                "The 'google-generativeai' package is required for GeminiAdapter. "
                "Please install it via 'pip install google-generativeai'."
            ) from exc

        if not self._sdk_configured:
            api_key = self.config.api_key
            if not api_key:
                import os

                api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise LLMConfigurationError(
                    "No API key provided for GeminiAdapter. Set 'api_key' in config "
                    "or export the GEMINI_API_KEY environment variable."
                )

            genai.configure(api_key=api_key)
            self._sdk_configured = True

        self._client = genai
        return self._client

    def _build_generation_config(
        self, override_config: Optional[GenerationConfig] = None
    ) -> dict[str, Any]:
        """Merges default provider generation config with optional overrides.

        Args:
            override_config: Optional GenerationConfig overriding defaults.

        Returns:
            A dictionary formatted for Gemini generation configuration.
        """
        base = self.config.default_generation_config.to_dict()
        if override_config:
            base.update(override_config.to_dict())

        sdk_config: dict[str, Any] = {}
        if "temperature" in base:
            sdk_config["temperature"] = base["temperature"]
        if "max_output_tokens" in base:
            sdk_config["max_output_tokens"] = base["max_output_tokens"]
        if "top_p" in base:
            sdk_config["top_p"] = base["top_p"]
        if "top_k" in base:
            sdk_config["top_k"] = base["top_k"]
        if "stop_sequences" in base:
            sdk_config["stop_sequences"] = base["stop_sequences"]
        if "response_mime_type" in base:
            sdk_config["response_mime_type"] = base["response_mime_type"]

        return sdk_config

    def _build_model_instance(
        self,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Any:
        """Instantiates a GenerativeModel with optional system instructions and settings.

        Args:
            system_prompt: Optional system instruction for the model.
            generation_config: Optional generation settings.

        Returns:
            A GenerativeModel instance ready for generation.
        """
        sdk = self._get_sdk()
        gen_config_dict = self._build_generation_config(generation_config)

        model_kwargs: dict[str, Any] = {
            "model_name": self.model_name,
        }
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt
        if gen_config_dict:
            # If genai.types.GenerationConfig is available, wrap the dictionary
            if hasattr(sdk, "types") and hasattr(sdk.types, "GenerationConfig"):
                model_kwargs["generation_config"] = sdk.types.GenerationConfig(**gen_config_dict)
            else:
                model_kwargs["generation_config"] = gen_config_dict

        return sdk.GenerativeModel(**model_kwargs)

    def _convert_exception(self, exc: Exception) -> Exception:
        """Converts upstream SDK exceptions into standardized LLM exceptions.

        Args:
            exc: The caught exception from the SDK or network call.

        Returns:
            A standardized LLMException subclass.
        """
        err_msg = str(exc).lower()
        if "429" in err_msg or "rate limit" in err_msg or "quota" in err_msg:
            return LLMRateLimitError(
                f"Gemini rate limit exceeded: {exc}",
                provider=self.provider_name,
            )
        if "401" in err_msg or "403" in err_msg or "permission" in err_msg or "auth" in err_msg:
            return LLMAuthenticationError(
                f"Gemini authentication failed: {exc}",
                provider=self.provider_name,
            )
        if "timeout" in err_msg or "deadline" in err_msg:
            return LLMTimeoutError(
                f"Gemini request timed out: {exc}",
                provider=self.provider_name,
            )
        return LLMProviderError(
            f"Gemini generation error: {exc}",
            provider=self.provider_name,
        )

    def _extract_response_metadata(self, response: Any) -> tuple[str, Optional[str], UsageMetadata]:
        """Extracts text content, finish reason, and usage statistics from SDK response.

        Args:
            response: The raw response object from Gemini SDK.

        Returns:
            A tuple of (content_text, finish_reason, UsageMetadata).
        """
        content_text = ""
        try:
            content_text = getattr(response, "text", "") or ""
        except Exception:
            # Fallback if safety settings blocked text attribute access
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    parts = [getattr(p, "text", "") for p in candidate.content.parts]
                    content_text = "".join(parts)

        finish_reason: Optional[str] = None
        if hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            finish_reason = str(getattr(cand, "finish_reason", None))

        usage = UsageMetadata()
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            usage.prompt_tokens = int(getattr(usage_meta, "prompt_token_count", 0))
            usage.completion_tokens = int(getattr(usage_meta, "candidates_token_count", 0))
            usage.total_tokens = int(getattr(usage_meta, "total_token_count", 0))

        return content_text, finish_reason, usage

    def generate_content(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """Synchronously generates content from Google Gemini.

        Args:
            user_message: Input text prompt.
            system_prompt: Optional system persona or instruction.
            generation_config: Optional generation configuration overrides.

        Returns:
            A populated LLMResponse object.

        Raises:
            LLMProviderError: If generation fails.
        """
        try:
            model = self._build_model_instance(system_prompt, generation_config)
            response = model.generate_content(user_message)
            content_text, finish_reason, usage = self._extract_response_metadata(response)

            return LLMResponse(
                content=content_text,
                model_name=self.model_name,
                provider_name=self.provider_name,
                finish_reason=finish_reason,
                usage=usage,
                raw_response=response,
            )
        except Exception as exc:
            raise self._convert_exception(exc) from exc

    async def agenerate_content(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """Asynchronously generates content from Google Gemini.

        Args:
            user_message: Input text prompt.
            system_prompt: Optional system persona or instruction.
            generation_config: Optional generation configuration overrides.

        Returns:
            A populated LLMResponse object.

        Raises:
            LLMProviderError: If generation fails.
        """
        try:
            model = self._build_model_instance(system_prompt, generation_config)
            response = await model.generate_content_async(user_message)
            content_text, finish_reason, usage = self._extract_response_metadata(response)

            return LLMResponse(
                content=content_text,
                model_name=self.model_name,
                provider_name=self.provider_name,
                finish_reason=finish_reason,
                usage=usage,
                raw_response=response,
            )
        except Exception as exc:
            raise self._convert_exception(exc) from exc

    def generate_stream(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Synchronously streams generated content chunks from Google Gemini.

        Args:
            user_message: Input text prompt.
            system_prompt: Optional system persona or instruction.
            generation_config: Optional generation configuration overrides.

        Yields:
            String chunks of generated text.

        Raises:
            LLMProviderError: If streaming fails.
        """
        try:
            model = self._build_model_instance(system_prompt, generation_config)
            response = model.generate_content(user_message, stream=True)
            for chunk in response:
                text_chunk = getattr(chunk, "text", None)
                if text_chunk:
                    yield text_chunk
        except Exception as exc:
            raise self._convert_exception(exc) from exc

    async def agenerate_stream(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Asynchronously streams generated content chunks from Google Gemini.

        Args:
            user_message: Input text prompt.
            system_prompt: Optional system persona or instruction.
            generation_config: Optional generation configuration overrides.

        Yields:
            String chunks of generated text.

        Raises:
            LLMProviderError: If streaming fails.
        """
        try:
            model = self._build_model_instance(system_prompt, generation_config)
            response = await model.generate_content_async(user_message, stream=True)
            async for chunk in response:
                text_chunk = getattr(chunk, "text", None)
                if text_chunk:
                    yield text_chunk
        except Exception as exc:
            raise self._convert_exception(exc) from exc
