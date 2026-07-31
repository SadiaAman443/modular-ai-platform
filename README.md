# AI Core Platform - LLM Foundation

A domain-agnostic, provider-independent Python foundation designed to power enterprise AI applications (e.g., Swargaseema AI, Hospital AI, Banking AI, Restaurant AI).

## Architectural Principles

1. **Zero Domain Coupling**: Contains **no** application-specific logic, prompts, or domain vocabulary.
2. **Provider Independence**: All LLM adapters implement the abstract `BaseLLM` interface, enabling effortless provider substitution.
3. **Dependency Injection**: Adapters and factories accept injected client instances and configuration objects, making unit testing and mocking trivial.
4. **Configurable Generation**: Generation settings and system prompts are supplied externally by the application layer.

---

## Architecture Overview

```
ai-core/
├── ai_core/
│   ├── config/
│   │   ├── models.py       # GenerationConfig, LLMProviderConfig, LLMConfig
│   │   └── loader.py       # ConfigLoader (dict, JSON, YAML, environment variables)
│   └── llm/
│       ├── base.py         # BaseLLM abstract interface, LLMResponse, UsageMetadata
│       ├── exceptions.py   # Structured LLM exception hierarchy
│       ├── gemini_adapter.py # Google Gemini SDK adapter implementing BaseLLM
│       └── factory.py      # LLMFactory & provider registry
```

---

## Quick Usage Example

```python
from ai_core.config import ConfigLoader, GenerationConfig
from ai_core.llm import LLMFactory

# 1. Load configuration from YAML, JSON, dictionary, or environment variables
config = ConfigLoader.load_from_env(default_provider="gemini")

# 2. Instantiate an LLM adapter via the factory
llm = LLMFactory.create_from_config(config, provider_name="gemini")

# 3. Generate text synchronously or asynchronously
response = llm.generate_content(
    user_message="Hello, how can you assist me today?",
    system_prompt="You are a helpful assistant.",
    generation_config=GenerationConfig(temperature=0.2, max_output_tokens=512)
)

print(response.content)
print(response.usage)
```

### Dependency Injection Usage (for Testing / Mocking)

```python
from ai_core.config import LLMProviderConfig
from ai_core.llm import GeminiAdapter

# Pass a custom or mock SDK client into the adapter constructor
mock_client = MyMockGenAIClient()
config = LLMProviderConfig(provider_name="gemini", model_name="gemini-2.5-pro", api_key="test-key")

adapter = GeminiAdapter(config, client=mock_client)
```
