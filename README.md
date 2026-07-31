# AI Core Platform - Reusable Foundation

A domain-agnostic, provider-independent Python foundation designed to power enterprise AI applications.

## Architectural Principles

1. **Zero Domain Coupling**: Contains **no** application-specific logic, prompts, or domain vocabulary.
2. **Provider Independence**: All LLM adapters implement the abstract `BaseLLM` interface, enabling effortless provider substitution.
3. **Strict Layer Separation**:
   - `prompts/`: Manages system prompts, templates, rendering, and loading. Never invokes LLM providers directly.
   - `conversation/`: Manages conversation lifecycle, state, history, and message flow. Never builds prompts and communicates exclusively with `BaseLLM`.
4. **Dependency Injection**: Engines, adapters, and factories accept injected clients and configurations, making unit testing and mocking trivial.

---

## Architecture Overview

```
ai-core/
├── ai_core/
│   ├── config/
│   │   ├── models.py          # GenerationConfig, LLMProviderConfig, LLMConfig
│   │   └── loader.py          # ConfigLoader (dict, JSON, YAML, environment variables)
│   ├── llm/
│   │   ├── base.py            # BaseLLM abstract interface, LLMResponse, UsageMetadata
│   │   ├── exceptions.py      # Structured LLM exception hierarchy
│   │   ├── gemini_adapter.py  # Google Gemini SDK adapter implementing BaseLLM
│   │   └── factory.py         # LLMFactory & provider registry
│   ├── prompts/
│   │   ├── models.py          # PromptTemplate, RenderedPrompt, PromptVariable
│   │   ├── loader.py          # PromptLoader (text, JSON, YAML, directory loading)
│   │   ├── engine.py          # PromptEngine template registry & renderer
│   │   └── exceptions.py      # Structured prompt exception hierarchy
│   └── conversation/
│       ├── models.py          # ConversationContext, Message, MessageRole, ConversationState
│       ├── state.py           # ConversationStateManager lifecycle & history controller
│       ├── engine.py          # ConversationEngine orchestrator (communicates only with BaseLLM)
│       └── exceptions.py      # Structured conversation exception hierarchy
```

---

## Quick Usage Example

### 1. LLM Layer & Factory
```python
from ai_core.config import ConfigLoader, GenerationConfig
from ai_core.llm import LLMFactory

config = ConfigLoader.load_from_env(default_provider="gemini")
llm = LLMFactory.create_from_config(config, provider_name="gemini")
```

### 2. Prompt Engine
```python
from ai_core.prompts import PromptEngine, PromptTemplate

engine = PromptEngine()
engine.register_template(
    PromptTemplate(
        template_id="assistant_system",
        template_text="You are a helpful assistant for {domain}.",
        is_system_prompt=True,
    )
)

rendered_system = engine.render("assistant_system", domain="Enterprise Operations")
```

### 3. Conversation Engine
```python
from ai_core.conversation import ConversationEngine

# Inject BaseLLM into the ConversationEngine
convo_engine = ConversationEngine(llm=llm)

# Process message turns synchronously, asynchronously, or streaming
assistant_msg = convo_engine.process_message(
    "How can you assist me today?",
    system_prompt=rendered_system.text,
    generation_config=GenerationConfig(temperature=0.3)
)

print(f"[{assistant_msg.role.value}] {assistant_msg.content}")
print(convo_engine.get_history())
```
