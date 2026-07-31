# MODULAR AI PLATFORM — ARCHITECTURE SPECIFICATION

**Document Version**: 1.0.0  
**Status**: Approved Architecture Contract  
**Scope**: Authoritative Specification for `ai_core` and `projects/*`  

---

## 1. Vision

### 1.1 What is the Modular AI Platform?
The **Modular AI Platform** is an enterprise-grade, foundation-first architectural framework designed to power multi-domain, multi-modal conversational AI assistants from a single, unified codebase. It separates conversational AI infrastructure—such as session state machines, prompt templating, event-driven execution, and language model networking—from domain-specific business rules and transport protocols.

### 1.2 What Problems Does It Solve?
1. **Monolithic Duplication**: Eliminates redundant, copy-pasted implementations of LLM networking, audio framing, and conversation state tracking across different AI applications (e.g., educational support, real estate receptionists, hospital triage, banking assistants).
2. **Vendor Lock-In**: Decouples AI applications from proprietary AI provider APIs (e.g., Google Gemini, OpenAI, Anthropic, open-source local models) and telephony/streaming platforms (e.g., Twilio, Bolna, WebRTC, SIP, HTTP).
3. **Fragile Prompt Coupling**: Prevents hardcoded business vocabulary and ad-hoc string formatting from entangling with core execution engines.
4. **Unsafe Evolutions**: Establishes strict architectural boundaries and directed dependency laws, ensuring that optimizations or security enhancements in the platform core instantly benefit all downstream projects without risking regressions in domain logic.

---

## 2. Core Principles

### 2.1 Separation of Concerns
Every module within the platform is responsible for exactly one distinct aspect of the conversational AI lifecycle:
- Model networking and token counting are isolated to the **LLM layer**.
- Prompt template discovery, variable validation, and string rendering are isolated to the **Prompt layer**.
- Session state transitions and message history tracking are isolated to the **Conversation layer**.
- Event-driven coordination and transport bridging are isolated to the **Runtime layer**.

### 2.2 Provider Independence
The `ai_core` framework must never import or depend on proprietary vendor SDKs, web frameworks, or telephony libraries (including Twilio, Bolna, FastAPI, WebSockets, or Flask). All external providers communicate with the core through abstract interfaces and standardized domain-agnostic data models.

### 2.3 Dependency Injection
Modules do not instantiate concrete dependencies internally. Instead, components receive their collaborators—such as language model adapters, state managers, and prompt loaders—via constructor injection. This guarantees testability, modularity, and runtime adaptability.

### 2.4 Plugin Architecture
Domain applications (`projects/*`) and specialized capabilities (`tools`, `memory`, `knowledge`) act as autonomous plugins around the core kernel. They attach to the platform through declarative contracts and pub-sub event subscriptions without modifying `ai_core`.

### 2.5 Domain Agnostic Design
The `ai_core` namespace must remain completely free of business-specific terminology. Vocabulary relating to students, colleges, sandalwood farms, hospitals, campaigns, or attendance is strictly prohibited inside `ai_core`.

---

## 3. Folder Responsibilities

### 3.1 `llm`
- **Purpose**: Provides a provider-agnostic abstraction layer for large language models.
- **Responsibilities**: Defines the abstract `BaseLLM` interface, concrete provider adapters (`GeminiAdapter`), factory creation logic (`LLMFactory`), and standardized usage/response models (`LLMResponse`, `UsageMetadata`).

### 3.2 `prompts`
- **Purpose**: Manages domain-agnostic prompt template discovery, validation, and rendering.
- **Responsibilities**: Implements template file loaders (`PromptLoader`), variable schema validators (`PromptVariable`), and dynamic rendering engines (`PromptEngine`). Never interacts with LLMs or session history directly.

### 3.3 `conversation`
- **Purpose**: Tracks conversation session lifecycle and chronological message history.
- **Responsibilities**: Implements state machines (`ConversationStateManager`), lifecycle states (`INITIALIZED`, `ACTIVE`, `PAUSED`, `ENDED`, `ERROR`), message data structures (`Message`, `MessageRole`), and turn orchestration (`ConversationEngine`).

### 3.4 `runtime`
- **Purpose**: Orchestrates event-driven execution between external adapters and the conversation engine.
- **Responsibilities**: Implements `ConversationRuntime`, pub-sub event dispatching (`EventDispatcher`), and lifecycle event models (`SESSION_STARTED`, `USER_MESSAGE`, `ASSISTANT_RESPONSE`, `INTERRUPTION`, `SESSION_ENDED`, `ERROR`).

### 3.5 `memory`
- **Purpose**: Long-term and working memory persistence layer (planned).
- **Responsibilities**: Summarizing multi-turn session context, storing user preferences across sessions, and querying episodic or relational memory stores without coupling to specific database vendors.

### 3.6 `knowledge`
- **Purpose**: Retrieval-Augmented Generation (RAG) and semantic grounding layer (planned).
- **Responsibilities**: Document ingestion, chunking, embedding generation, and contextual similarity search to ground assistant responses in verified facts.

### 3.7 `tools`
- **Purpose**: Function-calling and external tool execution registry (planned).
- **Responsibilities**: Declarative tool schema discovery, parameter validation, authorization checks, and deterministic tool execution on behalf of language models.

### 3.8 `voice`
- **Purpose**: Provider-agnostic speech and audio processing interface (planned).
- **Responsibilities**: Defining speech-to-text (STT), text-to-speech (TTS), audio resampling, and voice activity detection (VAD) contracts for telephony and real-time streaming adapters.

### 3.9 `workflow`
- **Purpose**: Multi-step task orchestration and state transition engine (planned).
- **Responsibilities**: Managing deterministic, multi-turn conversational workflows, branch routing, retry policies, and human-in-the-loop approvals.

### 3.10 `sessions`
- **Purpose**: Persistent session storage and concurrent session governance (planned).
- **Responsibilities**: Distributed session locking, inactivity timeouts, state serialization, and multi-tenant isolation across horizontal backend deployments.

### 3.11 `plugins`
- **Purpose**: Dynamic module discovery and extension registration mechanism (planned).
- **Responsibilities**: Registering community or enterprise add-ons, third-party tool packs, and custom storage backends at runtime.

### 3.12 `config`
- **Purpose**: Unified, type-safe configuration management.
- **Responsibilities**: Loading and validating configuration models (`LLMProviderConfig`, `GenerationConfig`) from environment variables, dictionaries, and JSON/YAML configuration files.

### 3.13 `utils`
- **Purpose**: Common platform-wide helper libraries.
- **Responsibilities**: Structured logging formatters, telemetry and tracing helpers, retry/exponential backoff wrappers, and domain-agnostic exception hierarchies.

---

## 4. Dependency Rules

### 4.1 Directed Acyclic Hierarchy
Module communication within `ai_core` must strictly adhere to a one-way directed acyclic hierarchy. Circular dependencies or upward imports are forbidden.

```
                  +-----------------------------------+
                  |             projects/*            |
                  +-----------------+-----------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                                 ai_core                                 |
|                                                                         |
|       +---------------------------------------------------------+       |
|       |                         runtime                         |       |
|       +-------+--------------------+--------------------+-------+       |
|               |                    |                    |               |
|               v                    v                    v               |
|      +-----------------+   +---------------+   +-----------------+      |
|      |  conversation   |   |    prompts    |   |  tools / voice  |      |
|      +--------+--------+   +-------+-------+   +--------+--------+      |
|               |                    |                    |               |
|               +--------------------+--------------------+               |
|                                    |                                    |
|                                    v                                    |
|       +---------------------------------------------------------+       |
|       |                           llm                           |       |
|       +----------------------------+----------------------------+       |
|                                    |                                    |
|                                    v                                    |
|       +---------------------------------------------------------+       |
|       |                     config / utils                      |       |
|       +---------------------------------------------------------+       |
+-------------------------------------------------------------------------+
```

### 4.2 Module Communication Contracts
- `runtime`
  - **Allowed to import**: `conversation`, `prompts`, `llm`, `tools`, `voice`, `memory`, `knowledge`, `sessions`, `config`, `utils`.
  - **Forbidden to import**: `projects/*`.
- `conversation`
  - **Allowed to import**: `llm`, `memory`, `config`, `utils`.
  - **Forbidden to import**: `runtime`, `prompts`, `tools`, `voice`, `projects/*`.
- `prompts`
  - **Allowed to import**: `config`, `utils`.
  - **Forbidden to import**: `llm`, `conversation`, `runtime`, `tools`, `voice`, `projects/*`.
- `llm`
  - **Allowed to import**: `config`, `utils`.
  - **Forbidden to import**: `prompts`, `conversation`, `runtime`, `tools`, `voice`, `projects/*`.
- `tools`
  - **Allowed to import**: `llm`, `config`, `utils`.
  - **Forbidden to import**: `conversation`, `runtime`, `prompts`, `projects/*`.
- `voice`
  - **Allowed to import**: `runtime`, `config`, `utils`.
  - **Forbidden to import**: `llm`, `prompts`, `conversation`, `projects/*`.

### 4.3 The Critical Boundary Law
**No module within `ai_core` may ever import, reference, or inspect any file, class, or module from `projects/*` or client applications.** Dependency flow is strictly unidirectional: projects consume `ai_core`.

---

## 5. Project Boundary

### 5.1 Inside `ai_core` (Platform Kernel)
The platform kernel contains only universal, domain-independent capabilities:
- Abstract provider interfaces (`BaseLLM`, abstract memory stores, abstract tool definitions).
- Execution engines (`ConversationRuntime`, `ConversationEngine`, `PromptEngine`).
- Lifecycle state machines and pub-sub event dispatchers.
- Common data transfer structures (`Message`, `Event`, `LLMResponse`, `PromptTemplate`).
- Configuration loaders and general utilities.

### 5.2 Inside Client Applications (`projects/<domain>_ai/`)
Client applications represent specific business use cases and contain all domain vocabulary and transport framing:
- Domain-specific system prompt templates and persona instructions.
- Business entities, lead schemas, database queries, and CRM service integrations.
- Telephony and streaming adapter bridges (e.g., Twilio WebSocket framing, Bolna audio hooks, WebRTC endpoints).
- Project-specific configuration settings, environment variables, and REST/WebSocket API endpoints.

---

## 6. Extension Guidelines

Developers extending the platform must follow these integration patterns without modifying existing `ai_core` modules:

### 6.1 Registering a New LLM Provider
1. Implement the `BaseLLM` interface in a self-contained adapter class.
2. Provide implementations for synchronous, asynchronous, and streaming generation methods.
3. Register the provider at application startup via the factory:
   `LLMFactory.register_provider("custom_provider", CustomLLMAdapter)`

### 6.2 Creating a New Domain AI Assistant
1. Create a dedicated package under `projects/<project_name>_ai/`.
2. Place plaintext system prompt templates under `projects/<project_name>_ai/prompts/templates/`.
3. Implement a domain `Assistant` class that instantiates `ai_core.runtime.ConversationRuntime` with custom prompt rendering.
4. Implement a provider-agnostic `Bridge` class that exposes session configuration and message routing methods without importing vendor SDKs.

### 6.3 Defining and Loading Custom Prompts
1. Write plaintext prompt files using `{variable}` placeholders.
2. Load templates via `PromptLoader.load_from_text_file(...)` and define `PromptVariable` schemas with fallback defaults.
3. Register the loaded template in `PromptEngine` and render it by passing key-value keyword arguments.

### 6.4 Subscribing to Conversation Events
1. Create a listener callback function accepting an `Event` object.
2. Subscribe to specific runtime lifecycle transitions:
   `runtime.subscribe(EventType.ASSISTANT_RESPONSE, my_analytics_listener)`
3. Use `subscribe_all(callback)` for cross-cutting concerns such as audit logging or WebSocket broadcasting.

---

## 7. Future Modules

The following planned modules define the future roadmap of `ai_core`. They represent architectural contracts and design intentions without current implementation:

### 7.1 `memory`
- **Architectural Scope**: Provides short-term working memory and long-term episodic memory persistence.
- **Design Intent**: Will define abstract store contracts allowing session history summarization and preference retrieval across sessions without locking into specific SQL or NoSQL databases.

### 7.2 `knowledge`
- **Architectural Scope**: Provides Retrieval-Augmented Generation (RAG) capabilities.
- **Design Intent**: Will define standardized document ingestion, chunking, embedding generation, and similarity search interfaces to dynamically ground LLM prompts in verified knowledge bases.

### 7.3 `tools`
- **Architectural Scope**: Provides deterministic tool and function execution.
- **Design Intent**: Will define declarative JSON Schema tool definitions, parameter validators, and sandboxed invocation handlers so assistants can interact with external APIs safely.

### 7.4 `voice`
- **Architectural Scope**: Provides real-time audio streaming and speech processing abstractions.
- **Design Intent**: Will standardize audio chunk buffering, sample rate resampling, speech-to-text (STT) transcription hooks, and text-to-speech (TTS) synthesis contracts for telephony adapters.

### 7.5 `workflow`
- **Architectural Scope**: Provides deterministic state-machine workflow orchestration.
- **Design Intent**: Will enable multi-stage conversations (e.g., identity verification -> qualification -> booking) with conditional branching, guardrail enforcement, and mandatory human-in-the-loop checkpoints.

### 7.6 `sessions`
- **Architectural Scope**: Provides distributed session governance for horizontal scaling.
- **Design Intent**: Will define abstract backend contracts for concurrent session locking, inactivity timeouts, and state rehydration across distributed server clusters.

### 7.7 `plugins`
- **Architectural Scope**: Provides dynamic runtime extensibility.
- **Design Intent**: Will define standard lifecycle hooks and discovery manifests allowing community or enterprise extensions to plug into `ai_core` without modifying core repositories.
