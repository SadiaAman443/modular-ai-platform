"""Fluent builder pattern for assembling AIAssistant instances.

This module provides `AssistantBuilder`, enabling step-by-step fluent construction
and configuration of an `AIAssistant` with optional injected components.
"""

from typing import Any, Optional

from ai_core.assistant.assistant import AIAssistant
from ai_core.assistant.models import AssistantConfig


class AssistantBuilder:
    """Fluent builder for creating configured AIAssistant instances.

    Example:
        >>> assistant = (
        ...     AssistantBuilder()
        ...     .with_llm(my_llm)
        ...     .with_memory(my_memory)
        ...     .with_tools(my_tools)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initializes a new AssistantBuilder with default empty components."""
        self._llm: Optional[Any] = None
        self._memory: Optional[Any] = None
        self._knowledge: Optional[Any] = None
        self._workflow: Optional[Any] = None
        self._voice: Optional[Any] = None
        self._tools: Optional[Any] = None
        self._runtime: Optional[Any] = None
        self._conversation: Optional[Any] = None
        self._prompt_engine: Optional[Any] = None
        self._config: AssistantConfig = AssistantConfig()

    def with_llm(self, llm: Any) -> "AssistantBuilder":
        """Injects a language model provider.

        Args:
            llm: An instance implementing BaseLLM.

        Returns:
            The current builder instance.
        """
        self._llm = llm
        return self

    def with_memory(self, memory: Any) -> "AssistantBuilder":
        """Injects a conversational memory manager.

        Args:
            memory: A MemoryManager or compatible memory interface.

        Returns:
            The current builder instance.
        """
        self._memory = memory
        return self

    def with_knowledge(self, knowledge: Any) -> "AssistantBuilder":
        """Injects a knowledge base / RAG component.

        Args:
            knowledge: A knowledge retrieval pipeline.

        Returns:
            The current builder instance.
        """
        self._knowledge = knowledge
        return self

    def with_workflow(self, workflow: Any) -> "AssistantBuilder":
        """Injects a multi-step workflow orchestration engine.

        Args:
            workflow: A workflow manager instance.

        Returns:
            The current builder instance.
        """
        self._workflow = workflow
        return self

    def with_voice(self, voice: Any) -> "AssistantBuilder":
        """Injects a voice/speech streaming provider interface.

        Args:
            voice: A voice adapter instance.

        Returns:
            The current builder instance.
        """
        self._voice = voice
        return self

    def with_tools(self, tools: Any) -> "AssistantBuilder":
        """Injects a tool registry.

        Args:
            tools: A tool registry instance.

        Returns:
            The current builder instance.
        """
        self._tools = tools
        return self

    def with_runtime(self, runtime: Any) -> "AssistantBuilder":
        """Injects a conversation runtime orchestrator.

        Args:
            runtime: A ConversationRuntime instance.

        Returns:
            The current builder instance.
        """
        self._runtime = runtime
        return self

    def with_conversation(self, conversation: Any) -> "AssistantBuilder":
        """Injects a conversation engine.

        Args:
            conversation: A ConversationEngine instance.

        Returns:
            The current builder instance.
        """
        self._conversation = conversation
        return self

    def with_prompt_engine(self, prompt_engine: Any) -> "AssistantBuilder":
        """Injects a prompt template engine.

        Args:
            prompt_engine: A PromptEngine instance.

        Returns:
            The current builder instance.
        """
        self._prompt_engine = prompt_engine
        return self

    def with_config(self, config: AssistantConfig) -> "AssistantBuilder":
        """Sets the assistant configuration object.

        Args:
            config: An AssistantConfig instance.

        Returns:
            The current builder instance.
        """
        self._config = config
        return self

    def with_system_prompt(self, system_prompt: str) -> "AssistantBuilder":
        """Sets the default system prompt instruction text.

        Args:
            system_prompt: System prompt string.

        Returns:
            The current builder instance.
        """
        self._config.system_prompt = system_prompt
        return self

    def with_session_id(self, session_id: str) -> "AssistantBuilder":
        """Sets the default conversation session ID.

        Args:
            session_id: Session identifier string.

        Returns:
            The current builder instance.
        """
        self._config.session_id = session_id
        return self

    def with_name(self, name: str) -> "AssistantBuilder":
        """Sets the human-readable assistant instance name.

        Args:
            name: Assistant name string.

        Returns:
            The current builder instance.
        """
        self._config.name = name
        return self

    def build(self) -> AIAssistant:
        """Assembles and validates the AIAssistant instance.

        Returns:
            An initialized AIAssistant configured with the injected components.

        Raises:
            AssistantConfigurationError: If required execution dependencies are missing.
        """
        return AIAssistant(
            llm=self._llm,
            memory=self._memory,
            knowledge=self._knowledge,
            workflow=self._workflow,
            voice=self._voice,
            tools=self._tools,
            runtime=self._runtime,
            conversation=self._conversation,
            prompt_engine=self._prompt_engine,
            config=self._config,
        )
