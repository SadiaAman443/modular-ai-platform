"""Primary orchestration entry point for AI Core.

This module provides `AIAssistant`, which coordinates language models, conversation
engines, memory managers, prompt engines, and runtime dispatchers via Dependency Injection
without implementing any business logic or vendor SDKs.
"""

from typing import Any, Optional

from ai_core.assistant.exceptions import (
    AssistantConfigurationError,
    AssistantExecutionError,
)
from ai_core.assistant.models import AssistantConfig, AssistantTurnResult
from ai_core.runtime.engine import ConversationRuntime
from ai_core.runtime.events import Event, EventType


class AIAssistant:
    """Orchestrates AI Core modules via Dependency Injection.

    `AIAssistant` serves as the primary entry point for conversational applications,
    coordinating lifecycle events, memory retrieval, prompt rendering, and language
    model turns without embedding business logic or vendor-specific code.

    Attributes:
        llm: Optional BaseLLM provider instance.
        memory: Optional memory manager instance.
        knowledge: Optional RAG/knowledge retrieval component.
        workflow: Optional workflow orchestration engine.
        voice: Optional voice/audio streaming interface.
        tools: Optional tool execution registry.
        runtime: Optional ConversationRuntime instance.
        conversation: Optional ConversationEngine instance.
        prompt_engine: Optional PromptEngine instance.
        config: Assistant configuration metadata.
    """

    def __init__(
        self,
        *,
        llm: Optional[Any] = None,
        memory: Optional[Any] = None,
        knowledge: Optional[Any] = None,
        workflow: Optional[Any] = None,
        voice: Optional[Any] = None,
        tools: Optional[Any] = None,
        runtime: Optional[Any] = None,
        conversation: Optional[Any] = None,
        prompt_engine: Optional[Any] = None,
        config: Optional[AssistantConfig] = None,
    ) -> None:
        """Initializes AIAssistant with optional injected components.

        Args:
            llm: Injected language model provider.
            memory: Injected conversational memory manager.
            knowledge: Injected knowledge base / RAG pipeline.
            workflow: Injected multi-step workflow engine.
            voice: Injected voice/speech provider interface.
            tools: Injected tool registry.
            runtime: Injected conversation runtime orchestrator.
            conversation: Injected conversation engine.
            prompt_engine: Injected prompt template engine.
            config: Optional assistant configuration metadata.

        Raises:
            AssistantConfigurationError: If no execution engine (`llm`, `conversation`,
                or `runtime`) is provided.
        """
        if not llm and not conversation and not runtime:
            raise AssistantConfigurationError(
                "AIAssistant requires at least an 'llm', 'conversation', or 'runtime' "
                "component to execute turns."
            )

        self.config = config or AssistantConfig()
        self.memory = memory
        self.knowledge = knowledge
        self.workflow = workflow
        self.voice = voice
        self.tools = tools

        if runtime:
            self.runtime = runtime
            self.llm = llm or getattr(runtime, "llm", None)
            self.conversation = conversation or getattr(
                runtime, "conversation_engine", None
            )
            self.prompt_engine = prompt_engine or getattr(
                runtime, "prompt_engine", None
            )
        else:
            self.llm = llm
            self.conversation = conversation
            self.prompt_engine = prompt_engine
            self.runtime = ConversationRuntime(
                llm=llm,
                conversation_engine=conversation,
                prompt_engine=prompt_engine,
                session_id=self.config.session_id,
                system_prompt=self.config.system_prompt,
            )

    def start_session(self, *, session_id: Optional[str] = None) -> list[Any]:
        """Starts a new conversation session on the underlying runtime.

        Args:
            session_id: Optional session identifier override.

        Returns:
            List of runtime events emitted during session startup.
        """
        if session_id:
            self.config.session_id = session_id
            if hasattr(self.runtime, "session_id"):
                self.runtime.session_id = session_id
        return self.runtime.start_session(system_prompt=self.config.system_prompt)

    def end_session(self) -> list[Any]:
        """Terminates the active conversation session.

        Returns:
            List of runtime events emitted during session shutdown.
        """
        return self.runtime.stop_session()

    def process_turn(
        self,
        user_message: str,
        *,
        entity_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AssistantTurnResult:
        """Orchestrates a synchronous conversational turn.

        Coordinates memory retrieval, runtime execution, and memory storage.

        Args:
            user_message: Input text from the user.
            entity_id: Optional user or entity identifier.
            metadata: Optional turn metadata dictionary.

        Returns:
            An AssistantTurnResult containing response text, events, and memory records.

        Raises:
            AssistantExecutionError: If turn coordination fails.
        """
        memory_records: list[Any] = []
        session_id = self.config.session_id or getattr(
            self.runtime, "session_id", None
        )

        try:
            # Step 1: Retrieve conversational memory context if available
            if self.memory and hasattr(self.memory, "retrieve_context"):
                mem_ctx = self.memory.retrieve_context(
                    query=user_message,
                    session_id=session_id,
                    entity_id=entity_id,
                )
                if hasattr(mem_ctx, "records"):
                    memory_records.extend(mem_ctx.records)

            # Step 2: Execute turn on runtime via event dispatch
            event = Event.create_user_message(
                session_id,
                content=user_message,
                metadata=metadata,
            )
            events = self.runtime.process_event(event)

            # Step 3: Extract main assistant response from events
            assistant_response = ""
            for ev in events:
                if (
                    getattr(ev, "event_type", None) == EventType.ASSISTANT_RESPONSE
                    and hasattr(ev, "payload")
                ):
                    assistant_response = str(
                        ev.payload.get("text")
                        or ev.payload.get("content")
                        or ""
                    )
                    break

            # Step 4: Store user turn into memory if available
            if (
                self.memory
                and hasattr(self.memory, "add_memory")
                and user_message.strip()
            ):
                try:
                    mem_type = (
                        getattr(self.memory, "default_type", None) or "working"
                    )
                    rec = self.memory.add_memory(
                        content=user_message.strip(),
                        memory_type=mem_type,
                        session_id=session_id,
                        entity_id=entity_id,
                    )
                    memory_records.append(rec)
                except Exception:
                    pass

            return AssistantTurnResult(
                user_message=user_message,
                assistant_response=assistant_response,
                session_id=session_id,
                memory_records=memory_records,
                events=events,
                metadata=metadata or {},
            )
        except Exception as exc:
            raise AssistantExecutionError(
                f"Failed to process assistant turn: {exc}"
            ) from exc

    async def aprocess_turn(
        self,
        user_message: str,
        *,
        entity_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AssistantTurnResult:
        """Orchestrates an asynchronous conversational turn.

        Args:
            user_message: Input text from the user.
            entity_id: Optional user or entity identifier.
            metadata: Optional turn metadata dictionary.

        Returns:
            An AssistantTurnResult containing response text, events, and memory records.

        Raises:
            AssistantExecutionError: If turn coordination fails.
        """
        memory_records: list[Any] = []
        session_id = self.config.session_id or getattr(
            self.runtime, "session_id", None
        )

        try:
            # Step 1: Retrieve conversational memory context if available
            if self.memory and hasattr(self.memory, "aretrieve_context"):
                mem_ctx = await self.memory.aretrieve_context(
                    query=user_message,
                    session_id=session_id,
                    entity_id=entity_id,
                )
                if hasattr(mem_ctx, "records"):
                    memory_records.extend(mem_ctx.records)
            elif self.memory and hasattr(self.memory, "retrieve_context"):
                mem_ctx = self.memory.retrieve_context(
                    query=user_message,
                    session_id=session_id,
                    entity_id=entity_id,
                )
                if hasattr(mem_ctx, "records"):
                    memory_records.extend(mem_ctx.records)

            # Step 2: Execute turn on runtime asynchronously via event dispatch
            event = Event.create_user_message(
                session_id,
                content=user_message,
                metadata=metadata,
            )
            events = await self.runtime.aprocess_event(event)

            # Step 3: Extract main assistant response from events
            assistant_response = ""
            for ev in events:
                if (
                    getattr(ev, "event_type", None) == EventType.ASSISTANT_RESPONSE
                    and hasattr(ev, "payload")
                ):
                    assistant_response = str(
                        ev.payload.get("text")
                        or ev.payload.get("content")
                        or ""
                    )
                    break

            # Step 4: Store user turn into memory asynchronously if available
            if (
                self.memory
                and hasattr(self.memory, "aadd_memory")
                and user_message.strip()
            ):
                try:
                    mem_type = (
                        getattr(self.memory, "default_type", None) or "working"
                    )
                    rec = await self.memory.aadd_memory(
                        content=user_message.strip(),
                        memory_type=mem_type,
                        session_id=session_id,
                        entity_id=entity_id,
                    )
                    memory_records.append(rec)
                except Exception:
                    pass

            return AssistantTurnResult(
                user_message=user_message,
                assistant_response=assistant_response,
                session_id=session_id,
                memory_records=memory_records,
                events=events,
                metadata=metadata or {},
            )
        except Exception as exc:
            raise AssistantExecutionError(
                f"Failed to asynchronously process assistant turn: {exc}"
            ) from exc

    def get_history(self) -> list[Any]:
        """Returns the chronological message history from the underlying runtime.

        Returns:
            List of Message instances in the active conversation session.
        """
        if self.runtime and hasattr(self.runtime, "conversation_engine"):
            ce = self.runtime.conversation_engine
            if ce and hasattr(ce, "get_history"):
                return ce.get_history()
        return []

    def subscribe(self, event_type: Any, callback: Any) -> None:
        """Registers an event listener for a specific runtime EventType.

        Args:
            event_type: The EventType enum value to observe.
            callback: Function to invoke when the event occurs.
        """
        if self.runtime and hasattr(self.runtime, "subscribe"):
            self.runtime.subscribe(event_type, callback)

    def subscribe_all(self, callback: Any) -> None:
        """Registers a global listener for all runtime events.

        Args:
            callback: Function to invoke for every emitted event.
        """
        if self.runtime and hasattr(self.runtime, "subscribe_all"):
            self.runtime.subscribe_all(callback)
