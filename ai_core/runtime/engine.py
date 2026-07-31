"""Conversation Runtime execution engine.

This module implements the provider-agnostic ConversationRuntime that coordinates
events between external transport providers (e.g., Twilio, Bolna, WebRTC) and the
underlying AI Core foundation (ConversationEngine, PromptEngine, BaseLLM).
"""

from typing import Any, Optional

from ai_core.conversation import ConversationEngine, ConversationState
from ai_core.llm import BaseLLM
from ai_core.prompts import PromptEngine
from ai_core.runtime.dispatcher import EventCallback, EventDispatcher
from ai_core.runtime.events import Event, EventType
from ai_core.runtime.exceptions import RuntimeException, RuntimeStateError
from ai_core.runtime.models import RuntimeContext, RuntimeState


class ConversationRuntime:
    """Execution layer bridging external conversation transport and AI Core.

    The ConversationRuntime owns a ConversationEngine, PromptEngine, and BaseLLM
    instance. It processes incoming conversation events, manages runtime state,
    and produces outgoing events via an EventDispatcher.

    Attributes:
        llm: The injected domain-agnostic LLM adapter.
        conversation_engine: Orchestrator for message turns and history.
        prompt_engine: Registry and renderer for prompt templates.
        dispatcher: Pub-sub dispatcher for emitting outgoing events.
        context: Runtime execution context tracking session state and metadata.
    """

    def __init__(
        self,
        llm: BaseLLM,
        conversation_engine: Optional[ConversationEngine] = None,
        prompt_engine: Optional[PromptEngine] = None,
        dispatcher: Optional[EventDispatcher] = None,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Initializes the ConversationRuntime.

        Args:
            llm: Instance implementing BaseLLM.
            conversation_engine: Optional ConversationEngine instance.
            prompt_engine: Optional PromptEngine instance.
            dispatcher: Optional EventDispatcher instance.
            session_id: Optional custom identifier for this runtime session.
            system_prompt: Optional default system prompt string for this session.

        Raises:
            RuntimeException: If llm does not implement BaseLLM.
        """
        if not isinstance(llm, BaseLLM):
            raise RuntimeException(
                f"ConversationRuntime requires an instance of BaseLLM, got {type(llm).__name__}."
            )
        self.llm = llm
        self.conversation_engine = conversation_engine or ConversationEngine(llm=llm)
        self.prompt_engine = prompt_engine or PromptEngine()
        self.dispatcher = dispatcher or EventDispatcher()

        self.context = RuntimeContext(
            session_id=session_id or self.conversation_engine.conversation_id,
            system_prompt=system_prompt,
        )

    @property
    def session_id(self) -> str:
        """Returns the unique runtime session identifier."""
        return self.context.session_id

    @property
    def state(self) -> RuntimeState:
        """Returns the current runtime lifecycle state."""
        return self.context.state

    def subscribe(self, event_type: EventType, callback: EventCallback) -> None:
        """Subscribes a callback to a specific outgoing EventType."""
        self.dispatcher.subscribe(event_type, callback)

    def subscribe_all(self, callback: EventCallback) -> None:
        """Subscribes a callback to all outgoing events."""
        self.dispatcher.subscribe_all(callback)

    def _ensure_running(self) -> None:
        """Enforces that the runtime is not stopped or in an error state.

        Raises:
            RuntimeStateError: If the runtime is STOPPED or ERROR.
        """
        if self.state in (RuntimeState.STOPPED, RuntimeState.ERROR):
            raise RuntimeStateError(
                f"Cannot execute event in terminal runtime state '{self.state.value}'."
            )
        if self.state in (RuntimeState.IDLE, RuntimeState.PAUSED):
            self.context.state = RuntimeState.RUNNING
            self.context.touch()
            if self.conversation_engine.state in (
                ConversationState.INITIALIZED,
                ConversationState.PAUSED,
            ):
                self.conversation_engine.start()

    def process_event(self, event: Event) -> list[Event]:
        """Synchronously processes an incoming Event and returns outgoing Events.

        Args:
            event: The incoming Event to process.

        Returns:
            A list of outgoing Event instances produced by processing or listeners.

        Raises:
            RuntimeStateError: If an illegal state transition is attempted.
        """
        if event.event_type == EventType.SESSION_STARTED:
            if self.state in (RuntimeState.STOPPED, RuntimeState.ERROR):
                raise RuntimeStateError(
                    f"Cannot start session from terminal state '{self.state.value}'."
                )
            self.context.state = RuntimeState.RUNNING
            if "system_prompt" in event.payload and event.payload["system_prompt"]:
                self.context.system_prompt = str(event.payload["system_prompt"])
            self.context.touch()
            if self.conversation_engine.state == ConversationState.INITIALIZED:
                self.conversation_engine.start()

            out_event = Event.create_session_started(
                self.session_id, payload=event.payload, metadata=event.metadata
            )
            dispatched = self.dispatcher.dispatch(out_event)
            return [out_event] + dispatched

        elif event.event_type == EventType.USER_MESSAGE:
            self._ensure_running()
            content = str(
                event.payload.get("content") or event.payload.get("text") or ""
            )

            try:
                assistant_msg = self.conversation_engine.process_message(
                    content,
                    system_prompt=self.context.system_prompt,
                )
                out_event = Event.create_assistant_response(
                    self.session_id,
                    content=assistant_msg.content,
                    usage=assistant_msg.metadata.get("usage"),
                    metadata=assistant_msg.metadata,
                )
                dispatched = self.dispatcher.dispatch(out_event)
                return [out_event] + dispatched
            except Exception as exc:
                self.context.state = RuntimeState.ERROR
                self.context.touch()
                err_event = Event.create_error(
                    self.session_id,
                    error_message=str(exc),
                    details={"exception": type(exc).__name__},
                )
                dispatched = self.dispatcher.dispatch(err_event)
                return [err_event] + dispatched

        elif event.event_type == EventType.INTERRUPTION:
            self._ensure_running()
            self.context.metadata.setdefault("interruptions", []).append(event.payload)
            self.context.touch()
            out_event = Event.create_interruption(
                self.session_id, payload=event.payload, metadata=event.metadata
            )
            dispatched = self.dispatcher.dispatch(out_event)
            return [out_event] + dispatched

        elif event.event_type == EventType.SESSION_ENDED:
            self.conversation_engine.end()
            self.context.state = RuntimeState.STOPPED
            self.context.touch()
            out_event = Event.create_session_ended(
                self.session_id, payload=event.payload, metadata=event.metadata
            )
            dispatched = self.dispatcher.dispatch(out_event)
            return [out_event] + dispatched

        elif event.event_type == EventType.ERROR:
            self.context.state = RuntimeState.ERROR
            self.context.metadata.setdefault("errors", []).append(event.payload)
            self.context.touch()
            out_event = Event.create_error(
                self.session_id,
                error_message=str(event.payload.get("message", "Unknown runtime error")),
                details=event.payload.get("details", {}),
                metadata=event.metadata,
            )
            dispatched = self.dispatcher.dispatch(out_event)
            return [out_event] + dispatched

        else:
            out_event = Event(
                event_type=event.event_type,
                session_id=self.session_id,
                payload=event.payload,
                metadata=event.metadata,
            )
            dispatched = self.dispatcher.dispatch(out_event)
            return [out_event] + dispatched

    async def aprocess_event(self, event: Event) -> list[Event]:
        """Asynchronously processes an incoming Event and returns outgoing Events.

        Args:
            event: The incoming Event to process.

        Returns:
            A list of outgoing Event instances produced by processing or listeners.

        Raises:
            RuntimeStateError: If an illegal state transition is attempted.
        """
        if event.event_type == EventType.SESSION_STARTED:
            if self.state in (RuntimeState.STOPPED, RuntimeState.ERROR):
                raise RuntimeStateError(
                    f"Cannot start session from terminal state '{self.state.value}'."
                )
            self.context.state = RuntimeState.RUNNING
            if "system_prompt" in event.payload and event.payload["system_prompt"]:
                self.context.system_prompt = str(event.payload["system_prompt"])
            self.context.touch()
            if self.conversation_engine.state == ConversationState.INITIALIZED:
                self.conversation_engine.start()

            out_event = Event.create_session_started(
                self.session_id, payload=event.payload, metadata=event.metadata
            )
            dispatched = await self.dispatcher.adispatch(out_event)
            return [out_event] + dispatched

        elif event.event_type == EventType.USER_MESSAGE:
            self._ensure_running()
            content = str(
                event.payload.get("content") or event.payload.get("text") or ""
            )

            try:
                assistant_msg = await self.conversation_engine.aprocess_message(
                    content,
                    system_prompt=self.context.system_prompt,
                )
                out_event = Event.create_assistant_response(
                    self.session_id,
                    content=assistant_msg.content,
                    usage=assistant_msg.metadata.get("usage"),
                    metadata=assistant_msg.metadata,
                )
                dispatched = await self.dispatcher.adispatch(out_event)
                return [out_event] + dispatched
            except Exception as exc:
                self.context.state = RuntimeState.ERROR
                self.context.touch()
                err_event = Event.create_error(
                    self.session_id,
                    error_message=str(exc),
                    details={"exception": type(exc).__name__},
                )
                dispatched = await self.dispatcher.adispatch(err_event)
                return [err_event] + dispatched

        elif event.event_type == EventType.INTERRUPTION:
            self._ensure_running()
            self.context.metadata.setdefault("interruptions", []).append(event.payload)
            self.context.touch()
            out_event = Event.create_interruption(
                self.session_id, payload=event.payload, metadata=event.metadata
            )
            dispatched = await self.dispatcher.adispatch(out_event)
            return [out_event] + dispatched

        elif event.event_type == EventType.SESSION_ENDED:
            self.conversation_engine.end()
            self.context.state = RuntimeState.STOPPED
            self.context.touch()
            out_event = Event.create_session_ended(
                self.session_id, payload=event.payload, metadata=event.metadata
            )
            dispatched = await self.dispatcher.adispatch(out_event)
            return [out_event] + dispatched

        elif event.event_type == EventType.ERROR:
            self.context.state = RuntimeState.ERROR
            self.context.metadata.setdefault("errors", []).append(event.payload)
            self.context.touch()
            out_event = Event.create_error(
                self.session_id,
                error_message=str(event.payload.get("message", "Unknown runtime error")),
                details=event.payload.get("details", {}),
                metadata=event.metadata,
            )
            dispatched = await self.dispatcher.adispatch(out_event)
            return [out_event] + dispatched

        else:
            out_event = Event(
                event_type=event.event_type,
                session_id=self.session_id,
                payload=event.payload,
                metadata=event.metadata,
            )
            dispatched = await self.dispatcher.adispatch(out_event)
            return [out_event] + dispatched

    def start_session(self, system_prompt: Optional[str] = None) -> list[Event]:
        """Convenience method to dispatch a SESSION_STARTED event."""
        payload = {"system_prompt": system_prompt} if system_prompt else {}
        return self.process_event(
            Event.create_session_started(self.session_id, payload=payload)
        )

    def stop_session(self) -> list[Event]:
        """Convenience method to dispatch a SESSION_ENDED event."""
        return self.process_event(Event.create_session_ended(self.session_id))

    def send_user_message(
        self, content: str, metadata: Optional[dict[str, Any]] = None
    ) -> list[Event]:
        """Convenience method to dispatch a USER_MESSAGE event."""
        return self.process_event(
            Event.create_user_message(self.session_id, content, metadata=metadata)
        )
