"""Unit tests for the AI Core Conversation Runtime package."""

import asyncio
from typing import AsyncIterator, Iterator, Optional

import pytest

from ai_core.config.models import GenerationConfig, LLMProviderConfig
from ai_core.conversation import ConversationState
from ai_core.llm.base import BaseLLM, LLMResponse, UsageMetadata
from ai_core.runtime import (
    ConversationRuntime,
    Event,
    EventDispatcher,
    EventType,
    RuntimeContext,
    RuntimeException,
    RuntimeState,
    RuntimeStateError,
)


class MockRuntimeLLM(BaseLLM):
    """Mock BaseLLM implementation for runtime unit testing."""

    def __init__(self, should_fail: bool = False) -> None:
        cfg = LLMProviderConfig(provider_name="mock-runtime", model_name="mock-model")
        super().__init__(cfg)
        self.should_fail = should_fail
        self.last_prompt: Optional[str] = None
        self.last_system_prompt: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return "mock-runtime"

    @property
    def model_name(self) -> str:
        return "mock-model"

    def generate_content(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        if self.should_fail:
            raise RuntimeError("Simulated LLM failure")
        self.last_prompt = user_message
        self.last_system_prompt = system_prompt
        return LLMResponse(
            content=f"Runtime response to: {user_message}",
            model_name=self.model_name,
            provider_name=self.provider_name,
            finish_reason="STOP",
            usage=UsageMetadata(prompt_tokens=10, completion_tokens=15, total_tokens=25),
        )

    async def agenerate_content(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        return self.generate_content(
            user_message,
            system_prompt=system_prompt,
            generation_config=generation_config,
        )

    def generate_stream(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        yield f"Runtime stream: {user_message}"

    async def agenerate_stream(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        yield f"Runtime async stream: {user_message}"


def test_event_factories_and_serialization():
    evt = Event.create_user_message("sess-123", "Hello AI", metadata={"source": "twilio"})
    assert evt.event_type == EventType.USER_MESSAGE
    assert evt.session_id == "sess-123"
    assert evt.payload["content"] == "Hello AI"
    assert evt.metadata["source"] == "twilio"

    data = evt.to_dict()
    assert data["event_type"] == "user_message"

    evt2 = Event.from_dict(data)
    assert evt2.session_id == "sess-123"
    assert evt2.payload["content"] == "Hello AI"


def test_runtime_context_lifecycle():
    ctx = RuntimeContext(session_id="ctx-1")
    assert ctx.state == RuntimeState.IDLE
    assert ctx.system_prompt is None

    data = ctx.to_dict()
    assert data["state"] == "idle"

    ctx2 = RuntimeContext.from_dict(data)
    assert ctx2.session_id == "ctx-1"
    assert ctx2.state == RuntimeState.IDLE


def test_event_dispatcher_sync():
    dispatcher = EventDispatcher()
    received: list[Event] = []

    def on_user_message(event: Event):
        received.append(event)
        # return a secondary event
        return Event.create_assistant_response(event.session_id, "Ack")

    dispatcher.subscribe(EventType.USER_MESSAGE, on_user_message)
    user_event = Event.create_user_message("s-1", "Hi")

    secondary = dispatcher.dispatch(user_event)
    assert len(received) == 1
    assert len(secondary) == 1
    assert secondary[0].event_type == EventType.ASSISTANT_RESPONSE
    assert secondary[0].payload["content"] == "Ack"


def test_event_dispatcher_async():
    dispatcher = EventDispatcher()
    received: list[Event] = []

    async def on_error(event: Event):
        received.append(event)

    dispatcher.subscribe(EventType.ERROR, on_error)
    err_event = Event.create_error("s-2", "Something broke")

    asyncio.run(dispatcher.adispatch(err_event))
    assert len(received) == 1


def test_event_dispatcher_sync_async_callback_error():
    dispatcher = EventDispatcher()

    async def async_cb(event: Event):
        pass

    dispatcher.subscribe(EventType.USER_MESSAGE, async_cb)
    with pytest.raises(RuntimeError):
        dispatcher.dispatch(Event.create_user_message("s-1", "test"))


def test_conversation_runtime_lifecycle_and_messages():
    llm = MockRuntimeLLM()
    runtime = ConversationRuntime(llm=llm, session_id="rt-100")
    assert runtime.state == RuntimeState.IDLE

    outgoing: list[Event] = []
    runtime.subscribe_all(lambda evt: outgoing.append(evt))

    # Start session
    start_events = runtime.start_session(system_prompt="Be concise")
    assert runtime.state == RuntimeState.RUNNING
    assert runtime.context.system_prompt == "Be concise"
    assert runtime.conversation_engine.state == ConversationState.ACTIVE
    assert any(e.event_type == EventType.SESSION_STARTED for e in start_events)

    # Process user message
    msg_events = runtime.send_user_message("What is the attendance?")
    assert any(e.event_type == EventType.ASSISTANT_RESPONSE for e in msg_events)
    assert llm.last_system_prompt == "Be concise"

    resp_event = [e for e in msg_events if e.event_type == EventType.ASSISTANT_RESPONSE][0]
    assert resp_event.payload["content"] == "Runtime response to: What is the attendance?"
    assert resp_event.payload["usage"]["total_tokens"] == 25

    # Interruption event
    int_events = runtime.process_event(Event.create_interruption("rt-100", {"reason": "user_spoke"}))
    assert any(e.event_type == EventType.INTERRUPTION for e in int_events)
    assert len(runtime.context.metadata["interruptions"]) == 1

    # End session
    end_events = runtime.stop_session()
    assert runtime.state == RuntimeState.STOPPED
    assert runtime.conversation_engine.state == ConversationState.ENDED
    assert any(e.event_type == EventType.SESSION_ENDED for e in end_events)

    # Attempting to send a message when stopped should raise RuntimeStateError
    with pytest.raises(RuntimeStateError):
        runtime.send_user_message("Too late")


def test_conversation_runtime_error_handling():
    llm = MockRuntimeLLM(should_fail=True)
    runtime = ConversationRuntime(llm=llm, session_id="rt-err")
    runtime.start_session()

    events = runtime.send_user_message("Will fail")
    assert runtime.state == RuntimeState.ERROR
    assert any(e.event_type == EventType.ERROR for e in events)

    err_event = [e for e in events if e.event_type == EventType.ERROR][0]
    assert "Simulated LLM failure" in err_event.payload["message"]


def test_conversation_runtime_async_execution():
    llm = MockRuntimeLLM()
    runtime = ConversationRuntime(llm=llm, session_id="rt-async")

    async def run_async():
        await runtime.aprocess_event(Event.create_session_started("rt-async"))
        events = await runtime.aprocess_event(
            Event.create_user_message("rt-async", "Hello async")
        )
        return events

    results = asyncio.run(run_async())
    assert runtime.state == RuntimeState.RUNNING
    assert any(e.event_type == EventType.ASSISTANT_RESPONSE for e in results)
