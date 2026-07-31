"""Unit tests for the AI Core Conversation foundation module."""

import asyncio
from typing import AsyncIterator, Iterator, Optional

import pytest

from ai_core.config.models import GenerationConfig, LLMProviderConfig
from ai_core.conversation.engine import ConversationEngine
from ai_core.conversation.exceptions import (
    ConversationLifecycleError,
    ConversationStateError,
)
from ai_core.conversation.models import (
    ConversationContext,
    ConversationState,
    Message,
    MessageRole,
)
from ai_core.conversation.state import ConversationStateManager
from ai_core.llm.base import BaseLLM, LLMResponse, UsageMetadata


class MockConversationLLM(BaseLLM):
    """Mock LLM implementation for testing ConversationEngine."""

    def __init__(self) -> None:
        cfg = LLMProviderConfig(provider_name="mock", model_name="mock-model")
        super().__init__(cfg)
        self.last_prompt: Optional[str] = None
        self.last_system_prompt: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return "mock"

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
        self.last_prompt = user_message
        self.last_system_prompt = system_prompt
        return LLMResponse(
            content=f"Echo: {user_message}",
            model_name=self.model_name,
            provider_name=self.provider_name,
            finish_reason="STOP",
            usage=UsageMetadata(prompt_tokens=5, completion_tokens=5, total_tokens=10),
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
        words = [f"Stream: ", user_message, " !"]
        for w in words:
            yield w

    async def agenerate_stream(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        words = [f"AsyncStream: ", user_message, " !"]
        for w in words:
            yield w


def test_conversation_state_manager_lifecycle():
    mgr = ConversationStateManager()
    assert mgr.state == ConversationState.INITIALIZED

    mgr.start()
    assert mgr.state == ConversationState.ACTIVE

    mgr.pause()
    assert mgr.state == ConversationState.PAUSED

    mgr.resume()
    assert mgr.state == ConversationState.ACTIVE

    mgr.end()
    assert mgr.state == ConversationState.ENDED

    with pytest.raises(ConversationStateError):
        mgr.start()


def test_conversation_state_manager_messages():
    mgr = ConversationStateManager()
    mgr.start()

    user_msg = mgr.add_user_message("Hello there")
    assert user_msg.role == MessageRole.USER
    assert user_msg.content == "Hello there"

    assistant_msg = mgr.add_assistant_message("Hello back")
    assert assistant_msg.role == MessageRole.ASSISTANT

    history = mgr.get_history()
    assert len(history) == 2
    assert mgr.get_last_message() == assistant_msg
    assert len(mgr.get_messages_by_role(MessageRole.USER)) == 1


def test_conversation_lifecycle_error_on_ended():
    mgr = ConversationStateManager()
    mgr.end()
    with pytest.raises(ConversationLifecycleError):
        mgr.add_user_message("Too late")


def test_conversation_context_serialization():
    ctx = ConversationContext()
    ctx.state = ConversationState.ACTIVE
    ctx.history.append(Message(role=MessageRole.USER, content="Test msg"))

    data = ctx.to_dict()
    assert data["state"] == "active"
    assert len(data["history"]) == 1

    ctx2 = ConversationContext.from_dict(data)
    assert ctx2.conversation_id == ctx.conversation_id
    assert ctx2.state == ConversationState.ACTIVE
    assert ctx2.history[0].content == "Test msg"


def test_conversation_engine_sync_process():
    llm = MockConversationLLM()
    engine = ConversationEngine(llm=llm)

    msg = engine.process_message("How are you?", system_prompt="Be polite")
    assert msg.role == MessageRole.ASSISTANT
    assert msg.content == "Echo: How are you?"
    assert llm.last_system_prompt == "Be polite"

    history = engine.get_history()
    assert len(history) == 2
    assert history[0].role == MessageRole.USER
    assert history[1].role == MessageRole.ASSISTANT


def test_conversation_engine_async_process():
    llm = MockConversationLLM()
    engine = ConversationEngine(llm=llm)

    msg = asyncio.run(engine.aprocess_message("Async Hi"))
    assert msg.content == "Echo: Async Hi"
    assert len(engine.get_history()) == 2


def test_conversation_engine_sync_stream():
    llm = MockConversationLLM()
    engine = ConversationEngine(llm=llm)

    chunks = list(engine.stream_message("Stream Test"))
    assert "".join(chunks) == "Stream: Stream Test !"

    history = engine.get_history()
    assert len(history) == 2
    assert history[-1].content == "Stream: Stream Test !"


def test_conversation_engine_async_stream():
    llm = MockConversationLLM()
    engine = ConversationEngine(llm=llm)

    async def consume():
        chunks = []
        async for c in engine.astream_message("Async Stream Test"):
            chunks.append(c)
        return "".join(chunks)

    result = asyncio.run(consume())
    assert result == "AsyncStream: Async Stream Test !"
    assert len(engine.get_history()) == 2
