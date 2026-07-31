"""Unit tests for the AI Core Assistant Framework.

Tests the AIAssistant orchestrator, AssistantBuilder fluent construction,
turn coordination with memory and runtime, and exception handling.
"""

import ast
from pathlib import Path
from typing import Any, Optional
import pytest

from ai_core.assistant import (
    AIAssistant,
    AssistantBuilder,
    AssistantConfig,
    AssistantConfigurationError,
    AssistantException,
    AssistantTurnResult,
)
from ai_core.config import GenerationConfig, LLMProviderConfig
from ai_core.llm import BaseLLM, LLMResponse, UsageMetadata
from ai_core.memory import (
    BaseMemoryProvider,
    MemoryContext,
    MemoryManager,
    MemoryRecord,
    MemoryType,
)
from ai_core.runtime.events import EventType


class MockAssistantLLM(BaseLLM):
    """Mock BaseLLM for testing AIAssistant coordination."""

    def __init__(self) -> None:
        cfg = LLMProviderConfig(provider_name="mock-assistant", model_name="mock-model")
        super().__init__(cfg)

    @property
    def provider_name(self) -> str:
        return "mock-assistant"

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
        return LLMResponse(
            content=f"Assistant reply to: {user_message}",
            model_name=self.model_name,
            provider_name=self.provider_name,
            usage=UsageMetadata(prompt_tokens=10, completion_tokens=5, total_tokens=15),
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
    ) -> Any:
        yield "Assistant stream reply"

    async def agenerate_stream(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Any:
        yield "Assistant stream reply"


class InMemoryAssistantMemoryProvider(BaseMemoryProvider):
    """Simple in-memory provider for testing assistant-memory integration."""

    def __init__(self) -> None:
        self.storage: dict[str, MemoryRecord] = {}

    def store_record(self, record: MemoryRecord) -> MemoryRecord:
        self.storage[record.record_id] = record
        return record

    async def astore_record(self, record: MemoryRecord) -> MemoryRecord:
        return self.store_record(record)

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        return self.storage.get(record_id)

    async def aget_record(self, record_id: str) -> Optional[MemoryRecord]:
        return self.get_record(record_id)

    def delete_record(self, record_id: str) -> bool:
        if record_id in self.storage:
            del self.storage[record_id]
            return True
        return False

    async def adelete_record(self, record_id: str) -> bool:
        return self.delete_record(record_id)

    def search_records(
        self,
        query: str,
        *,
        memory_type: Optional[MemoryType] = None,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        limit: int = 10,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[MemoryRecord]:
        results: list[MemoryRecord] = []
        for r in self.storage.values():
            if memory_type and r.memory_type != memory_type:
                continue
            if session_id and r.session_id != session_id:
                continue
            if entity_id and r.entity_id != entity_id:
                continue
            if query and not any(
                w in r.content.lower() for w in query.lower().split()
            ):
                continue
            results.append(r)
            if len(results) >= limit:
                break
        return results

    async def asearch_records(
        self,
        query: str,
        *,
        memory_type: Optional[MemoryType] = None,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        limit: int = 10,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[MemoryRecord]:
        return self.search_records(
            query=query,
            memory_type=memory_type,
            session_id=session_id,
            entity_id=entity_id,
            limit=limit,
            metadata_filter=metadata_filter,
        )

    def clear(
        self,
        *,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        keys_to_delete = list(self.storage.keys())
        for k in keys_to_delete:
            del self.storage[k]
        return len(keys_to_delete)

    async def aclear(
        self,
        *,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        return self.clear(
            session_id=session_id,
            entity_id=entity_id,
            memory_type=memory_type,
        )


def test_assistant_models_serialization():
    """Tests AssistantConfig and AssistantTurnResult serialization and deserialization."""
    config = AssistantConfig(
        name="TestBot",
        system_prompt="You are helpful.",
        session_id="sess-001",
        metadata={"version": "1.0"},
    )
    c_dict = config.to_dict()
    assert c_dict["name"] == "TestBot"
    assert c_dict["system_prompt"] == "You are helpful."

    deserialized_config = AssistantConfig.from_dict(c_dict)
    assert deserialized_config.name == "TestBot"
    assert deserialized_config.system_prompt == "You are helpful."

    result = AssistantTurnResult(
        user_message="hello",
        assistant_response="hi there",
        session_id="sess-001",
        metadata={"tokens": 12},
    )
    r_dict = result.to_dict()
    assert r_dict["user_message"] == "hello"
    assert r_dict["assistant_response"] == "hi there"
    assert r_dict["metadata"]["tokens"] == 12

    deserialized_result = AssistantTurnResult.from_dict(r_dict)
    assert deserialized_result.user_message == "hello"
    assert deserialized_result.assistant_response == "hi there"


def test_assistant_builder_validation():
    """Tests that building an assistant without execution engines raises AssistantConfigurationError."""
    with pytest.raises(AssistantConfigurationError):
        AssistantBuilder().with_name("EmptyBot").build()


def test_assistant_builder_fluent_construction():
    """Tests fluent builder chaining with LLM and MemoryManager."""
    llm = MockAssistantLLM()
    memory = MemoryManager(provider=InMemoryAssistantMemoryProvider())
    assistant = (
        AssistantBuilder()
        .with_name("ConciergeBot")
        .with_system_prompt("System prompt rules.")
        .with_session_id("session-888")
        .with_llm(llm)
        .with_memory(memory)
        .with_knowledge("mock-knowledge")
        .with_workflow("mock-workflow")
        .with_voice("mock-voice")
        .with_tools("mock-tools")
        .build()
    )
    assert isinstance(assistant, AIAssistant)
    assert assistant.config.name == "ConciergeBot"
    assert assistant.config.session_id == "session-888"
    assert assistant.llm is llm
    assert assistant.memory is memory
    assert assistant.knowledge == "mock-knowledge"
    assert assistant.workflow == "mock-workflow"
    assert assistant.voice == "mock-voice"
    assert assistant.tools == "mock-tools"
    assert assistant.runtime is not None


def test_assistant_synchronous_turn_orchestration():
    """Tests starting a session, executing a turn, storing memories, and subscribing to events."""
    llm = MockAssistantLLM()
    memory_provider = InMemoryAssistantMemoryProvider()
    memory = MemoryManager(provider=memory_provider)

    # Pre-seed episodic memory
    memory.add_memory(
        content="Customer has an active sandalwood query.",
        memory_type=MemoryType.EPISODIC,
        session_id="sess-convo-1",
        entity_id="cust-10",
    )

    assistant = (
        AssistantBuilder()
        .with_llm(llm)
        .with_memory(memory)
        .with_session_id("sess-convo-1")
        .build()
    )

    events_received: list[Any] = []

    def on_event(ev: Any) -> None:
        events_received.append(ev)

    assistant.subscribe_all(on_event)

    start_events = assistant.start_session()
    assert len(start_events) > 0
    assert len(events_received) >= len(start_events)

    turn_res = assistant.process_turn(
        "sandalwood plot options",
        entity_id="cust-10",
        metadata={"channel": "phone"},
    )

    assert turn_res.user_message == "sandalwood plot options"
    assert "Assistant reply to: sandalwood plot options" in turn_res.assistant_response
    assert turn_res.session_id == "sess-convo-1"
    # Check that pre-seeded episodic memory was retrieved in memory_records
    assert any(
        "active sandalwood query" in getattr(r, "content", "")
        for r in turn_res.memory_records
    )

    # Check that new user message was stored in working memory
    working_recs = memory_provider.search_records(
        "sandalwood plot options", session_id="sess-convo-1"
    )
    assert len(working_recs) >= 1

    history = assistant.get_history()
    assert len(history) == 2  # user input + assistant reply

    stop_events = assistant.end_session()
    assert len(stop_events) > 0


@pytest.mark.anyio
async def test_assistant_asynchronous_turn_orchestration():
    """Tests async session execution and memory coordination via AIAssistant."""
    llm = MockAssistantLLM()
    memory = MemoryManager(provider=InMemoryAssistantMemoryProvider())

    assistant = (
        AssistantBuilder()
        .with_llm(llm)
        .with_memory(memory)
        .with_session_id("async-sess-2")
        .build()
    )

    turn_res = await assistant.aprocess_turn("hello async world")
    assert turn_res.user_message == "hello async world"
    assert "Assistant reply to: hello async world" in turn_res.assistant_response

    history = assistant.get_history()
    assert len(history) == 2


def test_assistant_package_has_no_forbidden_imports():
    """Verifies that ai_core/assistant imports no database, vendor SDKs, or projects."""
    assistant_dir = Path(__file__).parent.parent / "ai_core" / "assistant"
    forbidden_terms = {
        "sqlite3",
        "psycopg2",
        "sqlalchemy",
        "redis",
        "chroma",
        "twilio",
        "bolna",
        "fastapi",
        "flask",
        "projects",
    }

    for py_file in assistant_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.lower()
                    assert not any(f in name for f in forbidden_terms), (
                        f"Forbidden import '{alias.name}' in {py_file}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    name = node.module.lower()
                    assert not any(f in name for f in forbidden_terms), (
                        f"Forbidden from-import '{node.module}' in {py_file}"
                    )
