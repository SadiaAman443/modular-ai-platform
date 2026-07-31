"""Unit tests for the AI Core Memory Framework.

Tests the domain-agnostic memory contracts, data models, exception handling,
and MemoryManager orchestration using an in-memory test provider.
"""

import ast
from pathlib import Path
from typing import Any, Optional
import pytest

from ai_core.memory import (
    BaseMemoryProvider,
    MemoryContext,
    MemoryException,
    MemoryManager,
    MemoryNotFoundError,
    MemoryProviderError,
    MemoryRecord,
    MemoryType,
    MemoryValidationError,
)


class InMemoryTestProvider(BaseMemoryProvider):
    """In-memory implementation of BaseMemoryProvider for unit testing."""

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
            if query and query.lower() not in r.content.lower():
                continue
            if metadata_filter:
                match = all(r.metadata.get(k) == v for k, v in metadata_filter.items())
                if not match:
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
        keys_to_delete: list[str] = []
        for rid, r in self.storage.items():
            if session_id and r.session_id != session_id:
                continue
            if entity_id and r.entity_id != entity_id:
                continue
            if memory_type and r.memory_type != memory_type:
                continue
            keys_to_delete.append(rid)

        for rid in keys_to_delete:
            del self.storage[rid]
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


class ErrorTestProvider(InMemoryTestProvider):
    """Test provider that raises runtime errors to test MemoryProviderError wrapping."""

    def store_record(self, record: MemoryRecord) -> MemoryRecord:
        raise RuntimeError("Simulated storage error")

    async def astore_record(self, record: MemoryRecord) -> MemoryRecord:
        raise RuntimeError("Simulated async storage error")

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        raise RuntimeError("Simulated read error")

    async def aget_record(self, record_id: str) -> Optional[MemoryRecord]:
        raise RuntimeError("Simulated async read error")

    def delete_record(self, record_id: str) -> bool:
        raise RuntimeError("Simulated delete error")

    async def adelete_record(self, record_id: str) -> bool:
        raise RuntimeError("Simulated async delete error")

    def search_records(self, *args: Any, **kwargs: Any) -> list[MemoryRecord]:
        raise RuntimeError("Simulated search error")

    async def asearch_records(self, *args: Any, **kwargs: Any) -> list[MemoryRecord]:
        raise RuntimeError("Simulated async search error")

    def clear(self, *args: Any, **kwargs: Any) -> int:
        raise RuntimeError("Simulated clear error")

    async def aclear(self, *args: Any, **kwargs: Any) -> int:
        raise RuntimeError("Simulated async clear error")


def test_memory_record_serialization():
    """Tests MemoryRecord dictionary serialization and deserialization."""
    record = MemoryRecord(
        record_id="rec-1",
        memory_type=MemoryType.EPISODIC,
        content="User prefers email summaries over calls.",
        session_id="session-101",
        entity_id="user-99",
        metadata={"priority": "high"},
    )
    data = record.to_dict()
    assert data["record_id"] == "rec-1"
    assert data["memory_type"] == "episodic"
    assert data["content"] == "User prefers email summaries over calls."

    deserialized = MemoryRecord.from_dict(data)
    assert deserialized.record_id == record.record_id
    assert deserialized.memory_type == record.memory_type
    assert deserialized.content == record.content
    assert deserialized.session_id == record.session_id
    assert deserialized.entity_id == record.entity_id
    assert deserialized.metadata == record.metadata


def test_memory_context_helpers():
    """Tests MemoryContext filtering and string formatting."""
    record1 = MemoryRecord(
        record_id="r1",
        memory_type=MemoryType.WORKING,
        content="Active query about farm plot availability.",
    )
    record2 = MemoryRecord(
        record_id="r2",
        memory_type=MemoryType.SEMANTIC,
        content="Customer language preference is Telugu.",
    )
    ctx = MemoryContext(session_id="s1", records=[record1, record2])

    working_records = ctx.get_records_by_type(MemoryType.WORKING)
    assert len(working_records) == 1
    assert working_records[0].record_id == "r1"

    text_block = ctx.to_string(separator=" | ")
    assert "Active query about farm plot availability." in text_block
    assert "Customer language preference is Telugu." in text_block
    assert " | " in text_block


def test_memory_manager_sync_lifecycle():
    """Tests synchronous CRUD, search, and clear operations via MemoryManager."""
    provider = InMemoryTestProvider()
    manager = MemoryManager(provider=provider)

    # Add memory
    record = manager.add_memory(
        content="User interested in Sandalwood Plot A12.",
        memory_type=MemoryType.EPISODIC,
        session_id="sess-1",
        entity_id="cust-1",
        metadata={"source": "call"},
    )
    assert record.record_id in provider.storage

    # Retrieve memory by id
    fetched = manager.get_memory(record.record_id)
    assert fetched is not None
    assert fetched.content == "User interested in Sandalwood Plot A12."

    # Search via retrieve_context
    ctx = manager.retrieve_context("Plot A12", session_id="sess-1")
    assert len(ctx.records) == 1
    assert ctx.records[0].record_id == record.record_id

    # Clear memory
    deleted_count = manager.clear_memory(session_id="sess-1")
    assert deleted_count == 1
    assert manager.get_memory(record.record_id) is None


@pytest.mark.anyio
async def test_memory_manager_async_lifecycle():
    """Tests asynchronous CRUD, search, and clear operations via MemoryManager."""
    provider = InMemoryTestProvider()
    manager = MemoryManager(provider=provider)

    # Async add memory
    record = await manager.aadd_memory(
        content="Preferred meeting hour is 10 AM.",
        memory_type=MemoryType.SEMANTIC,
        entity_id="cust-2",
    )
    assert record.record_id in provider.storage

    # Async retrieve by id
    fetched = await manager.aget_memory(record.record_id)
    assert fetched is not None
    assert fetched.content == "Preferred meeting hour is 10 AM."

    # Async retrieve context
    ctx = await manager.aretrieve_context("10 AM", entity_id="cust-2")
    assert len(ctx.records) == 1
    assert ctx.records[0].record_id == record.record_id

    # Async delete
    deleted = await manager.adelete_memory(record.record_id)
    assert deleted is True
    assert await manager.aget_memory(record.record_id) is None


def test_memory_manager_invalid_provider_and_content():
    """Tests constructor validation and content validation."""
    with pytest.raises(MemoryException):
        MemoryManager(provider="not-a-provider")  # type: ignore[arg-type]

    provider = InMemoryTestProvider()
    manager = MemoryManager(provider=provider)

    with pytest.raises(MemoryValidationError):
        manager.add_memory(content="", memory_type=MemoryType.WORKING)


def test_memory_manager_error_wrapping():
    """Tests that provider errors are properly wrapped in MemoryProviderError."""
    provider = ErrorTestProvider()
    manager = MemoryManager(provider=provider)

    with pytest.raises(MemoryProviderError):
        manager.add_memory("Some text", MemoryType.WORKING)

    with pytest.raises(MemoryProviderError):
        manager.get_memory("some-id")

    with pytest.raises(MemoryProviderError):
        manager.delete_memory("some-id")

    with pytest.raises(MemoryProviderError):
        manager.retrieve_context("query")

    with pytest.raises(MemoryProviderError):
        manager.clear_memory()


def test_memory_package_has_no_forbidden_imports():
    """Verifies that ai_core/memory imports no database, vector store, Redis, Chroma, or projects."""
    memory_dir = Path(__file__).parent.parent / "ai_core" / "memory"
    forbidden_terms = {
        "sqlite3",
        "psycopg2",
        "sqlalchemy",
        "redis",
        "chroma",
        "chromadb",
        "pinecone",
        "qdrant",
        "weaviate",
        "projects",
    }

    for py_file in memory_dir.glob("*.py"):
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
