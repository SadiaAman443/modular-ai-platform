"""Conversational memory manager and orchestrator.

This module provides `MemoryManager`, which coordinates the lifecycle of working,
episodic, and semantic memories by delegating storage and retrieval to an injected
`BaseMemoryProvider` without coupling to specific databases.
"""

from typing import Any, Optional
import uuid

from ai_core.memory.base import BaseMemoryProvider
from ai_core.memory.exceptions import (
    MemoryException,
    MemoryNotFoundError,
    MemoryProviderError,
    MemoryValidationError,
)
from ai_core.memory.models import MemoryContext, MemoryRecord, MemoryType


class MemoryManager:
    """Orchestrates conversational memory persistence, search, and context assembly.

    `MemoryManager` communicates exclusively through an injected `BaseMemoryProvider`.
    It provides synchronous and asynchronous helper methods to store records,
    retrieve records by ID, assemble context containers for prompt injection,
    and clear stored memories.

    Attributes:
        provider: The injected domain-agnostic memory storage provider.
    """

    def __init__(self, provider: BaseMemoryProvider) -> None:
        """Initializes the MemoryManager.

        Args:
            provider: An instance implementing BaseMemoryProvider.

        Raises:
            MemoryException: If `provider` does not implement BaseMemoryProvider.
        """
        if not isinstance(provider, BaseMemoryProvider):
            raise MemoryException(
                f"MemoryManager requires an instance of BaseMemoryProvider, "
                f"got {type(provider).__name__}."
            )
        self.provider = provider

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        *,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        record_id: Optional[str] = None,
    ) -> MemoryRecord:
        """Stores a new conversational memory entry.

        Args:
            content: Text summary, observation, or content of the memory.
            memory_type: Category of the memory (`WORKING`, `EPISODIC`, etc.).
            session_id: Optional associated conversation session ID.
            entity_id: Optional associated entity or user ID.
            metadata: Optional dictionary of metadata.
            record_id: Optional custom identifier; auto-generated if omitted.

        Returns:
            The stored MemoryRecord instance.

        Raises:
            MemoryValidationError: If content is empty.
            MemoryProviderError: If storage fails in the underlying provider.
        """
        if not content or not content.strip():
            raise MemoryValidationError("Memory content cannot be empty.")

        rec_id = record_id or uuid.uuid4().hex
        record = MemoryRecord(
            record_id=rec_id,
            memory_type=memory_type,
            content=content.strip(),
            session_id=session_id,
            entity_id=entity_id,
            metadata=metadata or {},
        )
        try:
            return self.provider.store_record(record)
        except Exception as exc:
            raise MemoryProviderError(
                f"Failed to store memory record '{rec_id}': {exc}"
            ) from exc

    async def aadd_memory(
        self,
        content: str,
        memory_type: MemoryType,
        *,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        record_id: Optional[str] = None,
    ) -> MemoryRecord:
        """Asynchronously stores a new conversational memory entry.

        Args:
            content: Text summary, observation, or content of the memory.
            memory_type: Category of the memory (`WORKING`, `EPISODIC`, etc.).
            session_id: Optional associated conversation session ID.
            entity_id: Optional associated entity or user ID.
            metadata: Optional dictionary of metadata.
            record_id: Optional custom identifier; auto-generated if omitted.

        Returns:
            The stored MemoryRecord instance.

        Raises:
            MemoryValidationError: If content is empty.
            MemoryProviderError: If storage fails in the underlying provider.
        """
        if not content or not content.strip():
            raise MemoryValidationError("Memory content cannot be empty.")

        rec_id = record_id or uuid.uuid4().hex
        record = MemoryRecord(
            record_id=rec_id,
            memory_type=memory_type,
            content=content.strip(),
            session_id=session_id,
            entity_id=entity_id,
            metadata=metadata or {},
        )
        try:
            return await self.provider.astore_record(record)
        except Exception as exc:
            raise MemoryProviderError(
                f"Failed to asynchronously store memory record '{rec_id}': {exc}"
            ) from exc

    def get_memory(self, record_id: str) -> Optional[MemoryRecord]:
        """Retrieves a memory record by its unique identifier.

        Args:
            record_id: The identifier of the memory record.

        Returns:
            The MemoryRecord if found, or None.

        Raises:
            MemoryProviderError: If retrieval fails in the underlying provider.
        """
        try:
            return self.provider.get_record(record_id)
        except Exception as exc:
            raise MemoryProviderError(
                f"Failed to retrieve memory record '{record_id}': {exc}"
            ) from exc

    async def aget_memory(self, record_id: str) -> Optional[MemoryRecord]:
        """Asynchronously retrieves a memory record by its unique identifier.

        Args:
            record_id: The identifier of the memory record.

        Returns:
            The MemoryRecord if found, or None.

        Raises:
            MemoryProviderError: If retrieval fails in the underlying provider.
        """
        try:
            return await self.provider.aget_record(record_id)
        except Exception as exc:
            raise MemoryProviderError(
                f"Failed to asynchronously retrieve memory record '{record_id}': {exc}"
            ) from exc

    def delete_memory(self, record_id: str) -> bool:
        """Deletes a memory record by its unique identifier.

        Args:
            record_id: The identifier of the memory record to remove.

        Returns:
            True if deleted, False if not found.

        Raises:
            MemoryProviderError: If deletion fails in the underlying provider.
        """
        try:
            return self.provider.delete_record(record_id)
        except Exception as exc:
            raise MemoryProviderError(
                f"Failed to delete memory record '{record_id}': {exc}"
            ) from exc

    async def adelete_memory(self, record_id: str) -> bool:
        """Asynchronously deletes a memory record by its unique identifier.

        Args:
            record_id: The identifier of the memory record to remove.

        Returns:
            True if deleted, False if not found.

        Raises:
            MemoryProviderError: If deletion fails in the underlying provider.
        """
        try:
            return await self.provider.adelete_record(record_id)
        except Exception as exc:
            raise MemoryProviderError(
                f"Failed to asynchronously delete memory record '{record_id}': {exc}"
            ) from exc

    def retrieve_context(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> MemoryContext:
        """Searches memory records and packages results in a MemoryContext container.

        Args:
            query: Search query string or filter keywords.
            session_id: Optional filter by conversation session.
            entity_id: Optional filter by entity or user ID.
            memory_type: Optional filter by memory category.
            limit: Maximum number of records to retrieve.
            metadata_filter: Optional required metadata key-value filters.

        Returns:
            A MemoryContext containing the retrieved MemoryRecord items.

        Raises:
            MemoryProviderError: If search fails in the underlying provider.
        """
        try:
            records = self.provider.search_records(
                query=query,
                memory_type=memory_type,
                session_id=session_id,
                entity_id=entity_id,
                limit=limit,
                metadata_filter=metadata_filter,
            )
            return MemoryContext(
                session_id=session_id,
                entity_id=entity_id,
                records=records,
            )
        except Exception as exc:
            raise MemoryProviderError(
                f"Failed to retrieve memory context for query '{query}': {exc}"
            ) from exc

    async def aretrieve_context(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> MemoryContext:
        """Asynchronously searches memory records and returns a MemoryContext.

        Args:
            query: Search query string or filter keywords.
            session_id: Optional filter by conversation session.
            entity_id: Optional filter by entity or user ID.
            memory_type: Optional filter by memory category.
            limit: Maximum number of records to retrieve.
            metadata_filter: Optional required metadata key-value filters.

        Returns:
            A MemoryContext containing the retrieved MemoryRecord items.

        Raises:
            MemoryProviderError: If search fails in the underlying provider.
        """
        try:
            records = await self.provider.asearch_records(
                query=query,
                memory_type=memory_type,
                session_id=session_id,
                entity_id=entity_id,
                limit=limit,
                metadata_filter=metadata_filter,
            )
            return MemoryContext(
                session_id=session_id,
                entity_id=entity_id,
                records=records,
            )
        except Exception as exc:
            raise MemoryProviderError(
                f"Failed to asynchronously retrieve memory context for query '{query}': {exc}"
            ) from exc

    def clear_memory(
        self,
        *,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """Clears memory records matching the optional filters.

        Args:
            session_id: Optional filter to clear only records for a session.
            entity_id: Optional filter to clear only records for an entity.
            memory_type: Optional filter to clear only records of a specific type.

        Returns:
            The number of records deleted.

        Raises:
            MemoryProviderError: If clearing fails in the underlying provider.
        """
        try:
            return self.provider.clear(
                session_id=session_id,
                entity_id=entity_id,
                memory_type=memory_type,
            )
        except Exception as exc:
            raise MemoryProviderError(f"Failed to clear memory: {exc}") from exc

    async def aclear_memory(
        self,
        *,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """Asynchronously clears memory records matching the optional filters.

        Args:
            session_id: Optional filter to clear only records for a session.
            entity_id: Optional filter to clear only records for an entity.
            memory_type: Optional filter to clear only records of a specific type.

        Returns:
            The number of records deleted.

        Raises:
            MemoryProviderError: If clearing fails in the underlying provider.
        """
        try:
            return await self.provider.aclear(
                session_id=session_id,
                entity_id=entity_id,
                memory_type=memory_type,
            )
        except Exception as exc:
            raise MemoryProviderError(
                f"Failed to asynchronously clear memory: {exc}"
            ) from exc
