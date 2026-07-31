"""Abstract base class contract for conversational memory providers.

This module defines `BaseMemoryProvider`, specifying the synchronous and
asynchronous contract that all persistence or caching backends must implement
without coupling AI Core to specific databases or vector stores.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ai_core.memory.models import MemoryRecord, MemoryType


class BaseMemoryProvider(ABC):
    """Abstract interface contract for conversational memory storage providers.

    All custom memory storage adapters (e.g., in-memory, relational, or vector
    stores) must inherit from `BaseMemoryProvider` and implement its synchronous
    and asynchronous CRUD and search methods.
    """

    @abstractmethod
    def store_record(self, record: MemoryRecord) -> MemoryRecord:
        """Stores or updates a single memory record.

        Args:
            record: The memory record to persist.

        Returns:
            The stored MemoryRecord (potentially with updated timestamps or IDs).
        """
        raise NotImplementedError

    @abstractmethod
    async def astore_record(self, record: MemoryRecord) -> MemoryRecord:
        """Asynchronously stores or updates a single memory record.

        Args:
            record: The memory record to persist.

        Returns:
            The stored MemoryRecord.
        """
        raise NotImplementedError

    @abstractmethod
    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        """Retrieves a memory record by its unique identifier.

        Args:
            record_id: The unique identifier of the record to retrieve.

        Returns:
            The matching MemoryRecord, or None if not found.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_record(self, record_id: str) -> Optional[MemoryRecord]:
        """Asynchronously retrieves a memory record by its unique identifier.

        Args:
            record_id: The unique identifier of the record to retrieve.

        Returns:
            The matching MemoryRecord, or None if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_record(self, record_id: str) -> bool:
        """Deletes a memory record by its unique identifier.

        Args:
            record_id: The unique identifier of the record to delete.

        Returns:
            True if the record was successfully deleted, False if not found.
        """
        raise NotImplementedError

    @abstractmethod
    async def adelete_record(self, record_id: str) -> bool:
        """Asynchronously deletes a memory record by its unique identifier.

        Args:
            record_id: The unique identifier of the record to delete.

        Returns:
            True if deleted, False if not found.
        """
        raise NotImplementedError

    @abstractmethod
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
        """Queries stored memory records matching criteria or search text.

        Args:
            query: Search string or keyword filter.
            memory_type: Optional filter by memory category.
            session_id: Optional filter by conversation session.
            entity_id: Optional filter by entity/user identifier.
            limit: Maximum number of records to return.
            metadata_filter: Optional dictionary of required metadata key-values.

        Returns:
            List of matching MemoryRecord instances.
        """
        raise NotImplementedError

    @abstractmethod
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
        """Asynchronously queries stored memory records matching criteria.

        Args:
            query: Search string or keyword filter.
            memory_type: Optional filter by memory category.
            session_id: Optional filter by conversation session.
            entity_id: Optional filter by entity/user identifier.
            limit: Maximum number of records to return.
            metadata_filter: Optional dictionary of required metadata key-values.

        Returns:
            List of matching MemoryRecord instances.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(
        self,
        *,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """Clears stored memory records matching the optional filters.

        Args:
            session_id: Optional filter to clear only records for a session.
            entity_id: Optional filter to clear only records for an entity.
            memory_type: Optional filter to clear only records of a specific type.

        Returns:
            The count of deleted memory records.
        """
        raise NotImplementedError

    @abstractmethod
    async def aclear(
        self,
        *,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """Asynchronously clears stored memory records matching the optional filters.

        Args:
            session_id: Optional filter to clear only records for a session.
            entity_id: Optional filter to clear only records for an entity.
            memory_type: Optional filter to clear only records of a specific type.

        Returns:
            The count of deleted memory records.
        """
        raise NotImplementedError
