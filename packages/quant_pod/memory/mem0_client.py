# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Mem0 Client - Native Mem0 integration for persistent agent memory.

Uses the mem0ai library with local Qdrant vector store for:
- Market observations (price movements, volume, gaps)
- Wave scenarios (wave counts, invalidation levels, targets)
- Regime context (volatility state, trend direction)
- Trade decisions (why trades taken/rejected, outcomes)
- Performance patterns (what strategies work in which conditions)
"""

from __future__ import annotations

import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from threading import Lock

from loguru import logger
from pydantic import BaseModel, Field


class MemoryCategory(str, Enum):
    """Categories for organizing agent memories."""

    MARKET_OBSERVATIONS = "market_observations"
    WAVE_SCENARIOS = "wave_scenarios"
    REGIME_CONTEXT = "regime_context"
    TRADE_DECISIONS = "trade_decisions"
    PERFORMANCE_PATTERNS = "performance_patterns"


class Memory(BaseModel):
    """A single memory entry."""

    id: Optional[str] = None
    content: str
    category: MemoryCategory
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    relevance_score: Optional[float] = None


class Mem0Client:
    """
    Native Mem0 client using local Qdrant vector store.

    Provides persistent memory storage for agents to share observations,
    learnings, and context across sessions.

    Usage:
        client = Mem0Client()

        # Store a memory
        client.store_memory(
            category=MemoryCategory.MARKET_OBSERVATIONS,
            content="SPY broke above resistance at 450",
            metadata={"symbol": "SPY", "price": 451.50}
        )

        # Search memories
        results = client.search_memory("SPY resistance levels")

        # Get recent memories
        recent = client.get_recent(MemoryCategory.WAVE_SCENARIOS, limit=5)
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to avoid multiple Mem0 instances."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        user_id: str = "quant_pod",
    ):
        """
        Initialize Mem0 client with local Qdrant storage.

        Args:
            user_id: User identifier for memory isolation
        """
        if self._initialized:
            return

        self.user_id = user_id
        self._mem0 = None
        self._available = None

        # Initialize Mem0
        self._init_mem0()
        self._initialized = True

    def _init_mem0(self) -> None:
        """Initialize the Mem0 Memory instance."""
        try:
            # Disable posthog telemetry
            os.environ["POSTHOG_DISABLED"] = "true"
            os.environ["ANONYMIZED_TELEMETRY"] = "false"

            from mem0 import Memory as Mem0Memory

            # Configure Mem0 with local Qdrant - disable graph store to avoid SQLite threading issues
            config = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "gpt-4o-mini",
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small",
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "quant_pod_memory",
                        "path": os.path.expanduser("~/.quant_pod/qdrant"),
                    },
                },
                # Disable graph store to avoid SQLite threading issues
                "graph_store": {
                    "provider": "none",
                },
                "version": "v1.1",  # Use newer version without graph by default
            }

            self._mem0 = Mem0Memory.from_config(config)
            self._available = True
            logger.info("Mem0 initialized with local Qdrant storage (graph disabled)")

        except Exception as e:
            logger.error(f"Failed to initialize Mem0: {e}")
            self._available = False

    def is_available(self) -> bool:
        """Check if Mem0 is available."""
        return self._available and self._mem0 is not None

    def store_memory(
        self,
        category: MemoryCategory,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store a memory in Mem0.

        Args:
            category: Memory category for organization
            content: The memory content (text)
            metadata: Additional metadata (symbol, price, etc.)
            agent_id: ID of the agent storing the memory

        Returns:
            Memory ID if successful, None otherwise
        """
        if not self.is_available():
            logger.warning("Mem0 not available - memory not stored")
            return None

        try:
            # Build metadata
            full_metadata = {
                "category": category.value,
                "agent_id": agent_id or "unknown",
                "timestamp": datetime.now().isoformat(),
                **(metadata or {}),
            }

            # Add to Mem0
            result = self._mem0.add(
                content,
                user_id=self.user_id,
                metadata=full_metadata,
            )

            # Extract memory ID from result
            memory_id = None
            if result and isinstance(result, dict):
                results = result.get("results", [])
                if results and len(results) > 0:
                    memory_id = results[0].get("id")

            logger.debug(f"Stored memory: {content[:50]}... (id={memory_id})")
            return memory_id

        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return None

    def search_memory(
        self,
        query: str,
        category: Optional[MemoryCategory] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """
        Search memories by semantic similarity.

        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum results to return

        Returns:
            List of matching memories
        """
        if not self.is_available():
            return []

        try:
            # Search in Mem0
            result = self._mem0.search(
                query,
                user_id=self.user_id,
                limit=limit,
            )

            memories = []
            results_list = result.get("results", []) if isinstance(result, dict) else []

            for item in results_list:
                item_metadata = item.get("metadata", {})
                item_category = item_metadata.get("category", "trade_decisions")

                # Filter by category if specified
                if category and item_category != category.value:
                    continue

                memories.append(
                    Memory(
                        id=item.get("id"),
                        content=item.get("memory", ""),
                        category=(
                            MemoryCategory(item_category)
                            if item_category
                            else MemoryCategory.TRADE_DECISIONS
                        ),
                        metadata=item_metadata,
                        agent_id=item_metadata.get("agent_id"),
                        relevance_score=item.get("score"),
                    )
                )

            return memories

        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return []

    def get_recent(
        self,
        category: Optional[MemoryCategory] = None,
        limit: int = 10,
        agent_id: Optional[str] = None,
    ) -> List[Memory]:
        """
        Get recent memories.

        Args:
            category: Optional category filter
            limit: Maximum results to return
            agent_id: Optional agent filter

        Returns:
            List of recent memories
        """
        if not self.is_available():
            return []

        try:
            # Get all memories for user
            result = self._mem0.get_all(user_id=self.user_id)

            memories = []
            results_list = result.get("results", []) if isinstance(result, dict) else []

            for item in results_list:
                item_metadata = item.get("metadata", {})
                item_category = item_metadata.get("category")
                item_agent = item_metadata.get("agent_id")

                # Filter by category if specified
                if category and item_category != category.value:
                    continue

                # Filter by agent if specified
                if agent_id and item_agent != agent_id:
                    continue

                memories.append(
                    Memory(
                        id=item.get("id"),
                        content=item.get("memory", ""),
                        category=(
                            MemoryCategory(item_category)
                            if item_category
                            else MemoryCategory.TRADE_DECISIONS
                        ),
                        metadata=item_metadata,
                        agent_id=item_agent,
                    )
                )

            # Sort by created_at descending and limit
            memories.sort(key=lambda m: m.created_at, reverse=True)
            return memories[:limit]

        except Exception as e:
            logger.error(f"Error getting recent memories: {e}")
            return []

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory if found, None otherwise
        """
        if not self.is_available():
            return None

        try:
            result = self._mem0.get(memory_id)

            if result:
                item_metadata = result.get("metadata", {})
                return Memory(
                    id=result.get("id"),
                    content=result.get("memory", ""),
                    category=MemoryCategory(
                        item_metadata.get("category", "trade_decisions")
                    ),
                    metadata=item_metadata,
                    agent_id=item_metadata.get("agent_id"),
                )
            return None

        except Exception as e:
            logger.error(f"Error getting memory: {e}")
            return None

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory identifier

        Returns:
            True if deleted successfully
        """
        if not self.is_available():
            return False

        try:
            self._mem0.delete(memory_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False

    def clear_all(self) -> bool:
        """Clear all memories for the user."""
        if not self.is_available():
            return False

        try:
            self._mem0.delete_all(user_id=self.user_id)
            logger.info(f"Cleared all memories for user {self.user_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return False


# Global client instance
_mem0_client: Optional[Mem0Client] = None


def get_mem0_client() -> Mem0Client:
    """Get or create the global Mem0 client instance."""
    global _mem0_client
    if _mem0_client is None:
        _mem0_client = Mem0Client()
    return _mem0_client


# Convenience functions for synchronous usage
def store_memory_sync(
    category: MemoryCategory,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    agent_id: Optional[str] = None,
) -> Optional[str]:
    """Store a memory synchronously."""
    client = get_mem0_client()
    return client.store_memory(category, content, metadata, agent_id)


def search_memory_sync(
    query: str,
    category: Optional[MemoryCategory] = None,
    limit: int = 10,
) -> List[Memory]:
    """Search memories synchronously."""
    client = get_mem0_client()
    return client.search_memory(query, category, limit)


def get_recent_sync(
    category: Optional[MemoryCategory] = None,
    limit: int = 10,
    agent_id: Optional[str] = None,
) -> List[Memory]:
    """Get recent memories synchronously."""
    client = get_mem0_client()
    return client.get_recent(category, limit, agent_id)
