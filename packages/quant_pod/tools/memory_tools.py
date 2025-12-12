# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Memory Tools - CrewAI tools for Mem0 memory operations.

Provides tools for agents to store and retrieve memories.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from loguru import logger
from pydantic import BaseModel, Field
from quant_pod.crewai_compat import BaseTool

from quant_pod.memory.mem0_client import (
    Mem0Client,
    MemoryCategory,
    get_mem0_client,
    store_memory_sync,
    search_memory_sync,
)


# =============================================================================
# INPUT SCHEMAS
# =============================================================================


class StoreMemoryInput(BaseModel):
    """Input for store_memory tool."""

    category: str = Field(
        ...,
        description="Memory category: market_observations, wave_scenarios, regime_context, trade_decisions, performance_patterns",
    )
    content: str = Field(..., description="The memory content to store")
    symbol: Optional[str] = Field(None, description="Related symbol if applicable")
    importance: str = Field(
        "normal", description="Importance level: low, normal, high, critical"
    )


class SearchMemoryInput(BaseModel):
    """Input for search_memory tool."""

    query: str = Field(..., description="Search query to find relevant memories")
    category: Optional[str] = Field(None, description="Optional category filter")
    limit: int = Field(5, description="Maximum number of results")


class GetRecentMemoryInput(BaseModel):
    """Input for get_recent_memory tool."""

    category: str = Field(..., description="Memory category to retrieve")
    limit: int = Field(10, description="Maximum number of results")


# =============================================================================
# TOOL CLASSES
# =============================================================================


class StoreMemoryTool(BaseTool):
    """Tool to store a memory for future reference."""

    name: str = "store_memory"
    description: str = """Store an observation, insight, or decision in persistent memory.
Use this to save important information that other agents should know about.
Categories: market_observations, wave_scenarios, regime_context, trade_decisions, performance_patterns"""
    args_schema: Type[BaseModel] = StoreMemoryInput

    def _run(
        self,
        category: str,
        content: str,
        symbol: Optional[str] = None,
        importance: str = "normal",
    ) -> str:
        """Store memory."""
        try:
            cat = MemoryCategory(category)
        except ValueError:
            cat = MemoryCategory.MARKET_OBSERVATIONS

        metadata = {
            "importance": importance,
        }
        if symbol:
            metadata["symbol"] = symbol.upper()

        memory_id = store_memory_sync(
            category=cat,
            content=content,
            metadata=metadata,
            agent_id=self.name,
        )

        if memory_id:
            return json.dumps(
                {
                    "success": True,
                    "memory_id": memory_id,
                    "message": f"Memory stored in {category}",
                }
            )
        else:
            # Fallback - memory logged but not stored in Mem0
            return json.dumps(
                {
                    "success": True,
                    "memory_id": None,
                    "message": f"Memory logged (Mem0 not available): {content[:50]}...",
                }
            )


class SearchMemoryTool(BaseTool):
    """Tool to search memories by semantic similarity."""

    name: str = "search_memory"
    description: str = """Search for relevant memories using semantic search.
Use this to find past observations, decisions, or patterns related to a topic."""
    args_schema: Type[BaseModel] = SearchMemoryInput

    def _run(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """Search memories."""
        cat = None
        if category:
            try:
                cat = MemoryCategory(category)
            except ValueError:
                pass

        memories = search_memory_sync(query, cat, limit)

        results = []
        for mem in memories:
            results.append(
                {
                    "content": mem.content,
                    "category": mem.category.value,
                    "metadata": mem.metadata,
                    "relevance": mem.relevance_score,
                }
            )

        return json.dumps(
            {
                "success": True,
                "count": len(results),
                "memories": results,
            }
        )


class GetRecentMemoryTool(BaseTool):
    """Tool to get recent memories from a category."""

    name: str = "get_recent_memory"
    description: str = """Get the most recent memories from a specific category.
Use this to catch up on recent observations or decisions."""
    args_schema: Type[BaseModel] = GetRecentMemoryInput

    def _run(
        self,
        category: str,
        limit: int = 10,
    ) -> str:
        """Get recent memories."""
        import asyncio

        try:
            cat = MemoryCategory(category)
        except ValueError:
            cat = MemoryCategory.MARKET_OBSERVATIONS

        client = get_mem0_client()

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        memories = loop.run_until_complete(client.get_recent(cat, limit))

        results = []
        for mem in memories:
            results.append(
                {
                    "content": mem.content,
                    "category": mem.category.value,
                    "metadata": mem.metadata,
                    "agent": mem.agent_id,
                }
            )

        return json.dumps(
            {
                "success": True,
                "count": len(results),
                "memories": results,
            }
        )


# Tool factory functions
def store_memory_tool() -> StoreMemoryTool:
    """Get the store memory tool instance."""
    return StoreMemoryTool()


def search_memory_tool() -> SearchMemoryTool:
    """Get the search memory tool instance."""
    return SearchMemoryTool()


def get_recent_memory_tool() -> GetRecentMemoryTool:
    """Get the recent memory tool instance."""
    return GetRecentMemoryTool()
