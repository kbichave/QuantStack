"""Tests for priority-based message pruning (section 11).

Validates:
  - P2 messages pruned before P1 when over budget
  - P1 messages truncated (not dropped) when P2 exhausted
  - P0 messages never pruned or summarized
  - P3 messages excluded from context
  - Type overrides (risk gate → P0 regardless of source agent tier)
  - Haiku summarization timeout fallback to truncation
  - Priority tag correctly set in metadata
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from quantstack.graphs.agent_executor import (
    MAX_MESSAGE_CHARS,
    PRIORITY_P0,
    PRIORITY_P1,
    PRIORITY_P2,
    PRIORITY_P3,
    SUMMARIZE_TRUNCATE_CHARS,
    _get_message_priority,
    _prune_messages,
    _summarize_message,
    _truncate_content,
    tag_message_priority,
)


def _make_msg(content: str, priority: str, msg_type=AIMessage) -> AIMessage:
    """Create a message with priority metadata."""
    msg = msg_type(content=content)
    msg.metadata = {"priority_tier": priority}
    return msg


def _make_oversized_messages(
    p0_count: int = 1,
    p1_count: int = 1,
    p2_count: int = 5,
    p3_count: int = 0,
    chars_per_msg: int = 40000,
) -> list:
    """Build a message list that exceeds MAX_MESSAGE_CHARS."""
    msgs = [
        SystemMessage(content="system"),
        HumanMessage(content="user prompt"),
    ]
    filler = "x" * chars_per_msg
    for _ in range(p0_count):
        msgs.append(_make_msg(filler, PRIORITY_P0))
    for _ in range(p1_count):
        msgs.append(_make_msg(filler, PRIORITY_P1))
    for _ in range(p2_count):
        msgs.append(_make_msg(filler, PRIORITY_P2))
    for _ in range(p3_count):
        msgs.append(_make_msg(filler, PRIORITY_P3))
    return msgs


class TestPriorityTagging:
    """Priority tier tagging on messages."""

    def test_default_priority_is_p2(self):
        msg = AIMessage(content="hello")
        assert _get_message_priority(msg) == PRIORITY_P2

    def test_tag_message_with_agent_priority(self):
        msg = AIMessage(content="hello")
        tag_message_priority(msg, agent_priority=PRIORITY_P1)
        assert _get_message_priority(msg) == PRIORITY_P1

    def test_type_override_risk_gate_is_p0(self):
        """Risk gate output is always P0 regardless of agent tier."""
        msg = AIMessage(content="risk gate output")
        tag_message_priority(msg, agent_priority=PRIORITY_P1, message_type="risk_gate_output")
        assert _get_message_priority(msg) == PRIORITY_P0

    def test_type_override_kill_switch_is_p0(self):
        msg = AIMessage(content="kill switch status")
        tag_message_priority(msg, agent_priority=PRIORITY_P2, message_type="kill_switch_active")
        assert _get_message_priority(msg) == PRIORITY_P0

    def test_type_override_blocking_node_error_is_p0(self):
        msg = AIMessage(content="data_refresh failed")
        tag_message_priority(msg, agent_priority=PRIORITY_P2, message_type="blocking_node_error")
        assert _get_message_priority(msg) == PRIORITY_P0

    def test_no_type_override_uses_agent_priority(self):
        msg = AIMessage(content="analysis")
        tag_message_priority(msg, agent_priority=PRIORITY_P2, message_type="analysis_output")
        assert _get_message_priority(msg) == PRIORITY_P2


class TestPruneMessages:
    """Priority-aware pruning algorithm."""

    def test_under_budget_returns_unchanged(self):
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="user"),
            _make_msg("short", PRIORITY_P2),
        ]
        result = _prune_messages(msgs)
        assert len(result) == 3

    def test_p3_removed_first(self):
        """P3 messages are swept even before checking P2."""
        msgs = _make_oversized_messages(p0_count=0, p1_count=0, p2_count=0, p3_count=5)
        result = _prune_messages(msgs)
        for m in result[2:]:
            assert _get_message_priority(m) != PRIORITY_P3

    def test_p2_pruned_before_p1(self):
        """P2 messages are removed before P1 messages are touched."""
        msgs = _make_oversized_messages(p0_count=0, p1_count=2, p2_count=3, chars_per_msg=40000)
        result = _prune_messages(msgs)
        priorities = [_get_message_priority(m) for m in result[2:]]
        # P2 count should be reduced (some or all removed)
        p2_remaining = priorities.count(PRIORITY_P2)
        p1_remaining = priorities.count(PRIORITY_P1)
        # All P1 should still be present since P2 was pruned first
        assert p1_remaining == 2

    def test_p0_never_pruned(self):
        """P0 messages survive even when over budget after all pruning."""
        msgs = _make_oversized_messages(p0_count=3, p1_count=0, p2_count=0, chars_per_msg=60000)
        result = _prune_messages(msgs)
        p0_msgs = [m for m in result[2:] if _get_message_priority(m) == PRIORITY_P0]
        assert len(p0_msgs) == 3

    def test_p1_truncated_when_p2_exhausted(self):
        """P1 messages are truncated (not removed) after all P2 removed."""
        # Build: lots of P1, no P2, over budget
        msgs = _make_oversized_messages(p0_count=0, p1_count=5, p2_count=0, chars_per_msg=40000)
        result = _prune_messages(msgs)
        p1_msgs = [m for m in result[2:] if _get_message_priority(m) == PRIORITY_P1 or
                    (hasattr(m, "metadata") and isinstance(m.metadata, dict) and m.metadata.get("summarized"))]
        # P1 messages should still be present (truncated, not removed)
        assert len(p1_msgs) > 0
        # At least one should be truncated
        truncated = [m for m in result[2:] if hasattr(m, "content") and
                     isinstance(m.content, str) and "[truncated]" in m.content]
        assert len(truncated) > 0

    def test_header_always_preserved(self):
        """System and user messages are never pruned."""
        msgs = _make_oversized_messages(p0_count=0, p1_count=0, p2_count=10, chars_per_msg=20000)
        result = _prune_messages(msgs)
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)


class TestTruncateContent:
    """Content truncation helper."""

    def test_short_content_unchanged(self):
        assert _truncate_content("hello", 500) == "hello"

    def test_long_content_truncated(self):
        result = _truncate_content("x" * 1000, 100)
        assert len(result) == 100 + len(" [truncated]")
        assert result.endswith("[truncated]")


class TestSummarizeMessage:
    """Haiku summarization with timeout fallback."""

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_truncation(self):
        """When Haiku times out, falls back to truncation."""
        import asyncio
        import sys

        mock_routing = MagicMock()
        mock_llm = AsyncMock()

        async def slow_invoke(*args, **kwargs):
            await asyncio.sleep(10)
            return MagicMock(content="summary")
        mock_llm.ainvoke = slow_invoke
        mock_routing.get_llm.return_value = mock_llm
        sys.modules["quantstack.llm.routing"] = mock_routing

        try:
            content = "x" * 2000
            result = await _summarize_message(content)
            assert "[truncated]" in result
        finally:
            del sys.modules["quantstack.llm.routing"]

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_truncation(self):
        """When Haiku raises, falls back to truncation."""
        import sys

        mock_routing = MagicMock()
        mock_routing.get_llm.side_effect = ConnectionError("LLM down")
        sys.modules["quantstack.llm.routing"] = mock_routing

        try:
            content = "x" * 2000
            result = await _summarize_message(content)
            assert "[truncated]" in result
        finally:
            del sys.modules["quantstack.llm.routing"]
