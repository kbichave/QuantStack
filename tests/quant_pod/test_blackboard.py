# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the DuckDB-backed Blackboard (agent memory).

All tests use an in-memory DB via TradingContext — zero file-system side-effects.
"""

from __future__ import annotations

from datetime import date

import pytest
from quantstack.context import create_trading_context
from quantstack.memory.blackboard import Blackboard


@pytest.fixture
def ctx():
    context = create_trading_context(db_path=":memory:")
    yield context
    context.db.close()


@pytest.fixture
def bb(ctx) -> Blackboard:
    return ctx.blackboard


class TestWrite:
    def test_write_single_entry(self, bb):
        bb.write("TrendIC", "SPY", "Strong uptrend detected")
        entries = bb.read_recent(limit=10)
        assert len(entries) == 1

    def test_write_multiple_entries(self, bb):
        bb.write("TrendIC", "SPY", "msg1")
        bb.write("TrendIC", "QQQ", "msg2")
        bb.write("MomIC", "AAPL", "msg3")
        entries = bb.read_recent(limit=10)
        assert len(entries) == 3

    def test_entry_fields_correct(self, bb):
        bb.write("TrendIC", "SPY", "uptrend", category="analysis")
        entries = bb.read_recent(limit=1)
        e = entries[0]
        assert e.agent == "TrendIC"
        assert e.symbol == "SPY"
        assert "uptrend" in e.message
        assert e.category == "analysis"

    def test_write_with_extra_data(self, bb):
        bb.write("TrendIC", "SPY", "signal", extra={"confidence": 0.82})
        entries = bb.read_recent(limit=1)
        assert len(entries) == 1

    def test_write_with_sim_date(self, bb):
        d = date(2024, 3, 15)
        bb.write("TrendIC", "SPY", "msg", sim_date=d)
        entries = bb.read_recent(limit=1)
        assert len(entries) == 1


class TestReadRecent:
    def test_filter_by_symbol(self, bb):
        bb.write("IC", "SPY", "spy msg")
        bb.write("IC", "QQQ", "qqq msg")
        bb.write("IC", "AAPL", "aapl msg")
        spy_entries = bb.read_recent(symbol="SPY", limit=10)
        assert all(e.symbol == "SPY" for e in spy_entries)
        assert len(spy_entries) == 1

    def test_filter_by_agent(self, bb):
        bb.write("TrendIC", "SPY", "trend")
        bb.write("MomIC", "SPY", "momentum")
        bb.write("TrendIC", "QQQ", "trend qqq")
        trend_entries = bb.read_recent(agent="TrendIC", limit=10)
        assert all(e.agent == "TrendIC" for e in trend_entries)
        assert len(trend_entries) == 2

    def test_filter_by_category(self, bb):
        bb.write("IC", "SPY", "decision", category="decision")
        bb.write("IC", "SPY", "analysis", category="analysis")
        decisions = bb.read_recent(category="decision", limit=10)
        assert all(e.category == "decision" for e in decisions)
        assert len(decisions) == 1

    def test_limit_respected(self, bb):
        for i in range(10):
            bb.write("IC", "SPY", f"msg {i}")
        entries = bb.read_recent(limit=3)
        assert len(entries) == 3

    def test_returns_most_recent_first(self, bb):
        bb.write("IC", "SPY", "first")
        bb.write("IC", "SPY", "second")
        bb.write("IC", "SPY", "third")
        entries = bb.read_recent(symbol="SPY", limit=3)
        # Most recent first
        assert entries[0].message == "third"

    def test_empty_result_when_no_match(self, bb):
        bb.write("IC", "SPY", "msg")
        entries = bb.read_recent(symbol="NONEXISTENT", limit=10)
        assert entries == []


class TestReadAsContext:
    def test_returns_markdown_string(self, bb):
        bb.write("TrendIC", "SPY", "uptrend detected")
        ctx_str = bb.read_as_context("SPY")
        assert isinstance(ctx_str, str)
        assert len(ctx_str) > 0

    def test_context_contains_symbol(self, bb):
        bb.write("IC", "AAPL", "strong move")
        ctx_str = bb.read_as_context("AAPL")
        assert "AAPL" in ctx_str or "strong move" in ctx_str

    def test_empty_context_when_no_entries(self, bb):
        ctx_str = bb.read_as_context("UNKNOWN")
        # Should return empty string or placeholder, not raise
        assert isinstance(ctx_str, str)


class TestClear:
    def test_clear_removes_all_entries(self, bb):
        bb.write("IC", "SPY", "msg1")
        bb.write("IC", "QQQ", "msg2")
        bb.clear()
        assert bb.read_recent(limit=100) == []

    def test_clear_scoped_to_session(self, bb, ctx):
        bb.write("IC", "SPY", "session1 msg")
        bb.clear(session_id=ctx.session_id)
        assert bb.read_recent(limit=100) == []

    def test_clear_before_date(self, bb):
        bb.write("IC", "SPY", "old", sim_date=date(2024, 1, 1))
        bb.write("IC", "SPY", "new", sim_date=date(2025, 1, 1))
        bb.clear_before_date(date(2024, 6, 1))
        entries = bb.read_recent(limit=10)
        # Only the 2025 entry should remain
        assert len(entries) == 1
        assert "new" in entries[0].message


class TestSessionScoping:
    def test_set_session_updates_session_id(self, ctx):
        bb = ctx.blackboard
        new_sid = "test-session-123"
        bb.set_session(new_sid)
        bb.write("IC", "SPY", "msg")
        entries = bb.read_recent(session_id=new_sid, limit=10)
        assert len(entries) == 1

    def test_multiple_sessions_isolated(self, ctx):
        bb = ctx.blackboard
        bb.set_session("session-A")
        bb.write("IC", "SPY", "session A message")
        bb.set_session("session-B")
        bb.write("IC", "SPY", "session B message")

        a_entries = bb.read_recent(session_id="session-A", limit=10)
        b_entries = bb.read_recent(session_id="session-B", limit=10)
        assert len(a_entries) == 1
        assert len(b_entries) == 1
        assert "session A" in a_entries[0].message
        assert "session B" in b_entries[0].message


class TestToMarkdown:
    def test_entry_markdown_format(self, bb):
        bb.write("TrendIC", "SPY", "uptrend detected")
        entries = bb.read_recent(limit=1)
        md = entries[0].to_markdown()
        assert "TrendIC" in md
        assert "SPY" in md
        assert "uptrend" in md
