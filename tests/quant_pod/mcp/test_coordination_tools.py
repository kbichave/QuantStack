# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for coordination MCP tools — event bus, heartbeats, preflight.

These are sync tools (not async). They use open_db() for PostgreSQL access.
"""

from __future__ import annotations

import pytest

from quantstack.mcp.tools.coordination import (
    poll_events,
    publish_event,
    record_heartbeat,
    get_loop_health,
    run_preflight_check,
)
from tests.quantstack.mcp.conftest import _fn, assert_standard_response


class TestPublishEvent:
    def test_publish_happy_path(self):
        result = _fn(publish_event)(
            event_type="loop_heartbeat",
            source="test_runner",
            payload={"test": True},
        )
        assert_standard_response(result)
        assert result["success"] is True
        assert "event_id" in result

    def test_publish_unknown_event_type(self):
        """Unknown event types should still work (stored as string)."""
        result = _fn(publish_event)(
            event_type="custom_event_type",
            source="test_runner",
        )
        assert_standard_response(result)
        assert result["success"] is True


class TestPollEvents:
    def test_poll_empty(self):
        result = _fn(poll_events)(
            consumer_id="test_consumer",
            since_minutes=1,
        )
        assert_standard_response(result)
        assert "events" in result

    def test_poll_after_publish(self):
        _fn(publish_event)(
            event_type="model_trained",
            source="test_publisher",
            payload={"model": "test_model"},
        )
        result = _fn(poll_events)(
            consumer_id="test_poller",
            event_types=["model_trained"],
            since_minutes=5,
        )
        assert result["success"] is True
        assert len(result["events"]) >= 1


class TestRecordHeartbeat:
    def test_heartbeat(self):
        result = _fn(record_heartbeat)(
            loop_name="test_loop",
            status="running",
            iteration=1,
        )
        assert_standard_response(result)
        assert result["success"] is True


class TestGetLoopHealth:
    def test_health_after_heartbeat(self):
        _fn(record_heartbeat)(
            loop_name="health_test_loop",
            status="running",
            iteration=1,
        )
        result = _fn(get_loop_health)()
        assert_standard_response(result)
        assert "loops" in result


class TestRunPreflightCheck:
    def test_preflight(self):
        result = _fn(run_preflight_check)(target_symbols=["SPY"], target_wallet=1000.0)
        assert_standard_response(result)
        assert "checks" in result or "ready" in result
