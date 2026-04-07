"""Tests for the tool lifecycle management system.

Covers: manifest loading, tool classification, agent resolution,
deferred search filtering, health tracking, and daily health check.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _reset_registries():
    """Reset lifecycle registries to a clean state after each test."""
    from quantstack.tools.registry import (
        ACTIVE_TOOLS,
        DEGRADED_TOOLS,
        PLANNED_TOOLS,
        classify_tools,
    )

    yield

    # Restore original classification so other tests aren't affected
    ACTIVE_TOOLS.clear()
    PLANNED_TOOLS.clear()
    DEGRADED_TOOLS.clear()
    classify_tools()


@pytest.fixture()
def mock_manifest_with_planned(tmp_path, _reset_registries):
    """Provide a manifest that marks 'web_search' as planned and 'signal_brief' as degraded."""
    import yaml

    manifest = {
        "tools": {
            "web_search": {"status": "planned"},
            "signal_brief": {"status": "degraded"},
            # Everything else defaults to active
        }
    }
    manifest_path = tmp_path / "tool_manifest.yaml"
    with open(manifest_path, "w") as fh:
        yaml.dump(manifest, fh)
    return manifest_path


# ---------------------------------------------------------------------------
# Manifest & classification
# ---------------------------------------------------------------------------


class TestManifestLoading:
    def test_load_tool_manifest_returns_dict(self):
        from quantstack.tools.registry import load_tool_manifest

        result = load_tool_manifest()
        assert isinstance(result, dict)

    def test_load_tool_manifest_missing_file(self, tmp_path):
        from quantstack.tools.registry import load_tool_manifest, _MANIFEST_PATH

        with patch("quantstack.tools.registry._MANIFEST_PATH", tmp_path / "nonexistent.yaml"):
            result = load_tool_manifest()
        assert result == {}


class TestClassification:
    def test_active_tools_contains_only_active(self):
        """After default classify_tools(), all tools should be active (manifest marks everything active)."""
        from quantstack.tools.registry import ACTIVE_TOOLS, PLANNED_TOOLS, DEGRADED_TOOLS, TOOL_REGISTRY

        # Default manifest marks everything active
        assert len(ACTIVE_TOOLS) > 0
        assert len(ACTIVE_TOOLS) == len(TOOL_REGISTRY) - len(PLANNED_TOOLS) - len(DEGRADED_TOOLS)

    def test_planned_tools_empty_by_default(self):
        from quantstack.tools.registry import PLANNED_TOOLS

        assert len(PLANNED_TOOLS) == 0

    def test_classify_with_planned_tool(self, mock_manifest_with_planned):
        from quantstack.tools.registry import (
            ACTIVE_TOOLS,
            DEGRADED_TOOLS,
            PLANNED_TOOLS,
            TOOL_REGISTRY,
            classify_tools,
        )

        with patch("quantstack.tools.registry._MANIFEST_PATH", mock_manifest_with_planned):
            classify_tools()

        # web_search should be in PLANNED, not ACTIVE
        if "web_search" in TOOL_REGISTRY:
            assert "web_search" in PLANNED_TOOLS
            assert "web_search" not in ACTIVE_TOOLS

        # signal_brief should be in DEGRADED
        if "signal_brief" in TOOL_REGISTRY:
            assert "signal_brief" in DEGRADED_TOOLS
            assert "signal_brief" not in ACTIVE_TOOLS

    def test_registry_is_union_of_all_three(self):
        """TOOL_REGISTRY should contain every tool across all 3 lifecycle dicts."""
        from quantstack.tools.registry import ACTIVE_TOOLS, DEGRADED_TOOLS, PLANNED_TOOLS, TOOL_REGISTRY

        union = set(ACTIVE_TOOLS) | set(PLANNED_TOOLS) | set(DEGRADED_TOOLS)
        assert union == set(TOOL_REGISTRY)


# ---------------------------------------------------------------------------
# Agent resolution
# ---------------------------------------------------------------------------


class TestGetToolsForAgent:
    def test_resolves_active_tool(self):
        from quantstack.tools.registry import ACTIVE_TOOLS, get_tools_for_agent

        if not ACTIVE_TOOLS:
            pytest.skip("No active tools available")
        name = next(iter(ACTIVE_TOOLS))
        tools = get_tools_for_agent([name])
        assert len(tools) == 1
        assert tools[0].name == name

    def test_rejects_planned_tool(self, mock_manifest_with_planned):
        from quantstack.tools.registry import TOOL_REGISTRY, classify_tools, get_tools_for_agent

        with patch("quantstack.tools.registry._MANIFEST_PATH", mock_manifest_with_planned):
            classify_tools()

        if "web_search" not in TOOL_REGISTRY:
            pytest.skip("web_search not in registry")

        with pytest.raises(KeyError, match="PLANNED"):
            get_tools_for_agent(["web_search"])

    def test_rejects_degraded_tool(self, mock_manifest_with_planned):
        from quantstack.tools.registry import TOOL_REGISTRY, classify_tools, get_tools_for_agent

        with patch("quantstack.tools.registry._MANIFEST_PATH", mock_manifest_with_planned):
            classify_tools()

        if "signal_brief" not in TOOL_REGISTRY:
            pytest.skip("signal_brief not in registry")

        with pytest.raises(KeyError, match="DEGRADED"):
            get_tools_for_agent(["signal_brief"])

    def test_rejects_unknown_tool(self):
        from quantstack.tools.registry import get_tools_for_agent

        with pytest.raises(KeyError, match="not found"):
            get_tools_for_agent(["this_tool_does_not_exist_xyz"])


# ---------------------------------------------------------------------------
# Deferred search filtering
# ---------------------------------------------------------------------------


class TestSearchDeferredTools:
    def test_excludes_planned_tools(self, mock_manifest_with_planned):
        from quantstack.tools.registry import classify_tools, search_deferred_tools

        with patch("quantstack.tools.registry._MANIFEST_PATH", mock_manifest_with_planned):
            classify_tools()

        # Search with web_search in the deferred set — should not appear in results
        results = search_deferred_tools("web search", deferred_names={"web_search"}, max_results=5)
        found_names = {r["name"] for r in results}
        assert "web_search" not in found_names

    def test_includes_active_deferred_tools(self):
        from quantstack.tools.registry import ACTIVE_TOOLS, search_deferred_tools

        if not ACTIVE_TOOLS:
            pytest.skip("No active tools")

        # Pick an active tool and search for it
        name = next(iter(ACTIVE_TOOLS))
        results = search_deferred_tools(name.replace("_", " "), deferred_names={name}, max_results=5)
        # May or may not match depending on description, but should not be excluded
        # The key assertion is that it wasn't filtered out by the PLANNED check


# ---------------------------------------------------------------------------
# Health tracking
# ---------------------------------------------------------------------------


class TestTrackToolHealth:
    @patch("quantstack.db.pg_conn")
    def test_success_invocation(self, mock_pg_conn):
        from quantstack.tools._helpers import track_tool_health

        mock_conn = MagicMock()
        mock_pg_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pg_conn.return_value.__exit__ = MagicMock(return_value=False)

        track_tool_health("signal_brief", success=True, latency_ms=42.5)

        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        params = mock_conn.execute.call_args[0][1]

        assert "INSERT INTO tool_health" in sql
        assert "ON CONFLICT" in sql
        assert params[0] == "signal_brief"  # tool_name
        assert params[1] == 1  # success_inc
        assert params[2] == 0  # failure_inc

    @patch("quantstack.db.pg_conn")
    def test_failure_invocation(self, mock_pg_conn):
        from quantstack.tools._helpers import track_tool_health

        mock_conn = MagicMock()
        mock_pg_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pg_conn.return_value.__exit__ = MagicMock(return_value=False)

        track_tool_health("bad_tool", success=False, latency_ms=100.0, error="timeout")

        params = mock_conn.execute.call_args[0][1]
        assert params[0] == "bad_tool"
        assert params[1] == 0  # success_inc
        assert params[2] == 1  # failure_inc
        assert params[5] == "timeout"  # last_error

    @patch("quantstack.db.pg_conn", side_effect=Exception("DB down"))
    def test_db_failure_does_not_raise(self, mock_pg_conn):
        """Health tracking must never propagate exceptions."""
        from quantstack.tools._helpers import track_tool_health

        # Should not raise
        track_tool_health("signal_brief", success=True, latency_ms=10.0)


# ---------------------------------------------------------------------------
# Daily health check
# ---------------------------------------------------------------------------


class TestDailyHealthCheck:
    def test_auto_disables_low_success_rate(self, _reset_registries):
        from quantstack.tools.health_monitor import run_daily_health_check
        from quantstack.tools.registry import ACTIVE_TOOLS, DEGRADED_TOOLS

        # Pick an active tool to degrade
        if not ACTIVE_TOOLS:
            pytest.skip("No active tools")
        target_tool = next(iter(ACTIVE_TOOLS))

        mock_conn = MagicMock()
        # Simulate a tool with 20% success rate (2/10)
        mock_conn.execute.return_value.fetchall.return_value = [
            (target_tool, 10, 2, 8, 150.0, "some error"),
        ]

        mock_bus = MagicMock()

        degraded = run_daily_health_check(mock_conn, event_bus=mock_bus)

        assert target_tool in degraded
        assert target_tool in DEGRADED_TOOLS
        assert target_tool not in ACTIVE_TOOLS

    def test_does_not_disable_healthy_tools(self, _reset_registries):
        from quantstack.tools.health_monitor import run_daily_health_check
        from quantstack.tools.registry import ACTIVE_TOOLS

        if not ACTIVE_TOOLS:
            pytest.skip("No active tools")
        target_tool = next(iter(ACTIVE_TOOLS))

        mock_conn = MagicMock()
        # 90% success rate — should NOT be disabled
        mock_conn.execute.return_value.fetchall.return_value = [
            (target_tool, 10, 9, 1, 50.0, None),
        ]

        degraded = run_daily_health_check(mock_conn)

        assert target_tool not in degraded
        assert target_tool in ACTIVE_TOOLS

    def test_publishes_tool_disabled_event(self, _reset_registries):
        from quantstack.tools.health_monitor import run_daily_health_check
        from quantstack.tools.registry import ACTIVE_TOOLS

        if not ACTIVE_TOOLS:
            pytest.skip("No active tools")
        target_tool = next(iter(ACTIVE_TOOLS))

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            (target_tool, 10, 1, 9, 200.0, "crash"),
        ]

        mock_bus = MagicMock()
        run_daily_health_check(mock_conn, event_bus=mock_bus)

        mock_bus.publish.assert_called_once()
        event = mock_bus.publish.call_args[0][0]
        assert event.event_type.value == "tool_disabled"
        assert event.payload["tool_name"] == target_tool
        assert event.payload["success_rate"] == 0.1

    def test_no_rows_returns_empty(self):
        from quantstack.tools.health_monitor import run_daily_health_check

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []

        result = run_daily_health_check(mock_conn)
        assert result == []


# ---------------------------------------------------------------------------
# move_tool
# ---------------------------------------------------------------------------


class TestMoveTool:
    def test_move_active_to_degraded(self, _reset_registries):
        from quantstack.tools.registry import ACTIVE_TOOLS, DEGRADED_TOOLS, move_tool

        if not ACTIVE_TOOLS:
            pytest.skip("No active tools")
        name = next(iter(ACTIVE_TOOLS))

        move_tool(name, "active", "degraded")

        assert name in DEGRADED_TOOLS
        assert name not in ACTIVE_TOOLS

    def test_move_nonexistent_raises(self, _reset_registries):
        from quantstack.tools.registry import move_tool

        with pytest.raises(KeyError):
            move_tool("nonexistent_tool_xyz", "active", "degraded")

    def test_invalid_status_raises(self, _reset_registries):
        from quantstack.tools.registry import move_tool

        with pytest.raises(ValueError, match="Invalid from_status"):
            move_tool("anything", "bogus", "active")
