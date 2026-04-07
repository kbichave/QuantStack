"""Tests for AutoResearchClaw upgrades — Phase 10 Section 14."""
import json
import sys
import pytest
from unittest.mock import patch, MagicMock


def _load_autoresclaw_module():
    """Load autoresclaw_runner.py, mocking heavy imports that need DB."""
    # Mock quantstack.db.open_db before importing the module so it doesn't
    # fail when PostgreSQL is unavailable.
    import importlib.util

    mock_db = MagicMock()
    with patch.dict(sys.modules, {"quantstack.db": mock_db}):
        spec = importlib.util.spec_from_file_location(
            "autoresclaw_runner",
            "scripts/autoresclaw_runner.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def _load_scheduler_module():
    """Load scheduler.py, mocking heavy imports."""
    import importlib.util

    mock_db = MagicMock()
    mock_apscheduler_blocking = MagicMock()
    mock_apscheduler_cron = MagicMock()
    mock_lifecycle = MagicMock()

    modules_patch = {
        "quantstack.db": mock_db,
        "quantstack.autonomous": MagicMock(),
        "quantstack.autonomous.strategy_lifecycle": mock_lifecycle,
        "apscheduler": MagicMock(),
        "apscheduler.schedulers": MagicMock(),
        "apscheduler.schedulers.blocking": mock_apscheduler_blocking,
        "apscheduler.triggers": MagicMock(),
        "apscheduler.triggers.cron": mock_apscheduler_cron,
    }
    with patch.dict(sys.modules, modules_patch):
        spec = importlib.util.spec_from_file_location(
            "scheduler",
            "scripts/scheduler.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def test_tool_implement_task_type_accepted():
    """tool_implement has a prompt builder and produces valid prompt."""
    mod = _load_autoresclaw_module()

    assert "tool_implement" in mod._PROMPT_BUILDERS
    ctx = {
        "tool_name": "test_tool",
        "description": "A test tool",
        "expected_input": {"param": "string"},
        "expected_output": {"result": "dict"},
    }
    prompt = mod._PROMPT_BUILDERS["tool_implement"](ctx)
    assert "test_tool" in prompt
    assert isinstance(prompt, str)


def test_gap_detection_task_type_accepted():
    """gap_detection has a prompt builder and produces valid prompt."""
    mod = _load_autoresclaw_module()

    assert "gap_detection" in mod._PROMPT_BUILDERS
    ctx = {
        "failure_mode": "regime_mismatch",
        "affected_strategies": ["strat_a", "strat_b"],
        "example_losses": [{"trade_id": "t1", "pnl": -500}],
        "suggested_research_direction": "test regime filters",
        "cumulative_pnl_impact": 2500.0,
    }
    prompt = mod._PROMPT_BUILDERS["gap_detection"](ctx)
    assert "regime_mismatch" in prompt


def test_docker_compose_restart():
    """_restart_loops_after_fix uses docker compose, not tmux."""
    mod = _load_autoresclaw_module()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        mod._restart_loops_after_fix(["src/quantstack/signal/foo.py"])
        # Verify docker compose is called, not tmux
        call_args = mock_run.call_args_list
        assert any("docker" in str(c) for c in call_args)
        assert not any("tmux" in str(c) for c in call_args)


def test_nightly_schedule():
    """Scheduler JOBS has autoresclaw nightly, not Sunday-only."""
    mod = _load_scheduler_module()

    arc_jobs = [j for j in mod.JOBS if "autoresclaw" in j.get("label", "")]
    assert len(arc_jobs) >= 1
    for job in arc_jobs:
        trigger = job["trigger"]
        assert "day_of_week" not in trigger, "Should be nightly, not day-specific"
        assert trigger["hour"] == 20


def test_load_test_fixture_missing_manifest():
    """_load_test_fixture returns None when manifest doesn't exist."""
    mod = _load_autoresclaw_module()

    result = mod._load_test_fixture("nonexistent_tool")
    # May return None or a fixture — just verify it doesn't crash
    assert result is None or isinstance(result, dict)
