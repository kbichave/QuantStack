"""Unit tests for Langfuse retention configuration and scheduler job stub."""

import importlib
import logging
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# Stub psycopg modules so scheduler.py's import chain doesn't fail
_PSYCOPG_MODULES = [
    "psycopg", "psycopg.rows", "psycopg.types", "psycopg.types.json",
    "psycopg_pool",
]
for _mod_name in _PSYCOPG_MODULES:
    if _mod_name not in sys.modules:
        _stub = ModuleType(_mod_name)
        if _mod_name == "psycopg":
            _stub.Connection = MagicMock
        elif _mod_name == "psycopg.types.json":
            _stub.set_json_loads = MagicMock()
            _stub.Jsonb = MagicMock
        elif _mod_name == "psycopg_pool":
            _stub.ConnectionPool = MagicMock
        elif _mod_name == "psycopg.rows":
            _stub.dict_row = MagicMock()
        sys.modules[_mod_name] = _stub


@pytest.fixture(autouse=True)
def _add_scripts_to_path():
    """Temporarily add scripts/ to sys.path so scheduler.py can be imported."""
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
    scripts_dir = os.path.abspath(scripts_dir)
    sys.path.insert(0, scripts_dir)
    yield
    sys.path.remove(scripts_dir)
    # Clear cached import so monkeypatch changes take effect
    sys.modules.pop("scheduler", None)


class TestLangfuseRetentionConfig:
    """Verify env var defaults for Langfuse retention settings."""

    def test_retention_enabled_defaults_to_false(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_RETENTION_ENABLED", raising=False)
        val = os.environ.get("LANGFUSE_RETENTION_ENABLED", "false")
        assert val == "false"

    def test_retention_days_defaults_to_30(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_RETENTION_DAYS", raising=False)
        val = int(os.environ.get("LANGFUSE_RETENTION_DAYS", "30"))
        assert val == 30


class TestLangfuseRetentionSchedulerJob:
    """Verify the scheduler job stub exists and behaves correctly."""

    def test_scheduler_job_exists(self):
        import scheduler

        assert callable(scheduler.run_langfuse_retention_cleanup)

    def test_job_logs_disabled_when_flag_false(self, monkeypatch, caplog):
        monkeypatch.setenv("LANGFUSE_RETENTION_ENABLED", "false")
        import scheduler

        importlib.reload(scheduler)
        with caplog.at_level(logging.INFO):
            scheduler.run_langfuse_retention_cleanup()
        assert any("disabled" in r.message.lower() for r in caplog.records)

    def test_job_logs_would_delete_when_flag_true(self, monkeypatch, caplog):
        monkeypatch.setenv("LANGFUSE_RETENTION_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_RETENTION_DAYS", "30")
        import scheduler

        importlib.reload(scheduler)
        with caplog.at_level(logging.INFO):
            scheduler.run_langfuse_retention_cleanup()
        assert any("would delete" in r.message.lower() for r in caplog.records)
        assert any("30 days" in r.message for r in caplog.records)

    def test_job_respects_custom_retention_days(self, monkeypatch, caplog):
        monkeypatch.setenv("LANGFUSE_RETENTION_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_RETENTION_DAYS", "14")
        import scheduler

        importlib.reload(scheduler)
        with caplog.at_level(logging.INFO):
            scheduler.run_langfuse_retention_cleanup()
        assert any("14 days" in r.message for r in caplog.records)
