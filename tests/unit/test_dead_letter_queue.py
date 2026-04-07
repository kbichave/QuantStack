"""Tests for Dead Letter Queue (section 10).

Validates:
  - DLQ writes on parse failure with full context
  - prompt_hash computation (SHA-256, 16-char truncated)
  - DLQ write failure doesn't break parse_json_response fallback
  - DLQ monitor rate computation and alert thresholds
  - No DLQ write when agent_name is empty (backward compat)
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from quantstack.graphs.agent_executor import parse_json_response
from quantstack.observability.dlq_monitor import (
    DLQ_CRITICAL_RATE_PCT,
    DLQ_WARN_RATE_PCT,
    check_dlq_alerts,
    compute_dlq_rate,
    count_dlq_entries,
)


# ── parse_json_response DLQ integration ──────────────────────────────────


class TestDLQWriteOnParseFailure:
    """DLQ rows are written when parse_json_response fails with context."""

    @patch("quantstack.db.db_conn")
    def test_dlq_write_on_failure(self, mock_db_conn):
        """Parse failure with agent context writes a DLQ row."""
        mock_conn = MagicMock()
        mock_db_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_conn.return_value.__exit__ = MagicMock(return_value=False)

        result = parse_json_response(
            "this is not json at all",
            fallback={},
            agent_name="test_agent",
            graph_name="trading",
            run_id="run-123",
            model_used="sonnet",
            prompt_text="What is the market doing?",
        )

        assert result == {}
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params[0] == "test_agent"
        assert params[1] == "trading"
        assert params[2] == "run-123"
        # input_summary is prompt_text[:500]
        assert params[3] == "What is the market doing?"
        # raw_output is truncated to 10000
        assert params[4] == "this is not json at all"
        assert params[5] == "parse_error"
        assert params[8] == "sonnet"

    @patch("quantstack.db.db_conn")
    def test_prompt_hash_computation(self, mock_db_conn):
        """prompt_hash is SHA-256 truncated to 16 chars."""
        mock_conn = MagicMock()
        mock_db_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_conn.return_value.__exit__ = MagicMock(return_value=False)

        prompt = "What is the market doing?"
        expected_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        parse_json_response(
            "not json",
            fallback={},
            agent_name="test_agent",
            prompt_text=prompt,
        )

        params = mock_conn.execute.call_args[0][1]
        assert params[7] == expected_hash  # prompt_hash position

    def test_no_dlq_write_without_agent_name(self):
        """Backward compat: no DLQ write when agent_name is empty."""
        # Should not raise or attempt DB write
        result = parse_json_response("not json", fallback={"default": True})
        assert result == {"default": True}

    @patch("quantstack.db.db_conn")
    def test_dlq_write_failure_returns_fallback(self, mock_db_conn):
        """DLQ write failure doesn't prevent fallback return."""
        mock_db_conn.return_value.__enter__ = MagicMock(
            side_effect=Exception("DB down")
        )

        result = parse_json_response(
            "garbage",
            fallback={"safe": True},
            agent_name="test_agent",
        )
        assert result == {"safe": True}

    def test_valid_json_no_dlq_write(self):
        """Successful parse never triggers DLQ write."""
        result = parse_json_response(
            '{"status": "ok"}',
            agent_name="test_agent",
            graph_name="trading",
        )
        assert result == {"status": "ok"}

    @patch("quantstack.db.db_conn")
    def test_raw_output_truncated_to_10000(self, mock_db_conn):
        """Raw output stored in DLQ is capped at 10000 chars."""
        mock_conn = MagicMock()
        mock_db_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_conn.return_value.__exit__ = MagicMock(return_value=False)

        long_text = "x" * 20000
        parse_json_response(
            long_text,
            fallback={},
            agent_name="test_agent",
        )

        params = mock_conn.execute.call_args[0][1]
        assert len(params[4]) == 10000  # raw_output truncated

    @patch("quantstack.db.db_conn")
    def test_empty_prompt_text_gives_empty_hash(self, mock_db_conn):
        """Empty prompt_text produces empty prompt_hash."""
        mock_conn = MagicMock()
        mock_db_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_conn.return_value.__exit__ = MagicMock(return_value=False)

        parse_json_response(
            "not json",
            fallback={},
            agent_name="test_agent",
            prompt_text="",
        )

        params = mock_conn.execute.call_args[0][1]
        assert params[7] == ""  # prompt_hash is empty


# ── DLQ Monitor ──────────────────────────────────────────────────────────


class TestDLQMonitor:
    """DLQ rate computation and alert thresholds."""

    @patch("quantstack.observability.dlq_monitor.count_dlq_entries", return_value=5)
    def test_compute_rate(self, _mock_count):
        rate = compute_dlq_rate("test_agent", 100)
        assert rate == pytest.approx(5.0)

    @patch("quantstack.observability.dlq_monitor.count_dlq_entries", return_value=0)
    def test_compute_rate_zero_failures(self, _mock_count):
        rate = compute_dlq_rate("test_agent", 100)
        assert rate == 0.0

    def test_compute_rate_zero_attempts(self):
        """Zero attempts returns 0.0, no division error."""
        rate = compute_dlq_rate("test_agent", 0)
        assert rate == 0.0

    def test_compute_rate_negative_attempts(self):
        """Negative attempts treated as zero."""
        rate = compute_dlq_rate("test_agent", -5)
        assert rate == 0.0

    @patch("quantstack.observability.dlq_monitor.compute_dlq_rate", return_value=12.0)
    def test_alert_critical(self, _mock_rate):
        assert check_dlq_alerts("agent", 100) == "critical"

    @patch("quantstack.observability.dlq_monitor.compute_dlq_rate", return_value=7.0)
    def test_alert_warn(self, _mock_rate):
        assert check_dlq_alerts("agent", 100) == "warn"

    @patch("quantstack.observability.dlq_monitor.compute_dlq_rate", return_value=2.0)
    def test_alert_none(self, _mock_rate):
        assert check_dlq_alerts("agent", 100) is None

    @patch("quantstack.observability.dlq_monitor.compute_dlq_rate")
    def test_alert_at_exact_threshold(self, mock_rate):
        """Exactly at warn threshold triggers warn."""
        mock_rate.return_value = DLQ_WARN_RATE_PCT
        assert check_dlq_alerts("agent", 100) == "warn"

    @patch("quantstack.observability.dlq_monitor.compute_dlq_rate")
    def test_alert_at_exact_critical(self, mock_rate):
        """Exactly at critical threshold triggers critical."""
        mock_rate.return_value = DLQ_CRITICAL_RATE_PCT
        assert check_dlq_alerts("agent", 100) == "critical"
