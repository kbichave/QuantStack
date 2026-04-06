"""Tests for EWF analyzer core (Section 02) and prompts (Section 03)."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import after path setup
from scripts.ewf_analyzer import (
    _acquire_pid_lock,
    _build_vision_prompt,
    _detect_mime_type,
    _fallback_result,
    _get_images_to_analyze,
    _parse_vision_response,
    _query_ohlcv_context,
    _upsert_analysis,
)


def _make_ewf_fixture(tmp_path, update_type, symbols, fetched_at_str):
    """Create data/ewf/{date}/{update_type}/ with images + metadata.json."""
    ut_dir = tmp_path / "2026-04-04" / update_type
    ut_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata.json
    results = {s: "ok" for s in symbols}
    meta = {
        "update_type": update_type,
        "date": "2026-04-04",
        "fetched_at_utc": fetched_at_str,
        "results": results,
    }
    (ut_dir / "metadata.json").write_text(json.dumps(meta))

    # Create dummy PNG files
    for sym in symbols:
        (ut_dir / f"{sym}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    return ut_dir


@pytest.fixture(autouse=True)
def _clean_ewf_table(trading_ctx):
    """Delete test rows before each test."""
    trading_ctx.db.execute("DELETE FROM ewf_chart_analyses")
    trading_ctx.db.execute("DELETE FROM screener_results WHERE regime_used = 'ewf_blue_box'")
    trading_ctx.db.execute("DELETE FROM signal_state WHERE action = 'ewf_blue_box_alert'")
    trading_ctx.db.commit()
    yield


class TestGetImagesToAnalyze:
    def test_returns_empty_when_dir_missing(self, trading_ctx, tmp_path):
        with patch("scripts.ewf_analyzer.DATA_DIR", tmp_path / "nonexistent"):
            result = _get_images_to_analyze("2026-04-04", "4h", None, trading_ctx.db)
        assert result == []

    def test_skips_already_analyzed(self, trading_ctx, tmp_path):
        ts_str = "2026-04-04T12:00:00"
        ts = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone.utc)
        _make_ewf_fixture(tmp_path, "4h", ["AAPL"], ts_str)

        # Pre-insert the row
        trading_ctx.db.execute(
            "INSERT INTO ewf_chart_analyses (symbol, timeframe, fetched_at, bias) "
            "VALUES (%s, %s, %s, %s)",
            ("AAPL", "4h", ts, "bullish"),
        )
        trading_ctx.db.commit()

        with patch("scripts.ewf_analyzer.DATA_DIR", tmp_path):
            result = _get_images_to_analyze("2026-04-04", "4h", None, trading_ctx.db)
        assert len(result) == 0

    def test_returns_new_images(self, trading_ctx, tmp_path):
        _make_ewf_fixture(tmp_path, "4h", ["AAPL", "MSFT"], "2026-04-04T12:00:00")

        with patch("scripts.ewf_analyzer.DATA_DIR", tmp_path):
            result = _get_images_to_analyze("2026-04-04", "4h", None, trading_ctx.db)
        assert len(result) == 2
        symbols = {r["symbol"] for r in result}
        assert symbols == {"AAPL", "MSFT"}


class TestParseVisionResponse:
    def test_valid_json(self):
        raw = json.dumps({
            "reasoning": "Y-axis shows 175-210 range. Turning Up box bottom-right.",
            "bias": "bullish",
            "turning_signal": "turning_up",
            "wave_position": "completing wave (2) correction",
            "wave_degree": "minor",
            "current_wave_label": "(2)",
            "completed_wave_sequence": "1 → 2 → 3 → 4 → 5 → (1) → (2)",
            "projected_path": "wave (3) up toward 192 after (2) completes near 172",
            "key_levels": {"support": [172.0], "resistance": [192.0],
                          "invalidation": 163.5, "target": 192.0},
            "blue_box_active": False,
            "blue_box_zone": None,
            "confidence": 0.85,
            "invalidation_rule_violated": False,
            "analyst_notes": "Strong impulse count intact",
            "summary": "Turning Up. Wave (2) correction completing near 172. Invalidation 163.50.",
        })
        result = _parse_vision_response(raw, "NVDA", "4h")
        assert result["bias"] == "bullish"
        assert result["turning_signal"] == "turning_up"
        assert result["confidence"] == 0.85
        assert result["completed_wave_sequence"] is not None
        assert result["projected_path"] is not None
        assert result["reasoning"] is not None

    def test_invalid_json_returns_fallback(self):
        result = _parse_vision_response("not json at all", "AAPL", "4h")
        assert result["bias"] == "unknown"
        assert result["turning_signal"] == "none"
        assert result["confidence"] == 0.0

    def test_markdown_fences_stripped(self):
        raw = '```json\n{"bias": "bearish", "turning_signal": "turning_down", "confidence": 0.7}\n```'
        result = _parse_vision_response(raw, "AAPL", "4h")
        assert result["bias"] == "bearish"
        assert result["turning_signal"] == "turning_down"

    def test_empty_response_returns_fallback(self):
        result = _parse_vision_response("", "AAPL", "4h")
        assert result["bias"] == "unknown"
        assert result["turning_signal"] == "none"

    def test_backwards_compatible_without_new_fields(self):
        """Old-format responses (missing new fields) still parse correctly."""
        raw = json.dumps({
            "bias": "bullish",
            "wave_position": "wave 3 of 5",
            "wave_degree": "minor",
            "current_wave_label": "3",
            "key_levels": {"support": [180.0], "resistance": [200.0],
                          "invalidation": None, "target": 200.0},
            "blue_box_active": False,
            "blue_box_zone": None,
            "confidence": 0.8,
            "invalidation_rule_violated": False,
            "analyst_notes": "Strong impulse",
            "summary": "Bullish impulse in progress",
        })
        result = _parse_vision_response(raw, "AAPL", "4h")
        assert result["bias"] == "bullish"
        assert result["turning_signal"] == "none"  # default
        assert result["completed_wave_sequence"] is None  # default
        assert result["projected_path"] is None  # default
        assert result["reasoning"] is None  # default


class TestUpsertAnalysis:
    def test_insert_new_row(self, trading_ctx):
        ts = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone.utc)
        row = {
            "bias": "bullish",
            "turning_signal": "turning_up",
            "confidence": 0.85,
            "wave_position": "wave 3",
            "wave_degree": "minor",
            "current_wave_label": "3",
            "completed_wave_sequence": "1 → 2 → 3",
            "projected_path": "wave 4 correction then 5 up",
            "key_levels": {"support": [180.0]},
            "blue_box_active": False,
            "blue_box_zone": None,
            "invalidation_rule_violated": False,
            "analyst_notes": None,
            "summary": "Test",
            "reasoning": "Step-by-step reading...",
            "raw_analysis": "{}",
            "model_used": "test",
            "image_path": "data/ewf/test.png",
        }
        _upsert_analysis(trading_ctx.db, "AAPL", "4h", ts, row)
        trading_ctx.db.execute(
            "SELECT bias, confidence, turning_signal FROM ewf_chart_analyses "
            "WHERE symbol = 'AAPL' AND timeframe = '4h'"
        )
        result = trading_ctx.db.fetchone()
        assert result == ("bullish", 0.85, "turning_up")

    def test_upsert_updates_existing(self, trading_ctx):
        ts = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone.utc)
        row1 = _fallback_result("AAPL", "4h", "first")
        row1["image_path"] = "test.png"
        _upsert_analysis(trading_ctx.db, "AAPL", "4h", ts, row1)

        # Re-analyze with different bias
        row2 = _fallback_result("AAPL", "4h", "second")
        row2["bias"] = "bullish"
        row2["confidence"] = 0.9
        row2["image_path"] = "test.png"
        _upsert_analysis(trading_ctx.db, "AAPL", "4h", ts, row2)

        # Should be exactly one row, updated
        trading_ctx.db.execute(
            "SELECT COUNT(*), MAX(bias), MAX(confidence) "
            "FROM ewf_chart_analyses "
            "WHERE symbol = 'AAPL' AND timeframe = '4h' AND fetched_at = %s",
            (ts,),
        )
        count, bias, conf = trading_ctx.db.fetchone()
        assert count == 1
        assert bias == "bullish"
        assert conf == 0.9


class TestOhlcvContext:
    def test_returns_empty_for_unknown_symbol(self, trading_ctx):
        result = _query_ohlcv_context("ZZZZZ", trading_ctx.db)
        assert result == []

    def test_returns_empty_for_market_overview(self, trading_ctx):
        result = _query_ohlcv_context("$MKT", trading_ctx.db)
        assert result == []


class TestPidLock:
    def test_exits_when_pid_alive(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        pid_path.write_text(str(os.getpid()))

        with patch("scripts.ewf_analyzer.Path") as mock_path_cls:
            # Make the Path() constructor return our tmp file
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.read_text.return_value = str(os.getpid())
            mock_path_cls.return_value = mock_path_instance

            with pytest.raises(SystemExit) as exc_info:
                _acquire_pid_lock("test")
            assert exc_info.value.code == 0

    def test_continues_when_pid_stale(self, tmp_path):
        """Stale PID (process not running) is overwritten."""
        pid_path = tmp_path / "test.pid"
        pid_path.write_text("99999999")  # Very unlikely to be a real PID

        with patch("scripts.ewf_analyzer.Path") as mock_path_cls:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.read_text.return_value = "99999999"
            mock_path_cls.return_value = mock_path_instance

            # Should not exit — stale PID is overwritten
            result = _acquire_pid_lock("test")
            mock_path_instance.write_text.assert_called_once_with(str(os.getpid()))


class TestBuildVisionPrompt:
    def test_includes_ohlcv_price_table(self):
        rows = [("2026-04-01", 180.0), ("2026-04-02", 182.5)]
        _sys, user = _build_vision_prompt("AAPL", "4h", rows)
        assert "180.00" in user
        assert "182.50" in user
        assert "Date" in user
        assert "Close" in user
        assert "Most recent close: 182.50" in user

    def test_includes_reading_procedure(self):
        rows = [("2026-04-01", 180.0)]
        _sys, user = _build_vision_prompt("AAPL", "4h", rows)
        assert "Turning Up/Down" in user
        assert "invalidation" in user.lower()
        assert "reasoning" in user

    def test_valid_prompt_when_ohlcv_empty(self):
        _sys, user = _build_vision_prompt("AAPL", "4h", [])
        assert "No OHLCV data available" in user
        assert "AAPL" in user
        assert "4h" in user

    def test_unavailable_note_when_no_data(self):
        _sys, user = _build_vision_prompt("MSFT", "daily", [])
        assert "No OHLCV data available" in user

    def test_symbol_in_price_context_header(self):
        rows = [("2026-04-01", 100.0)]
        _sys, user = _build_vision_prompt("TSLA", "weekly", rows)
        assert "TSLA" in user
        assert "weekly" in user

    def test_system_prompt_has_visual_vocabulary(self):
        sys_prompt, _ = _build_vision_prompt("SPY", "4h", [])
        assert "Visual Vocabulary" in sys_prompt
        assert "Turning Up" in sys_prompt
        assert "Turning Down" in sys_prompt
        assert "Invalidation" in sys_prompt
        assert "dashed" in sys_prompt.lower() or "Dashed" in sys_prompt
        assert "Red labels" in sys_prompt
        assert "Blue labels" in sys_prompt


class TestDetectMimeType:
    def test_png(self, tmp_path):
        p = tmp_path / "chart.png"
        p.write_bytes(b"\x89PNG")
        assert _detect_mime_type(p) == "image/png"

    def test_jpeg(self, tmp_path):
        p = tmp_path / "chart.jpg"
        p.write_bytes(b"\xff\xd8")
        assert _detect_mime_type(p) == "image/jpeg"

    def test_unknown_extension_falls_back(self, tmp_path):
        p = tmp_path / "chart.xyz"
        p.write_bytes(b"data")
        assert _detect_mime_type(p) == "image/jpeg"


class TestHandleBlueBoxAlerts:
    """Section 04: Blue Box alert handler tests."""

    def _insert_blue_box_row(self, conn, symbol, bias="bullish", confidence=0.85,
                              zone=None, minutes_ago=5):
        """Insert a blue_box analysis row analyzed `minutes_ago` minutes ago."""
        zone = zone or {"low": 150.0, "high": 160.0}
        conn.execute(
            """
            INSERT INTO ewf_chart_analyses
                (symbol, timeframe, fetched_at, analyzed_at, bias,
                 blue_box_active, blue_box_zone, confidence, summary)
            VALUES (%s, 'blue_box', NOW(), NOW() - INTERVAL '%s minutes',
                    %s, TRUE, %s, %s, %s)
            """,
            (symbol, minutes_ago, bias, json.dumps(zone), confidence,
             f"Blue box setup for {symbol}"),
        )
        conn.commit()

    def test_adds_to_screener_results(self, trading_ctx):
        self._insert_blue_box_row(trading_ctx.db, "AAPL")
        from scripts.ewf_analyzer import _handle_blue_box_alerts
        _handle_blue_box_alerts(trading_ctx.db, "blue_box")

        trading_ctx.db.execute(
            "SELECT symbol, regime_used FROM screener_results "
            "WHERE symbol = 'AAPL' AND regime_used = 'ewf_blue_box'"
        )
        row = trading_ctx.db.fetchone()
        assert row is not None
        assert row[0] == "AAPL"

    def test_upserts_signal_state(self, trading_ctx):
        self._insert_blue_box_row(trading_ctx.db, "MSFT", bias="bearish", confidence=0.9)
        from scripts.ewf_analyzer import _handle_blue_box_alerts
        _handle_blue_box_alerts(trading_ctx.db, "blue_box")

        trading_ctx.db.execute(
            "SELECT action, confidence FROM signal_state WHERE symbol = 'MSFT'"
        )
        row = trading_ctx.db.fetchone()
        assert row is not None
        assert row[0] == "ewf_blue_box_alert"
        assert row[1] == 0.9

    def test_appends_session_handoffs(self, trading_ctx, tmp_path):
        handoff_file = tmp_path / "session_handoffs.md"
        self._insert_blue_box_row(trading_ctx.db, "TSLA", bias="bullish", confidence=0.82)

        from scripts.ewf_analyzer import _handle_blue_box_alerts
        with patch("scripts.ewf_analyzer._SESSION_HANDOFFS_PATH", handoff_file):
            _handle_blue_box_alerts(trading_ctx.db, "blue_box")

        content = handoff_file.read_text()
        assert "## EWF Blue Box — TSLA bullish" in content
        assert "Zone: 150.0" in content
        assert "160.0" in content
        assert "82%" in content

    def test_noop_when_no_recent_blue_box(self, trading_ctx, tmp_path):
        # Insert a row analyzed 60 minutes ago — outside the 30-min window
        self._insert_blue_box_row(trading_ctx.db, "AAPL", minutes_ago=60)

        handoff_file = tmp_path / "session_handoffs.md"
        handoff_file.write_text("# existing\n")

        from scripts.ewf_analyzer import _handle_blue_box_alerts
        with patch("scripts.ewf_analyzer._SESSION_HANDOFFS_PATH", handoff_file):
            _handle_blue_box_alerts(trading_ctx.db, "blue_box")

        # No screener_results row
        trading_ctx.db.execute(
            "SELECT 1 FROM screener_results WHERE regime_used = 'ewf_blue_box'"
        )
        assert trading_ctx.db.fetchone() is None
        # File unchanged
        assert handoff_file.read_text() == "# existing\n"

    def test_handoffs_entry_includes_all_fields(self, trading_ctx, tmp_path):
        handoff_file = tmp_path / "session_handoffs.md"
        self._insert_blue_box_row(
            trading_ctx.db, "NVDA", bias="bearish", confidence=0.75,
            zone={"low": 800.0, "high": 850.0},
        )

        from scripts.ewf_analyzer import _handle_blue_box_alerts
        with patch("scripts.ewf_analyzer._SESSION_HANDOFFS_PATH", handoff_file):
            _handle_blue_box_alerts(trading_ctx.db, "blue_box")

        content = handoff_file.read_text()
        assert "## EWF Blue Box — NVDA bearish" in content
        assert "Zone: 800.0" in content
        assert "850.0" in content
        assert "75%" in content
        assert "Blue box setup for NVDA" in content
