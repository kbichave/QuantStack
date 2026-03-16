# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Decoder Crew — verifies pattern extraction from trade signals.

Feeds synthetic signals with known patterns and verifies the ICs recover them.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from quant_pod.crews.decoder_crew import (
    _analyze_entry_patterns,
    _analyze_exit_patterns,
    _analyze_regime_affinity,
    _analyze_sizing_patterns,
    _parse_signal,
    decode_signals,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic signal sets with known patterns
# ---------------------------------------------------------------------------


def _make_morning_long_signals(n: int = 40) -> list:
    """Generate signals that always enter at 10:30 AM, long direction."""
    signals = []
    base_date = datetime(2024, 1, 2, 10, 30)
    for i in range(n):
        entry = base_date + timedelta(days=i)
        exit_time = entry + timedelta(hours=2)
        entry_price = 100 + i * 0.1
        # 70% winners, 30% losers
        if i % 10 < 7:
            exit_price = entry_price * 1.02  # +2% win
        else:
            exit_price = entry_price * 0.99  # -1% loss

        signals.append({
            "symbol": "SPY",
            "direction": "long",
            "entry_time": entry.strftime("%Y-%m-%d %H:%M"),
            "entry_price": entry_price,
            "exit_time": exit_time.strftime("%Y-%m-%d %H:%M"),
            "exit_price": exit_price,
        })
    return signals


def _make_swing_mixed_signals(n: int = 30) -> list:
    """Generate swing-length signals with mixed long/short."""
    signals = []
    base_date = datetime(2024, 1, 2, 14, 0)
    for i in range(n):
        entry = base_date + timedelta(days=i * 3)
        exit_time = entry + timedelta(days=3)
        direction = "long" if i % 2 == 0 else "short"
        entry_price = 100 + i * 0.5
        if direction == "long":
            exit_price = entry_price * (1.03 if i % 4 != 0 else 0.98)
        else:
            exit_price = entry_price * (0.97 if i % 5 != 0 else 1.02)

        signals.append({
            "symbol": "AAPL",
            "direction": direction,
            "entry_time": entry.strftime("%Y-%m-%d %H:%M"),
            "entry_price": entry_price,
            "exit_time": exit_time.strftime("%Y-%m-%d %H:%M"),
            "exit_price": exit_price,
        })
    return signals


# ---------------------------------------------------------------------------
# Signal parsing
# ---------------------------------------------------------------------------


class TestParseSignal:
    def test_parses_valid_signal(self):
        sig = _parse_signal({
            "symbol": "SPY", "direction": "long",
            "entry_time": "2024-01-15 10:30", "entry_price": 470.5,
            "exit_time": "2024-01-15 14:00", "exit_price": 473.2,
        })
        assert sig is not None
        assert sig["symbol"] == "SPY"
        assert sig["pnl_pct"] > 0
        assert sig["is_winner"] is True
        assert sig["entry_hour"] == 10

    def test_rejects_missing_price(self):
        sig = _parse_signal({
            "symbol": "SPY", "direction": "long",
            "entry_time": "2024-01-15 10:30", "entry_price": 0,
            "exit_time": "2024-01-15 14:00", "exit_price": 473.2,
        })
        assert sig is None

    def test_short_pnl_calculation(self):
        sig = _parse_signal({
            "symbol": "SPY", "direction": "short",
            "entry_time": "2024-01-15 10:30", "entry_price": 470.5,
            "exit_time": "2024-01-15 14:00", "exit_price": 465.0,
        })
        assert sig is not None
        assert sig["pnl_pct"] > 0  # Short: entry > exit = winner


# ---------------------------------------------------------------------------
# Entry pattern IC
# ---------------------------------------------------------------------------


class TestEntryPatterns:
    def test_detects_morning_pattern(self):
        signals = [_parse_signal(s) for s in _make_morning_long_signals()]
        result = _analyze_entry_patterns([s for s in signals if s])
        assert result["timing_pattern"] == "morning_trader"
        assert result["peak_entry_hour"] == 10
        assert result["direction_bias"] == "long"
        assert result["long_pct"] == 100.0

    def test_detects_long_bias(self):
        signals = [_parse_signal(s) for s in _make_morning_long_signals()]
        result = _analyze_entry_patterns([s for s in signals if s])
        assert result["direction_bias"] == "long"

    def test_detects_mixed_direction(self):
        signals = [_parse_signal(s) for s in _make_swing_mixed_signals()]
        result = _analyze_entry_patterns([s for s in signals if s])
        assert result["direction_bias"] == "mixed"


# ---------------------------------------------------------------------------
# Exit pattern IC
# ---------------------------------------------------------------------------


class TestExitPatterns:
    def test_detects_intraday_style(self):
        signals = [_parse_signal(s) for s in _make_morning_long_signals()]
        result = _analyze_exit_patterns([s for s in signals if s])
        assert result["style"] == "intraday"
        assert result["avg_holding_minutes"] > 0

    def test_detects_swing_style(self):
        signals = [_parse_signal(s) for s in _make_swing_mixed_signals()]
        result = _analyze_exit_patterns([s for s in signals if s])
        assert result["style"] == "swing"

    def test_computes_win_loss_averages(self):
        signals = [_parse_signal(s) for s in _make_morning_long_signals()]
        result = _analyze_exit_patterns([s for s in signals if s])
        assert result["avg_win_pct"] > 0
        assert result["avg_loss_pct"] < 0


# ---------------------------------------------------------------------------
# Sizing pattern IC
# ---------------------------------------------------------------------------


class TestSizingPatterns:
    def test_no_size_data_returns_patience_ratio(self):
        signals = [_parse_signal(s) for s in _make_morning_long_signals()]
        result = _analyze_sizing_patterns([s for s in signals if s])
        assert result["has_size_data"] is False
        assert "patience_ratio" in result

    def test_with_size_data(self):
        raw = _make_morning_long_signals(20)
        for i, s in enumerate(raw):
            s["size"] = 100 + (i * 10)  # Increasing size
        signals = [_parse_signal(s) for s in raw]
        result = _analyze_sizing_patterns([s for s in signals if s])
        assert result["has_size_data"] is True
        assert "conviction_model" in result


# ---------------------------------------------------------------------------
# Regime affinity IC
# ---------------------------------------------------------------------------


class TestRegimeAffinity:
    def test_returns_regime_affinity(self):
        signals = [_parse_signal(s) for s in _make_morning_long_signals()]
        result = _analyze_regime_affinity([s for s in signals if s])
        assert "regime_affinity" in result
        assert isinstance(result["regime_affinity"], dict)
        assert len(result["regime_affinity"]) > 0

    def test_computes_win_rates_by_direction(self):
        signals = [_parse_signal(s) for s in _make_swing_mixed_signals()]
        result = _analyze_regime_affinity([s for s in signals if s])
        assert "long_win_rate" in result
        assert "short_win_rate" in result


# ---------------------------------------------------------------------------
# Full decode pipeline
# ---------------------------------------------------------------------------


class TestDecodeSignals:
    def test_full_decode_morning_longs(self):
        result = decode_signals(_make_morning_long_signals(), source_name="test_trader")
        assert result["success"] is True
        decoded = result["decoded_strategy"]
        assert decoded["source_trader"] == "test_trader"
        assert decoded["style"] == "intraday"
        assert decoded["timing_pattern"] == "morning_trader"
        assert decoded["win_rate"] > 60
        assert decoded["confidence"] > 0

    def test_full_decode_swing_mixed(self):
        result = decode_signals(_make_swing_mixed_signals(), source_name="swing_trader")
        assert result["success"] is True
        decoded = result["decoded_strategy"]
        assert decoded["style"] == "swing"
        assert decoded["sample_size"] == 30

    def test_low_confidence_warning(self):
        # Only 5 signals — below 20 threshold
        signals = _make_morning_long_signals(5)
        result = decode_signals(signals)
        assert result["success"] is True
        assert result["low_confidence_warning"] is True

    def test_empty_signals_fails(self):
        result = decode_signals([])
        assert result["success"] is False

    def test_invalid_signals_fails(self):
        result = decode_signals([{"bad": "data"}, {"also": "bad"}])
        assert result["success"] is False

    def test_decoded_has_edge_hypothesis(self):
        result = decode_signals(_make_morning_long_signals())
        assert "edge_hypothesis" in result["decoded_strategy"]
        assert len(result["decoded_strategy"]["edge_hypothesis"]) > 20

    def test_decoded_has_regime_affinity(self):
        result = decode_signals(_make_morning_long_signals())
        affinity = result["decoded_strategy"]["regime_affinity"]
        assert isinstance(affinity, dict)
        assert len(affinity) > 0
