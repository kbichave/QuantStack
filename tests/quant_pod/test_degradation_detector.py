# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for quant_pod.monitoring.degradation_detector — Sprint 4.

Tests IS/OOS Sharpe classification, benchmark registration, and
_classify() logic. Uses in-memory DuckDB.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
import pytest
from quant_pod.monitoring.degradation_detector import (
    DegradationDetector,
    DegradationReport,
    DegradationStatus,
    ISBenchmark,
)


@pytest.fixture
def conn():
    return duckdb.connect(":memory:")


@pytest.fixture
def detector(conn) -> DegradationDetector:
    return DegradationDetector(conn=conn)


def _register_benchmark(detector: DegradationDetector, sharpe=1.8, dd=0.08, wr=0.57):
    benchmark = ISBenchmark(
        strategy_id="test_strategy",
        predicted_annual_sharpe=sharpe,
        predicted_max_drawdown=dd,
        predicted_win_rate=wr,
    )
    detector.register_benchmark(benchmark)
    return benchmark


def _seed_trades(conn, pnls: list, strategy_id: str = "test_strategy"):
    """Insert dummy closed trades into the DB for the detector to pick up."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS closed_trades (
            trade_id VARCHAR,
            strategy_id VARCHAR,
            realized_pnl DOUBLE,
            closed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    for i, pnl in enumerate(pnls):
        conn.execute(
            "INSERT INTO closed_trades VALUES (?, ?, ?, NOW())",
            [f"t{i}", strategy_id, pnl],
        )


# ---------------------------------------------------------------------------
# Benchmark registration
# ---------------------------------------------------------------------------


class TestBenchmarkRegistration:
    def test_register_persists_benchmark(self, detector):
        _register_benchmark(detector)
        assert "test_strategy" in detector._benchmarks

    def test_register_sets_fields(self, detector):
        _register_benchmark(detector, sharpe=2.0, dd=0.10, wr=0.60)
        stored = detector._benchmarks["test_strategy"]
        assert stored.predicted_annual_sharpe == 2.0
        assert stored.predicted_max_drawdown == 0.10


# ---------------------------------------------------------------------------
# Insufficient data path
# ---------------------------------------------------------------------------


class TestInsufficientData:
    def test_fewer_than_min_trades_returns_insufficient(self, detector):
        """With no closed trades, expect INSUFFICIENT_DATA."""
        report = detector.check("test_strategy")
        assert report.status == DegradationStatus.INSUFFICIENT_DATA
        assert report.recommended_size_multiplier == 1.0

    def test_nine_trades_still_insufficient(self, conn, detector):
        _seed_trades(conn, [100.0] * 9)
        report = detector.check("test_strategy")
        assert report.status == DegradationStatus.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# _classify — direct unit tests on the classification logic
# ---------------------------------------------------------------------------


class TestClassify:
    def test_negative_sharpe_is_critical(self, detector):
        status, findings, mult = detector._classify(
            live_sharpe=-0.5,
            live_win_rate=0.45,
            live_max_dd=-0.15,
            oos_is_ratio=None,
            dd_ratio=None,
            benchmark=None,
        )
        assert status == DegradationStatus.CRITICAL
        assert mult <= 0.25

    def test_good_oos_is_ratio_is_clean(self, detector):
        bm = ISBenchmark(
            strategy_id="s",
            predicted_annual_sharpe=2.0,
            predicted_max_drawdown=0.08,
            predicted_win_rate=0.55,
        )
        status, findings, mult = detector._classify(
            live_sharpe=1.5,
            live_win_rate=0.54,
            live_max_dd=-0.07,
            oos_is_ratio=0.75,  # 75% of IS — healthy
            dd_ratio=0.9,
            benchmark=bm,
        )
        assert status == DegradationStatus.CLEAN
        assert mult == 1.0

    def test_low_oos_is_ratio_is_warning(self, detector):
        bm = ISBenchmark(
            strategy_id="s",
            predicted_annual_sharpe=2.0,
            predicted_max_drawdown=0.08,
            predicted_win_rate=0.55,
        )
        status, findings, mult = detector._classify(
            live_sharpe=0.8,
            live_win_rate=0.52,
            live_max_dd=-0.10,
            oos_is_ratio=0.40,  # < 0.5 → WARNING
            dd_ratio=1.2,
            benchmark=bm,
        )
        assert status == DegradationStatus.WARNING
        assert mult <= 0.5

    def test_very_low_oos_is_ratio_is_critical(self, detector):
        bm = ISBenchmark(
            strategy_id="s",
            predicted_annual_sharpe=2.0,
            predicted_max_drawdown=0.08,
            predicted_win_rate=0.55,
        )
        status, findings, mult = detector._classify(
            live_sharpe=0.3,
            live_win_rate=0.48,
            live_max_dd=-0.10,
            oos_is_ratio=0.10,  # < 0.25 → CRITICAL
            dd_ratio=1.0,
            benchmark=bm,
        )
        assert status == DegradationStatus.CRITICAL
        assert mult <= 0.25

    def test_drawdown_ratio_warning(self, detector):
        bm = ISBenchmark(
            strategy_id="s",
            predicted_annual_sharpe=1.5,
            predicted_max_drawdown=0.05,
            predicted_win_rate=0.55,
        )
        status, findings, mult = detector._classify(
            live_sharpe=1.2,
            live_win_rate=0.54,
            live_max_dd=-0.12,  # actual 12% vs predicted 5%
            oos_is_ratio=0.8,
            dd_ratio=2.4,  # > 2.0 → WARNING
            benchmark=bm,
        )
        assert status == DegradationStatus.WARNING


# ---------------------------------------------------------------------------
# DegradationReport properties
# ---------------------------------------------------------------------------


class TestDegradationReportProperties:
    def test_emoji_critical(self):
        report = DegradationReport(
            strategy_id="s",
            status=DegradationStatus.CRITICAL,
            checked_at=datetime.now(),
            live_sharpe=-0.5,
            live_win_rate=0.40,
            live_max_drawdown=-0.15,
            live_n_trades=15,
            rolling_window_days=60,
            is_benchmark=None,
            sharpe_ratio_oos_vs_is=None,
            drawdown_ratio_vs_predicted=None,
        )
        assert "🚨" in report.emoji

    def test_emoji_clean(self):
        report = DegradationReport(
            strategy_id="s",
            status=DegradationStatus.CLEAN,
            checked_at=datetime.now(),
            live_sharpe=1.5,
            live_win_rate=0.55,
            live_max_drawdown=-0.05,
            live_n_trades=20,
            rolling_window_days=60,
            is_benchmark=None,
            sharpe_ratio_oos_vs_is=None,
            drawdown_ratio_vs_predicted=None,
        )
        assert "✅" in report.emoji
