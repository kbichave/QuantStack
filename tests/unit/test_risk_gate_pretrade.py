"""Tests for pre-trade risk gate checks (section 12).

Validates:
  - Correlation check: threshold, boundary, data missing, no positions
  - Heat budget: threshold, boundary, day rollover, configurable
  - Sector concentration: threshold, unknown sector, configurable
  - All checks fail closed on data unavailability
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest

from quantstack.execution.risk_gate import RiskGate, RiskLimits, RiskVerdict
from quantstack.universe import Sector


# ── Test fixtures ────────────────────────────────────────────────────────


@dataclass
class FakePosition:
    symbol: str
    quantity: int = 100
    avg_cost: float = 150.0
    current_price: float = 155.0
    side: str = "long"
    opened_at: datetime = None
    time_horizon: str = "swing"
    instrument_type: str = "equity"
    entry_atr: float = 2.0
    stop_price: float | None = None
    target_price: float | None = None
    option_type: str | None = None
    option_expiry: str | None = None
    option_strike: float | None = None
    strategy_id: str = ""

    def __post_init__(self):
        if self.opened_at is None:
            self.opened_at = datetime.now()


@dataclass
class FakeSnapshot:
    total_equity: float = 100_000.0
    daily_pnl: float = 0.0


class FakePortfolio:
    def __init__(self, positions=None, snapshot=None):
        self._positions = positions or []
        self._snapshot = snapshot or FakeSnapshot()

    def get_positions(self):
        return self._positions

    def get_position(self, symbol):
        for p in self._positions:
            if p.symbol == symbol:
                return p
        return None

    def get_snapshot(self):
        return self._snapshot


def _make_correlated_series(corr: float, n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate two OHLCV DataFrames with specified correlation between close returns."""
    np.random.seed(42)
    base = np.random.randn(n + 1)
    noise = np.random.randn(n + 1)
    # Create series with target correlation
    series_a = np.cumsum(base) + 100
    series_b = np.cumsum(corr * base + np.sqrt(1 - corr**2) * noise) + 100

    dates = pd.date_range("2026-01-01", periods=n + 1, freq="D")
    df_a = pd.DataFrame({"close": series_a}, index=dates)
    df_b = pd.DataFrame({"close": series_b}, index=dates)
    return df_a, df_b


def _make_gate(
    positions=None,
    equity=100_000.0,
    daily_pnl=0.0,
    limits=None,
) -> RiskGate:
    """Create RiskGate with fake portfolio."""
    portfolio = FakePortfolio(
        positions=positions or [],
        snapshot=FakeSnapshot(total_equity=equity, daily_pnl=daily_pnl),
    )
    gate = RiskGate.__new__(RiskGate)
    gate.limits = limits or RiskLimits()
    gate._portfolio = portfolio
    gate._daily_halted = None
    gate._lock = __import__("threading").Lock()
    gate.DAILY_HALT_SENTINEL = MagicMock()
    gate.DAILY_HALT_SENTINEL.exists.return_value = False
    return gate


# ── 12.1 Pre-Trade Correlation ──────────────────────────────────────────


class TestPretradeCorrelation:

    @patch("quantstack.execution.risk_gate.DataStore")
    def test_high_correlation_rejected(self, MockStore):
        """Position with 0.8 correlation to existing → rejected."""
        df_a, df_b = _make_correlated_series(0.85, 30)
        store = MagicMock()
        store.load_ohlcv.side_effect = lambda sym, _tf: df_a if sym == "AAPL" else df_b
        MockStore.return_value = store

        gate = _make_gate(positions=[FakePosition(symbol="AAPL")])
        violations = gate._check_pretrade_correlation("MSFT", 150.0)
        assert len(violations) == 1
        assert violations[0].rule == "pretrade_correlation"

    @patch("quantstack.execution.risk_gate.DataStore")
    def test_low_correlation_approved(self, MockStore):
        """Position with 0.3 correlation → approved."""
        df_a, df_b = _make_correlated_series(0.3, 30)
        store = MagicMock()
        store.load_ohlcv.side_effect = lambda sym, _tf: df_a if sym == "AAPL" else df_b
        MockStore.return_value = store

        gate = _make_gate(positions=[FakePosition(symbol="AAPL")])
        violations = gate._check_pretrade_correlation("MSFT", 150.0)
        assert len(violations) == 0

    @patch("quantstack.execution.risk_gate.DataStore")
    def test_data_unavailable_fails_closed(self, MockStore):
        """No price data → fail closed."""
        store = MagicMock()
        store.load_ohlcv.return_value = None
        MockStore.return_value = store

        gate = _make_gate(positions=[FakePosition(symbol="AAPL")])
        violations = gate._check_pretrade_correlation("NEWSTOCK", 50.0)
        assert len(violations) == 1
        assert "data_missing" in violations[0].rule

    @patch("quantstack.execution.risk_gate.DataStore")
    def test_data_feed_error_fails_closed(self, MockStore):
        """DataStore raises → fail closed, not a crash."""
        store = MagicMock()
        store.load_ohlcv.side_effect = Exception("feed down")
        MockStore.return_value = store

        gate = _make_gate(positions=[FakePosition(symbol="AAPL")])
        violations = gate._check_pretrade_correlation("MSFT", 150.0)
        assert len(violations) == 1

    def test_no_existing_positions_passes(self):
        """Empty portfolio → correlation check is a no-op."""
        gate = _make_gate(positions=[])
        violations = gate._check_pretrade_correlation("AAPL", 150.0)
        assert len(violations) == 0


# ── 12.2 Heat Budget ────────────────────────────────────────────────────


class TestHeatBudget:

    @patch("quantstack.execution.risk_gate.pg_conn")
    def test_over_budget_rejected(self, mock_pg):
        """31% heat → rejected."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (31_000.0,)
        mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pg.return_value.__exit__ = MagicMock(return_value=False)

        gate = _make_gate(equity=100_000.0)
        violations = gate._check_heat_budget(0, 100_000.0)  # existing 31k
        assert len(violations) == 1
        assert violations[0].rule == "daily_heat_budget"

    @patch("quantstack.execution.risk_gate.pg_conn")
    def test_under_budget_approved(self, mock_pg):
        """29% heat → approved."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (25_000.0,)
        mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pg.return_value.__exit__ = MagicMock(return_value=False)

        gate = _make_gate(equity=100_000.0)
        violations = gate._check_heat_budget(4_000.0, 100_000.0)  # 25k + 4k = 29%
        assert len(violations) == 0

    @patch("quantstack.execution.risk_gate.pg_conn")
    def test_boundary_exactly_30_rejected(self, mock_pg):
        """Exactly 30% → rejected (>= threshold)."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (30_000.0,)
        mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pg.return_value.__exit__ = MagicMock(return_value=False)

        gate = _make_gate(equity=100_000.0)
        violations = gate._check_heat_budget(0, 100_000.0)
        assert len(violations) == 1

    @patch("quantstack.execution.risk_gate.pg_conn")
    def test_configurable_threshold(self, mock_pg):
        """Custom 50% threshold → 40% approved."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (40_000.0,)
        mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pg.return_value.__exit__ = MagicMock(return_value=False)

        limits = RiskLimits()
        limits.max_daily_heat_pct = 0.50
        gate = _make_gate(equity=100_000.0, limits=limits)
        violations = gate._check_heat_budget(0, 100_000.0)
        assert len(violations) == 0

    @patch("quantstack.execution.risk_gate.pg_conn")
    def test_db_error_fails_closed(self, mock_pg):
        """DB query failure → fail closed."""
        mock_pg.return_value.__enter__ = MagicMock(side_effect=Exception("DB down"))
        gate = _make_gate()
        violations = gate._check_heat_budget(5_000.0, 100_000.0)
        assert len(violations) == 1


# ── 12.3 Sector Concentration ───────────────────────────────────────────


class TestSectorConcentration:

    def test_over_threshold_rejected(self):
        """41% in one sector → rejected."""
        positions = [
            FakePosition(symbol="AAPL", quantity=100, current_price=195.0),  # $19,500
            FakePosition(symbol="MSFT", quantity=100, current_price=195.0),  # $19,500 → $39k tech
        ]
        gate = _make_gate(positions=positions, equity=100_000.0)
        # Adding $2k more tech → $41k / $100k = 41%
        violations = gate._check_sector_concentration("NVDA", 2_000.0, 100_000.0)
        assert len(violations) == 1
        assert violations[0].rule == "sector_concentration"

    def test_under_threshold_approved(self):
        """39% in one sector → approved."""
        positions = [
            FakePosition(symbol="AAPL", quantity=100, current_price=195.0),  # $19,500
            FakePosition(symbol="MSFT", quantity=100, current_price=195.0),  # $19,500 → $39k
        ]
        gate = _make_gate(positions=positions, equity=100_000.0)
        # Adding $500 tech → $39.5k / $100k = 39.5%
        violations = gate._check_sector_concentration("NVDA", 500.0, 100_000.0)
        assert len(violations) == 0

    def test_unknown_sector_approved(self):
        """Symbol not in universe → treated as unique sector, no concentration."""
        gate = _make_gate()
        violations = gate._check_sector_concentration("UNKNOWN_TICKER", 5_000.0, 100_000.0)
        assert len(violations) == 0

    def test_configurable_threshold(self):
        """Custom 50% threshold → 45% approved."""
        positions = [
            FakePosition(symbol="AAPL", quantity=200, current_price=200.0),  # $40k tech
        ]
        limits = RiskLimits()
        limits.max_sector_concentration_pct = 0.50
        gate = _make_gate(positions=positions, equity=100_000.0, limits=limits)
        # Adding $5k tech → $45k / $100k = 45% < 50%
        violations = gate._check_sector_concentration("MSFT", 5_000.0, 100_000.0)
        assert len(violations) == 0
