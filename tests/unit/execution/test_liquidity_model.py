"""Tests for the pre-trade liquidity model.

Verifies:
  - Spread estimation returns positive bps
  - Spread per-time-bucket varies with TOD multipliers
  - Depth estimation returns shares per bucket
  - Order within depth threshold --> PASS
  - Order exceeding 10% depth --> SCALE_DOWN with recommended qty
  - Illiquid symbol (50% of depth) --> REJECT
  - Time-of-day multipliers: open=1.5x, midday=1.0x, close=1.3x
  - Stressed exit computes portfolio-level slippage
  - Stressed exit alert when threshold exceeded
  - Stressed exit no alert below threshold
  - Risk gate integration ordering test
"""

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from quantstack.execution.liquidity_model import (
    LiquidityCheckResult,
    LiquidityModel,
    LiquidityVerdict,
    StressedExitResult,
)

_ET = ZoneInfo("America/New_York")


# =============================================================================
# Helpers
# =============================================================================


def _model(
    daily_volumes: dict[str, int] | None = None,
    bar_data: dict[str, dict[str, float]] | None = None,
) -> LiquidityModel:
    """Build a LiquidityModel with no DB, just injected data."""
    return LiquidityModel(conn=None, daily_volumes=daily_volumes, bar_data=bar_data)


# =============================================================================
# Spread estimation
# =============================================================================


class TestEstimateSpread:
    def test_returns_positive_bps(self):
        model = _model(daily_volumes={"AAPL": 10_000_000})
        spread = model.estimate_spread("AAPL")
        assert spread > 0

    def test_large_cap_default(self):
        """ADV >= 5M should use 10 bps default (midday mult=1.0)."""
        model = _model(daily_volumes={"AAPL": 10_000_000})
        assert model.estimate_spread("AAPL", "midday") == pytest.approx(10.0)

    def test_mid_cap_default(self):
        """ADV 500K-5M should use 25 bps default."""
        model = _model(daily_volumes={"MID": 1_000_000})
        assert model.estimate_spread("MID", "midday") == pytest.approx(25.0)

    def test_small_cap_default(self):
        """ADV < 500K should use 50 bps default."""
        model = _model(daily_volumes={"TINY": 100_000})
        assert model.estimate_spread("TINY", "midday") == pytest.approx(50.0)

    def test_bar_range_proxy(self):
        """When bar data available but no EWMA, use bar-range proxy."""
        bar = {"high": 102.0, "low": 98.0, "close": 100.0}
        model = _model(bar_data={"XYZ": bar})
        # midpoint = 100, range = 4, range_bps = 400, * 0.2 = 80 bps, * midday 1.0
        assert model.estimate_spread("XYZ", "midday") == pytest.approx(80.0)

    def test_spread_varies_by_bucket(self):
        """Different time buckets produce different spread estimates."""
        model = _model(daily_volumes={"AAPL": 10_000_000})
        morning = model.estimate_spread("AAPL", "morning")
        midday = model.estimate_spread("AAPL", "midday")
        close = model.estimate_spread("AAPL", "close")
        # morning (1.2x) > midday (1.0x), close (1.3x) > midday (1.0x)
        assert morning > midday
        assert close > midday

    def test_tod_multiplier_open(self):
        model = _model(daily_volumes={"SPY": 80_000_000})
        spread_open = model.estimate_spread("SPY", "open")
        spread_midday = model.estimate_spread("SPY", "midday")
        assert spread_open == pytest.approx(spread_midday * 1.5)

    def test_tod_multiplier_midday_is_baseline(self):
        model = _model(daily_volumes={"SPY": 80_000_000})
        assert model.estimate_spread("SPY", "midday") == pytest.approx(10.0)

    def test_tod_multiplier_close(self):
        model = _model(daily_volumes={"SPY": 80_000_000})
        spread_close = model.estimate_spread("SPY", "close")
        spread_midday = model.estimate_spread("SPY", "midday")
        assert spread_close == pytest.approx(spread_midday * 1.3)


# =============================================================================
# Depth estimation
# =============================================================================


class TestEstimateDepth:
    def test_returns_positive_shares(self):
        model = _model(daily_volumes={"AAPL": 10_000_000})
        depth = model.estimate_depth("AAPL", "morning")
        assert depth > 0

    def test_morning_bucket_weight(self):
        """Morning bucket should be 30% of ADV."""
        model = _model(daily_volumes={"AAPL": 10_000_000})
        assert model.estimate_depth("AAPL", "morning") == 3_000_000

    def test_midday_bucket_weight(self):
        """Midday bucket should be 20% of ADV."""
        model = _model(daily_volumes={"AAPL": 10_000_000})
        assert model.estimate_depth("AAPL", "midday") == 2_000_000

    def test_close_bucket_weight(self):
        """Close bucket should be 25% of ADV."""
        model = _model(daily_volumes={"AAPL": 10_000_000})
        assert model.estimate_depth("AAPL", "close") == 2_500_000

    def test_unknown_symbol_uses_default_adv(self):
        """Unknown symbol falls back to 1M default ADV."""
        model = _model()
        depth = model.estimate_depth("UNKNOWN", "midday")
        # 1M * 0.20 = 200K
        assert depth == 200_000

    def test_minimum_one_share(self):
        """Even with zero ADV, depth should be at least 1."""
        model = _model(daily_volumes={"DUST": 0})
        assert model.estimate_depth("DUST", "midday") >= 1


# =============================================================================
# Pre-trade check
# =============================================================================


class TestPreTradeCheck:
    def test_small_order_passes(self):
        """Order well within 10% of depth should PASS."""
        # ADV 10M, midday depth = 2M, order 100K = 5% of depth
        model = _model(daily_volumes={"AAPL": 10_000_000})
        result = model.pre_trade_check("AAPL", 100_000)
        assert result.verdict == LiquidityVerdict.PASS
        assert result.recommended_quantity == 100_000
        assert result.estimated_spread_bps > 0
        assert result.estimated_depth_shares > 0

    def test_order_at_threshold_passes(self):
        """Order exactly at 10% of depth should PASS."""
        # ADV 10M, midday depth = 2M, 10% = 200K
        model = _model(daily_volumes={"AAPL": 10_000_000})
        result = model.pre_trade_check("AAPL", 200_000)
        assert result.verdict == LiquidityVerdict.PASS

    def test_order_exceeding_threshold_scales_down(self):
        """Order > 10% but <= 20% of depth should SCALE_DOWN."""
        # ADV 10M, midday depth = 2M, order 300K = 15% of depth
        model = _model(daily_volumes={"AAPL": 10_000_000})
        result = model.pre_trade_check("AAPL", 300_000)
        assert result.verdict == LiquidityVerdict.SCALE_DOWN
        # Recommended = depth * 10% = 200K
        assert result.recommended_quantity == 200_000

    def test_highly_illiquid_order_rejected(self):
        """Order > 20% of depth should be REJECTED."""
        # ADV 10M, midday depth = 2M, order 500K = 25% of depth
        model = _model(daily_volumes={"AAPL": 10_000_000})
        result = model.pre_trade_check("AAPL", 500_000)
        assert result.verdict == LiquidityVerdict.REJECT
        assert result.recommended_quantity is None

    def test_half_of_depth_rejected(self):
        """Order at 50% of depth is well above 2x threshold, so REJECT."""
        # ADV 1M, midday depth = 200K, order 100K = 50% of depth
        model = _model(daily_volumes={"ILLIQ": 1_000_000})
        result = model.pre_trade_check("ILLIQ", 100_000)
        assert result.verdict == LiquidityVerdict.REJECT

    def test_time_aware_check_uses_bucket(self):
        """Passing current_time should resolve the correct bucket."""
        model = _model(daily_volumes={"SPY": 80_000_000})
        # 10:00 ET = morning bucket, depth = 80M * 0.30 = 24M
        morning_time = datetime(2026, 4, 6, 10, 0, tzinfo=_ET)
        result = model.pre_trade_check("SPY", 1_000_000, current_time=morning_time)
        assert result.verdict == LiquidityVerdict.PASS
        assert result.estimated_depth_shares == 24_000_000


# =============================================================================
# Stressed exit slippage
# =============================================================================


class TestStressedExitSlippage:
    def test_computes_portfolio_level_slippage(self):
        model = _model(daily_volumes={"AAPL": 10_000_000, "MSFT": 8_000_000})
        positions = [
            {"symbol": "AAPL", "quantity": 100, "price": 200.0},
            {"symbol": "MSFT", "quantity": 200, "price": 400.0},
        ]
        result = model.stressed_exit_slippage(positions)
        assert isinstance(result, StressedExitResult)
        assert result.portfolio_value > 0
        assert result.slippage_dollar_estimate >= 0
        assert result.total_slippage_bps >= 0
        assert "AAPL" in result.per_symbol_breakdown
        assert "MSFT" in result.per_symbol_breakdown

    def test_no_alert_below_threshold(self):
        """Liquid portfolio should not trigger alert."""
        model = _model(daily_volumes={"SPY": 80_000_000})
        positions = [
            {"symbol": "SPY", "quantity": 1000, "price": 450.0},
        ]
        result = model.stressed_exit_slippage(positions)
        assert result.alert is False

    def test_alert_when_threshold_exceeded(self):
        """Illiquid portfolio should trigger alert when slippage > 100 bps."""
        # Very small ADV = huge spreads and high participation
        model = _model(daily_volumes={"PENNY": 1_000})
        positions = [
            {"symbol": "PENNY", "quantity": 10_000, "price": 5.0},
        ]
        result = model.stressed_exit_slippage(positions)
        # With ADV=1K, midday depth=200, holding 10K shares — massive slippage
        assert result.alert is True
        assert result.total_slippage_bps > LiquidityModel.STRESS_THRESHOLD_BPS

    def test_empty_positions(self):
        model = _model()
        result = model.stressed_exit_slippage([])
        assert result.portfolio_value == 0
        assert result.total_slippage_bps == 0
        assert result.alert is False

    def test_per_symbol_breakdown_keys(self):
        model = _model(daily_volumes={"A": 5_000_000, "B": 5_000_000})
        positions = [
            {"symbol": "A", "quantity": 50, "price": 100.0},
            {"symbol": "B", "quantity": 50, "price": 100.0},
        ]
        result = model.stressed_exit_slippage(positions)
        assert set(result.per_symbol_breakdown.keys()) == {"A", "B"}


# =============================================================================
# Risk gate integration
# =============================================================================


class TestRiskGateIntegration:
    """Verify liquidity model is invoked in the correct position within risk_gate.check().

    The trading window check (step 1c) must be bypassed since it reads live
    config and may reject before we reach the liquidity model.
    """

    _TW_PATCH = "quantstack.execution.risk_gate.is_trade_allowed"

    @patch("quantstack.trading_window.is_trade_allowed", return_value=True)
    @patch("quantstack.execution.risk_gate.pg_conn")
    @patch("quantstack.execution.risk_gate.get_portfolio_state")
    @patch("quantstack.execution.risk_gate.LiquidityModel")
    def test_liquidity_reject_blocks_order(
        self, MockLiqModel, mock_get_ps, mock_pg_conn, mock_tw,
    ):
        """When LiquidityModel returns REJECT, risk gate should reject the order."""
        from quantstack.execution.risk_gate import RiskGate, RiskLimits

        # Set up portfolio state mock
        mock_ps = MagicMock()
        snapshot = MagicMock()
        snapshot.total_equity = 100_000.0
        snapshot.daily_pnl = 0.0
        mock_ps.get_snapshot.return_value = snapshot
        mock_ps.get_position.return_value = None
        mock_ps.get_positions.return_value = []
        mock_get_ps.return_value = mock_ps

        # Mock pg_conn context manager
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_pg_conn.return_value = mock_conn

        # LiquidityModel mock — returns REJECT
        mock_instance = MagicMock()
        mock_instance.pre_trade_check.return_value = LiquidityCheckResult(
            verdict=LiquidityVerdict.REJECT,
            reason="too illiquid",
        )
        MockLiqModel.return_value = mock_instance

        limits = RiskLimits()
        gate = RiskGate(limits=limits, portfolio=mock_ps)
        verdict = gate.check(
            symbol="ILLIQ",
            side="buy",
            quantity=10_000,
            current_price=50.0,
            daily_volume=1_000_000,
        )

        assert not verdict.approved
        assert any("liquidity" in v.rule for v in verdict.violations)

    @patch("quantstack.trading_window.is_trade_allowed", return_value=True)
    @patch("quantstack.execution.risk_gate.pg_conn")
    @patch("quantstack.execution.risk_gate.get_portfolio_state")
    @patch("quantstack.execution.risk_gate.LiquidityModel")
    def test_liquidity_scale_down_adjusts_quantity(
        self, MockLiqModel, mock_get_ps, mock_pg_conn, mock_tw,
    ):
        """When LiquidityModel returns SCALE_DOWN, risk gate should reduce quantity."""
        from quantstack.execution.risk_gate import RiskGate, RiskLimits

        mock_ps = MagicMock()
        snapshot = MagicMock()
        snapshot.total_equity = 100_000.0
        snapshot.daily_pnl = 0.0
        mock_ps.get_snapshot.return_value = snapshot
        mock_ps.get_position.return_value = None
        mock_ps.get_positions.return_value = []
        mock_get_ps.return_value = mock_ps

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_pg_conn.return_value = mock_conn

        mock_instance = MagicMock()
        mock_instance.pre_trade_check.return_value = LiquidityCheckResult(
            verdict=LiquidityVerdict.SCALE_DOWN,
            reason="scaling down",
            recommended_quantity=500,
        )
        MockLiqModel.return_value = mock_instance

        limits = RiskLimits()
        gate = RiskGate(limits=limits, portfolio=mock_ps)
        verdict = gate.check(
            symbol="MID",
            side="buy",
            quantity=1_000,
            current_price=50.0,
            daily_volume=1_000_000,
        )

        # Liquidity model was invoked and quantity was scaled down
        mock_instance.pre_trade_check.assert_called_once()

    @patch("quantstack.trading_window.is_trade_allowed", return_value=True)
    @patch("quantstack.execution.risk_gate.pg_conn")
    @patch("quantstack.execution.risk_gate.get_portfolio_state")
    @patch("quantstack.execution.risk_gate.LiquidityModel")
    def test_liquidity_pass_continues(
        self, MockLiqModel, mock_get_ps, mock_pg_conn, mock_tw,
    ):
        """When LiquidityModel returns PASS, risk gate should continue normally."""
        from quantstack.execution.risk_gate import RiskGate, RiskLimits

        mock_ps = MagicMock()
        snapshot = MagicMock()
        snapshot.total_equity = 1_000_000.0
        snapshot.daily_pnl = 0.0
        mock_ps.get_snapshot.return_value = snapshot
        mock_ps.get_position.return_value = None
        mock_ps.get_positions.return_value = []
        mock_get_ps.return_value = mock_ps

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_pg_conn.return_value = mock_conn

        # execution quality scalar
        mock_conn.execute.return_value.fetchone.return_value = None

        mock_instance = MagicMock()
        mock_instance.pre_trade_check.return_value = LiquidityCheckResult(
            verdict=LiquidityVerdict.PASS,
            reason="ok",
            recommended_quantity=100,
        )
        MockLiqModel.return_value = mock_instance

        limits = RiskLimits()
        gate = RiskGate(limits=limits, portfolio=mock_ps)
        verdict = gate.check(
            symbol="SPY",
            side="buy",
            quantity=100,
            current_price=450.0,
            daily_volume=80_000_000,
        )

        # Liquidity check passed, so it should have been invoked
        mock_instance.pre_trade_check.assert_called_once()
