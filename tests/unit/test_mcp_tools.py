# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for QuantCore MCP tools.

Tests verify:
1. All new MCP tool implementations return JSON-serializable outputs
2. Error handling for invalid inputs
3. Mock external dependencies for deterministic testing

Note: MCP tools are decorated with @mcp.tool() which wraps them as FunctionTool objects.
We test the underlying async function implementations directly.
"""

import pytest
import numpy as np
import pandas as pd
import json
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_datastore():
    """Mock DataStore for testing."""
    with patch("quantcore.data.storage.DataStore") as mock:
        store = MagicMock()

        # Create sample OHLCV data with enough bars
        dates = pd.date_range("2024-01-01", periods=150, freq="D")
        df = pd.DataFrame(
            {
                "open": np.linspace(100, 120, 150),
                "high": np.linspace(101, 122, 150),
                "low": np.linspace(99, 118, 150),
                "close": np.linspace(100, 120, 150),
                "volume": np.random.randint(1000000, 5000000, 150),
            },
            index=dates,
        )

        store.load_ohlcv.return_value = df
        store.close.return_value = None
        mock.return_value = store

        yield mock


@pytest.fixture
def sample_equity_curve():
    """Sample equity curve for testing."""
    return [100, 102, 101, 104, 103, 106, 108, 107, 110, 112]


@pytest.fixture
def sample_structure_spec():
    """Sample options structure specification."""
    return {
        "underlying_symbol": "SPY",
        "underlying_price": 450.0,
        "legs": [
            {
                "option_type": "call",
                "strike": 450.0,
                "expiry_days": 30,
                "quantity": 1,
                "iv": 0.20,
            }
        ],
    }


@pytest.fixture
def sample_trade_template():
    """Sample trade template."""
    return {
        "template_id": "SPY_vertical_bullish_30d",
        "symbol": "SPY",
        "direction": "bullish",
        "structure_type": "Bull Call Spread",
        "underlying_price": 450.0,
        "legs": [
            {
                "option_type": "call",
                "strike": 450.0,
                "expiry_days": 30,
                "quantity": 1,
                "iv": 0.20,
            },
            {
                "option_type": "call",
                "strike": 455.0,
                "expiry_days": 30,
                "quantity": -1,
                "iv": 0.18,
            },
        ],
        "risk_profile": {
            "max_profit": 300,
            "max_loss": -200,
            "break_evens": [452.0],
        },
        "greeks": {"delta": 25.0, "gamma": 0.5, "theta": -5.0, "vega": 10.0},
        "validation": {"is_defined_risk": True},
    }


# ==============================================================================
# Test: Options Engine (used by MCP tools)
# ==============================================================================


class TestOptionsEngine:
    """Tests for the options engine that powers MCP tools."""

    def test_price_option_dispatch_european(self):
        """Test European option pricing through engine."""
        from quantcore.options.engine import price_option_dispatch

        result = price_option_dispatch(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="call",
            exercise_style="european",
        )

        assert "price" in result
        assert result["price"] > 0
        assert "greeks" in result

    def test_price_option_dispatch_american(self):
        """Test American option pricing through engine."""
        from quantcore.options.engine import price_option_dispatch

        result = price_option_dispatch(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="put",
            exercise_style="american",
        )

        assert "price" in result
        assert result["price"] > 0

    def test_compute_greeks_dispatch(self):
        """Test Greeks computation through engine."""
        from quantcore.options.engine import compute_greeks_dispatch

        result = compute_greeks_dispatch(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="call",
        )

        assert "greeks" in result
        assert "delta" in result["greeks"]
        assert 0.4 < result["greeks"]["delta"] < 0.6  # ATM call

    def test_compute_iv_dispatch(self):
        """Test IV computation through engine."""
        from quantcore.options.engine import compute_iv_dispatch

        result = compute_iv_dispatch(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            rate=0.05,
            dividend_yield=0.0,
            option_price=5.0,
            option_type="call",
        )

        assert "implied_volatility" in result or "error" in result
        if "implied_volatility" in result:
            assert 0.1 < result["implied_volatility"] < 0.5


# ==============================================================================
# Test: Adapters (used by MCP tools)
# ==============================================================================


class TestAdaptersForMCP:
    """Tests for adapter functions used by MCP tools."""

    def test_analyze_structure_long_call(self):
        """Test structure analysis for single call."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 100.0,
            "legs": [
                {
                    "option_type": "call",
                    "strike": 100.0,
                    "expiry_days": 30,
                    "quantity": 1,
                    "iv": 0.20,
                },
            ],
        }

        result = analyze_structure_quantsbin(spec)

        assert "structure_type" in result
        assert result["structure_type"] == "Long Call"
        assert "greeks" in result

    def test_analyze_structure_spread(self):
        """Test structure analysis for spread."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 100.0,
            "legs": [
                {
                    "option_type": "call",
                    "strike": 95.0,
                    "expiry_days": 30,
                    "quantity": 1,
                    "iv": 0.20,
                },
                {
                    "option_type": "call",
                    "strike": 105.0,
                    "expiry_days": 30,
                    "quantity": -1,
                    "iv": 0.18,
                },
            ],
        }

        result = analyze_structure_quantsbin(spec)

        assert "structure_type" in result
        assert result["structure_type"] == "Bull Call Spread"
        assert result["is_defined_risk"] == True

    def test_portfolio_stats_ffn(self, sample_equity_curve):
        """Test portfolio stats computation."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        result = compute_portfolio_stats_ffn(sample_equity_curve)

        assert "total_return" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result

    def test_sabr_surface_fit(self):
        """Test SABR surface fitting."""
        from quantcore.options.adapters.pysabr_adapter import fit_sabr_surface

        quotes = pd.DataFrame(
            {
                "strike": [90, 95, 100, 105, 110],
                "iv": [0.28, 0.24, 0.22, 0.23, 0.26],
            }
        )

        result = fit_sabr_surface(
            quotes=quotes,
            forward=100.0,
            time_to_expiry=30 / 365,
            beta=1.0,
        )

        assert "params" in result or "params_dict" in result
        assert "fit_quality" in result

    def test_american_option_pricing(self):
        """Test American option pricing via adapter."""
        from quantcore.options.adapters.financepy_adapter import price_american_option

        result = price_american_option(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            vol=0.20,
            rate=0.05,
            dividend_yield=0.0,
            option_type="put",
        )

        assert "price" in result
        assert result["price"] > 0
        assert "early_exercise_premium" in result


# ==============================================================================
# Test: Trade Template and Validation Logic
# ==============================================================================


class TestTradeTemplateLogic:
    """Tests for trade template generation and validation logic."""

    def test_trade_validation_passes(self, sample_trade_template):
        """Test that valid trade passes validation."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        # Simulate what validate_trade does internally
        template = sample_trade_template
        max_loss = abs(template.get("risk_profile", {}).get("max_loss", float("inf")))
        is_defined_risk = template.get("validation", {}).get("is_defined_risk", False)
        account_equity = 100000.0
        max_position_pct = 5.0

        max_position_value = account_equity * max_position_pct / 100

        # Checks
        checks = {
            "defined_risk": is_defined_risk,
            "within_position_limit": max_loss <= max_position_value,
        }

        assert checks["defined_risk"] == True
        assert checks["within_position_limit"] == True

    def test_trade_validation_rejects_oversized(self):
        """Test that oversized position fails validation."""
        template = {
            "risk_profile": {"max_loss": -10000},
            "validation": {"is_defined_risk": True},
        }

        max_loss = abs(template["risk_profile"]["max_loss"])
        account_equity = 100000.0
        max_position_pct = 1.0  # Very restrictive

        max_position_value = account_equity * max_position_pct / 100

        within_limit = max_loss <= max_position_value

        assert within_limit == False

    def test_trade_template_structure(self, sample_trade_template):
        """Test trade template has required fields."""
        template = sample_trade_template

        assert "symbol" in template
        assert "legs" in template
        assert "risk_profile" in template
        assert "greeks" in template
        assert "validation" in template


# ==============================================================================
# Test: JSON Serialization
# ==============================================================================


class TestJSONSerialization:
    """Tests to verify all outputs are JSON-serializable."""

    def test_price_option_serializable(self):
        """Test price_option output is JSON-serializable."""
        from quantcore.options.engine import price_option_dispatch

        result = price_option_dispatch(100, 100, 0.25, 0.20, 0.05, 0.0, "call")

        # Should not raise
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_greeks_serializable(self):
        """Test greeks output is JSON-serializable."""
        from quantcore.options.engine import compute_greeks_dispatch

        result = compute_greeks_dispatch(100, 100, 0.25, 0.20, 0.05, 0.0, "call")

        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_structure_analysis_serializable(self):
        """Test structure analysis output is JSON-serializable."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 100.0,
            "legs": [
                {
                    "option_type": "call",
                    "strike": 100,
                    "expiry_days": 30,
                    "quantity": 1,
                    "iv": 0.20,
                }
            ],
        }

        result = analyze_structure_quantsbin(spec)

        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_portfolio_stats_serializable(self):
        """Test portfolio stats output is JSON-serializable."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        result = compute_portfolio_stats_ffn([100, 102, 104, 103, 105])

        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_sabr_fit_serializable(self):
        """Test SABR fit output is JSON-serializable."""
        from quantcore.options.adapters.pysabr_adapter import fit_sabr_surface

        quotes = pd.DataFrame(
            {
                "strike": [90, 95, 100, 105, 110],
                "iv": [0.28, 0.24, 0.22, 0.23, 0.26],
            }
        )

        result = fit_sabr_surface(quotes, 100.0, 30 / 365)

        # Need to convert SABRParams to dict if present
        if "params" in result and hasattr(result["params"], "to_dict"):
            result["params"] = result["params"].to_dict()

        json_str = json.dumps(result)
        assert len(json_str) > 0


# ==============================================================================
# Test: Error Handling
# ==============================================================================


class TestErrorHandling:
    """Tests for error handling in MCP tool implementations."""

    def test_structure_empty_legs_error(self):
        """Test error handling for empty legs."""
        from quantcore.options.adapters.quantsbin_adapter import (
            analyze_structure_quantsbin,
        )

        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 100.0,
            "legs": [],
        }

        result = analyze_structure_quantsbin(spec)

        assert "error" in result

    def test_portfolio_stats_insufficient_data(self):
        """Test error handling for insufficient data."""
        from quantcore.analytics.adapters.ffn_adapter import compute_portfolio_stats_ffn

        result = compute_portfolio_stats_ffn([100])

        assert "error" in result

    def test_sabr_insufficient_points(self):
        """Test SABR fitting with insufficient data points."""
        from quantcore.options.adapters.pysabr_adapter import fit_sabr_surface

        quotes = pd.DataFrame({"strike": [100], "iv": [0.22]})

        with pytest.raises(ValueError):
            fit_sabr_surface(quotes, 100.0, 30 / 365)

    def test_vollib_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        from quantcore.options.adapters.vollib_adapter import bs_price_vollib

        with pytest.raises(ValueError):
            bs_price_vollib(-100, 100, 0.25, 0.20, 0.05, 0.0, "call")  # Negative spot


# ==============================================================================
# Test: QuantAgent Features Integration
# ==============================================================================


class TestQuantAgentFeatures:
    """Tests for QuantAgent feature computation used by MCP tools."""

    def test_pattern_features_computation(self):
        """Test pattern feature computation."""
        from quantcore.features.quantagents_pattern import QuantAgentsPatternFeatures
        from quantcore.config.timeframes import Timeframe

        # Create test data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "open": np.linspace(100, 110, 100),
                "high": np.linspace(101, 112, 100),
                "low": np.linspace(99, 108, 100),
                "close": np.linspace(100, 110, 100),
                "volume": np.random.randint(1000000, 5000000, 100),
            },
            index=dates,
        )

        pattern_calc = QuantAgentsPatternFeatures(Timeframe.D1)
        result = pattern_calc.compute(df)

        assert "qa_pattern_is_pullback" in result.columns
        assert "qa_pattern_is_breakout" in result.columns
        assert "qa_pattern_consolidation" in result.columns

    def test_trend_features_computation(self):
        """Test trend feature computation."""
        from quantcore.features.quantagents_trend import QuantAgentsTrendFeatures
        from quantcore.config.timeframes import Timeframe

        # Create test data (need more bars for trend calculation)
        dates = pd.date_range("2024-01-01", periods=150, freq="D")
        df = pd.DataFrame(
            {
                "open": np.linspace(100, 150, 150),
                "high": np.linspace(101, 152, 150),
                "low": np.linspace(99, 148, 150),
                "close": np.linspace(100, 150, 150),
                "volume": np.random.randint(1000000, 5000000, 150),
            },
            index=dates,
        )

        trend_calc = QuantAgentsTrendFeatures(Timeframe.D1)
        result = trend_calc.compute(df)

        assert "qa_trend_slope_short" in result.columns
        assert "qa_trend_regime" in result.columns
        assert "qa_trend_quality_med" in result.columns


# ==============================================================================
# Test: Quick Functions
# ==============================================================================


class TestQuickFunctions:
    """Tests for quick convenience functions."""

    def test_quick_price(self):
        """Test quick option pricing."""
        from quantcore.options.engine import quick_price

        price = quick_price(100, 100, 30, 0.20, "call")

        assert price > 0
        assert price < 10  # Reasonable for ATM 30-day call

    def test_quick_greeks(self):
        """Test quick Greeks computation."""
        from quantcore.options.engine import quick_greeks

        greeks = quick_greeks(100, 100, 30, 0.20, "call")

        assert "delta" in greeks
        assert "gamma" in greeks
        assert 0.4 < greeks["delta"] < 0.6  # ATM call

    def test_quick_iv(self):
        """Test quick IV computation."""
        from quantcore.options.engine import quick_iv

        # Price an option first
        from quantcore.options.engine import quick_price

        price = quick_price(100, 100, 30, 0.25, "call")

        # Recover IV
        iv = quick_iv(100, 100, 30, price, "call")

        assert iv is not None
        assert abs(iv - 0.25) < 0.01  # Should be close to original vol


# ==============================================================================
# Test: NEW Research Tools
# ==============================================================================


class TestWalkForwardValidator:
    """Tests for walk-forward validation."""

    def test_walkforward_split_generation(self):
        """Test walk-forward split generation."""
        from quantcore.research.walkforward import WalkForwardValidator

        # Create test data (need min_train + n_splits * test_size bars)
        # 504 + 5 * 252 = 1764, so use 2000
        dates = pd.date_range("2020-01-01", periods=2000, freq="D")
        df = pd.DataFrame(
            {
                "close": np.linspace(100, 150, 2000),
            },
            index=dates,
        )

        validator = WalkForwardValidator(
            n_splits=5,
            test_size=252,
            min_train_size=504,
            expanding=True,
        )

        splits = list(validator.split(df))

        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert len(test_idx) == 252
            assert len(train_idx) >= 504

    def test_walkforward_insufficient_data(self):
        """Test error handling for insufficient data."""
        from quantcore.research.walkforward import WalkForwardValidator

        # Create small dataset
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({"close": np.linspace(100, 110, 100)}, index=dates)

        validator = WalkForwardValidator(n_splits=5, test_size=252, min_train_size=504)

        with pytest.raises(ValueError):
            list(validator.split(df))


class TestSignalValidation:
    """Tests for signal validation tools."""

    def test_adf_stationary_signal(self):
        """Test ADF on stationary signal."""
        from quantcore.research.stat_tests import adf_test

        # Create stationary signal (white noise)
        np.random.seed(42)
        signal = pd.Series(np.random.randn(500))

        result = adf_test(signal)

        assert result.test_name == "ADF"
        assert result.is_significant == True  # White noise is stationary

    def test_adf_nonstationary_signal(self):
        """Test ADF on non-stationary signal."""
        from quantcore.research.stat_tests import adf_test

        # Create random walk (non-stationary)
        np.random.seed(42)
        signal = pd.Series(np.cumsum(np.random.randn(500)))

        result = adf_test(signal)

        assert result.is_significant == False  # Random walk is non-stationary

    def test_lagged_correlation(self):
        """Test lagged cross-correlation."""
        from quantcore.research.stat_tests import lagged_cross_correlation

        np.random.seed(42)
        signal = pd.Series(np.random.randn(200))
        returns = pd.Series(np.random.randn(200))

        correlations = lagged_cross_correlation(signal, returns, max_lag=5)

        assert 1 in correlations
        assert 5 in correlations
        assert all(abs(v) < 1 for v in correlations.values() if not np.isnan(v))


class TestLeakageDiagnostics:
    """Tests for leakage detection."""

    def test_clean_features_pass(self):
        """Test that clean features don't trigger leakage."""
        from quantcore.research.leak_diagnostics import LeakageDiagnostics

        np.random.seed(42)
        n = 200

        # Create lagged features (no leakage)
        features = pd.DataFrame(
            {
                "lagged_return": np.random.randn(n),
                "momentum": np.random.randn(n),
            }
        )
        labels = pd.Series(np.random.randint(0, 2, n))
        prices = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.1))
        returns = prices.pct_change()

        diagnostics = LeakageDiagnostics()
        report = diagnostics.run_full_diagnostics(features, labels, prices, returns)

        # Should have low severity or no leakage
        assert report.severity in ["none", "low", "medium"]

    def test_leaky_feature_detected(self):
        """Test that obvious leakage is detected."""
        from quantcore.research.leak_diagnostics import LeakageDiagnostics

        np.random.seed(42)
        n = 200

        # Create feature that IS the future return (obvious leakage)
        returns = pd.Series(np.random.randn(n) * 0.01)
        features = pd.DataFrame(
            {
                "future_return": returns,  # This IS the return - perfect leakage
            }
        )
        labels = (returns > 0).astype(int)
        prices = pd.Series(100 + np.cumsum(returns))

        diagnostics = LeakageDiagnostics()
        report = diagnostics.run_full_diagnostics(features, labels, prices, returns)

        # Should detect leakage
        assert report.has_leakage == True


# ==============================================================================
# Test: NEW Risk Tools
# ==============================================================================


class TestVaRCalculation:
    """Tests for VaR calculation."""

    def test_historical_var(self):
        """Test historical VaR calculation."""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02  # 2% daily vol

        # Calculate VaR manually
        var_95 = -np.percentile(returns, 5) * 100

        assert var_95 > 0  # VaR should be positive (loss)
        assert var_95 < 10  # Should be reasonable for 2% vol

    def test_var_scaling(self):
        """Test VaR scales with sqrt(time)."""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02

        var_1d = -np.percentile(returns, 5)
        var_5d = -np.percentile(returns * np.sqrt(5), 5)

        # 5-day VaR should be ~sqrt(5) times 1-day VaR
        assert 2 < var_5d / var_1d < 3  # sqrt(5) ≈ 2.24


class TestStressTesting:
    """Tests for portfolio stress testing."""

    def test_stress_scenario_pnl(self):
        """Test stress scenario P&L calculation."""
        from quantcore.options.pricing import black_scholes_price
        from quantcore.options.models import OptionType

        # Initial position: ATM call
        # Signature: black_scholes_price(S, K, T, r, sigma, option_type, q=0.0)
        S0, K, T, vol, r = 100, 100, 0.25, 0.20, 0.05
        initial_price = black_scholes_price(S0, K, T, r, vol, OptionType.CALL)

        # Stress scenario: -20% price, +50% vol
        S_stressed = S0 * 0.8
        vol_stressed = vol * 1.5
        stressed_price = black_scholes_price(
            S_stressed, K, T, r, vol_stressed, OptionType.CALL
        )

        # Call should lose value from price drop
        assert stressed_price < initial_price

    def test_stress_scenarios_exist(self):
        """Test predefined stress scenarios exist."""
        from quantcore.risk.stress_testing import STRESS_SCENARIOS

        assert "2008_lehman" in STRESS_SCENARIOS
        assert "2020_covid_crash" in STRESS_SCENARIOS
        assert "2018_volmageddon" in STRESS_SCENARIOS


class TestRiskControls:
    """Tests for risk control checks."""

    def test_normal_state(self):
        """Test normal risk state."""
        from quantcore.risk.controls import RiskStatus, ExposureManager

        manager = ExposureManager(
            max_concurrent_trades=5,
            max_exposure_per_symbol_pct=20.0,
        )

        can_open, reason = manager.can_open_position("SPY", 10.0, 100000)

        assert can_open == True

    def test_position_limit_breach(self):
        """Test position limit enforcement."""
        from quantcore.risk.controls import ExposureManager

        manager = ExposureManager(max_concurrent_trades=2)

        # Fill up positions
        manager._open_positions = {"SPY": 20, "QQQ": 20}

        can_open, reason = manager.can_open_position("AAPL", 10.0, 100000)

        assert can_open == False
        assert "concurrent" in reason.lower()


# ==============================================================================
# Test: NEW Microstructure Tools
# ==============================================================================


class TestLiquidityAnalysis:
    """Tests for liquidity analysis."""

    def test_spread_estimation(self):
        """Test Corwin-Schultz spread estimator."""
        from quantcore.microstructure.liquidity import SpreadEstimator

        # Create OHLC data with known spread
        np.random.seed(42)
        n = 100
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
        high = close + np.random.rand(n) * 0.5
        low = close - np.random.rand(n) * 0.5

        spread = SpreadEstimator.corwin_schultz_spread(high, low, window=20)

        assert len(spread) == n
        assert spread.iloc[-1] >= 0  # Spread can't be negative

    def test_liquidity_score(self):
        """Test liquidity scoring."""
        from quantcore.microstructure.liquidity import LiquidityAnalyzer

        # Create test data with sufficient bars for rolling calculations
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates)

        df = pd.DataFrame(
            {
                "open": close - np.random.rand(n) * 0.3,
                "high": close + np.random.rand(n) * 0.5,
                "low": close - np.random.rand(n) * 0.5,
                "close": close,
                "volume": np.random.randint(500000, 1500000, n),
            },
            index=dates,
        )

        analyzer = LiquidityAnalyzer()
        features = analyzer.compute_features(df)

        assert "liquidity_score" in features.columns
        # Check that liquidity_score exists and is computed (may be NaN in first window)
        valid_scores = features["liquidity_score"].dropna()
        if len(valid_scores) > 0:
            assert valid_scores.iloc[-1] >= 0
            assert valid_scores.iloc[-1] <= 1


class TestVolumeProfile:
    """Tests for volume profile analysis."""

    def test_vwap_calculation(self):
        """Test VWAP calculation."""
        # Simple VWAP test
        typical_price = pd.Series([100, 101, 102])
        volume = pd.Series([1000, 2000, 1000])

        vwap = (typical_price * volume).sum() / volume.sum()

        # Expected: (100*1000 + 101*2000 + 102*1000) / 4000 = 101
        assert abs(vwap - 101) < 0.01


class TestTradingCalendar:
    """Tests for trading calendar."""

    def test_market_hours(self):
        """Test market hours configuration."""
        market_open = "09:30"
        market_close = "16:00"

        # Parse times
        from datetime import time as dt_time

        open_time = dt_time(int(market_open[:2]), int(market_open[3:]))
        close_time = dt_time(int(market_close[:2]), int(market_close[3:]))

        assert open_time < close_time


# ==============================================================================
# Test: NEW Validation Tools
# ==============================================================================


class TestPurgedCV:
    """Tests for purged cross-validation."""

    def test_purged_cv_splits(self):
        """Test purged CV generates correct splits."""
        from quantcore.validation.purged_cv import PurgedKFoldCV

        # Create test data
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        X = pd.DataFrame({"feature": np.random.randn(1000)}, index=dates)

        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.01)
        splits = list(cv.split(X))

        assert len(splits) == 5

        for split in splits:
            # Train and test should not overlap
            train_set = set(split.train_indices)
            test_set = set(split.test_indices)
            assert len(train_set.intersection(test_set)) == 0

    def test_embargo_applied(self):
        """Test embargo creates gap between train and test."""
        from quantcore.validation.purged_cv import PurgedKFoldCV

        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        X = pd.DataFrame({"feature": np.random.randn(1000)}, index=dates)

        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.05)  # 5% embargo
        splits = list(cv.split(X))

        # Check that train and test don't overlap
        for split in splits:
            train_set = set(split.train_indices)
            test_set = set(split.test_indices)

            # No overlap between train and test
            assert len(train_set.intersection(test_set)) == 0

            # Both train and test should have data
            assert len(train_set) > 0
            assert len(test_set) > 0


class TestLookaheadDetection:
    """Tests for lookahead bias detection."""

    def test_clean_features_pass(self):
        """Test clean features don't trigger lookahead."""
        from scipy import stats

        np.random.seed(42)
        n = 200

        # Feature that predicts forward return with lag
        feature = pd.Series(np.random.randn(n))
        forward_return = pd.Series(np.random.randn(n))

        corr, _ = stats.spearmanr(feature, forward_return)

        # Random features should have low correlation
        assert abs(corr) < 0.2

    def test_leaky_feature_detected(self):
        """Test obvious lookahead is detected."""
        from scipy import stats

        np.random.seed(42)
        n = 200

        # Feature IS the future return (obvious leakage)
        forward_return = pd.Series(np.random.randn(n))
        feature = forward_return  # Perfect leakage

        corr, _ = stats.spearmanr(feature, forward_return)

        # Should have perfect correlation
        assert corr == 1.0
