# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the cross-asset SignalEngine collector.

Tests verify key output contracts — presence of expected signal keys
and value-range invariants — using a mock DataStore.
"""

import asyncio

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from quantstack.signal_engine.collectors.cross_asset import collect_cross_asset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    n: int = 60, seed: int = 42, start_close: float = 100.0
) -> pd.DataFrame:
    """Return a realistic OHLCV DataFrame of length n."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = start_close + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3) + 0.2
    low = close - np.abs(np.random.randn(n) * 0.3) - 0.2
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.randint(500_000, 2_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_store(n: int = 60) -> MagicMock:
    """Return a mock DataStore providing OHLCV for all required tickers."""
    store = MagicMock()

    dfs = {
        "SPY": _make_ohlcv(n, seed=1, start_close=450.0),
        "QQQ": _make_ohlcv(n, seed=2, start_close=380.0),
        "IWM": _make_ohlcv(n, seed=3, start_close=190.0),
        "TLT": _make_ohlcv(n, seed=4, start_close=95.0),
        "GLD": _make_ohlcv(n, seed=5, start_close=185.0),
        "ES=F": _make_ohlcv(n, seed=6, start_close=4510.0),
        "VVIX": _make_ohlcv(n, seed=7, start_close=90.0),
        "AAPL": _make_ohlcv(n, seed=8, start_close=175.0),
    }

    store.load_ohlcv.side_effect = lambda sym, tf: dfs.get(sym)
    return store


# Removed: _run helper is replaced by run_async fixture from conftest.py
def _run_deprecated(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Core contract
# ---------------------------------------------------------------------------


class TestCrossAssetCollectorCore:
    def test_returns_dict(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        assert isinstance(result, dict)

    def test_empty_when_spy_missing(self, run_async):
        store = MagicMock()
        store.load_ohlcv.return_value = None
        result = run_async(collect_cross_asset("AAPL", store))
        assert result == {}

    def test_empty_when_spy_insufficient_bars(self, run_async):
        store = MagicMock()
        store.load_ohlcv.return_value = _make_ohlcv(n=5)  # too few bars
        result = run_async(collect_cross_asset("AAPL", store))
        assert result == {}

    def test_baseline_keys_present(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        for key in ("spy_regime_5d", "risk_on_score", "cross_asset_regime"):
            assert key in result, f"missing baseline key: {key}"


# ---------------------------------------------------------------------------
# ETF return signals
# ---------------------------------------------------------------------------


class TestCrossAssetETFSignals:
    def test_qqq_vs_spy_key_present(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        assert "qqq_vs_spy_5d" in result

    def test_iwm_vs_spy_key_present(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        assert "iwm_vs_spy_5d" in result

    def test_tlt_return_key_present(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        assert "tlt_return_5d" in result

    def test_gld_return_key_present(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        assert "gld_return_5d" in result

    def test_spy_regime_is_float_or_none(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        v = result.get("spy_regime_5d")
        assert v is None or isinstance(v, float)


# ---------------------------------------------------------------------------
# Risk-on score and regime
# ---------------------------------------------------------------------------


class TestCrossAssetRiskScore:
    def test_risk_on_score_in_zero_one(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        score = result.get("risk_on_score")
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_regime_valid_value(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        assert result.get("cross_asset_regime") in ("risk_on", "risk_off", "mixed")

    def test_risk_on_score_is_float(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        assert isinstance(result.get("risk_on_score"), float)


# ---------------------------------------------------------------------------
# SMT Divergence signals
# ---------------------------------------------------------------------------


class TestCrossAssetSMT:
    def test_smt_keys_present(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        assert "smt_bearish" in result
        assert "smt_bullish" in result

    def test_smt_binary_values(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        assert result.get("smt_bearish") in (0, 1)
        assert result.get("smt_bullish") in (0, 1)

    def test_smt_not_simultaneously_bullish_and_bearish(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        # Cannot be both bullish and bearish at the same time
        assert not (result.get("smt_bearish") == 1 and result.get("smt_bullish") == 1)

    def test_smt_strength_key_present(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        assert "smt_strength" in result

    def test_smt_spy_vs_qqq_when_symbol_is_spy(self, run_async):
        """When symbol=SPY, should compare SPY vs QQQ (not SPY vs SPY)."""
        result = run_async(collect_cross_asset("SPY", _make_store()))
        # Should return a non-empty result (no division by zero / self-comparison)
        assert isinstance(result, dict)
        assert "smt_bearish" in result


# ---------------------------------------------------------------------------
# Futures Basis (ES=F vs SPY)
# ---------------------------------------------------------------------------


class TestCrossAssetFuturesBasis:
    def test_es_basis_pct_key_present(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        assert "es_basis_pct" in result

    def test_es_basis_zscore_key_present(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        assert "es_basis_zscore" in result

    def test_es_contango_binary(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        if "es_contango" in result:
            assert result["es_contango"] in (0, 1)

    def test_es_basis_pct_is_float_or_none(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        v = result.get("es_basis_pct")
        assert v is None or isinstance(v, float)

    def test_es_basis_absent_when_es_missing(self, run_async):
        """When ES=F data is unavailable, es_basis_pct should be absent (not crash)."""
        store = _make_store()
        # Override to return None for ES=F only
        dfs = {
            "SPY": _make_ohlcv(60, seed=1, start_close=450.0),
            "QQQ": _make_ohlcv(60, seed=2, start_close=380.0),
            "IWM": _make_ohlcv(60, seed=3, start_close=190.0),
            "TLT": _make_ohlcv(60, seed=4, start_close=95.0),
            "GLD": _make_ohlcv(60, seed=5, start_close=185.0),
            "ES=F": None,
            "VVIX": _make_ohlcv(60, seed=7, start_close=90.0),
            "AAPL": _make_ohlcv(60, seed=8, start_close=175.0),
        }
        store.load_ohlcv.side_effect = lambda sym, tf: dfs.get(sym)
        result = run_async(collect_cross_asset("AAPL", store))
        assert isinstance(result, dict)
        assert "es_basis_pct" not in result
        # Baseline keys should still be present
        assert "spy_regime_5d" in result


# ---------------------------------------------------------------------------
# VVIX signals
# ---------------------------------------------------------------------------


class TestCrossAssetVVIX:
    def test_vvix_key_present(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        assert "vvix" in result

    def test_vvix_above_sma20_binary(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        if "vvix_above_sma20" in result:
            assert result["vvix_above_sma20"] in (0, 1)

    def test_vvix_positive(self, run_async):
        result = run_async(collect_cross_asset("AAPL", _make_store()))
        v = result.get("vvix")
        if v is not None:
            assert v > 0

    def test_vvix_absent_when_missing(self, run_async):
        """When VVIX data is unavailable, vvix should be absent (not crash)."""
        dfs = {
            "SPY": _make_ohlcv(60, seed=1, start_close=450.0),
            "QQQ": _make_ohlcv(60, seed=2, start_close=380.0),
            "IWM": _make_ohlcv(60, seed=3, start_close=190.0),
            "TLT": _make_ohlcv(60, seed=4, start_close=95.0),
            "GLD": _make_ohlcv(60, seed=5, start_close=185.0),
            "ES=F": _make_ohlcv(60, seed=6, start_close=4510.0),
            "VVIX": None,
            "AAPL": _make_ohlcv(60, seed=8, start_close=175.0),
        }
        store = MagicMock()
        store.load_ohlcv.side_effect = lambda sym, tf: dfs.get(sym)
        result = run_async(collect_cross_asset("AAPL", store))
        assert isinstance(result, dict)
        assert "vvix" not in result
        assert "spy_regime_5d" in result

    def test_vvix_absent_when_insufficient_bars(self, run_async):
        """VVIX with fewer than 20 bars should be absent."""
        dfs = {
            "SPY": _make_ohlcv(60, seed=1, start_close=450.0),
            "QQQ": _make_ohlcv(60, seed=2, start_close=380.0),
            "IWM": _make_ohlcv(60, seed=3, start_close=190.0),
            "TLT": _make_ohlcv(60, seed=4, start_close=95.0),
            "GLD": _make_ohlcv(60, seed=5, start_close=185.0),
            "ES=F": _make_ohlcv(60, seed=6, start_close=4510.0),
            "VVIX": _make_ohlcv(10, seed=7, start_close=90.0),  # < 20 bars
            "AAPL": _make_ohlcv(60, seed=8, start_close=175.0),
        }
        store = MagicMock()
        store.load_ohlcv.side_effect = lambda sym, tf: dfs.get(sym)
        result = run_async(collect_cross_asset("AAPL", store))
        assert "vvix" not in result
