"""Tests for quantstack.finrl.features.RLFeatureExtractor."""

import numpy as np
import pytest

from quantstack.finrl.features import RLFeatureExtractor


class TestExecutionFeatures:
    def test_shape(self):
        feats = RLFeatureExtractor.execution_features(
            remaining_qty=500, total_qty=1000,
            remaining_time=10, time_horizon=20,
            current_price=100, arrival_price=99.5,
            spread_bps=5, volatility=0.02,
            volume_ratio=1.2, vwap=99.8, shortfall=0.001,
        )
        assert feats.shape == (8,)
        assert feats.dtype == np.float32

    def test_values_clipped(self):
        feats = RLFeatureExtractor.execution_features(
            remaining_qty=2000, total_qty=1000,  # >1 ratio
            remaining_time=10, time_horizon=20,
            current_price=200, arrival_price=100,  # huge deviation
            spread_bps=5, volatility=0.5,  # huge vol
            volume_ratio=20, vwap=100, shortfall=0.5,
        )
        assert feats[0] <= 1.0  # qty_frac clipped
        assert feats[2] <= 0.1 + 1e-7  # price_dev clipped to [-0.1, 0.1]
        assert feats[4] <= 0.2 + 1e-7  # volatility clipped to [0, 0.2]

    def test_zero_vwap(self):
        feats = RLFeatureExtractor.execution_features(
            remaining_qty=500, total_qty=1000,
            remaining_time=10, time_horizon=20,
            current_price=100, arrival_price=99.5,
            spread_bps=5, volatility=0.02,
            volume_ratio=1.0, vwap=0.0, shortfall=0.0,
        )
        assert feats[6] == 0.0  # vwap_dev when vwap=0


class TestSizingFeatures:
    def test_shape(self):
        feats = RLFeatureExtractor.sizing_features(
            signal_confidence=0.7, signal_direction="LONG",
            returns_window=[0.01, -0.005, 0.003, 0.002, -0.001] * 4,
            current_position_pct=0.05, drawdown=0.02,
            risk_budget_used=0.4, time_since_trade=3,
            regime_label="trending_up", win_rate=0.55, rolling_sharpe=0.8,
        )
        assert feats.shape == (10,)
        assert feats.dtype == np.float32

    def test_direction_encoding(self):
        long = RLFeatureExtractor.sizing_features(
            signal_confidence=0.5, signal_direction="LONG",
            returns_window=[0.01] * 5, current_position_pct=0, drawdown=0,
            risk_budget_used=0, time_since_trade=0, regime_label="normal",
            win_rate=0.5, rolling_sharpe=0,
        )
        short = RLFeatureExtractor.sizing_features(
            signal_confidence=0.5, signal_direction="SHORT",
            returns_window=[0.01] * 5, current_position_pct=0, drawdown=0,
            risk_budget_used=0, time_since_trade=0, regime_label="normal",
            win_rate=0.5, rolling_sharpe=0,
        )
        assert long[1] == 1.0
        assert short[1] == -1.0


class TestAlphaSelectionFeatures:
    def test_shape(self):
        names = ["alpha_a", "alpha_b"]
        returns = {"alpha_a": [0.01] * 25, "alpha_b": [-0.005] * 25}
        alignments = {"alpha_a": 0.8, "alpha_b": 0.3}

        feats = RLFeatureExtractor.alpha_selection_features(
            regime_idx=0, alpha_names=names,
            alpha_returns_history=returns,
            alpha_regime_alignments=alignments,
            market_volatility=0.3, vix_normalized=0.4,
        )
        expected = 4 + 4 * 2 + 4
        assert feats.shape == (expected,)

    def test_regime_one_hot(self):
        feats = RLFeatureExtractor.alpha_selection_features(
            regime_idx=2, alpha_names=["a"],
            alpha_returns_history={"a": []},
            alpha_regime_alignments={"a": 0.5},
            market_volatility=0.3, vix_normalized=0.4,
        )
        assert feats[2] == 1.0  # index 2 should be hot
        assert feats[0] == 0.0
        assert feats[1] == 0.0
        assert feats[3] == 0.0

    def test_expected_dims(self):
        dims = RLFeatureExtractor.expected_dims(n_alphas=7)
        assert dims["execution"] == 8
        assert dims["sizing"] == 10
        assert dims["alpha_selection"] == 4 + 28 + 4
