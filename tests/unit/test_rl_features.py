# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for RLFeatureExtractor.

Critical invariant being tested: the feature vectors produced at training time
(in environment._get_state()) and at inference time (in rl_tools._run()) must
be byte-for-byte identical when given the same inputs. This module tests the
canonical extractor in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.rl.features import RLFeatureExtractor


class TestExecutionFeatures:
    def test_output_shape(self):
        feats = RLFeatureExtractor.execution_features(
            remaining_qty=500,
            total_qty=1000,
            remaining_time=10,
            time_horizon=20,
            current_price=100.0,
            arrival_price=100.0,
            spread_bps=5.0,
            volatility=0.02,
            volume_ratio=1.0,
            vwap=100.0,
            shortfall=0.0,
        )
        assert feats.shape == (8,)
        assert feats.dtype == np.float32

    def test_qty_fraction(self):
        feats = RLFeatureExtractor.execution_features(
            remaining_qty=300,
            total_qty=1000,
            remaining_time=5,
            time_horizon=10,
            current_price=100.0,
            arrival_price=100.0,
            spread_bps=5.0,
            volatility=0.02,
            volume_ratio=1.0,
            vwap=100.0,
            shortfall=0.0,
        )
        assert abs(feats[0] - 0.3) < 1e-5  # remaining_qty / total_qty

    def test_time_fraction(self):
        feats = RLFeatureExtractor.execution_features(
            remaining_qty=500,
            total_qty=1000,
            remaining_time=5,
            time_horizon=10,
            current_price=100.0,
            arrival_price=100.0,
            spread_bps=5.0,
            volatility=0.02,
            volume_ratio=1.0,
            vwap=100.0,
            shortfall=0.0,
        )
        assert abs(feats[1] - 0.5) < 1e-5  # remaining_time / time_horizon

    def test_price_drift_zero_when_equal(self):
        feats = RLFeatureExtractor.execution_features(
            remaining_qty=500,
            total_qty=1000,
            remaining_time=5,
            time_horizon=10,
            current_price=100.0,
            arrival_price=100.0,
            spread_bps=5.0,
            volatility=0.02,
            volume_ratio=1.0,
            vwap=100.0,
            shortfall=0.0,
        )
        assert abs(feats[2]) < 1e-6  # (current - arrival) / arrival

    def test_div_by_zero_guard_total_qty(self):
        feats = RLFeatureExtractor.execution_features(
            remaining_qty=0,
            total_qty=0,  # pathological
            remaining_time=5,
            time_horizon=10,
            current_price=100.0,
            arrival_price=100.0,
            spread_bps=5.0,
            volatility=0.02,
            volume_ratio=1.0,
            vwap=100.0,
            shortfall=0.0,
        )
        assert np.isfinite(feats).all()

    def test_spread_bps_normalised_and_clipped(self):
        # spread is clipped to [0.0, 0.005] in the implementation
        feats_large = RLFeatureExtractor.execution_features(
            remaining_qty=500,
            total_qty=1000,
            remaining_time=5,
            time_horizon=10,
            current_price=100.0,
            arrival_price=100.0,
            spread_bps=10_000.0,  # large → clipped to 0.005
            volatility=0.02,
            volume_ratio=1.0,
            vwap=100.0,
            shortfall=0.0,
        )
        assert abs(feats_large[3] - 0.005) < 1e-6  # clipped at 0.005

        feats_small = RLFeatureExtractor.execution_features(
            remaining_qty=500,
            total_qty=1000,
            remaining_time=5,
            time_horizon=10,
            current_price=100.0,
            arrival_price=100.0,
            spread_bps=10.0,  # 10 bps → 0.001
            volatility=0.02,
            volume_ratio=1.0,
            vwap=100.0,
            shortfall=0.0,
        )
        assert abs(feats_small[3] - 0.001) < 1e-6


class TestSizingFeatures:
    def test_output_shape(self):
        feats = RLFeatureExtractor.sizing_features(
            signal_confidence=0.7,
            signal_direction="LONG",
            returns_window=[0.01, -0.005, 0.02],
            current_position_pct=0.5,
            drawdown=0.05,
            risk_budget_used=0.3,
            time_since_trade=3,
            regime_label="normal",
            win_rate=0.55,
            rolling_sharpe=0.8,
        )
        assert feats.shape == (10,)
        assert feats.dtype == np.float32

    def test_long_direction_encodes_plus_one(self):
        feats = RLFeatureExtractor.sizing_features(
            signal_confidence=0.7,
            signal_direction="LONG",
            returns_window=[],
            current_position_pct=0.0,
            drawdown=0.0,
            risk_budget_used=0.0,
            time_since_trade=0,
            regime_label="normal",
            win_rate=0.5,
            rolling_sharpe=0.0,
        )
        assert feats[1] == 1.0

    def test_short_direction_encodes_minus_one(self):
        feats = RLFeatureExtractor.sizing_features(
            signal_confidence=0.7,
            signal_direction="SHORT",
            returns_window=[],
            current_position_pct=0.0,
            drawdown=0.0,
            risk_budget_used=0.0,
            time_since_trade=0,
            regime_label="normal",
            win_rate=0.5,
            rolling_sharpe=0.0,
        )
        assert feats[1] == -1.0

    def test_neutral_direction_encodes_zero(self):
        feats = RLFeatureExtractor.sizing_features(
            signal_confidence=0.7,
            signal_direction="NEUTRAL",
            returns_window=[],
            current_position_pct=0.0,
            drawdown=0.0,
            risk_budget_used=0.0,
            time_since_trade=0,
            regime_label="normal",
            win_rate=0.5,
            rolling_sharpe=0.0,
        )
        assert feats[1] == 0.0

    def test_confidence_clipped_to_01(self):
        feats_hi = RLFeatureExtractor.sizing_features(
            signal_confidence=2.0,  # out of range
            signal_direction="LONG",
            returns_window=[],
            current_position_pct=0.0,
            drawdown=0.0,
            risk_budget_used=0.0,
            time_since_trade=0,
            regime_label="normal",
            win_rate=0.5,
            rolling_sharpe=0.0,
        )
        feats_lo = RLFeatureExtractor.sizing_features(
            signal_confidence=-1.0,
            signal_direction="LONG",
            returns_window=[],
            current_position_pct=0.0,
            drawdown=0.0,
            risk_budget_used=0.0,
            time_since_trade=0,
            regime_label="normal",
            win_rate=0.5,
            rolling_sharpe=0.0,
        )
        assert feats_hi[0] == 1.0
        assert feats_lo[0] == 0.0

    def test_high_vol_regime_indicator(self):
        returns_high_vol = [0.05, -0.05, 0.05, -0.05, 0.05] * 6  # ~annualized vol > 25%
        feats = RLFeatureExtractor.sizing_features(
            signal_confidence=0.7,
            signal_direction="LONG",
            returns_window=returns_high_vol,
            current_position_pct=0.0,
            drawdown=0.0,
            risk_budget_used=0.0,
            time_since_trade=0,
            regime_label="high_vol",
            win_rate=0.5,
            rolling_sharpe=0.0,
        )
        # Feature 8 = regime indicator; high vol → 1
        assert feats[8] == 1.0

    def test_all_finite(self):
        feats = RLFeatureExtractor.sizing_features(
            signal_confidence=0.6,
            signal_direction="SHORT",
            returns_window=list(np.random.randn(30) * 0.01),
            current_position_pct=0.2,
            drawdown=0.03,
            risk_budget_used=0.5,
            time_since_trade=5,
            regime_label="normal",
            win_rate=0.45,
            rolling_sharpe=-0.3,
        )
        assert np.isfinite(feats).all()


class TestAlphaSelectionFeatures:
    ALPHA_NAMES = ["TREND", "MOMENTUM", "VOL"]

    def test_output_shape(self):
        n = len(self.ALPHA_NAMES)
        expected = 4 + 4 * n + 4
        feats = RLFeatureExtractor.alpha_selection_features(
            regime_idx=1,
            alpha_names=self.ALPHA_NAMES,
            alpha_returns_history={a: [] for a in self.ALPHA_NAMES},
            alpha_regime_alignments={a: 0.5 for a in self.ALPHA_NAMES},
            market_volatility=0.3,
            vix_normalized=0.4,
        )
        assert feats.shape == (expected,)

    def test_regime_one_hot_correct(self):
        feats = RLFeatureExtractor.alpha_selection_features(
            regime_idx=2,
            alpha_names=self.ALPHA_NAMES,
            alpha_returns_history={a: [] for a in self.ALPHA_NAMES},
            alpha_regime_alignments={a: 0.5 for a in self.ALPHA_NAMES},
            market_volatility=0.3,
            vix_normalized=0.4,
        )
        # First 4 features are one-hot for regime 2
        assert feats[2] == 1.0
        assert feats[0] == 0.0
        assert feats[1] == 0.0
        assert feats[3] == 0.0

    def test_regime_idx_clamped(self):
        feats = RLFeatureExtractor.alpha_selection_features(
            regime_idx=99,  # out of range → clamped to 3
            alpha_names=self.ALPHA_NAMES,
            alpha_returns_history={a: [] for a in self.ALPHA_NAMES},
            alpha_regime_alignments={a: 0.5 for a in self.ALPHA_NAMES},
            market_volatility=0.3,
            vix_normalized=0.4,
        )
        assert feats[3] == 1.0  # clamped to index 3

    def test_all_finite_with_real_returns(self):
        returns = list(np.random.randn(30) * 0.01)
        feats = RLFeatureExtractor.alpha_selection_features(
            regime_idx=0,
            alpha_names=self.ALPHA_NAMES,
            alpha_returns_history={a: returns for a in self.ALPHA_NAMES},
            alpha_regime_alignments={a: 0.7 for a in self.ALPHA_NAMES},
            market_volatility=0.25,
            vix_normalized=0.5,
        )
        assert np.isfinite(feats).all()

    def test_expected_dims_utility(self):
        dims = RLFeatureExtractor.expected_dims(n_alphas=7)
        assert dims["execution"] == 8
        assert dims["sizing"] == 10
        assert dims["alpha_selection"] == 4 + 4 * 7 + 4

    def test_empty_alpha_list(self):
        feats = RLFeatureExtractor.alpha_selection_features(
            regime_idx=0,
            alpha_names=[],
            alpha_returns_history={},
            alpha_regime_alignments={},
            market_volatility=0.3,
            vix_normalized=0.4,
        )
        assert feats.shape == (8,)  # 4 + 0 + 4


class TestFeatureExtractorFromOHLCV:
    def test_execution_features_from_ohlcv(self):
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        df = pd.DataFrame(
            {
                "close": prices,
                "high": prices * 1.005,
                "low": prices * 0.995,
                "volume": np.random.randint(1000, 5000, 50),
            }
        )
        feats = RLFeatureExtractor.execution_features_from_ohlcv(
            ohlcv_df=df,
            remaining_qty=500,
            total_qty=1000,
            remaining_time=5,
            time_horizon=10,
            arrival_price=prices[0],
            shortfall=0.0,
            data_idx=30,
        )
        assert feats.shape == (8,)
        assert np.isfinite(feats).all()
