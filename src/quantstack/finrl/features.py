"""
RL Feature Extractor — canonical feature vectors for all RL agents.

THIS IS THE SINGLE SOURCE OF TRUTH for feature computation.

Problem solved: Training-serving skew.
Both environments (training) and tools (inference) call these static methods
to produce byte-for-byte identical feature vectors.

Ported from quantstack.rl.features — same logic, no dependency on old RL base classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RLFeatureExtractor:
    """
    Canonical feature vectors for all RL agents.

    All methods are static. Each method corresponds to one environment's observation space.
    Normalization constants exactly match the Gymnasium environment implementations.

    Feature dimensions:
        execution_features: 8
        sizing_features:    10
        alpha_selection_features: 4 + 4*n_alphas + 4 (variable)
    """

    # ── EXECUTION AGENT — 8 features ──

    @staticmethod
    def execution_features(
        remaining_qty: float,
        total_qty: float,
        remaining_time: int,
        time_horizon: int,
        current_price: float,
        arrival_price: float,
        spread_bps: float,
        volatility: float,
        volume_ratio: float,
        vwap: float,
        shortfall: float,
    ) -> np.ndarray:
        """8-feature state vector for execution optimization."""
        qty_frac = remaining_qty / max(total_qty, 1e-8)
        time_frac = remaining_time / max(time_horizon, 1)
        price_dev = (current_price - arrival_price) / max(arrival_price, 1e-8)
        spread_frac = spread_bps / 10_000.0
        vwap_dev = (current_price - vwap) / max(vwap, 1e-8) if vwap > 0 else 0.0

        return np.array(
            [
                float(np.clip(qty_frac, 0.0, 1.0)),
                float(np.clip(time_frac, 0.0, 1.0)),
                float(np.clip(price_dev, -0.1, 0.1)),
                float(np.clip(spread_frac, 0.0, 0.005)),
                float(np.clip(volatility, 0.0, 0.2)),
                float(np.clip(volume_ratio, 0.0, 5.0)),
                float(np.clip(vwap_dev, -0.05, 0.05)),
                float(np.clip(shortfall, -0.05, 0.05)),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def execution_features_from_ohlcv(
        ohlcv_df: pd.DataFrame,
        remaining_qty: float,
        total_qty: float,
        remaining_time: int,
        time_horizon: int,
        arrival_price: float,
        shortfall: float,
        spread_bps: float = 5.0,
        data_idx: int = 0,
    ) -> np.ndarray:
        """Convenience overload computing volatility/volume from OHLCV."""
        current_price = float(
            ohlcv_df.iloc[min(data_idx, len(ohlcv_df) - 1)]["close"]
            if "close" in ohlcv_df.columns and len(ohlcv_df) > 0
            else arrival_price
        )

        if "close" in ohlcv_df.columns and data_idx >= 20:
            returns = ohlcv_df["close"].pct_change().iloc[data_idx - 20 : data_idx]
            volatility = float(returns.std()) if not returns.isna().all() else 0.02
        else:
            volatility = 0.02

        if "volume" in ohlcv_df.columns and data_idx >= 20:
            recent_vol = ohlcv_df["volume"].iloc[min(data_idx, len(ohlcv_df) - 1)]
            avg_vol = ohlcv_df["volume"].iloc[max(0, data_idx - 20) : data_idx].mean()
            volume_ratio = float(recent_vol / avg_vol) if avg_vol > 0 else 1.0
        else:
            volume_ratio = 1.0

        vwap = arrival_price
        if (
            all(c in ohlcv_df.columns for c in ["high", "low", "close", "volume"])
            and data_idx > 0
        ):
            recent = ohlcv_df.iloc[max(0, data_idx - 10) : data_idx + 1]
            tp = (recent["high"] + recent["low"] + recent["close"]) / 3
            vol_sum = recent["volume"].sum()
            if vol_sum > 0:
                vwap = float((tp * recent["volume"]).sum() / vol_sum)

        return RLFeatureExtractor.execution_features(
            remaining_qty=remaining_qty,
            total_qty=total_qty,
            remaining_time=remaining_time,
            time_horizon=time_horizon,
            current_price=current_price,
            arrival_price=arrival_price,
            spread_bps=spread_bps,
            volatility=volatility,
            volume_ratio=volume_ratio,
            vwap=vwap,
            shortfall=shortfall,
        )

    # ── SIZING AGENT — 10 features ──

    @staticmethod
    def sizing_features(
        signal_confidence: float,
        signal_direction: str,
        returns_window: list[float],
        current_position_pct: float,
        drawdown: float,
        risk_budget_used: float,
        time_since_trade: int,
        regime_label: str,
        win_rate: float,
        rolling_sharpe: float,
        max_drawdown_limit: float = 0.15,
        max_position_pct: float = 0.20,
    ) -> np.ndarray:
        """10-feature state vector for position sizing."""
        dir_enc = (
            1 if signal_direction == "LONG" else (-1 if signal_direction == "SHORT" else 0)
        )

        if len(returns_window) >= 5:
            vol = float(np.std(returns_window[-20:]) * np.sqrt(252))
        else:
            vol = 0.02 * np.sqrt(252)

        if vol > 0.25:
            regime = 1
        elif vol < 0.10:
            regime = -1
        else:
            regime = 0

        dd_normalized = drawdown / max_drawdown_limit

        return np.array(
            [
                float(np.clip(signal_confidence, 0.0, 1.0)),
                float(dir_enc),
                float(np.clip(vol / 0.3, 0.0, 5.0)),
                float(np.clip(dd_normalized, 0.0, 2.0)),
                float(np.clip(risk_budget_used, 0.0, 1.0)),
                float(np.clip(rolling_sharpe / 3.0, -2.0, 2.0)),
                float(np.clip(current_position_pct, -1.0, 1.0)),
                float(np.clip(time_since_trade / 10.0, 0.0, 1.0)),
                float(regime),
                float(np.clip(win_rate, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    # ── ALPHA SELECTION AGENT — variable features ──

    @staticmethod
    def alpha_selection_features(
        regime_idx: int,
        alpha_names: list[str],
        alpha_returns_history: dict[str, list[float]],
        alpha_regime_alignments: dict[str, float],
        market_volatility: float,
        vix_normalized: float,
        correlation_regime: float = 0.0,
        usd_regime: float = 0.0,
        lookback: int = 20,
    ) -> np.ndarray:
        """Variable-length state vector for alpha selection. Dims: 4 + 4*n_alphas + 4."""
        features: list[float] = []

        regime_one_hot = [0.0, 0.0, 0.0, 0.0]
        regime_one_hot[min(max(regime_idx, 0), 3)] = 1.0
        features.extend(regime_one_hot)

        for name in alpha_names:
            returns = alpha_returns_history.get(name, [])

            if len(returns) >= lookback:
                recent = returns[-lookback:]
                sharpe = float(np.mean(recent) / (np.std(recent) + 1e-8) * np.sqrt(252))
            else:
                sharpe = 0.0

            recent_return = float(np.sum(returns[-lookback:])) if returns else 0.0
            hit_rate = (
                float(np.mean([r > 0 for r in returns[-lookback:]])) if returns else 0.5
            )
            alignment = float(alpha_regime_alignments.get(name, 0.4))

            features.extend(
                [
                    float(np.clip(sharpe / 3.0, -2.0, 2.0)),
                    float(np.clip(recent_return * 10.0, -1.0, 1.0)),
                    float(np.clip(hit_rate, 0.0, 1.0)),
                    float(np.clip(alignment, 0.0, 1.0)),
                ]
            )

        features.extend(
            [
                float(np.clip(market_volatility, 0.0, 1.0)),
                float(np.clip(correlation_regime, -1.0, 1.0)),
                float(np.clip(usd_regime, -1.0, 1.0)),
                float(np.clip(vix_normalized, 0.0, 1.0)),
            ]
        )

        return np.array(features, dtype=np.float32)

    @staticmethod
    def expected_dims(n_alphas: int) -> dict:
        """Return expected feature dimensions for each environment."""
        return {
            "execution": 8,
            "sizing": 10,
            "alpha_selection": 4 + 4 * n_alphas + 4,
        }
