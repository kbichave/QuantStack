"""
RL Feature Extractor — canonical feature vectors for all RL agents.

THIS IS THE SINGLE SOURCE OF TRUTH for feature computation.

Problem solved: Training-serving skew.
Without this module, each RL environment computes features internally during
training (e.g. SizingEnvironment._get_state()), while the orchestrator computes
slightly different features at inference time (_build_sizing_state()). Even tiny
divergences (different normalization constants, missing clip bounds) cause the
agent to behave unpredictably in production.

Solution: The environments can call these static methods from their _get_state()
implementations, and the RL tools call the same methods at inference time.
Both paths produce byte-for-byte identical feature vectors.

Usage:
    from quantcore.rl.features import RLFeatureExtractor

    # At inference (in rl_tools.py):
    feats = RLFeatureExtractor.sizing_features(
        signal_confidence=0.72,
        signal_direction="LONG",
        returns_window=[0.01, -0.005, ...],
        current_position_pct=0.05,
        drawdown=0.02,
        risk_budget_used=0.4,
        time_since_trade=3,
        regime_label="trending_up",
        win_rate=0.55,
        rolling_sharpe=0.8,
    )

    # At training (in SizingEnvironment._get_state()):
    # Same call, same output — guaranteed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RLFeatureExtractor:
    """
    Canonical feature vectors for all RL agents.

    All methods are static. Each method corresponds to one RL agent's state space.
    Normalization constants exactly match the environment implementations to
    eliminate training-serving skew.

    Feature dimensions:
        execution_features: 8
        sizing_features:    10
        alpha_selection_features: 4 + 4*n_alphas + 4 (variable)
    """

    # -------------------------------------------------------------------------
    # EXECUTION AGENT — 8 features
    # Matches ExecutionEnvironment._get_state()
    # -------------------------------------------------------------------------

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
        """
        8-feature state vector for ExecutionRLAgent.

        Features (in order):
        0: remaining_quantity / total_quantity         — how much left to execute
        1: remaining_time / time_horizon               — how much time left
        2: (current_price - arrival_price) / arrival_price — price drift
        3: spread_bps / 10000                          — bid-ask spread as fraction
        4: volatility                                  — normalised 20-bar vol
        5: volume_ratio                                — current / avg volume
        6: (current_price - vwap) / vwap               — VWAP deviation
        7: shortfall                                   — implementation shortfall so far

        All values are left in their natural ranges — the agent's DQN normalizes
        inputs in its forward pass.  Do NOT add extra normalization here; it
        would break parity with the environment.
        """
        # Guard: avoid division by zero
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
        """
        Convenience overload that computes volatility and volume_ratio from
        an OHLCV DataFrame rather than requiring pre-computed values.

        Used by rl_tools.py at inference time when only raw market data is available.
        """
        # Current price from data
        current_price = float(
            ohlcv_df.iloc[min(data_idx, len(ohlcv_df) - 1)]["close"]
            if "close" in ohlcv_df.columns and len(ohlcv_df) > 0
            else arrival_price
        )

        # 20-bar volatility
        if "close" in ohlcv_df.columns and data_idx >= 20:
            returns = ohlcv_df["close"].pct_change().iloc[data_idx - 20 : data_idx]
            volatility = float(returns.std()) if not returns.isna().all() else 0.02
        else:
            volatility = 0.02

        # Volume ratio (current / avg-20)
        if "volume" in ohlcv_df.columns and data_idx >= 20:
            recent_vol = ohlcv_df["volume"].iloc[min(data_idx, len(ohlcv_df) - 1)]
            avg_vol = ohlcv_df["volume"].iloc[max(0, data_idx - 20) : data_idx].mean()
            volume_ratio = float(recent_vol / avg_vol) if avg_vol > 0 else 1.0
        else:
            volume_ratio = 1.0

        # VWAP from recent bars
        vwap = arrival_price
        if all(c in ohlcv_df.columns for c in ["high", "low", "close", "volume"]) and data_idx > 0:
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

    # -------------------------------------------------------------------------
    # SIZING AGENT — 10 features
    # Matches SizingEnvironment._get_state()
    # -------------------------------------------------------------------------

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
        """
        10-feature state vector for SizingRLAgent.

        Features (in order) — normalization matches SizingEnvironment._get_state():
        0: signal_confidence                           — 0..1
        1: signal_direction encoded                    — -1, 0, +1
        2: portfolio_volatility / 0.3                  — normalized annualized vol
        3: drawdown / max_drawdown_limit               — normalized current DD
        4: risk_budget_used                            — 0..1
        5: rolling_sharpe / 3                          — normalized
        6: current_position_pct                        — fraction of max position
        7: min(time_since_trade / 10, 1.0)             — normalized bars since trade
        8: regime indicator                            — -1 (low vol), 0 (normal), +1 (high vol)
        9: win_rate                                    — 0..1 rolling win rate
        """
        # Direction encoding — matches SizingEnvironment exactly
        dir_enc = 1 if signal_direction == "LONG" else (-1 if signal_direction == "SHORT" else 0)

        # Volatility from returns window
        if len(returns_window) >= 5:
            vol = float(np.std(returns_window[-20:]) * np.sqrt(252))
        else:
            vol = 0.02 * np.sqrt(252)  # default ~32%

        # Regime indicator from volatility — matches SizingEnvironment._get_state()
        if vol > 0.25:
            regime = 1
        elif vol < 0.10:
            regime = -1
        else:
            regime = 0

        # Normalize drawdown the same way the environment does
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

    # -------------------------------------------------------------------------
    # ALPHA SELECTION AGENT — 4 + 4*n_alphas + 4 features
    # Matches AlphaSelectionEnvironment._get_state()
    # -------------------------------------------------------------------------

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
        """
        Variable-length state vector for AlphaSelectionAgent.
        Total dims: 4 + 4*n_alphas + 4

        Features:
        [0:4]      Regime one-hot [low_vol_bull, high_vol_bull, low_vol_bear, high_vol_bear]
        [4:4+4n]   Per-alpha: sharpe/3, recent_return*10, hit_rate, regime_alignment
        [-4:]      Market: volatility, correlation_regime, usd_regime, vix

        normalization matches AlphaSelectionEnvironment._get_state() exactly.
        """
        features: list[float] = []

        # Regime one-hot (4) — regime_idx in [0, 1, 2, 3]
        regime_one_hot = [0.0, 0.0, 0.0, 0.0]
        regime_one_hot[min(max(regime_idx, 0), 3)] = 1.0
        features.extend(regime_one_hot)

        # Per-alpha features (4 each)
        for name in alpha_names:
            returns = alpha_returns_history.get(name, [])

            # Recent Sharpe — matches AlphaSelectionEnvironment
            if len(returns) >= lookback:
                recent = returns[-lookback:]
                sharpe = float(np.mean(recent) / (np.std(recent) + 1e-8) * np.sqrt(252))
            else:
                sharpe = 0.0

            # Recent cumulative return
            recent_return = float(np.sum(returns[-lookback:])) if returns else 0.0

            # Hit rate
            hit_rate = float(np.mean([r > 0 for r in returns[-lookback:]])) if returns else 0.5

            # Regime alignment (provided externally from KnowledgeStore or heuristic)
            alignment = float(alpha_regime_alignments.get(name, 0.4))

            features.extend(
                [
                    float(np.clip(sharpe / 3.0, -2.0, 2.0)),
                    float(np.clip(recent_return * 10.0, -1.0, 1.0)),
                    float(np.clip(hit_rate, 0.0, 1.0)),
                    float(np.clip(alignment, 0.0, 1.0)),
                ]
            )

        # Market features (4) — same as AlphaSelectionEnvironment._get_state()
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
        """Return expected feature dimensions for documentation and test assertions."""
        return {
            "execution": 8,
            "sizing": 10,
            "alpha_selection": 4 + 4 * n_alphas + 4,
        }
