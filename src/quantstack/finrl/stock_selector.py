"""
ML-based instrument selector — adopted from FinRL-Trading.

Two selectors:
  - MLStockSelector: RF + GBM ensemble for equity screening
  - OptionsSelector: scores underlyings for options suitability (IV rank,
    liquidity, predicted move magnitude), then recommends contract specs

Usage:
    from quantstack.finrl.stock_selector import MLStockSelector, OptionsSelector

    # Equity screening
    selector = MLStockSelector()
    picks = selector.select(data, candidates=["AAPL", "MSFT"], top_n=10)

    # Options screening
    opts = OptionsSelector()
    recommendations = opts.select(data, candidates=["SPY", "QQQ"], top_n=5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


@dataclass
class StockPick:
    """A selected stock with score and weight."""

    symbol: str
    score: float
    weight: float
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class OptionsPick:
    """A recommended options trade."""

    symbol: str
    score: float
    direction: str  # "bullish", "bearish", "neutral"
    strategy: str  # "long_call", "long_put", "bull_call_spread", "bear_put_spread", "straddle", etc.
    strike_pct: float  # strike as % of current price (e.g. 1.02 = 2% OTM call)
    dte_target: int  # recommended days to expiration
    iv_rank: float  # current IV percentile (0-1)
    predicted_move: float  # ML-predicted forward return magnitude
    liquidity_score: float  # 0-1, based on volume/OI
    features: dict[str, float] = field(default_factory=dict)


class MLStockSelector:
    """
    ML-based stock selection using RF + GBM ensemble.

    Features: technical indicators + fundamental ratios (when available).
    Scoring: Ensemble average of RF and GBM predicted return rank.
    Weighting: Equal-weight or min-variance.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        lookback_quarters: int = 16,
        test_quarters: int = 4,
    ):
        self.n_estimators = n_estimators
        self.lookback_quarters = lookback_quarters
        self.test_quarters = test_quarters

    def select(
        self,
        data: pd.DataFrame,
        candidates: list[str] | None = None,
        top_n: int = 10,
        weighting: str = "equal",
    ) -> list[StockPick]:
        """
        Select top stocks from candidates based on ML scoring.

        Args:
            data: FinRL-format DataFrame with date, tic, OHLCV + indicators
            candidates: Optional filter to specific tickers
            top_n: Number of stocks to select
            weighting: "equal" or "min_variance"

        Returns:
            List of StockPick with scores and weights
        """
        if candidates:
            data = data[data["tic"].isin(candidates)]

        if data.empty:
            logger.warning("[MLStockSelector] No data for candidates.")
            return []

        # Prepare features: all numeric columns except date, tic, close
        feature_cols = [
            c
            for c in data.columns
            if c not in ("date", "tic", "open", "high", "low", "volume")
            and data[c].dtype in ("float64", "float32", "int64")
        ]

        if not feature_cols:
            logger.warning("[MLStockSelector] No feature columns found.")
            return []

        # Target: forward 5-day return
        data = data.sort_values(["tic", "date"])
        data["fwd_return"] = data.groupby("tic")["close"].transform(
            lambda x: x.shift(-5) / x - 1
        )
        data = data.dropna(subset=["fwd_return"])

        if len(data) < 100:
            logger.warning("[MLStockSelector] Insufficient data for training.")
            return []

        # Train/test split (time-series aware)
        dates = sorted(data["date"].unique())
        split_idx = int(len(dates) * 0.8)
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]

        train = data[data["date"].isin(train_dates)]
        test = data[data["date"].isin(test_dates)]

        X_train = train[feature_cols].fillna(0)
        y_train = train["fwd_return"]
        X_test = test[feature_cols].fillna(0)

        # Ensemble: RF + GBM
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=6, random_state=42, n_jobs=-1
        )
        gbm = GradientBoostingRegressor(
            n_estimators=self.n_estimators, max_depth=4, learning_rate=0.1, random_state=42
        )

        rf.fit(X_train, y_train)
        gbm.fit(X_train, y_train)

        # Score on most recent test data
        latest_date = test_dates[-1] if test_dates else dates[-1]
        latest = test[test["date"] == latest_date]

        if latest.empty:
            latest = test.groupby("tic").last().reset_index()

        X_latest = latest[feature_cols].fillna(0)
        rf_scores = rf.predict(X_latest)
        gbm_scores = gbm.predict(X_latest)
        ensemble_scores = (rf_scores + gbm_scores) / 2

        # Rank and select top N
        latest = latest.copy()
        latest["ml_score"] = ensemble_scores
        latest = latest.sort_values("ml_score", ascending=False)
        top = latest.head(top_n)

        # Weighting
        symbols = top["tic"].tolist()
        scores = top["ml_score"].tolist()

        if weighting == "min_variance" and len(symbols) > 1:
            weights = self._min_variance_weights(data, symbols)
        else:
            weights = [1.0 / len(symbols)] * len(symbols)

        picks = []
        for sym, score, weight in zip(symbols, scores, weights):
            row = top[top["tic"] == sym].iloc[0]
            feats = {c: float(row[c]) for c in feature_cols[:5] if c in row.index}
            picks.append(
                StockPick(symbol=sym, score=float(score), weight=float(weight), features=feats)
            )

        return picks

    @staticmethod
    def _min_variance_weights(data: pd.DataFrame, symbols: list[str]) -> list[float]:
        """Compute minimum-variance portfolio weights."""
        # Compute returns
        returns = data.pivot_table(index="date", columns="tic", values="close").pct_change().dropna()
        returns = returns[[s for s in symbols if s in returns.columns]]

        if returns.shape[1] < 2:
            return [1.0 / len(symbols)] * len(symbols)

        cov = returns.cov().values
        n = cov.shape[0]

        def portfolio_variance(w):
            return w @ cov @ w

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1)] * n
        x0 = np.ones(n) / n

        result = minimize(
            portfolio_variance, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            weights = result.x.tolist()
        else:
            weights = [1.0 / n] * n

        # Pad for any missing symbols
        full_weights = []
        available = [s for s in symbols if s in returns.columns]
        w_map = dict(zip(available, weights))
        for s in symbols:
            full_weights.append(w_map.get(s, 1.0 / len(symbols)))

        # Renormalize
        total = sum(full_weights)
        return [w / total for w in full_weights]


class OptionsSelector:
    """
    ML-based options instrument selector.

    Scores underlyings on three axes:
      1. IV Rank: current IV percentile vs 1-year range (high IV → sell premium, low → buy)
      2. Liquidity: volume/OI score from recent options data
      3. Predicted move: RF+GBM predicted forward return magnitude

    Then recommends:
      - Direction (bullish/bearish/neutral) from ML predicted sign
      - Strategy (call/put/spread/straddle) from IV rank + direction
      - Strike (ATM, OTM %) from predicted move magnitude
      - DTE (7-60 days) from volatility regime
    """

    # Strategy selection matrix: (direction, iv_rank_bucket) → strategy
    STRATEGY_MATRIX = {
        ("bullish", "high"): "bull_call_spread",  # sell expensive premium
        ("bullish", "mid"): "long_call",
        ("bullish", "low"): "long_call",
        ("bearish", "high"): "bear_put_spread",
        ("bearish", "mid"): "long_put",
        ("bearish", "low"): "long_put",
        ("neutral", "high"): "iron_condor",  # sell premium both sides
        ("neutral", "mid"): "straddle",
        ("neutral", "low"): "calendar_spread",  # buy vol cheap
    }

    def __init__(self, n_estimators: int = 100):
        self.n_estimators = n_estimators

    def select(
        self,
        data: pd.DataFrame,
        candidates: list[str] | None = None,
        top_n: int = 5,
        min_dte: int = 7,
        max_dte: int = 60,
    ) -> list[OptionsPick]:
        """
        Score and rank underlyings for options trading, then recommend contracts.

        Args:
            data: FinRL-format DataFrame with date, tic, OHLCV + indicators
            candidates: Filter to specific tickers
            top_n: Number of recommendations
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration

        Returns:
            List of OptionsPick with strategy, strike, DTE recommendations
        """
        if candidates:
            data = data[data["tic"].isin(candidates)]

        if data.empty:
            logger.warning("[OptionsSelector] No data for candidates.")
            return []

        feature_cols = [
            c
            for c in data.columns
            if c not in ("date", "tic", "open", "high", "low", "volume")
            and data[c].dtype in ("float64", "float32", "int64")
        ]

        if not feature_cols:
            logger.warning("[OptionsSelector] No feature columns.")
            return []

        data = data.sort_values(["tic", "date"])

        # Forward 5-day return (target)
        data = data.copy()
        data["fwd_return"] = data.groupby("tic")["close"].transform(
            lambda x: x.shift(-5) / x - 1
        )
        data = data.dropna(subset=["fwd_return"])

        if len(data) < 100:
            logger.warning("[OptionsSelector] Insufficient data.")
            return []

        # Compute per-symbol IV rank (using realized vol as proxy when IV unavailable)
        data["realized_vol_20"] = data.groupby("tic")["close"].transform(
            lambda x: x.pct_change().rolling(20).std() * np.sqrt(252)
        )
        data["vol_1yr_high"] = data.groupby("tic")["realized_vol_20"].transform(
            lambda x: x.rolling(252, min_periods=20).max()
        )
        data["vol_1yr_low"] = data.groupby("tic")["realized_vol_20"].transform(
            lambda x: x.rolling(252, min_periods=20).min()
        )
        data["iv_rank"] = (data["realized_vol_20"] - data["vol_1yr_low"]) / (
            data["vol_1yr_high"] - data["vol_1yr_low"] + 1e-10
        )
        data["iv_rank"] = data["iv_rank"].clip(0, 1)

        # Liquidity score (volume-based proxy)
        data["vol_rank"] = data.groupby("tic")["volume"].transform(
            lambda x: x.rank(pct=True)
        )

        # Train ML model for predicted move
        dates = sorted(data["date"].unique())
        split_idx = int(len(dates) * 0.8)
        train_dates = dates[:split_idx]

        train = data[data["date"].isin(train_dates)]
        X_train = train[feature_cols].fillna(0)
        y_train = train["fwd_return"]

        rf = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=6, random_state=42, n_jobs=-1
        )
        gbm = GradientBoostingRegressor(
            n_estimators=self.n_estimators, max_depth=4, learning_rate=0.1, random_state=42
        )
        rf.fit(X_train, y_train)
        gbm.fit(X_train, y_train)

        # Score on latest data per symbol
        latest = data.groupby("tic").last().reset_index()
        X_latest = latest[feature_cols].fillna(0)
        rf_pred = rf.predict(X_latest)
        gbm_pred = gbm.predict(X_latest)
        predicted_move = (rf_pred + gbm_pred) / 2

        latest = latest.copy()
        latest["predicted_move"] = predicted_move
        latest["abs_move"] = np.abs(predicted_move)

        # Composite score: abs(predicted_move) * iv_rank_factor * liquidity
        # High IV rank is good for selling premium, high abs move is good for directional
        latest["options_score"] = (
            latest["abs_move"] * 100
            + latest["iv_rank"].fillna(0.5) * 2
            + latest["vol_rank"].fillna(0.5)
        )

        latest = latest.sort_values("options_score", ascending=False)
        top = latest.head(top_n)

        picks = []
        for _, row in top.iterrows():
            symbol = row["tic"]
            pred = float(row["predicted_move"])
            iv_rank = float(row.get("iv_rank", 0.5))
            liquidity = float(row.get("vol_rank", 0.5))
            current_vol = float(row.get("realized_vol_20", 0.2))

            # Direction from predicted sign
            if abs(pred) < 0.005:
                direction = "neutral"
            elif pred > 0:
                direction = "bullish"
            else:
                direction = "bearish"

            # IV rank bucket
            if iv_rank > 0.7:
                iv_bucket = "high"
            elif iv_rank > 0.3:
                iv_bucket = "mid"
            else:
                iv_bucket = "low"

            strategy = self.STRATEGY_MATRIX.get(
                (direction, iv_bucket), "long_call" if direction == "bullish" else "long_put"
            )

            # Strike: ATM for neutral, OTM proportional to predicted move for directional
            if direction == "neutral":
                strike_pct = 1.0
            elif direction == "bullish":
                strike_pct = 1.0 + min(abs(pred) * 2, 0.05)  # 0-5% OTM call
            else:
                strike_pct = 1.0 - min(abs(pred) * 2, 0.05)  # 0-5% OTM put

            # DTE: higher vol → shorter DTE (theta decay faster), lower vol → longer DTE
            if current_vol > 0.3:
                dte = max(min_dte, 14)  # high vol → 2 weeks
            elif current_vol > 0.15:
                dte = 30  # normal vol → 1 month
            else:
                dte = min(max_dte, 45)  # low vol → 45 days

            feats = {c: float(row[c]) for c in feature_cols[:5] if c in row.index}

            picks.append(
                OptionsPick(
                    symbol=symbol,
                    score=float(row["options_score"]),
                    direction=direction,
                    strategy=strategy,
                    strike_pct=round(strike_pct, 4),
                    dte_target=dte,
                    iv_rank=round(iv_rank, 4),
                    predicted_move=round(pred, 6),
                    liquidity_score=round(liquidity, 4),
                    features=feats,
                )
            )

        return picks
