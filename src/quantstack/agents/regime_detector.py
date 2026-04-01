# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Real RegimeDetectorAgent — replaces the stub with actual indicator-based detection.

Regime classification uses:
  - Trend: ADX (Average Directional Index)
      ADX > 25 → trending
      ADX < 20 → ranging
      +DI > -DI → up / +DI < -DI → down
  - Volatility: ATR percentile vs. 252-day history
      < 25th pct → low vol
      25–75th pct → normal
      > 75th pct → high vol
      > 90th pct → extreme

Regime labels:
  trend_regime:    "trending_up" | "trending_down" | "ranging" | "unknown"
  volatility_regime: "low" | "normal" | "high" | "extreme"

Falls back to stub values if market data is unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.data.pg_storage import PgDataStore
from quantstack.data.provider import get_provider


class RegimeDetectorAgent:
    """
    Indicator-based market regime classifier.

    Uses ADX for trend strength/direction and ATR percentile for volatility.
    No LLM — deterministic and fast.
    """

    ADX_TRENDING_THRESHOLD = 25
    ADX_RANGING_THRESHOLD = 20
    VOL_LOW_PCT = 25
    VOL_HIGH_PCT = 75
    VOL_EXTREME_PCT = 90

    def __init__(self, symbols: list[str] | None = None) -> None:
        self.symbols = list(symbols) if symbols else ["SPY"]

    def detect_regime(
        self, symbol: str, timeframe: str = "daily", bars: list[dict] | None = None
    ) -> dict[str, Any]:
        """
        Detect market regime for a symbol.

        Args:
            symbol: Ticker symbol
            timeframe: Timeframe string (informational)
            bars: Optional list of OHLCV dicts. If None, attempts to fetch
                  from the active DataProvider.

        Returns:
            {
                success: bool,
                symbol: str,
                timeframe: str,
                trend_regime: str (trending_up/trending_down/ranging),
                volatility_regime: str (low/normal/high/extreme),
                confidence: float,
                adx: float,
                plus_di: float,
                minus_di: float,
                atr: float,
                atr_percentile: float,
                hmm_regime: str (optional, from HMM classifier),
                regime_agreement: bool (whether ADX and HMM agree),
            }
        """
        ohlcv = bars

        if ohlcv is None:
            ohlcv = self._fetch_bars(symbol)

        if not ohlcv or len(ohlcv) < 30:
            logger.warning(
                f"[REGIME] {symbol}: insufficient data ({len(ohlcv or [])} bars)"
            )
            return self._fallback(symbol, timeframe, "insufficient data")

        try:
            closes = np.array([b["close"] for b in ohlcv], dtype=float)
            highs = np.array([b["high"] for b in ohlcv], dtype=float)
            lows = np.array([b["low"] for b in ohlcv], dtype=float)

            adx, plus_di, minus_di = self._adx(highs, lows, closes, period=14)
            atr = self._atr(highs, lows, closes, period=14)
            atr_pct = self._atr_percentile(highs, lows, closes, period=14, lookback=252)

            # Trend regime
            if adx >= self.ADX_TRENDING_THRESHOLD:
                if plus_di > minus_di:
                    trend_regime = "trending_up"
                else:
                    trend_regime = "trending_down"
            elif adx < self.ADX_RANGING_THRESHOLD:
                trend_regime = "ranging"
            else:
                # Transitional zone (20–25)
                if plus_di > minus_di:
                    trend_regime = "trending_up"
                else:
                    trend_regime = "trending_down"

            # Volatility regime
            if atr_pct < self.VOL_LOW_PCT:
                vol_regime = "low"
            elif atr_pct < self.VOL_HIGH_PCT:
                vol_regime = "normal"
            elif atr_pct < self.VOL_EXTREME_PCT:
                vol_regime = "high"
            else:
                vol_regime = "extreme"

            # Confidence: higher ADX = more confident in trend call
            confidence = min(1.0, adx / 50.0) if trend_regime != "ranging" else 0.5

            # Try to get HMM regime for comparison (optional)
            hmm_regime = None
            regime_agreement = True
            try:
                from quantstack.core.hierarchy.regime.hmm_model import (
                    HMMRegimeModel,
                    HMMRegimeState,
                )

                # Convert bars to DataFrame for HMM
                df_hmm = pd.DataFrame(ohlcv)
                if len(df_hmm) >= 100:  # HMM needs sufficient data
                    hmm_model = HMMRegimeModel()
                    hmm_model.fit(df_hmm)
                    hmm_result = hmm_model.predict(df_hmm)
                    hmm_regime = hmm_result.state.name  # e.g., "HIGH_VOL_BULL"

                    # Check agreement between ADX and HMM
                    hmm_trend = self._hmm_to_trend(hmm_result.state)
                    if trend_regime != "ranging" and hmm_trend != trend_regime:
                        regime_agreement = False
                        logger.warning(
                            f"[REGIME] {symbol}: ADX says {trend_regime}, "
                            f"HMM says {hmm_regime} ({hmm_trend}). Conflict detected."
                        )
            except Exception as e:
                logger.debug(f"[REGIME] {symbol}: HMM check failed: {e}")

            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "trend_regime": trend_regime,
                "volatility_regime": vol_regime,
                "confidence": round(confidence, 3),
                "adx": round(float(adx), 2),
                "plus_di": round(float(plus_di), 2),
                "minus_di": round(float(minus_di), 2),
                "atr": round(float(atr), 4),
                "atr_percentile": round(float(atr_pct), 1),
                "hmm_regime": hmm_regime,
                "regime_agreement": regime_agreement,
            }

        except Exception as e:
            logger.error(f"[REGIME] {symbol}: computation failed: {e}")
            return self._fallback(symbol, timeframe, str(e))

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _hmm_to_trend(hmm_state) -> str:
        """Map HMM regime state to ADX trend regime format.

        Args:
            hmm_state: HMMRegimeState enum value

        Returns:
            "trending_up", "trending_down", or "ranging"
        """
        # HMM states: LOW_VOL_BULL=0, HIGH_VOL_BULL=1, LOW_VOL_BEAR=2, HIGH_VOL_BEAR=3
        name = hmm_state.name if hasattr(hmm_state, "name") else str(hmm_state)
        if "BULL" in name:
            return "trending_up"
        elif "BEAR" in name:
            return "trending_down"
        else:
            return "ranging"

    # -------------------------------------------------------------------------
    # Indicator implementations
    # -------------------------------------------------------------------------

    @staticmethod
    def _atr(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float:
        """Average True Range — latest value."""
        n = len(closes)
        if n < period + 1:
            return float(np.mean(highs - lows))

        true_ranges = np.zeros(n)
        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            true_ranges[i] = max(hl, hc, lc)

        # Wilder's smoothed ATR
        atr = float(np.mean(true_ranges[1 : period + 1]))
        for i in range(period + 1, n):
            atr = (atr * (period - 1) + true_ranges[i]) / period
        return atr

    @staticmethod
    def _atr_percentile(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
        lookback: int = 252,
    ) -> float:
        """
        ATR percentile vs. historical ATR distribution.

        Returns where today's ATR sits in the last `lookback` bars.
        50 = median volatility, 90 = very high.
        """
        n = len(closes)
        if n < period + 10:
            return 50.0

        # Compute rolling ATR for lookback window
        start = max(1, n - lookback)
        true_ranges = np.zeros(n)
        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            true_ranges[i] = max(hl, hc, lc)

        # Rolling ATR values
        rolling_atrs = []
        for j in range(start, n):
            if j < period:
                continue
            window_tr = true_ranges[max(0, j - period + 1) : j + 1]
            rolling_atrs.append(float(np.mean(window_tr)))

        if not rolling_atrs:
            return 50.0

        current_atr = rolling_atrs[-1]
        pct = (
            float(np.sum(np.array(rolling_atrs) <= current_atr))
            / len(rolling_atrs)
            * 100
        )
        return pct

    @staticmethod
    def _adx(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> tuple[float, float, float]:
        """
        ADX, +DI, -DI using Wilder's smoothing.

        Returns (adx, plus_di, minus_di) — latest values.
        """
        n = len(closes)
        if n < period * 2:
            return 0.0, 0.0, 0.0

        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        tr = np.zeros(n)

        for i in range(1, n):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]

            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0

            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

        # Wilder's initial smoothing
        smooth_tr = float(np.sum(tr[1 : period + 1]))
        smooth_plus = float(np.sum(plus_dm[1 : period + 1]))
        smooth_minus = float(np.sum(minus_dm[1 : period + 1]))

        dx_values = []

        for i in range(period + 1, n):
            smooth_tr = smooth_tr - smooth_tr / period + tr[i]
            smooth_plus = smooth_plus - smooth_plus / period + plus_dm[i]
            smooth_minus = smooth_minus - smooth_minus / period + minus_dm[i]

            if smooth_tr == 0:
                continue

            pdi = 100 * smooth_plus / smooth_tr
            mdi = 100 * smooth_minus / smooth_tr
            denom = pdi + mdi
            dx = 100 * abs(pdi - mdi) / denom if denom > 0 else 0.0
            dx_values.append((dx, pdi, mdi))

        if not dx_values:
            return 0.0, 0.0, 0.0

        # ADX = smoothed DX
        if len(dx_values) < period:
            adx = float(np.mean([d[0] for d in dx_values]))
            plus_di = dx_values[-1][1]
            minus_di = dx_values[-1][2]
        else:
            adx = float(np.mean([d[0] for d in dx_values[-period:]]))
            plus_di = dx_values[-1][1]
            minus_di = dx_values[-1][2]

        return adx, plus_di, minus_di

    # -------------------------------------------------------------------------
    # Data fetching
    # -------------------------------------------------------------------------

    def _fetch_bars(self, symbol: str) -> list[dict]:
        """Fetch OHLCV bars, preferring the local PgDataStore over the live API.

        PgDataStore is tried first — it's the same source used by all other
        tools (compute_technical_indicators, get_signal_brief, etc.).  The live
        provider is only called if the local store returns fewer than 30 bars.
        """
        # --- 1. Try local PostgreSQL store ---
        try:
            store = PgDataStore()
            df = store.load_ohlcv(symbol, Timeframe.D1)
            if df is not None and len(df) >= 30:
                df = df.sort_index().tail(300)
                return [
                    {
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row.get("volume", 0)),
                    }
                    for _, row in df.iterrows()
                ]
        except Exception as e:
            logger.debug(f"[REGIME] PgDataStore miss for {symbol}: {e}")

        # --- 2. Fall back to live provider ---
        try:
            provider = get_provider()
            bars_obj = provider.get_bars(symbol, interval="1d", limit=300)
            return [
                {
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in bars_obj
            ]
        except Exception as e:
            logger.warning(f"[REGIME] Could not fetch bars for {symbol}: {e}")
            return []

    # -------------------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------------------

    def _fallback(self, symbol: str, timeframe: str, reason: str) -> dict[str, Any]:
        return {
            "success": False,
            "symbol": symbol,
            "timeframe": timeframe,
            "trend_regime": "unknown",
            "volatility_regime": "unknown",
            "confidence": 0.0,
            "adx": 0.0,
            "plus_di": 0.0,
            "minus_di": 0.0,
            "atr": 0.0,
            "atr_percentile": 50.0,
            "error": reason,
        }
