"""Volume and dollar bar generation from minute OHLCV data.

Implements information-driven bars from AFML Chapter 2. Volume and dollar bars
normalize for varying market activity, producing bars that carry roughly equal
information content regardless of time-of-day effects.

Honest limitation: Built from minute aggregation, these bars lose the fine-grained
information-arrival properties of tick-level bars. This is an incremental improvement
over time bars, not a paradigm shift. Tick imbalance bars require SIP feed data.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger


class BarGenerator:
    """Generate volume or dollar bars from minute OHLCV data.

    Args:
        bar_type: 'volume' or 'dollar'.
        threshold: Cumulative volume (or dollar volume) to trigger a new bar.
    """

    def __init__(self, bar_type: str, threshold: float):
        if bar_type not in ("volume", "dollar"):
            raise ValueError(f"bar_type must be 'volume' or 'dollar', got '{bar_type}'")
        self.bar_type = bar_type
        self.threshold = threshold

    def generate(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """Generate bars from minute OHLCV data.

        Args:
            minute_df: DataFrame with columns [timestamp, open, high, low, close, volume].

        Returns:
            Bar DataFrame with columns [timestamp, open, high, low, close, volume,
            dollar_volume, tick_count, vwap, bar_duration_seconds].
        """
        if minute_df.empty:
            return self._empty_bars_df()

        bars = []
        self._reset_accumulators()

        prev_date = None
        for _, row in minute_df.iterrows():
            row_date = row["timestamp"].date() if isinstance(row["timestamp"], datetime) else row["timestamp"]

            # Day boundary: emit partial bar and reset
            if prev_date is not None and row_date != prev_date:
                if self._tick_count > 0:
                    bars.append(self._emit_bar())
                self._reset_accumulators()

            prev_date = row_date
            self._accumulate(row)

            accumulator = self._dollar_accum if self.bar_type == "dollar" else self._vol_accum
            if accumulator >= self.threshold:
                bars.append(self._emit_bar())
                self._reset_accumulators()

        # Emit any remaining partial bar
        if self._tick_count > 0:
            bars.append(self._emit_bar())

        if not bars:
            return self._empty_bars_df()

        return pd.DataFrame(bars)

    def calibrate_threshold(
        self, daily_df: pd.DataFrame, target_bars_per_day: int = 50
    ) -> float:
        """Compute threshold from historical daily data.

        For volume bars: avg_daily_volume / target_bars_per_day.
        For dollar bars: avg_daily_dollar_volume / target_bars_per_day.
        """
        if self.bar_type == "dollar" and "dollar_volume" in daily_df.columns:
            avg = daily_df["dollar_volume"].mean()
        else:
            avg = daily_df["volume"].mean()
        return avg / target_bars_per_day

    # -- Internal state management --

    def _reset_accumulators(self) -> None:
        self._open = 0.0
        self._high = -np.inf
        self._low = np.inf
        self._close = 0.0
        self._vol_accum = 0
        self._dollar_accum = 0.0
        self._vwap_numer = 0.0
        self._tick_count = 0
        self._first_ts: datetime | None = None
        self._last_ts: datetime | None = None

    def _accumulate(self, row: pd.Series) -> None:
        if self._tick_count == 0:
            self._open = row["open"]
            self._first_ts = row["timestamp"]

        self._high = max(self._high, row["high"])
        self._low = min(self._low, row["low"])
        self._close = row["close"]
        self._last_ts = row["timestamp"]

        vol = int(row["volume"])
        self._vol_accum += vol
        self._dollar_accum += row["close"] * vol
        self._vwap_numer += row["close"] * vol
        self._tick_count += 1

    def _emit_bar(self) -> dict:
        vwap = self._vwap_numer / self._vol_accum if self._vol_accum > 0 else self._close

        duration = 0
        if self._first_ts is not None and self._last_ts is not None:
            delta = self._last_ts - self._first_ts
            duration = int(delta.total_seconds())

        return {
            "timestamp": self._last_ts,
            "open": self._open,
            "high": self._high,
            "low": self._low,
            "close": self._close,
            "volume": self._vol_accum,
            "dollar_volume": self._dollar_accum,
            "tick_count": self._tick_count,
            "vwap": vwap,
            "bar_duration_seconds": duration,
        }

    @staticmethod
    def _empty_bars_df() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "dollar_volume", "tick_count", "vwap", "bar_duration_seconds",
        ])
