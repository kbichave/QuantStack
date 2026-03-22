# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Koncorde — 6-component composite order flow indicator.

Koncorde (developed by Óscar Cagigas) is a Spanish-origin composite indicator
that attempts to separate "smart money" (institutional) from "dumb money"
(retail/speculative) flow. It combines six sub-signals into a visual
dashboard with a clear regime interpretation.

The six components
------------------
1. **RSI-based oscillator** — momentum measure of trend exhaustion
2. **MFI (Money Flow Index)** — volume-weighted RSI; captures buying/selling pressure
3. **Bollinger Band %B** — where price sits within the BB envelope
4. **Stochastic %K** — short-term momentum oscillator
5. **PVI (Positive Volume Index)** — accumulates price change on high-volume days
   (retail crowd tends to follow volume → PVI tracks dumb money)
6. **NVI (Negative Volume Index)** — accumulates price change on low-volume days
   (smart money moves on quiet days → NVI tracks smart money)

Composite signals
-----------------
- **green_line** (smart money): normalised NVI signal. Rising → smart money accumulating.
- **blue_line**  (dumb money):  RSI+MFI+BB+Stoch composite. Rising → retail buying.
- **agreement**: green and blue both positive / both negative = high-conviction signal.
- **divergence**: green and blue disagree = potential reversal or distribution.

All calculations are pure OHLCV — no additional data required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sub-signal helpers
# ---------------------------------------------------------------------------


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI — returns [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index — volume-weighted RSI, [0, 100]."""
    typical_price = (high + low + close) / 3
    raw_mf = typical_price * volume

    pos_mf = raw_mf.where(typical_price > typical_price.shift(1), 0.0)
    neg_mf = raw_mf.where(typical_price < typical_price.shift(1), 0.0)

    pos_roll = pos_mf.rolling(period).sum()
    neg_roll = neg_mf.rolling(period).sum()
    mfr = pos_roll / neg_roll.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def _bb_pct_b(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger Band %B: position of close within the band [0, 1]."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    band_width = (upper - lower).replace(0, np.nan)
    return (close - lower) / band_width


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.Series:
    """Stochastic %K — [0, 100]; smoothed with %D."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    range_hl = (highest_high - lowest_low).replace(0, np.nan)
    k = (close - lowest_low) / range_hl * 100
    return k.rolling(d_period).mean()  # %D = smooth %K


def _pvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Positive Volume Index — cumulates price % change on days where volume > prior volume.
    Tracks retail crowd (who trades on high-volume days).
    """
    ret = close.pct_change()
    pvi = np.full(len(close), 1000.0)  # arbitrary starting value
    for i in range(1, len(close)):
        if volume.iloc[i] > volume.iloc[i - 1]:
            pvi[i] = pvi[i - 1] * (1 + ret.iloc[i])
        else:
            pvi[i] = pvi[i - 1]
    return pd.Series(pvi, index=close.index)


def _nvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Negative Volume Index — cumulates price % change on days where volume < prior volume.
    Tracks smart money (who moves on low-volume, quiet days).
    """
    ret = close.pct_change()
    nvi = np.full(len(close), 1000.0)
    for i in range(1, len(close)):
        if volume.iloc[i] < volume.iloc[i - 1]:
            nvi[i] = nvi[i - 1] * (1 + ret.iloc[i])
        else:
            nvi[i] = nvi[i - 1]
    return pd.Series(nvi, index=close.index)


def _normalise_0_100(s: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling min-max normalise to [0, 100] over `window` bars.
    Preserves the relative position without lookahead.
    """
    roll_min = s.rolling(window, min_periods=10).min()
    roll_max = s.rolling(window, min_periods=10).max()
    span = (roll_max - roll_min).replace(0, np.nan)
    return (s - roll_min) / span * 100


# ---------------------------------------------------------------------------
# Koncorde
# ---------------------------------------------------------------------------


class Koncorde:
    """
    Koncorde composite indicator — full 6-component implementation.

    Separates smart money (NVI-based, green line) from retail flow
    (RSI + MFI + BB%B + Stoch composite, blue line).

    Parameters
    ----------
    rsi_period : int
        RSI lookback. Default 14.
    mfi_period : int
        MFI lookback. Default 14.
    bb_period : int
        Bollinger Band period. Default 20.
    bb_std : float
        Bollinger Band standard deviation multiplier. Default 2.0.
    stoch_k : int
        Stochastic %K period. Default 14.
    stoch_d : int
        Stochastic %D smoothing. Default 3.
    nvi_signal_period : int
        EMA period for NVI signal line. Default 255 (approximately 1 year).
    normalise_window : int
        Rolling window for min-max normalisation. Default 252.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        mfi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        stoch_k: int = 14,
        stoch_d: int = 3,
        nvi_signal_period: int = 255,
        normalise_window: int = 252,
    ) -> None:
        self.rsi_period = rsi_period
        self.mfi_period = mfi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.nvi_signal_period = nvi_signal_period
        self.normalise_window = normalise_window

    def compute(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        high, low, close, volume : pd.Series — aligned DatetimeIndex.

        Returns
        -------
        pd.DataFrame with columns:
            rsi           – RSI oscillator [0, 100]
            mfi           – Money Flow Index [0, 100]
            bb_pct_b      – Bollinger %B [0, 1]
            stochastic    – Stochastic %D [0, 100]
            pvi           – Positive Volume Index (raw)
            nvi           – Negative Volume Index (raw)
            nvi_signal    – EMA of NVI (smart money signal line)
            green_line    – smart money composite: normalised (NVI - nvi_signal) [0, 100]
            blue_line     – retail composite: mean(RSI, MFI, BB%B×100, Stoch) [0, 100]
            green_positive – 1 when green_line > 50 (smart money bullish)
            blue_positive  – 1 when blue_line > 50 (retail bullish)
            agreement     – 1 when both lines positive, -1 when both negative, 0 otherwise
            divergence    – 1 when green and blue disagree (potential reversal)
        """
        rsi = _rsi(close, self.rsi_period)
        mfi = _mfi(high, low, close, volume, self.mfi_period)
        bb_b = _bb_pct_b(close, self.bb_period, self.bb_std)
        stoch = _stochastic(high, low, close, self.stoch_k, self.stoch_d)
        pvi_series = _pvi(close, volume)
        nvi_series = _nvi(close, volume)

        # Smart money signal: NVI relative to its EMA (above EMA = accumulating)
        nvi_signal = nvi_series.ewm(span=self.nvi_signal_period, adjust=False).mean()
        nvi_diff = nvi_series - nvi_signal
        green_line = _normalise_0_100(nvi_diff, self.normalise_window)

        # Retail signal: mean of 4 normalised oscillators
        # Already in [0,100] range except bb_b which is [0,1] → scale ×100
        blue_line = (rsi + mfi + bb_b.clip(0, 1) * 100 + stoch.clip(0, 100)) / 4

        green_pos = (green_line > 50).astype(int)
        blue_pos = (blue_line > 50).astype(int)

        both_pos = (green_pos == 1) & (blue_pos == 1)
        both_neg = (green_pos == 0) & (blue_pos == 0)
        agreement = pd.Series(
            np.where(both_pos, 1, np.where(both_neg, -1, 0)),
            index=close.index,
        )
        divergence = (
            (green_pos != blue_pos) & green_line.notna() & blue_line.notna()
        ).astype(int)

        return pd.DataFrame(
            {
                "rsi": rsi,
                "mfi": mfi,
                "bb_pct_b": bb_b,
                "stochastic": stoch,
                "pvi": pvi_series,
                "nvi": nvi_series,
                "nvi_signal": nvi_signal,
                "green_line": green_line,
                "blue_line": blue_line,
                "green_positive": green_pos,
                "blue_positive": blue_pos,
                "agreement": agreement,
                "divergence": divergence,
            },
            index=close.index,
        )
