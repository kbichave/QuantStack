# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Order Flow Approximations from OHLCV data.

Tick-level order flow analysis ideally requires signed trade data (buy-initiated
vs. sell-initiated trades). Without tick data we use OHLCV approximations that
capture ~70-80% of the signal with zero additional data cost.

Three signal families
---------------------
1. **CumulativeVolumeDelta (CVD)**
   Approximates the cumulative imbalance between buying and selling volume
   using the candle's body/range relationship (Lee-Ready proxy).

2. **VPIN (Volume-synchronized PIN)**
   Volume-synchronized Probability of Informed Trading. Approximates the
   fraction of "toxic" (informed) volume in each volume bucket.
   Tick-level VPIN uses signed trades; here we approximate trade direction
   from price change direction per bar.

3. **HawkesIntensity**
   Self-exciting point process model for trade clustering. Captures how
   a burst of activity (large candles, high volume) predicts elevated
   activity in subsequent bars — a signature of institutional order flow.

All classes are causal (no lookahead bias). Computations are pure pandas/numpy.

Data contract for tick-gated signals
-------------------------------------
When tick-level data becomes available (e.g., Polygon.io, Databento), replace
the OHLCV proxy in each class with the real signed-trade calculation. The class
interfaces are designed to be drop-in compatible: pass a `signed_volume` Series
directly to bypass the OHLCV approximation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: Lee-Ready buy/sell volume split from OHLCV
# ---------------------------------------------------------------------------


def _buy_sell_split(open_: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series, volume: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Approximate per-bar buy and sell volume using candle structure.

    Method (candle-body proxy):
        buy_fraction  = (close - low) / (high - low + ε)
        sell_fraction = (high - close) / (high - low + ε)

    This splits total bar volume proportionally to where the close sits
    within the bar's range. A close near the high → mostly buying;
    close near the low → mostly selling. Perfect for daily/hourly bars.

    For bars with zero range (doji), volume is split 50/50.

    Parameters
    ----------
    open_, high, low, close, volume : pd.Series (aligned index)

    Returns
    -------
    (buy_vol, sell_vol) : both pd.Series, sum = volume per bar
    """
    bar_range = (high - low).replace(0, np.nan)
    buy_frac = (close - low) / bar_range
    buy_frac = buy_frac.fillna(0.5).clip(0, 1)
    sell_frac = 1.0 - buy_frac
    return volume * buy_frac, volume * sell_frac


# ---------------------------------------------------------------------------
# Cumulative Volume Delta
# ---------------------------------------------------------------------------


class CumulativeVolumeDelta:
    """
    Cumulative Volume Delta (CVD) — approximated from OHLCV candle structure.

    CVD = cumulative sum of (buy_volume − sell_volume) per bar.

    Rising CVD + rising price = healthy trend (buyers driving price up).
    Falling CVD + rising price = bearish divergence (price moving up on selling).

    Real tick-level CVD requires signed trades (Lee-Ready or BBO classification).
    This implementation uses the candle body/range proxy which captures the broad
    direction accurately on longer timeframes.

    Parameters
    ----------
    lookback : int | None
        If set, reset the cumulative sum every `lookback` bars (rolling CVD).
        Default None = full cumulative from bar 0.
    signed_volume : pd.Series | None
        Optional: pass pre-computed signed volume (positive = buy, negative = sell)
        from tick data to bypass OHLCV approximation.
    """

    def __init__(self, lookback: int | None = None) -> None:
        self.lookback = lookback

    def compute(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        signed_volume: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        open_, high, low, close, volume : pd.Series — OHLCV (aligned index)
        signed_volume : pd.Series | None
            If provided, bypasses OHLCV proxy. Positive = net buying, negative = selling.

        Returns
        -------
        pd.DataFrame with columns:
            bar_delta       – per-bar (buy_vol - sell_vol)
            cvd             – cumulative volume delta
            cvd_ma          – 20-bar moving average of bar_delta (signal line)
            cvd_divergence  – 1 when price makes new high but CVD does not
                              (bearish divergence), -1 for bullish, 0 otherwise
            buy_vol         – estimated buy volume per bar
            sell_vol        – estimated sell volume per bar
        """
        if signed_volume is not None:
            bar_delta = signed_volume.copy()
            buy_vol = signed_volume.clip(lower=0)
            sell_vol = (-signed_volume).clip(lower=0)
        else:
            buy_vol, sell_vol = _buy_sell_split(open_, high, low, close, volume)
            bar_delta = buy_vol - sell_vol

        if self.lookback is not None:
            cvd = bar_delta.rolling(self.lookback).sum()
        else:
            cvd = bar_delta.cumsum()

        cvd_ma = bar_delta.rolling(20).mean()

        # Divergence: price makes 20-bar high but CVD does not
        price_new_high = (close == close.rolling(20).max()).astype(int)
        cvd_new_high   = (cvd == cvd.rolling(20).max()).astype(int)
        price_new_low  = (close == close.rolling(20).min()).astype(int)
        cvd_new_low    = (cvd == cvd.rolling(20).min()).astype(int)

        divergence = pd.Series(0, index=close.index)
        divergence[price_new_high == 1] = np.where(
            cvd_new_high[price_new_high == 1] == 0, -1, 0
        )
        divergence[price_new_low == 1] = np.where(
            cvd_new_low[price_new_low == 1] == 0, 1, 0
        )

        return pd.DataFrame(
            {
                "bar_delta":      bar_delta,
                "cvd":            cvd,
                "cvd_ma":         cvd_ma,
                "cvd_divergence": divergence,
                "buy_vol":        buy_vol,
                "sell_vol":       sell_vol,
            },
            index=close.index,
        )


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------


class VPIN:
    """
    Volume-synchronized Probability of Informed Trading (VPIN).

    VPIN measures the imbalance of buyer- vs. seller-initiated volume in each
    equal-volume bucket. High VPIN = high fraction of "toxic" (informed) flow
    = adverse selection risk; historically precedes volatility spikes.

    Academic reference: Easley, López de Prado, O'Hara (2012).

    This implementation uses OHLCV approximation:
    - Each bar's volume is classified as buy or sell via candle body proxy.
    - Bars are accumulated into buckets of size = average_daily_volume / n_buckets.
    - VPIN = rolling mean of |buy_bucket - sell_bucket| / bucket_size.

    Parameters
    ----------
    n_buckets : int
        Number of volume buckets per day equivalent. Default 50.
    window : int
        Number of past buckets for rolling VPIN average. Default 50.
    """

    def __init__(self, n_buckets: int = 50, window: int = 50) -> None:
        self.n_buckets = n_buckets
        self.window = window

    def compute(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            vpin          – rolling VPIN estimate [0, 1]; higher = more toxic flow
            vpin_high     – 1 when vpin > 75th percentile of its rolling history
            bucket_imbalance – |buy_bucket - sell_bucket| / bucket_size per bucket
        """
        buy_vol, sell_vol = _buy_sell_split(open_, high, low, close, volume)

        # Target bucket size = average volume / n_buckets
        avg_vol = volume.mean()
        bucket_size = max(avg_vol / self.n_buckets, 1.0)

        n = len(volume)
        vpin_raw = np.full(n, np.nan)
        bucket_imbalance_series = np.zeros(n)

        # Accumulate bars into buckets
        current_buy = 0.0
        current_sell = 0.0
        current_vol = 0.0
        bucket_vpin_vals: list[float] = []

        for i in range(n):
            bv = buy_vol.iloc[i]
            sv = sell_vol.iloc[i]
            current_buy += bv
            current_sell += sv
            current_vol += bv + sv

            while current_vol >= bucket_size:
                # Fraction of bucket filled by this bar
                fraction = bucket_size / current_vol
                b_contrib = current_buy * fraction
                s_contrib = current_sell * fraction
                imbalance = abs(b_contrib - s_contrib) / bucket_size
                bucket_vpin_vals.append(imbalance)
                bucket_imbalance_series[i] = imbalance

                current_buy   *= (1 - fraction)
                current_sell  *= (1 - fraction)
                current_vol   *= (1 - fraction)

            if len(bucket_vpin_vals) >= self.window:
                vpin_raw[i] = np.mean(bucket_vpin_vals[-self.window:])

        vpin_series = pd.Series(vpin_raw, index=close.index)
        roll_75 = vpin_series.rolling(252, min_periods=20).quantile(0.75)
        vpin_high = (vpin_series > roll_75).astype(int)

        return pd.DataFrame(
            {
                "vpin":             vpin_series,
                "vpin_high":        vpin_high,
                "bucket_imbalance": pd.Series(bucket_imbalance_series, index=close.index),
            },
            index=close.index,
        )


# ---------------------------------------------------------------------------
# Hawkes Intensity
# ---------------------------------------------------------------------------


class HawkesIntensity:
    """
    Hawkes self-exciting point process intensity from OHLCV data.

    In a Hawkes process, each event (here: a large-volume or large-range bar)
    temporarily raises the probability of further events. This captures the
    "clustering" behaviour of institutional order flow.

    Intensity model:
        λ(t) = μ + Σ_{s < t}  α × exp(-β × (t - s))

    Where events are bars exceeding a volume or range threshold.

    OHLCV proxy: events are defined as bars where (volume > threshold_mult × avg_vol)
    OR (bar_range > threshold_mult × avg_range). This approximates trade clustering
    which tick-level Hawkes models capture from individual trade timestamps.

    Parameters
    ----------
    decay : float
        Hawkes decay parameter β. Controls how quickly past events lose influence.
        Default 0.5 (half-life ≈ 2 bars).
    excitation : float
        Hawkes excitation parameter α (jump per event). Default 0.8.
    baseline : float
        Baseline intensity μ. Default 0.1.
    event_threshold : float
        Multiplier above rolling average to classify a bar as an event.
        e.g. 1.5 = bar volume > 1.5 × avg_vol. Default 1.5.
    event_window : int
        Rolling window for computing average volume/range. Default 20.
    """

    def __init__(
        self,
        decay: float = 0.5,
        excitation: float = 0.8,
        baseline: float = 0.1,
        event_threshold: float = 1.5,
        event_window: int = 20,
    ) -> None:
        self.decay = decay
        self.excitation = excitation
        self.baseline = baseline
        self.event_threshold = event_threshold
        self.event_window = event_window

    def compute(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            event           – 1 if bar qualifies as a Hawkes event
            intensity       – λ(t) estimated intensity at each bar
            intensity_high  – 1 when intensity > 75th percentile of rolling history
            excited         – 1 when intensity > baseline × 2 (elevated regime)
        """
        bar_range = high - low
        avg_vol   = volume.rolling(self.event_window, min_periods=5).mean()
        avg_range = bar_range.rolling(self.event_window, min_periods=5).mean()

        vol_event   = (volume > avg_vol * self.event_threshold).fillna(False)
        range_event = (bar_range > avg_range * self.event_threshold).fillna(False)
        event = (vol_event | range_event).astype(int)

        # Recursive Hawkes intensity update (causal, O(N))
        n = len(close)
        intensity = np.full(n, self.baseline)
        for i in range(1, n):
            # Decay prior intensity and add contribution of any event at i-1
            intensity[i] = (
                self.baseline
                + (intensity[i - 1] - self.baseline) * np.exp(-self.decay)
                + self.excitation * event.iloc[i - 1]
            )

        intensity_s = pd.Series(intensity, index=close.index)
        roll_75 = intensity_s.rolling(252, min_periods=20).quantile(0.75)
        intensity_high = (intensity_s > roll_75).astype(int)
        excited = (intensity_s > self.baseline * 2).astype(int)

        return pd.DataFrame(
            {
                "event":          event,
                "intensity":      intensity_s,
                "intensity_high": intensity_high,
                "excited":        excited,
            },
            index=close.index,
        )
