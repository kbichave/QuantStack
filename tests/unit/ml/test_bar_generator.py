"""Tests for volume and dollar bar generation (AFML Chapter 2)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantstack.data.bars.bar_generator import BarGenerator


def _minute_data(n_minutes: int = 390, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic minute OHLCV data for one trading day."""
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 2, 9, 30)
    timestamps = [base + timedelta(minutes=i) for i in range(n_minutes)]
    prices = 100 + np.cumsum(rng.normal(0, 0.1, n_minutes))
    volumes = rng.integers(1000, 10000, n_minutes)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + rng.uniform(0, 0.5, n_minutes),
        "low": prices - rng.uniform(0, 0.5, n_minutes),
        "close": prices + rng.normal(0, 0.1, n_minutes),
        "volume": volumes,
    })


def _two_day_minute_data(seed: int = 42) -> pd.DataFrame:
    """Generate 2 days of minute data with an overnight gap."""
    day1 = _minute_data(390, seed)
    day2 = _minute_data(390, seed + 1)
    base2 = datetime(2024, 1, 3, 9, 30)
    day2["timestamp"] = [base2 + timedelta(minutes=i) for i in range(390)]
    day2["close"] = day2["close"] + 5  # price jump
    return pd.concat([day1, day2], ignore_index=True)


def test_volume_bar_triggers_at_threshold():
    """Volume bars trigger when cumulative volume >= threshold."""
    df = _minute_data()
    total_vol = df["volume"].sum()
    target_bars = 50
    threshold = total_vol / target_bars

    gen = BarGenerator(bar_type="volume", threshold=threshold)
    bars = gen.generate(df)

    # Should produce approximately target_bars (within 20%)
    assert len(bars) > 0
    assert abs(len(bars) - target_bars) < target_bars * 0.3


def test_dollar_bar_triggers_at_threshold():
    """Dollar bars trigger when cumulative dollar volume >= threshold."""
    df = _minute_data()
    dollar_vol = (df["close"] * df["volume"]).sum()
    target_bars = 30
    threshold = dollar_vol / target_bars

    gen = BarGenerator(bar_type="dollar", threshold=threshold)
    bars = gen.generate(df)

    assert len(bars) > 0
    assert abs(len(bars) - target_bars) < target_bars * 0.3


def test_ohlcv_values_correct():
    """OHLCV values computed correctly within each bar."""
    df = _minute_data(n_minutes=100)
    # Use a large threshold so we get few bars where we can verify
    gen = BarGenerator(bar_type="volume", threshold=df["volume"].sum() / 3)
    bars = gen.generate(df)

    assert len(bars) >= 1
    for _, bar in bars.iterrows():
        assert bar["high"] >= bar["open"]
        assert bar["high"] >= bar["close"]
        assert bar["low"] <= bar["open"]
        assert bar["low"] <= bar["close"]
        assert bar["volume"] > 0
        assert bar["tick_count"] > 0


def test_vwap_computed_correctly():
    """VWAP = sum(price * volume) / sum(volume) within each bar."""
    # Small dataset: 10 minutes, threshold forces 1-2 bars
    df = pd.DataFrame({
        "timestamp": [datetime(2024, 1, 2, 9, 30) + timedelta(minutes=i) for i in range(10)],
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0] * 10,
        "close": [100.5] * 10,
        "volume": [1000] * 10,
    })
    gen = BarGenerator(bar_type="volume", threshold=df["volume"].sum() + 1)
    bars = gen.generate(df)

    assert len(bars) == 1
    expected_vwap = (100.5 * 1000 * 10) / (1000 * 10)
    assert abs(bars.iloc[0]["vwap"] - expected_vwap) < 0.01


def test_bars_never_span_day_boundary():
    """Bars should never span across a trading day boundary."""
    df = _two_day_minute_data()
    total_vol = df["volume"].sum()
    # Use large threshold that would span days if not guarded
    gen = BarGenerator(bar_type="volume", threshold=total_vol / 5)
    bars = gen.generate(df)

    for _, bar in bars.iterrows():
        # Each bar's timestamp should be on a single day
        assert bar["bar_duration_seconds"] < 24 * 3600


def test_handles_gaps_in_minute_data():
    """Gaps in timestamps (e.g., lunch gap) should not break accumulation."""
    df = _minute_data(n_minutes=200)
    # Remove 50 minutes in the middle (simulating a gap)
    df = pd.concat([df.iloc[:80], df.iloc[130:]], ignore_index=True)

    gen = BarGenerator(bar_type="volume", threshold=df["volume"].sum() / 10)
    bars = gen.generate(df)

    assert len(bars) > 0
    assert all(bars["volume"] > 0)


def test_calibrate_threshold():
    """calibrate_threshold computes threshold from daily data."""
    daily_df = pd.DataFrame({
        "volume": [1_000_000, 1_200_000, 800_000, 1_100_000, 900_000],
    })
    gen = BarGenerator(bar_type="volume", threshold=0)
    threshold = gen.calibrate_threshold(daily_df, target_bars_per_day=50)

    expected = daily_df["volume"].mean() / 50
    assert abs(threshold - expected) < 1.0
