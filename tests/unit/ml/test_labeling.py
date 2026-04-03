"""Tests for triple-barrier labeling (AFML Chapter 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantstack.ml.labeling import triple_barrier_label, label_series


def test_upper_barrier_hit_first():
    """Price rises to profit-take barrier first -> label = 1."""
    # Price: 100, 101, 102, 103, 104, 105
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    volatility = 2.0  # barrier at 100 + 1.5*2 = 103
    label = triple_barrier_label(
        prices, entry_idx=0, pt_multiplier=1.5, sl_multiplier=3.0,
        max_holding_period=10, volatility=volatility,
    )
    assert label == 1


def test_lower_barrier_hit_first():
    """Price drops to stop-loss barrier first -> label = -1."""
    # Price: 100, 99, 98, 97, 96
    prices = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0])
    volatility = 2.0  # lower barrier at 100 - 1.5*2 = 97
    label = triple_barrier_label(
        prices, entry_idx=0, pt_multiplier=3.0, sl_multiplier=1.5,
        max_holding_period=10, volatility=volatility,
    )
    assert label == -1


def test_time_expired():
    """Price stays within barriers for full holding period -> label = 0."""
    # Price oscillates narrowly
    prices = pd.Series([100.0, 100.1, 99.9, 100.05, 99.95, 100.02])
    volatility = 5.0  # wide barriers: ±7.5 from entry
    label = triple_barrier_label(
        prices, entry_idx=0, pt_multiplier=1.5, sl_multiplier=1.5,
        max_holding_period=5, volatility=volatility,
    )
    assert label == 0


def test_barrier_widths_scale_with_atr():
    """Higher ATR -> wider barriers, harder to hit."""
    prices_up = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0])

    # Low ATR: barrier at 100 + 1.0*2 = 102 -> hits on bar 1
    label_low_atr = triple_barrier_label(
        prices_up, entry_idx=0, pt_multiplier=1.0, sl_multiplier=1.0,
        max_holding_period=10, volatility=2.0,
    )

    # High ATR: barrier at 100 + 1.0*20 = 120 -> never hits
    label_high_atr = triple_barrier_label(
        prices_up, entry_idx=0, pt_multiplier=1.0, sl_multiplier=1.0,
        max_holding_period=4, volatility=20.0,
    )

    assert label_low_atr == 1
    assert label_high_atr == 0  # time expired with wide barriers


def test_label_series_multiple_entries():
    """label_series applies triple-barrier labeling to multiple entry points."""
    n = 100
    rng = np.random.default_rng(42)
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    atr_series = pd.Series(np.full(n, 2.0))
    entry_indices = [0, 20, 40, 60]

    result = label_series(
        prices, entry_indices,
        pt_multiplier=2.0, sl_multiplier=2.0,
        max_holding_period=15, atr_series=atr_series,
    )

    assert len(result) == len(entry_indices)
    assert set(result.columns) >= {"entry_idx", "entry_price", "label", "exit_idx", "holding_period"}
    assert all(result["label"].isin([-1, 0, 1]))


def test_entry_mid_series():
    """Entry at a non-zero index works correctly."""
    prices = pd.Series([50, 51, 52, 100, 101, 102, 103, 104, 105, 106])
    label = triple_barrier_label(
        prices, entry_idx=3, pt_multiplier=1.0, sl_multiplier=1.0,
        max_holding_period=6, volatility=3.0,
    )
    # Upper barrier: 100 + 3 = 103, hits at index 6
    assert label == 1
