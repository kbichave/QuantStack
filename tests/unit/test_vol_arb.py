"""Tests for volatility arbitrage (Section 10.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantstack.alpha_discovery.vol_arb import (
    calibrate_vol_params,
    compute_vol_spread,
    generate_vol_signal,
    select_structure,
)


def test_vol_spread_computed_correctly():
    """IV - RV spread computed correctly."""
    assert abs(compute_vol_spread(0.35, 0.22) - 0.13) < 1e-6


def test_vol_spread_negative_when_iv_below_rv():
    """IV < RV -> negative spread."""
    spread = compute_vol_spread(0.20, 0.30)
    assert spread < 0


def test_calibration_returns_mean_std():
    """Trailing 252-day mean and std computed."""
    rng = np.random.default_rng(42)
    spread_history = pd.Series(rng.normal(0.08, 0.04, 300))
    params = calibrate_vol_params(spread_history, window=252)

    assert "mean" in params
    assert "std" in params
    assert params["std"] > 0


def test_entry_signal_sell_vol_above_1_std():
    """Spread > mean + 1*std -> sell_vol."""
    params = {"mean": 0.08, "std": 0.04}
    signal = generate_vol_signal(0.15, params)  # z = (0.15-0.08)/0.04 = 1.75
    assert signal == "sell_vol"


def test_no_signal_within_1_std():
    """Spread within 1 std -> no signal."""
    params = {"mean": 0.08, "std": 0.04}
    signal = generate_vol_signal(0.09, params)  # z = (0.09-0.08)/0.04 = 0.25
    assert signal is None


def test_entry_signal_buy_vol_below_neg_1_std():
    """Spread < mean - 1*std -> buy_vol."""
    params = {"mean": 0.08, "std": 0.04}
    signal = generate_vol_signal(-0.01, params)  # z = (-0.01-0.08)/0.04 = -2.25
    assert signal == "buy_vol"


def test_iron_condor_for_sell_vol():
    """sell_vol -> iron_condor structure."""
    structure = select_structure("sell_vol", equity=100_000)
    assert structure["type"] == "iron_condor"
    assert structure["max_loss"] <= 100_000 * 0.02


def test_straddle_for_buy_vol():
    """buy_vol -> straddle structure."""
    structure = select_structure("buy_vol", equity=100_000)
    assert structure["type"] == "straddle"
    assert structure["max_loss"] <= 100_000 * 0.02


def test_max_loss_per_position_constrained():
    """Max loss per position <= 2% of equity."""
    for signal in ("sell_vol", "buy_vol"):
        structure = select_structure(signal, equity=50_000, max_loss_pct=0.02)
        assert structure["max_loss"] <= 50_000 * 0.02
