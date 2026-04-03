"""Tests for PEAD (Post-Earnings Announcement Drift) strategy (Section 08)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantstack.alpha_discovery.pead import (
    compute_sue,
    generate_pead_signal,
)


# -- 8.1 SUE Computation --


def _make_earnings_history(
    surprises: list[float],
    estimates: list[float] | None = None,
    reported: list[float] | None = None,
) -> pd.DataFrame:
    """Build earnings history DataFrame from surprise values."""
    n = len(surprises)
    dates = pd.bdate_range("2023-01-01", periods=n, freq="QS")
    if estimates is None:
        estimates = [1.0] * n
    if reported is None:
        reported = [est + s for est, s in zip(estimates, surprises)]
    return pd.DataFrame({
        "report_date": dates,
        "estimate": estimates,
        "reported_eps": reported,
        "surprise": surprises,
    }).sort_values("report_date")


def test_sue_computation_correct():
    """SUE = (actual - estimate) / std(past_surprises). Verify with known values."""
    surprises = [0.10, 0.05, -0.05, 0.15, 0.20]
    df = _make_earnings_history(surprises)
    sue = compute_sue("TEST", df, min_quarters=4)

    # std of first 4 surprises: std([0.10, 0.05, -0.05, 0.15])
    past_std = np.std([0.10, 0.05, -0.05, 0.15], ddof=1)
    expected = 0.20 / past_std
    assert sue is not None
    assert abs(sue - expected) < 0.01


def test_sue_with_zero_std_returns_none():
    """All identical surprises -> std=0 -> return None."""
    surprises = [0.10, 0.10, 0.10, 0.10, 0.10]
    df = _make_earnings_history(surprises)
    sue = compute_sue("TEST", df, min_quarters=4)
    assert sue is None


def test_sue_with_insufficient_history_returns_none():
    """Fewer than 4 quarters -> None."""
    surprises = [0.10, 0.05]
    df = _make_earnings_history(surprises)
    sue = compute_sue("TEST", df, min_quarters=4)
    assert sue is None


def test_entry_signal_fires_when_sue_above_threshold():
    """SUE > threshold -> long entry signal."""
    signal = generate_pead_signal("TEST", sue=3.0, sue_threshold=2.0, holding_period_days=60)
    assert signal is not None
    assert signal["direction"] == "long"
    assert signal["holding_period_days"] == 60


def test_no_entry_signal_when_sue_below_threshold():
    """SUE below threshold -> no signal."""
    signal = generate_pead_signal("TEST", sue=1.5, sue_threshold=2.0, holding_period_days=60)
    assert signal is None


def test_no_entry_signal_when_sue_negative():
    """Negative SUE -> no long entry."""
    signal = generate_pead_signal("TEST", sue=-1.5, sue_threshold=2.0, holding_period_days=60)
    assert signal is None


def test_holding_period_matches_parameter():
    """Exit uses the holding_period parameter."""
    signal = generate_pead_signal("TEST", sue=3.0, sue_threshold=2.0, holding_period_days=80)
    assert signal is not None
    assert signal["holding_period_days"] == 80


def test_position_sizing_fallback_when_no_meta_label():
    """Without meta-label, signal uses default half-Kelly sizing."""
    signal = generate_pead_signal("TEST", sue=3.0, sue_threshold=2.0, holding_period_days=60)
    assert signal is not None
    assert signal["sizing_method"] == "fixed_fractional"
    assert 0 < signal["bet_size"] <= 1.0
