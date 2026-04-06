"""Unit tests for ic_calculator.py (section-03)."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantstack.core.ic_calculator import (
    compute_cross_sectional_ic,
    compute_rolling_icir,
    detect_ic_decay,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_aligned_frames(
    signal_values: list[float],
    return_values: list[float],
    date: str = "2024-01-02",
    symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build single-date signals + forward_returns DataFrames for testing."""
    n = len(signal_values)
    if symbols is None:
        symbols = [f"SYM{i:02d}" for i in range(n)]
    idx = pd.DatetimeIndex([date] * n)
    signals = pd.DataFrame({"symbol": symbols, "signal_value": signal_values}, index=idx)
    fwd = pd.DataFrame({"symbol": symbols, "forward_return": return_values}, index=idx)
    return signals, fwd


# ---------------------------------------------------------------------------
# Rank IC formula tests
# ---------------------------------------------------------------------------

def test_rank_ic_formula_matches_scipy():
    """compute_cross_sectional_ic must match scipy.stats.spearmanr on a known dataset."""
    np.random.seed(42)
    n = 20
    signal_vals = np.random.randn(n).tolist()
    return_vals = (np.array(signal_vals) * 0.6 + np.random.randn(n) * 0.4).tolist()

    signals, fwd = _make_aligned_frames(signal_vals, return_vals)
    ic_series = compute_cross_sectional_ic(signals, fwd)

    expected_ic, _ = stats.spearmanr(signal_vals, return_vals)
    assert ic_series.iloc[0] == pytest.approx(expected_ic, abs=1e-9)


def test_perfectly_ranked_signals_return_ic_one():
    """Perfect rank alignment → IC = 1.0."""
    n = 10
    signal_vals = list(range(n))
    return_vals = list(range(n))  # same rank order
    signals, fwd = _make_aligned_frames(signal_vals, return_vals)
    ic_series = compute_cross_sectional_ic(signals, fwd)
    assert ic_series.iloc[0] == pytest.approx(1.0, abs=1e-9)


def test_reversed_ranking_returns_ic_minus_one():
    """Perfectly reversed rank → IC = -1.0."""
    n = 10
    signal_vals = list(range(n))
    return_vals = list(reversed(range(n)))
    signals, fwd = _make_aligned_frames(signal_vals, return_vals)
    ic_series = compute_cross_sectional_ic(signals, fwd)
    assert ic_series.iloc[0] == pytest.approx(-1.0, abs=1e-9)


def test_fewer_than_five_symbols_returns_nan():
    """Cross-section with 4 symbols → NaN (not enough to be statistically meaningful)."""
    signal_vals = [1.0, 2.0, 3.0, 4.0]
    return_vals = [4.0, 3.0, 2.0, 1.0]
    signals, fwd = _make_aligned_frames(signal_vals, return_vals)
    ic_series = compute_cross_sectional_ic(signals, fwd)
    assert math.isnan(ic_series.iloc[0])


def test_multi_date_ic_computed_per_date():
    """With two dates, IC is computed independently per date."""
    np.random.seed(7)
    n = 10
    d1_sig = np.random.randn(n).tolist()
    d1_ret = (np.array(d1_sig) * 0.8 + np.random.randn(n) * 0.2).tolist()
    d2_sig = np.random.randn(n).tolist()
    d2_ret = (-np.array(d2_sig) * 0.5 + np.random.randn(n) * 0.5).tolist()

    dates1 = pd.DatetimeIndex(["2024-01-02"] * n)
    dates2 = pd.DatetimeIndex(["2024-01-03"] * n)
    syms = [f"S{i}" for i in range(n)]

    signals = pd.concat([
        pd.DataFrame({"symbol": syms, "signal_value": d1_sig}, index=dates1),
        pd.DataFrame({"symbol": syms, "signal_value": d2_sig}, index=dates2),
    ])
    fwd = pd.concat([
        pd.DataFrame({"symbol": syms, "forward_return": d1_ret}, index=dates1),
        pd.DataFrame({"symbol": syms, "forward_return": d2_ret}, index=dates2),
    ])

    ic_series = compute_cross_sectional_ic(signals, fwd)
    assert len(ic_series) == 2

    exp_ic1, _ = stats.spearmanr(d1_sig, d1_ret)
    exp_ic2, _ = stats.spearmanr(d2_sig, d2_ret)
    assert ic_series.iloc[0] == pytest.approx(exp_ic1, abs=1e-9)
    assert ic_series.iloc[1] == pytest.approx(exp_ic2, abs=1e-9)


# ---------------------------------------------------------------------------
# Rolling ICIR tests
# ---------------------------------------------------------------------------

def test_icir_rolling_window_equals_mean_over_std():
    """compute_rolling_icir output must equal pd.Series.rolling(w).mean() / .std()."""
    np.random.seed(1)
    ic = pd.Series(np.random.randn(60))
    window = 21

    result = compute_rolling_icir(ic, window)

    roll_mean = ic.rolling(window, min_periods=window).mean()
    roll_std = ic.rolling(window, min_periods=window).std()
    expected = roll_mean / roll_std
    expected[roll_std == 0] = float("nan")

    pd.testing.assert_series_equal(result, expected, check_names=False, atol=1e-9)


def test_icir_early_periods_return_nan():
    """First (window - 1) values must be NaN."""
    ic = pd.Series(np.random.randn(30))
    window = 21
    result = compute_rolling_icir(ic, window)
    assert result.iloc[:window - 1].isna().all()
    assert not math.isnan(result.iloc[window - 1])


def test_icir_std_zero_returns_nan():
    """All identical IC values → std=0 → ICIR = NaN (not inf or large number)."""
    ic = pd.Series([0.3] * 30)
    result = compute_rolling_icir(ic, window=21)
    assert result.iloc[20:].isna().all()


# ---------------------------------------------------------------------------
# IC decay detection tests
# ---------------------------------------------------------------------------

def test_ic_decay_both_windows_below_threshold():
    """Both icir_21d and icir_63d < 0.3 → decay detected."""
    assert detect_ic_decay(0.1, 0.2) is True


def test_ic_decay_both_at_threshold_not_triggered():
    """Exactly 0.3 is NOT below threshold (strict inequality) → no decay."""
    assert detect_ic_decay(0.3, 0.3) is False


def test_ic_decay_only_one_window_below_does_not_trigger():
    """icir_21d < 0.3 but icir_63d >= 0.3 → no decay (AND condition)."""
    assert detect_ic_decay(0.1, 0.4) is False


def test_ic_decay_only_other_window_below_does_not_trigger():
    """icir_63d < 0.3 but icir_21d >= 0.3 → no decay (AND condition)."""
    assert detect_ic_decay(0.5, 0.2) is False


def test_ic_decay_nan_input_returns_false():
    """NaN for either input → False (no action on missing data)."""
    assert detect_ic_decay(float("nan"), 0.1) is False
    assert detect_ic_decay(0.1, float("nan")) is False
    assert detect_ic_decay(float("nan"), float("nan")) is False


def test_hysteresis_repromotion_both_above_threshold():
    """repromotion_check=True: both > 0.5 → eligible for re-promotion (returns True)."""
    assert detect_ic_decay(0.6, 0.7, repromotion_check=True) is True


def test_hysteresis_repromotion_only_one_above_threshold():
    """repromotion_check=True: only one > 0.5 → not eligible (returns False)."""
    assert detect_ic_decay(0.6, 0.4, repromotion_check=True) is False
    assert detect_ic_decay(0.35, 0.6, repromotion_check=True) is False


def test_hysteresis_repromotion_at_threshold_not_eligible():
    """repromotion_check=True: exactly 0.5 is NOT above (strict) → not eligible."""
    assert detect_ic_decay(0.5, 0.5, repromotion_check=True) is False
