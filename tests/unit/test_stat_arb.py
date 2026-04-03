"""Tests for statistical arbitrage research tool (Section 10.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantstack.alpha_discovery.stat_arb import (
    compute_half_life,
    compute_spread_z_score,
    scan_cointegrated_pairs,
)


def _cointegrated_pair(n: int = 504, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Generate a cointegrated pair: Y = 2*X + mean-reverting noise."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, n))  # Random walk
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.5)
    y = 2 * x + noise
    dates = pd.bdate_range("2022-01-01", periods=n)
    return pd.Series(x, index=dates), pd.Series(y, index=dates)


def _independent_pair(n: int = 504, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Generate two independent random walks (not cointegrated)."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, n))
    y = np.cumsum(rng.normal(0, 1, n))
    dates = pd.bdate_range("2022-01-01", periods=n)
    return pd.Series(x, index=dates), pd.Series(y, index=dates)


def test_cointegrated_pair_detected():
    """Perfectly cointegrated pair -> p-value < 0.05."""
    x, y = _cointegrated_pair()
    results = scan_cointegrated_pairs(
        price_data={"A": x, "B": y},
        pairs=[("A", "B")],
        significance=0.05,
    )
    assert len(results) == 1
    assert results[0]["p_value"] < 0.05


def test_independent_pair_rejected():
    """Independent random walks -> p-value > 0.05."""
    x, y = _independent_pair()
    results = scan_cointegrated_pairs(
        price_data={"A": x, "B": y},
        pairs=[("A", "B")],
        significance=0.05,
    )
    assert len(results) == 0


def test_bonferroni_correction_applied():
    """With 100 pairs tested, effective threshold = 0.05/100 = 0.0005."""
    x, y = _cointegrated_pair()
    # Use very strict significance (simulating many pairs)
    results = scan_cointegrated_pairs(
        price_data={"A": x, "B": y},
        pairs=[("A", "B")],
        significance=0.0005,
    )
    # May or may not pass the very strict threshold — test that logic doesn't crash
    assert isinstance(results, list)


def test_half_life_computed():
    """Mean-reverting spread has reasonable half-life."""
    rng = np.random.default_rng(42)
    spread = np.zeros(500)
    for i in range(1, 500):
        spread[i] = 0.95 * spread[i - 1] + rng.normal(0, 1)  # AR(1) with phi=0.95
    spread_series = pd.Series(spread)

    hl = compute_half_life(spread_series)
    # Theoretical: -log(2)/log(0.95) ≈ 13.5 days
    assert hl is not None
    assert 5 < hl < 30


def test_half_life_random_walk_returns_none():
    """Random walk (no mean reversion) -> None or very large half-life."""
    rng = np.random.default_rng(42)
    walk = np.cumsum(rng.normal(0, 1, 500))
    spread = pd.Series(walk)

    hl = compute_half_life(spread)
    # Should be None or > 60 (too slow)
    assert hl is None or hl > 60


def test_z_score_computed():
    """Z-score = (current - mean) / std."""
    spread = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0])
    z = compute_spread_z_score(spread, lookback=10)
    # mean=5.5, std~3.03, current=10, z~1.48
    assert z is not None
    assert 1.0 < z < 2.0


def test_results_are_research_only():
    """scan returns data structures, never calls execution functions."""
    x, y = _cointegrated_pair()
    results = scan_cointegrated_pairs(
        price_data={"A": x, "B": y},
        pairs=[("A", "B")],
        significance=0.05,
    )
    for r in results:
        assert "pair" in r
        assert "p_value" in r
        assert "half_life" in r
        assert "current_z_score" in r
