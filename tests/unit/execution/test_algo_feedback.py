"""Tests for execution algo feedback loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantstack.core.execution.algo_feedback import (
    aggregate_tca_results,
    find_best_algo,
)


def _mock_tca_results(n_per_group: int = 40, seed: int = 42) -> pd.DataFrame:
    """Generate mock TCA results for two algos."""
    rng = np.random.default_rng(seed)
    rows = []
    for algo in ["IMMEDIATE", "TWAP"]:
        mean = 8.0 if algo == "IMMEDIATE" else 3.0
        for i in range(n_per_group):
            rows.append({
                "symbol_adv_bucket": "high_adv",
                "algo_used": algo,
                "time_bucket": "09:30-10:00",
                "shortfall_vs_arrival_bps": rng.normal(mean, 2.0),
            })
    return pd.DataFrame(rows)


def test_daily_aggregation_groups_correctly():
    """Aggregation groups by (symbol_bucket, algo, time_bucket)."""
    df = _mock_tca_results(n_per_group=40)
    agg = aggregate_tca_results(df)

    assert len(agg) == 2  # IMMEDIATE and TWAP
    assert all(col in agg.columns for col in ["algo_used", "mean_shortfall", "count"])


def test_ttest_identifies_significant_difference():
    """T-test flags TWAP as superior when IMMEDIATE has higher shortfall."""
    df = _mock_tca_results(n_per_group=35)
    result = find_best_algo(df, "high_adv", "09:30-10:00", min_fills=30)

    assert result is not None
    assert result["preferred_algo"] == "TWAP"
    assert result["p_value"] < 0.05


def test_insufficient_data_no_rule():
    """< 30 fills per group -> no rule created."""
    df = _mock_tca_results(n_per_group=15)
    result = find_best_algo(df, "high_adv", "09:30-10:00", min_fills=30)

    assert result is None


def test_no_significant_difference_no_rule():
    """When algos have similar shortfall, no rule created."""
    rng = np.random.default_rng(42)
    rows = []
    for algo in ["IMMEDIATE", "TWAP"]:
        for _ in range(40):
            rows.append({
                "symbol_adv_bucket": "mid_adv",
                "algo_used": algo,
                "time_bucket": "10:00-10:30",
                "shortfall_vs_arrival_bps": rng.normal(5.0, 2.0),  # same mean
            })
    df = pd.DataFrame(rows)
    result = find_best_algo(df, "mid_adv", "10:00-10:30", min_fills=30)

    # Should be None (or at least p_value >= 0.05)
    if result is not None:
        assert result["p_value"] >= 0.05
