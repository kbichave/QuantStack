# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for LSVHerding and InstitutionalConcentration from 13F filings.
"""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.features.institutional_signals import (
    LSVHerding,
    InstitutionalConcentration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ownership(n: int = 12, seed: int = 42) -> pd.DataFrame:
    """Return a quarterly 13F ownership snapshot history."""
    np.random.seed(seed)
    dates = pd.date_range("2020-03-31", periods=n, freq="QE")
    total_holders = np.random.randint(200, 500, n)
    # Simulate ~55% buying on average with some variation
    buying_frac = np.clip(np.random.normal(0.55, 0.1, n), 0, 1)
    holders_increased = (total_holders * buying_frac).astype(int)
    holders_decreased = total_holders - holders_increased
    return pd.DataFrame(
        {
            "period_end": dates,
            "total_holders": total_holders,
            "holders_increased": holders_increased,
            "holders_decreased": holders_decreased,
        }
    )


def _make_concentration(n: int = 12, seed: int = 7) -> pd.DataFrame:
    """Return quarterly holder count data for concentration signal."""
    np.random.seed(seed)
    dates = pd.date_range("2020-03-31", periods=n, freq="QE")
    total_holders = 200 + np.cumsum(np.random.randint(-10, 20, n))
    return pd.DataFrame(
        {
            "period_end": dates,
            "total_holders": np.maximum(total_holders, 1),
        }
    )


# ---------------------------------------------------------------------------
# LSVHerding
# ---------------------------------------------------------------------------


class TestLSVHerding:
    def test_returns_dataframe(self):
        result = LSVHerding().compute(_make_ownership())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = LSVHerding().compute(_make_ownership())
        for col in (
            "fraction_buying",
            "expected_buying",
            "herding_measure",
            "herding_buy_bias",
            "herding_high",
        ):
            assert col in result.columns, f"missing: {col}"

    def test_same_length(self):
        df = _make_ownership(16)
        result = LSVHerding().compute(df)
        assert len(result) == 16

    def test_fraction_buying_in_zero_one(self):
        result = LSVHerding().compute(_make_ownership())
        valid = result["fraction_buying"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_herding_measure_non_negative(self):
        result = LSVHerding().compute(_make_ownership())
        valid = result["herding_measure"].dropna()
        assert (valid >= 0.0).all()

    def test_herding_buy_bias_in_neg1_0_1(self):
        result = LSVHerding().compute(_make_ownership())
        valid = result["herding_buy_bias"].dropna()
        assert valid.isin([-1, 0, 1]).all()

    def test_herding_high_binary(self):
        result = LSVHerding().compute(_make_ownership())
        valid = result["herding_high"].dropna()
        assert valid.isin([0, 1]).all()

    def test_mostly_buying_gives_positive_or_zero_bias(self):
        """When fraction_buying consistently exceeds expected, bias should be >= 0."""
        n = 12
        # First 4 quarters: 50% buying (sets the expected baseline low)
        # Last 8 quarters: 90% buying (exceeds baseline → positive herding bias)
        holders = [200] * n
        increased = [100] * 4 + [180] * 8
        decreased = [100] * 4 + [20] * 8
        df = pd.DataFrame(
            {
                "period_end": pd.date_range("2020-03-31", periods=n, freq="QE"),
                "total_holders": holders,
                "holders_increased": increased,
                "holders_decreased": decreased,
            }
        )
        result = LSVHerding(rolling_window=4).compute(df)
        # In the 90%-buying phase, bias should be non-negative
        non_null = result["herding_buy_bias"].iloc[-4:].dropna()
        assert (non_null >= 0).all()

    def test_mostly_selling_gives_negative_or_zero_bias(self):
        """When fraction_buying consistently below expected, bias should be <= 0."""
        n = 12
        holders = [200] * n
        increased = [100] * 4 + [20] * 8
        decreased = [100] * 4 + [180] * 8
        df = pd.DataFrame(
            {
                "period_end": pd.date_range("2020-03-31", periods=n, freq="QE"),
                "total_holders": holders,
                "holders_increased": increased,
                "holders_decreased": decreased,
            }
        )
        result = LSVHerding(rolling_window=4).compute(df)
        non_null = result["herding_buy_bias"].iloc[-4:].dropna()
        assert (non_null <= 0).all()

    def test_sorted_by_period_end(self):
        df = _make_ownership()
        df_shuffled = df.sample(frac=1, random_state=55).reset_index(drop=True)
        result = LSVHerding().compute(df_shuffled)
        dates = pd.to_datetime(result["period_end"]).values
        assert (dates[1:] >= dates[:-1]).all()

    def test_zero_total_changes_no_crash(self):
        """When no institution increased or decreased, no crash."""
        df = pd.DataFrame(
            {
                "period_end": pd.date_range("2020-03-31", periods=4, freq="QE"),
                "total_holders": [300, 300, 300, 300],
                "holders_increased": [0, 0, 0, 0],
                "holders_decreased": [0, 0, 0, 0],
            }
        )
        result = LSVHerding().compute(df)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# InstitutionalConcentration
# ---------------------------------------------------------------------------


class TestInstitutionalConcentration:
    def test_returns_dataframe(self):
        result = InstitutionalConcentration().compute(_make_concentration())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = InstitutionalConcentration().compute(_make_concentration())
        # Core expected columns
        assert "holder_change" in result.columns
        assert "holder_change_pct" in result.columns

    def test_same_length(self):
        df = _make_concentration(16)
        result = InstitutionalConcentration().compute(df)
        assert len(result) == 16

    def test_holder_count_change_direction(self):
        """When holder count monotonically increases, change should be non-negative."""
        n = 8
        df = pd.DataFrame(
            {
                "period_end": pd.date_range("2020-03-31", periods=n, freq="QE"),
                "total_holders": 100 + np.arange(n) * 10,
            }
        )
        result = InstitutionalConcentration().compute(df)
        # Skip first row (NaN or zero change)
        changes = result["holder_change"].iloc[1:].dropna()
        assert (changes >= 0).all()

    def test_sorted_by_period_end(self):
        df = _make_concentration()
        df_shuffled = df.sample(frac=1, random_state=88).reset_index(drop=True)
        result = InstitutionalConcentration().compute(df_shuffled)
        if "period_end" in result.columns:
            dates = pd.to_datetime(result["period_end"]).values
        else:
            dates = pd.to_datetime(result.index).values
        assert (dates[1:] >= dates[:-1]).all()
