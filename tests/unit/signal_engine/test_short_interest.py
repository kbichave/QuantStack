"""Tests for short interest collector (Section 09)."""

from __future__ import annotations

import pytest

from quantstack.signal_engine.collectors.short_interest import (
    compute_short_interest_metrics,
)


def test_short_interest_ratio_computed():
    """SI shares / float -> ratio."""
    metrics = compute_short_interest_metrics(
        si_shares=5_000_000,
        float_shares=50_000_000,
        avg_daily_volume=1_000_000,
    )
    assert abs(metrics["short_interest_ratio"] - 0.10) < 0.001


def test_days_to_cover_computed():
    """DTC = SI / avg_volume."""
    metrics = compute_short_interest_metrics(
        si_shares=5_000_000,
        float_shares=50_000_000,
        avg_daily_volume=1_000_000,
    )
    assert abs(metrics["days_to_cover"] - 5.0) < 0.001


def test_squeeze_candidate_detected():
    """SI > 20% + DTC > 5 -> squeeze."""
    metrics = compute_short_interest_metrics(
        si_shares=12_000_000,
        float_shares=50_000_000,
        avg_daily_volume=1_000_000,
    )
    assert metrics["squeeze_candidate"] is True
    assert metrics["short_interest_ratio"] > 0.20


def test_no_squeeze_low_si():
    """Low SI -> not a squeeze candidate."""
    metrics = compute_short_interest_metrics(
        si_shares=1_000_000,
        float_shares=50_000_000,
        avg_daily_volume=1_000_000,
    )
    assert metrics["squeeze_candidate"] is False


def test_zero_volume_no_crash():
    """avg_volume=0 -> days_to_cover=None, no crash."""
    metrics = compute_short_interest_metrics(
        si_shares=5_000_000,
        float_shares=50_000_000,
        avg_daily_volume=0,
    )
    assert metrics["days_to_cover"] is None
    assert metrics["short_interest_ratio"] == 0.10


def test_zero_float_no_crash():
    """float_shares=0 -> ratio=None, no crash."""
    metrics = compute_short_interest_metrics(
        si_shares=5_000_000,
        float_shares=0,
        avg_daily_volume=1_000_000,
    )
    assert metrics["short_interest_ratio"] is None
