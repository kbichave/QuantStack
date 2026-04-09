"""Tests for 4.3 — Look-Ahead Bias Detection (QS-S4)."""

from datetime import datetime

import pandas as pd
import pytest

from quantstack.core.validation.lookahead_detector import (
    FeatureTimestamp,
    LookaheadViolation,
    check_feature_availability,
    check_lookahead,
)


class TestCheckLookahead:
    def test_violation_detected(self):
        """Feature known after signal time is flagged."""
        signal_time = datetime(2024, 3, 15, 9, 30)
        features = [
            FeatureTimestamp("eps_surprise", known_since=datetime(2024, 3, 17)),
        ]
        violations = check_lookahead(features, signal_time)
        assert len(violations) == 1
        assert violations[0].feature_name == "eps_surprise"
        assert violations[0].violation_days > 0

    def test_no_violation(self):
        """Feature known before signal time passes."""
        signal_time = datetime(2024, 6, 1, 16, 0)
        features = [
            FeatureTimestamp("rsi_14", known_since=datetime(2024, 5, 31)),
        ]
        violations = check_lookahead(features, signal_time)
        assert len(violations) == 0

    def test_exact_boundary(self):
        """Feature known at exactly signal time is NOT a violation."""
        t = datetime(2024, 6, 1, 9, 30)
        features = [FeatureTimestamp("momentum", known_since=t)]
        violations = check_lookahead(features, t)
        assert len(violations) == 0

    def test_multiple_features_mixed(self):
        """Mix of clean and violated features."""
        signal_time = datetime(2024, 3, 15)
        features = [
            FeatureTimestamp("clean_feature", known_since=datetime(2024, 3, 10)),
            FeatureTimestamp("bad_feature", known_since=datetime(2024, 3, 20)),
            FeatureTimestamp("also_bad", known_since=datetime(2024, 4, 1)),
        ]
        violations = check_lookahead(features, signal_time)
        assert len(violations) == 2
        names = {v.feature_name for v in violations}
        assert names == {"bad_feature", "also_bad"}


class TestCheckFeatureAvailability:
    def test_unshifted_fundamental_flagged(self):
        """A quarterly fundamental with no PIT shift should be flagged."""
        dates = pd.bdate_range("2024-01-02", periods=50)
        df = pd.DataFrame(
            {"sloan_accruals": range(50)},
            index=dates,
        )
        feature_sources = {"sloan_accruals": "fundamental_quarterly"}
        violations = check_feature_availability(df, feature_sources)
        assert len(violations) >= 1
        assert violations[0].feature_name == "sloan_accruals"

    def test_zero_delay_source_clean(self):
        """Earnings surprise (0-day delay) should never flag."""
        dates = pd.bdate_range("2024-01-02", periods=50)
        df = pd.DataFrame({"eps_surprise": range(50)}, index=dates)
        feature_sources = {"eps_surprise": "earnings_surprise"}
        violations = check_feature_availability(df, feature_sources)
        assert len(violations) == 0

    def test_missing_column_skipped(self):
        """Features not in DataFrame are silently skipped."""
        dates = pd.bdate_range("2024-01-02", periods=50)
        df = pd.DataFrame({"close": range(50)}, index=dates)
        feature_sources = {"nonexistent_col": "fundamental_quarterly"}
        violations = check_feature_availability(df, feature_sources)
        assert len(violations) == 0
