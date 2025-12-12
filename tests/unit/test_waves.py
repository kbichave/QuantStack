"""
Unit tests for wave pattern detection.

Tests verify:
- Impulse pattern detection (5-wave structures)
- ABC correction detection
- Wave grammar constraints (retrace limits, wave sizes)
- Partial pattern scoring
- Wave feature computation
"""

import pytest
import pandas as pd
import numpy as np

from quantcore.features.waves import (
    SwingDetector,
    WaveLabeler,
    WaveFeatures,
    PartialPatternScorer,
    WaveConfig,
    SwingPoint,
    SwingLeg,
    WavePattern,
)
from quantcore.config.timeframes import Timeframe
from tests.conftest import (
    make_impulse_wave_ohlcv,
    make_impulse_up_legs,
    make_swing_leg,
    add_atr_column,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def wave_config():
    """Default wave configuration for testing."""
    return WaveConfig()


@pytest.fixture
def swing_detector(wave_config):
    """SwingDetector instance with default config."""
    return SwingDetector(wave_config)


@pytest.fixture
def wave_labeler(wave_config):
    """WaveLabeler instance with default config."""
    return WaveLabeler(wave_config)


@pytest.fixture
def partial_scorer(wave_config):
    """PartialPatternScorer instance with default config."""
    return PartialPatternScorer(wave_config)


@pytest.fixture
def wave_features():
    """WaveFeatures instance for H4 timeframe."""
    return WaveFeatures(Timeframe.H4)


@pytest.fixture
def valid_impulse_up_legs():
    """5 legs forming a valid impulse-up pattern."""
    return make_impulse_up_legs()


@pytest.fixture
def impulse_wave_df():
    """DataFrame with impulse wave price pattern."""
    df = make_impulse_wave_ohlcv()
    return add_atr_column(df)


# =============================================================================
# Test: Impulse Pattern Detection
# =============================================================================


class TestImpulseDetection:
    """Tests for impulse pattern detection."""

    def test_detect_impulse_up_valid_pattern(self, wave_labeler, valid_impulse_up_legs):
        """
        Verify that a valid impulse-up pattern is detected.

        Scenario: 5 legs with correct directions and wave relationships.
        Expected: At least one impulse_up pattern detected.
        """
        patterns = wave_labeler.detect_impulse_up(valid_impulse_up_legs)

        assert len(patterns) >= 1, "Should detect at least one impulse-up pattern"
        assert patterns[0].type == "impulse_up"
        assert len(patterns[0].leg_indices) == 5
        assert patterns[0].stages == [1, 2, 3, 4, 5]

    def test_impulse_up_directions_correct(self, wave_labeler, valid_impulse_up_legs):
        """
        Verify detected impulse has correct leg directions.

        Invariant: Impulse-up must have directions [up, down, up, down, up].
        """
        patterns = wave_labeler.detect_impulse_up(valid_impulse_up_legs)

        if patterns:
            pattern = patterns[0]
            directions = [
                valid_impulse_up_legs[i].direction for i in pattern.leg_indices
            ]
            assert directions == ["up", "down", "up", "down", "up"]

    def test_no_impulse_wrong_directions(self, wave_labeler):
        """
        Verify no impulse detected when directions don't alternate correctly.

        Scenario: 5 legs but starting with down instead of up.
        Expected: No impulse-up detected.
        """
        # Create legs with wrong direction sequence
        legs = [
            make_swing_leg(0, 5, 100, 95, "down"),  # Should be up
            make_swing_leg(5, 10, 95, 98, "up"),
            make_swing_leg(10, 15, 98, 90, "down"),
            make_swing_leg(15, 20, 90, 92, "up"),
            make_swing_leg(20, 25, 92, 85, "down"),
        ]

        patterns = wave_labeler.detect_impulse_up(legs)
        assert len(patterns) == 0, "Should not detect impulse-up with wrong directions"

    def test_wave2_retrace_limit(self, wave_labeler, wave_config):
        """
        Verify wave 2 retrace constraint is enforced.

        Invariant: Wave 2 cannot retrace more than 100% of Wave 1.
        """
        # Create legs where wave 2 retraces more than 100% of wave 1
        legs = [
            make_swing_leg(0, 5, 100, 110, "up"),  # Wave 1: +10%
            make_swing_leg(5, 10, 110, 98, "down"),  # Wave 2: -10.9% (>100% retrace)
            make_swing_leg(10, 15, 98, 120, "up"),
            make_swing_leg(15, 20, 120, 115, "down"),
            make_swing_leg(20, 25, 115, 130, "up"),
        ]

        patterns = wave_labeler.detect_impulse_up(legs)
        # Pattern should not be detected due to wave 2 violation
        assert len(patterns) == 0, "Should reject pattern where wave 2 retraces >100%"

    def test_wave3_minimum_size(self, wave_labeler, wave_config):
        """
        Verify wave 3 must be at least as large as wave 1.

        Invariant: Wave 3 >= Wave 1 in magnitude.
        """
        # Create legs where wave 3 is smaller than wave 1
        legs = [
            make_swing_leg(0, 5, 100, 115, "up"),  # Wave 1: +15%
            make_swing_leg(5, 10, 115, 108, "down"),  # Wave 2
            make_swing_leg(10, 15, 108, 118, "up"),  # Wave 3: +9.3% (< wave 1)
            make_swing_leg(15, 20, 118, 114, "down"),
            make_swing_leg(20, 25, 114, 125, "up"),
        ]

        patterns = wave_labeler.detect_impulse_up(legs)
        assert len(patterns) == 0, "Should reject pattern where wave 3 < wave 1"


# =============================================================================
# Test: ABC Correction Detection
# =============================================================================


class TestABCDetection:
    """Tests for ABC correction pattern detection."""

    def test_detect_abc_down_valid(self, wave_labeler):
        """
        Verify valid ABC down pattern is detected.

        Scenario: 3 legs with directions [down, up, down].
        """
        legs = [
            make_swing_leg(0, 5, 100, 92, "down"),  # A wave
            make_swing_leg(5, 10, 92, 96, "up"),  # B wave (50% retrace)
            make_swing_leg(10, 15, 96, 86, "down"),  # C wave
        ]

        patterns = wave_labeler.detect_abc_down(legs)

        assert len(patterns) >= 1, "Should detect ABC down pattern"
        assert patterns[0].type == "abc_down"
        assert len(patterns[0].leg_indices) == 3

    def test_detect_abc_up_valid(self, wave_labeler):
        """
        Verify valid ABC up pattern is detected.

        Scenario: 3 legs with directions [up, down, up].
        """
        legs = [
            make_swing_leg(0, 5, 100, 108, "up"),  # A wave
            make_swing_leg(5, 10, 108, 104, "down"),  # B wave
            make_swing_leg(10, 15, 104, 115, "up"),  # C wave
        ]

        patterns = wave_labeler.detect_abc_up(legs)

        assert len(patterns) >= 1, "Should detect ABC up pattern"
        assert patterns[0].type == "abc_up"

    def test_c_wave_minimum_size(self, wave_labeler, wave_config):
        """
        Verify C wave minimum size constraint.

        Invariant: C wave must be at least 61.8% of A wave.
        """
        # Create legs where C wave is too small
        legs = [
            make_swing_leg(0, 5, 100, 90, "down"),  # A: -10%
            make_swing_leg(5, 10, 90, 95, "up"),  # B
            make_swing_leg(10, 15, 95, 93, "down"),  # C: -2.1% (only ~21% of A)
        ]

        patterns = wave_labeler.detect_abc_down(legs)
        assert len(patterns) == 0, "Should reject ABC where C < 61.8% of A"


# =============================================================================
# Test: Swing Alternation Invariant
# =============================================================================


class TestSwingAlternation:
    """Tests for swing direction alternation invariant."""

    def test_detected_swings_alternate(self, swing_detector, impulse_wave_df):
        """
        Verify detected swings strictly alternate direction.

        Invariant: No two consecutive swings can have the same direction.
        """
        swings = swing_detector.detect(impulse_wave_df)

        for i in range(1, len(swings)):
            prev_dir = swings[i - 1].direction
            curr_dir = swings[i].direction
            assert (
                prev_dir != curr_dir
            ), f"Swings at {swings[i-1].idx} and {swings[i].idx} both are '{curr_dir}'"

    def test_legs_have_consistent_direction(self, swing_detector, impulse_wave_df):
        """
        Verify leg direction matches price movement.

        Invariant: Up leg = positive return, down leg = negative return.
        """
        swings = swing_detector.detect(impulse_wave_df)
        legs = swing_detector.build_legs(swings)

        for leg in legs:
            if leg.direction == "up":
                assert leg.ret_pct > 0, f"Up leg has non-positive return: {leg.ret_pct}"
                assert leg.end_price > leg.start_price
            else:
                assert (
                    leg.ret_pct < 0
                ), f"Down leg has non-negative return: {leg.ret_pct}"
                assert leg.end_price < leg.start_price


# =============================================================================
# Test: Partial Pattern Scoring
# =============================================================================


class TestPartialPatternScoring:
    """Tests for partial pattern probability scoring."""

    def test_partial_impulse_up_score_increases_with_legs(self, partial_scorer):
        """
        Verify partial impulse score increases with more confirming legs.
        """
        # 2 legs
        legs_2 = [
            make_swing_leg(0, 5, 100, 110, "up"),
            make_swing_leg(5, 10, 110, 105, "down"),
        ]
        score_2 = partial_scorer.score_partial_impulse_up(legs_2)

        # 3 legs (adding wave 3)
        legs_3 = legs_2 + [make_swing_leg(10, 15, 105, 125, "up")]
        score_3 = partial_scorer.score_partial_impulse_up(legs_3)

        # 4 legs
        legs_4 = legs_3 + [make_swing_leg(15, 20, 125, 120, "down")]
        score_4 = partial_scorer.score_partial_impulse_up(legs_4)

        assert score_3 >= score_2, "Score should increase with wave 3 confirmation"
        assert score_4 >= score_3, "Score should increase with wave 4 confirmation"

    def test_partial_score_zero_for_wrong_directions(self, partial_scorer):
        """
        Verify partial score is zero when directions don't match impulse pattern.
        """
        # Wrong direction sequence for impulse up
        legs = [
            make_swing_leg(0, 5, 100, 95, "down"),
            make_swing_leg(5, 10, 95, 100, "up"),
        ]

        score = partial_scorer.score_partial_impulse_up(legs)
        assert score == 0.0, "Should return 0 for wrong direction sequence"


# =============================================================================
# Test: Wave Feature Computation
# =============================================================================


class TestWaveFeatureComputation:
    """Tests for wave feature computation via WaveFeatures class."""

    def test_compute_adds_expected_columns(self, wave_features, impulse_wave_df):
        """
        Verify compute() adds all expected wave feature columns.
        """
        result = wave_features.compute(impulse_wave_df)

        expected_cols = [
            "wave_role",
            "wave_stage",
            "wave_conf",
            "prob_impulse_up",
            "prob_impulse_down",
            "prob_corr_down",
            "prob_corr_up",
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_get_feature_names_returns_list(self, wave_features):
        """
        Verify get_feature_names() returns expected feature names.
        """
        names = wave_features.get_feature_names()

        assert isinstance(names, list)
        assert len(names) >= 7
        assert "wave_role" in names
        assert "wave_stage" in names

    def test_wave_probabilities_sum_reasonable(self, wave_features, impulse_wave_df):
        """
        Verify wave probability outputs are in valid range.
        """
        result = wave_features.compute(impulse_wave_df)

        prob_cols = [
            "prob_impulse_up",
            "prob_impulse_down",
            "prob_corr_down",
            "prob_corr_up",
        ]

        for col in prob_cols:
            assert (result[col] >= 0).all(), f"{col} has negative values"
            assert (result[col] <= 1).all(), f"{col} has values > 1"


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestWaveEdgeCases:
    """Edge case tests for wave detection."""

    def test_insufficient_legs_for_impulse(self, wave_labeler):
        """
        Verify no impulse detected with fewer than 5 legs.
        """
        legs = [
            make_swing_leg(0, 5, 100, 110, "up"),
            make_swing_leg(5, 10, 110, 105, "down"),
            make_swing_leg(10, 15, 105, 120, "up"),
        ]

        patterns = wave_labeler.detect_impulse_up(legs)
        assert len(patterns) == 0, "Should not detect impulse with only 3 legs"

    def test_empty_legs_list(self, wave_labeler, partial_scorer):
        """
        Verify graceful handling of empty legs list.
        """
        patterns = wave_labeler.detect_all_patterns([])
        assert patterns == []

        score = partial_scorer.score_partial_impulse_up([])
        assert score == 0.0

    def test_very_small_price_movements(self, wave_labeler, wave_config):
        """
        Verify handling of legs with very small price movements.
        """
        # Legs below minimum impulse threshold
        legs = [
            make_swing_leg(0, 5, 100, 100.5, "up"),  # 0.5%
            make_swing_leg(5, 10, 100.5, 100.3, "down"),
            make_swing_leg(10, 15, 100.3, 100.8, "up"),
            make_swing_leg(15, 20, 100.8, 100.6, "down"),
            make_swing_leg(20, 25, 100.6, 101.0, "up"),
        ]

        patterns = wave_labeler.detect_impulse_up(legs)
        # Should not detect patterns below minimum thresholds
        assert len(patterns) == 0, "Should not detect impulse with tiny movements"


# =============================================================================
# Test: Pattern Confidence Scoring
# =============================================================================


class TestPatternConfidence:
    """Tests for pattern confidence scoring."""

    def test_ideal_fib_retrace_higher_confidence(self, wave_labeler):
        """
        Verify patterns with ideal Fibonacci retraces get higher confidence.
        """
        # Pattern with ideal 50% wave 2 retrace
        ideal_legs = [
            make_swing_leg(0, 5, 100, 110, "up"),  # Wave 1: +10%
            make_swing_leg(5, 10, 110, 105, "down"),  # Wave 2: -50% retrace (ideal)
            make_swing_leg(10, 15, 105, 125, "up"),  # Wave 3
            make_swing_leg(15, 20, 125, 120, "down"),  # Wave 4: -25% retrace (ideal)
            make_swing_leg(20, 25, 120, 135, "up"),  # Wave 5
        ]

        # Pattern with non-ideal retraces
        non_ideal_legs = [
            make_swing_leg(0, 5, 100, 110, "up"),
            make_swing_leg(5, 10, 110, 101, "down"),  # ~90% retrace (poor)
            make_swing_leg(10, 15, 101, 125, "up"),
            make_swing_leg(15, 20, 125, 107, "down"),  # ~75% retrace (poor)
            make_swing_leg(20, 25, 107, 130, "up"),
        ]

        ideal_patterns = wave_labeler.detect_impulse_up(ideal_legs)
        non_ideal_patterns = wave_labeler.detect_impulse_up(non_ideal_legs)

        if ideal_patterns and non_ideal_patterns:
            assert (
                ideal_patterns[0].confidence >= non_ideal_patterns[0].confidence
            ), "Ideal Fib retraces should have higher or equal confidence"

    def test_confidence_in_valid_range(self, wave_labeler, valid_impulse_up_legs):
        """
        Verify confidence scores are in [0, 1] range.
        """
        patterns = wave_labeler.detect_all_patterns(valid_impulse_up_legs)

        for pattern in patterns:
            assert (
                0 <= pattern.confidence <= 1
            ), f"Confidence {pattern.confidence} outside [0, 1] range"
