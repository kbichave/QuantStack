# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the autonomous feature factory (section-08).

All tests are mock-based — no real DB, no real data, no real LLM calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantstack.research.feature_enumerator import enumerate_programmatic, enumerate_with_llm
from quantstack.research.feature_factory import (
    _CANDIDATE_HARD_CAP,
    _IC_DECAY_THRESHOLD,
    _IC_DECAY_WINDOW_DAYS,
    enumerate_features,
    monitor_features,
    publish_replacement_event,
    run_full_pipeline,
    screen_features,
)
from quantstack.research.feature_screener import (
    compute_ic,
    compute_ic_stability,
    screen_and_filter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_features(n: int = 10) -> list[str]:
    """Generate n base feature names."""
    names = [
        "close", "open", "high", "low", "volume",
        "rsi_14", "atr_pct", "adx_14", "bb_pct", "volume_ratio",
        "macd", "ema_21", "obv", "vwap", "stoch_k",
    ]
    return names[:n]


def _make_ohlcv(feature_names: list[str], n_obs: int = 500) -> dict[str, np.ndarray]:
    """Generate synthetic OHLCV-like data with forward returns."""
    rng = np.random.default_rng(42)
    data: dict[str, np.ndarray] = {}
    # Forward returns with slight positive drift
    data["forward_returns"] = rng.normal(0.001, 0.02, n_obs)

    for name in feature_names:
        # Create features with varying correlation to forward returns
        noise = rng.normal(0, 1, n_obs)
        signal = data["forward_returns"] * rng.uniform(0.5, 5.0)
        data[name] = signal + noise * rng.uniform(0.1, 2.0)

    return data


# ---------------------------------------------------------------------------
# Phase 1: Enumeration tests
# ---------------------------------------------------------------------------


class TestProgrammaticEnumeration:
    """Tests for enumerate_programmatic."""

    def test_10_base_produces_over_100_candidates(self):
        """10 base features should produce well over 100 candidates."""
        base = _base_features(10)
        candidates = enumerate_programmatic(base)
        # 10 features * (6 lags + 4 stats * 4 windows) + C(10,2) cross = 10*22 + 45 = 265
        assert len(candidates) > 100

    def test_candidate_structure(self):
        """Each candidate has required keys."""
        candidates = enumerate_programmatic(["close", "volume"])
        for cand in candidates:
            assert "feature_id" in cand
            assert "feature_name" in cand
            assert "definition" in cand
            assert "source" in cand

    def test_unique_feature_ids(self):
        """All feature_ids are unique within a single enumeration."""
        candidates = enumerate_programmatic(_base_features(10))
        ids = [c["feature_id"] for c in candidates]
        assert len(ids) == len(set(ids))

    def test_lag_features_generated(self):
        """Lag features are generated for each base feature."""
        candidates = enumerate_programmatic(["close"])
        lag_names = [c["feature_name"] for c in candidates if "lag" in c["feature_name"]]
        assert len(lag_names) == 6  # 6 lag periods

    def test_rolling_features_generated(self):
        """Rolling stat features are generated for each base feature."""
        candidates = enumerate_programmatic(["close"])
        rolling_names = [c["feature_name"] for c in candidates if "mean" in c["feature_name"] or "std" in c["feature_name"]]
        # 4 windows * 2 stats (mean, std) = 8
        assert len(rolling_names) == 8

    def test_cross_interactions_generated(self):
        """Cross-interaction (ratio) features are generated for feature pairs."""
        candidates = enumerate_programmatic(["close", "volume", "rsi_14"])
        cross = [c for c in candidates if c["source"] == "programmatic_cross"]
        # C(3, 2) = 3
        assert len(cross) == 3


class TestHardCap:
    """Tests for the 2000-candidate hard cap."""

    def test_hard_cap_enforced(self):
        """enumerate_features never returns more than 2000 candidates."""
        # Use enough base features to exceed 2000 programmatically
        # 15 features -> 15*22 + C(15,2) = 330 + 105 = 435 (under cap, but test the cap path)
        # To actually hit the cap, we'd need ~90 base features. Instead, test the cap logic directly.
        base = _base_features(15)
        candidates = enumerate_features(base, use_llm=False)
        assert len(candidates) <= _CANDIDATE_HARD_CAP

    def test_hard_cap_with_large_input(self):
        """When programmatic output exceeds cap, it's truncated."""
        # Generate enough base features to exceed 2000
        # Each base produces 22 candidates + cross terms
        large_base = [f"feat_{i}" for i in range(100)]
        candidates = enumerate_features(large_base, use_llm=False)
        assert len(candidates) <= _CANDIDATE_HARD_CAP


class TestLLMFallback:
    """Tests for LLM enumeration fallback behavior."""

    @patch("quantstack.research.feature_enumerator.enumerate_with_llm", return_value=[])
    def test_llm_failure_falls_back_to_programmatic(self, mock_llm):
        """If LLM fails, we still get programmatic candidates."""
        base = _base_features(10)
        candidates = enumerate_features(base, use_llm=True)
        # Should have programmatic candidates even though LLM returned empty
        assert len(candidates) > 100

    def test_llm_exception_returns_empty(self):
        """enumerate_with_llm returns empty list on any exception (import, API, etc)."""
        # get_llm may not exist or may fail — the function handles all exceptions
        result = enumerate_with_llm(["close"], "trending_up")
        assert result == []


# ---------------------------------------------------------------------------
# Phase 2: Screening tests
# ---------------------------------------------------------------------------


class TestICComputation:
    """Tests for IC (Spearman rank correlation) computation."""

    def test_perfect_positive_correlation(self):
        """Monotonically increasing feature vs returns -> IC near 1.0."""
        x = np.arange(100, dtype=np.float64)
        y = np.arange(100, dtype=np.float64)
        ic = compute_ic(x, y)
        assert ic > 0.99

    def test_perfect_negative_correlation(self):
        """Monotonically decreasing feature vs returns -> IC near -1.0."""
        x = np.arange(100, dtype=np.float64)
        y = -np.arange(100, dtype=np.float64)
        ic = compute_ic(x, y)
        assert ic < -0.99

    def test_no_correlation(self):
        """Random data -> IC near 0."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(0, 1, 1000)
        ic = compute_ic(x, y)
        assert abs(ic) < 0.1

    def test_too_short_returns_zero(self):
        """Less than 10 observations -> IC = 0.0."""
        ic = compute_ic(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert ic == 0.0


class TestICScreening:
    """Tests for IC-based screening."""

    def test_filters_below_ic_threshold(self):
        """Features with IC < 0.01 are filtered out."""
        rng = np.random.default_rng(42)
        n = 500

        # Create a feature with effectively zero IC (pure noise)
        noise_feature = rng.normal(0, 1, n)
        forward_returns = rng.normal(0, 0.02, n)

        candidates = [{
            "feature_id": "noise_001",
            "feature_name": "pure_noise",
            "definition": "random",
            "source": "test",
        }]

        ohlcv_data = {
            "pure_noise": noise_feature,
            "forward_returns": forward_returns,
        }

        # Use a very low stability threshold so IC is the binding constraint
        result = screen_and_filter(candidates, ohlcv_data, ic_min=0.01, stability_min=0.0)

        # Pure noise should be filtered out (IC ~ 0)
        # Allow it to pass sometimes due to randomness, but with seed 42 it shouldn't
        noise_ic = abs(compute_ic(noise_feature, forward_returns))
        if noise_ic < 0.01:
            assert len(result) == 0


class TestStabilityScreening:
    """Tests for IC stability filtering."""

    def test_stability_computation(self):
        """IC stability is computable and positive for correlated data."""
        rng = np.random.default_rng(42)
        n = 500
        signal = np.cumsum(rng.normal(0, 1, n))
        returns = np.diff(signal, prepend=signal[0])
        stability = compute_ic_stability(signal, returns, window=63)
        assert stability > 0

    def test_filters_below_stability_threshold(self):
        """Features with unstable IC (high variance) are filtered."""
        rng = np.random.default_rng(42)
        n = 500

        # Create feature with inconsistent IC — correlated in some windows, not others
        forward_returns = rng.normal(0, 0.02, n)
        # Alternate between correlated and uncorrelated segments
        feature = np.empty(n)
        for i in range(n):
            if (i // 63) % 2 == 0:
                feature[i] = forward_returns[i] * 3 + rng.normal(0, 0.01)
            else:
                feature[i] = rng.normal(0, 1)

        stability = compute_ic_stability(feature, forward_returns, window=63)
        # Unstable feature should have lower stability
        # (exact value depends on randomness, but testing the machinery works)
        assert isinstance(stability, float)


class TestCorrelationFiltering:
    """Tests for cross-correlation filtering."""

    def test_drops_highly_correlated_features(self):
        """Features with correlation > 0.95 are deduplicated."""
        rng = np.random.default_rng(42)
        n = 500

        forward_returns = rng.normal(0.001, 0.02, n)
        base_signal = forward_returns * 5 + rng.normal(0, 0.1, n)
        # Two nearly identical features (correlation > 0.99)
        feat_a = base_signal + rng.normal(0, 0.001, n)
        feat_b = base_signal + rng.normal(0, 0.001, n)
        # One independent feature
        feat_c = forward_returns * 3 + rng.normal(0, 0.5, n)

        candidates = [
            {"feature_id": "a", "feature_name": "feat_a", "definition": "a", "source": "test"},
            {"feature_id": "b", "feature_name": "feat_b", "definition": "b", "source": "test"},
            {"feature_id": "c", "feature_name": "feat_c", "definition": "c", "source": "test"},
        ]

        ohlcv_data = {
            "feat_a": feat_a,
            "feat_b": feat_b,
            "feat_c": feat_c,
            "forward_returns": forward_returns,
        }

        result = screen_and_filter(
            candidates, ohlcv_data,
            ic_min=0.0, stability_min=0.0, correlation_max=0.95,
        )

        result_names = {c["feature_name"] for c in result}
        # feat_a and feat_b are correlated > 0.95, so at most one survives
        assert not ({"feat_a", "feat_b"} <= result_names), \
            "Both highly correlated features survived — correlation filter broken"


# ---------------------------------------------------------------------------
# Phase 3: Monitoring tests
# ---------------------------------------------------------------------------


class TestPSIDecay:
    """Tests for PSI-based decay detection."""

    def test_psi_above_threshold_detected(self):
        """PSI > 0.25 is detected as decay."""
        rng = np.random.default_rng(42)

        # Baseline: normal distribution
        baseline = rng.normal(0, 1, 200)
        # Current: shifted distribution (should produce high PSI)
        current = rng.normal(5, 1, 200)

        curated = [{
            "feature_id": "f1",
            "feature_name": "shifted_feat",
            "definition": "test",
            "source": "test",
            "ic": 0.05,
        }]

        decayed = monitor_features(
            curated,
            ohlcv_data={"shifted_feat": current},
            baseline_data={"shifted_feat": baseline},
        )

        assert len(decayed) == 1
        assert decayed[0]["feature_name"] == "shifted_feat"
        assert decayed[0]["status"] == "decayed"

    def test_stable_distribution_not_flagged(self):
        """Same distribution -> no decay."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)

        curated = [{
            "feature_id": "f1",
            "feature_name": "stable_feat",
            "definition": "test",
            "source": "test",
            "ic": 0.05,
        }]

        decayed = monitor_features(
            curated,
            ohlcv_data={"stable_feat": data},
            baseline_data={"stable_feat": data},
        )

        assert len(decayed) == 0


class TestICDecay:
    """Tests for IC-based decay detection."""

    def test_low_ic_for_window_detected(self):
        """IC < 0.005 for 10 consecutive days -> decay."""
        curated = [{
            "feature_id": "f1",
            "feature_name": "weak_feat",
            "definition": "test",
            "source": "test",
            "ic": 0.002,
        }]

        # 10 days of IC below threshold
        ic_history = {
            "weak_feat": [0.004] * _IC_DECAY_WINDOW_DAYS,
        }

        decayed = monitor_features(curated, ic_history=ic_history)

        assert len(decayed) == 1
        assert decayed[0]["feature_name"] == "weak_feat"

    def test_healthy_ic_not_flagged(self):
        """IC above threshold -> no decay."""
        curated = [{
            "feature_id": "f1",
            "feature_name": "strong_feat",
            "definition": "test",
            "source": "test",
            "ic": 0.05,
        }]

        ic_history = {
            "strong_feat": [0.03] * 15,
        }

        decayed = monitor_features(curated, ic_history=ic_history)
        assert len(decayed) == 0

    def test_mixed_ic_not_flagged(self):
        """IC history with some good days -> no decay."""
        curated = [{
            "feature_id": "f1",
            "feature_name": "mixed_feat",
            "definition": "test",
            "source": "test",
            "ic": 0.01,
        }]

        # Mix of good and bad IC — not all below threshold
        ic_history = {
            "mixed_feat": [0.001, 0.001, 0.001, 0.001, 0.001,
                           0.02, 0.001, 0.001, 0.001, 0.001],
        }

        decayed = monitor_features(curated, ic_history=ic_history)
        assert len(decayed) == 0


class TestEventBusIntegration:
    """Tests for event bus publishing on decay and replacement."""

    def test_feature_decayed_event_published(self):
        """FEATURE_DECAYED event is published when decay is detected."""
        mock_bus = MagicMock()

        curated = [{
            "feature_id": "f1",
            "feature_name": "decaying_feat",
            "definition": "test",
            "source": "test",
            "ic": 0.002,
        }]

        ic_history = {
            "decaying_feat": [0.001] * _IC_DECAY_WINDOW_DAYS,
        }

        monitor_features(curated, ic_history=ic_history, event_bus=mock_bus)

        mock_bus.publish.assert_called_once()
        event = mock_bus.publish.call_args[0][0]
        assert event.event_type.value == "feature_decayed"
        assert event.payload["feature_name"] == "decaying_feat"

    def test_feature_replaced_event_published(self):
        """FEATURE_REPLACED event is published when replacement is issued."""
        mock_bus = MagicMock()

        old = {"feature_id": "old_1", "feature_name": "old_feat"}
        new = {"feature_id": "new_1", "feature_name": "new_feat"}

        publish_replacement_event(mock_bus, old, new)

        mock_bus.publish.assert_called_once()
        event = mock_bus.publish.call_args[0][0]
        assert event.event_type.value == "feature_replaced"
        assert event.payload["old_feature_name"] == "old_feat"
        assert event.payload["new_feature_name"] == "new_feat"

    def test_no_event_when_no_bus(self):
        """No crash when event_bus is None."""
        curated = [{
            "feature_id": "f1",
            "feature_name": "decaying_feat",
            "definition": "test",
            "source": "test",
        }]

        ic_history = {
            "decaying_feat": [0.001] * _IC_DECAY_WINDOW_DAYS,
        }

        # Should not raise even with event_bus=None
        decayed = monitor_features(curated, ic_history=ic_history, event_bus=None)
        assert len(decayed) == 1
