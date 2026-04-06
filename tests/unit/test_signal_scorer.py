"""Unit tests for signal_scorer.py (section-02)."""

import pytest
from quantstack.core.signal_scorer import score_signal


# ---------------------------------------------------------------------------
# Basic satisfaction tests
# ---------------------------------------------------------------------------

def test_all_rules_satisfied_returns_positive_signal():
    """All bullish rules satisfied → signal_value near +1.0."""
    rules = [
        {"field": "RSI_14", "operator": "<", "threshold": 30, "direction": "bullish"},
        {"field": "close", "operator": "<", "threshold": 100, "direction": "bullish"},
    ]
    market_data = {"RSI_14": 10, "close": 50}  # both well below threshold (2x distance → 1.0 each)
    signal_value, confidence = score_signal(rules, market_data)
    assert signal_value > 0.9
    assert confidence == 1.0


def test_no_rules_satisfied_returns_zero_signal():
    """Bullish rules where market is on the wrong side → score 0.0 each → signal_value = 0.0."""
    rules = [
        {"field": "RSI_14", "operator": "<", "threshold": 30, "direction": "bullish"},
        {"field": "close", "operator": "<", "threshold": 100, "direction": "bullish"},
    ]
    market_data = {"RSI_14": 60, "close": 150}  # both above threshold → rule not satisfied
    signal_value, confidence = score_signal(rules, market_data)
    assert signal_value == pytest.approx(0.0)
    assert confidence == 1.0


def test_half_evaluable_returns_correct_confidence():
    """2 of 4 rules have market_data → confidence = 0.5."""
    rules = [
        {"field": "RSI_14", "operator": "<", "threshold": 30, "direction": "bullish"},
        {"field": "MACD", "operator": ">", "threshold": 0, "direction": "bullish"},
        {"field": "MISSING_A", "operator": "<", "threshold": 50},
        {"field": "MISSING_B", "operator": ">", "threshold": 10},
    ]
    market_data = {"RSI_14": 20, "MACD": 1.0}
    signal_value, confidence = score_signal(rules, market_data)
    # confidence = 2/4 = 0.5 (exactly half, which is >= 0.5 threshold so not zeroed out)
    assert confidence == pytest.approx(0.5)
    assert -1.0 <= signal_value <= 1.0


def test_fewer_than_half_evaluable_returns_zero_zero():
    """1 of 4 rules evaluable → below 50% threshold → (0.0, 0.0)."""
    rules = [
        {"field": "RSI_14", "operator": "<", "threshold": 30, "direction": "bullish"},
        {"field": "MISSING_A", "operator": "<", "threshold": 50},
        {"field": "MISSING_B", "operator": ">", "threshold": 10},
        {"field": "MISSING_C", "operator": "<", "threshold": 20},
    ]
    market_data = {"RSI_14": 20}
    signal_value, confidence = score_signal(rules, market_data)
    assert signal_value == pytest.approx(0.0)
    assert confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RSI boundary scoring
# ---------------------------------------------------------------------------

def test_rsi_at_2x_distance_scores_1():
    """RSI=20, threshold=30 → distance=10, reference=10 → score=1.0."""
    rules = [{"field": "RSI_14", "operator": "<", "threshold": 30, "direction": "bullish"}]
    signal_value, _ = score_signal(rules, {"RSI_14": 20})
    assert signal_value == pytest.approx(1.0)


def test_rsi_at_midpoint_scores_half():
    """RSI=25, threshold=30 → distance=5, reference=10 → score=0.5."""
    rules = [{"field": "RSI_14", "operator": "<", "threshold": 30, "direction": "bullish"}]
    signal_value, _ = score_signal(rules, {"RSI_14": 25})
    assert signal_value == pytest.approx(0.5)


def test_rsi_at_threshold_scores_zero():
    """RSI=30, threshold=30 → score=0.0."""
    rules = [{"field": "RSI_14", "operator": "<", "threshold": 30, "direction": "bullish"}]
    signal_value, _ = score_signal(rules, {"RSI_14": 30})
    assert signal_value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Non-numeric / malformed rule handling
# ---------------------------------------------------------------------------

def test_non_numeric_field_value_is_skipped(caplog):
    """String market_data value → rule skipped, logged at DEBUG, confidence reflects exclusion."""
    import logging
    rules = [
        {"field": "pattern", "operator": "==", "threshold": 1, "direction": "bullish"},
        {"field": "RSI_14", "operator": "<", "threshold": 30, "direction": "bullish"},
    ]
    market_data = {"pattern": "bullish_engulfing", "RSI_14": 20}
    with caplog.at_level(logging.DEBUG, logger="quantstack.core.signal_scorer"):
        signal_value, confidence = score_signal(rules, market_data)
    # Only RSI_14 is evaluable
    assert confidence == pytest.approx(0.5)
    assert "pattern" in caplog.text or "non-numeric" in caplog.text.lower()


def test_missing_field_is_skipped():
    """Field absent from market_data → rule skipped, does not raise."""
    rules = [
        {"field": "NONEXISTENT", "operator": "<", "threshold": 50},
        {"field": "RSI_14", "operator": "<", "threshold": 30, "direction": "bullish"},
    ]
    market_data = {"RSI_14": 20}
    # 1/2 evaluable = 50% → not below threshold, should compute normally
    signal_value, confidence = score_signal(rules, market_data)
    assert confidence == pytest.approx(0.5)
    assert -1.0 <= signal_value <= 1.0


# ---------------------------------------------------------------------------
# Bounds guarantee
# ---------------------------------------------------------------------------

def test_signal_value_always_in_bounds():
    """Extreme inputs never produce values outside [-1, 1] and [0, 1]."""
    rules = [
        {"field": "RSI_14", "operator": "<", "threshold": 30, "direction": "bullish"},
        {"field": "close", "operator": "<", "threshold": 100, "direction": "bearish"},
    ]
    for rsi in [0, 10, 20, 29, 30, 50, 100]:
        for close in [1, 50, 99, 100, 200]:
            sv, conf = score_signal(rules, {"RSI_14": rsi, "close": close})
            assert -1.0 <= sv <= 1.0, f"signal_value={sv} out of bounds"
            assert 0.0 <= conf <= 1.0, f"confidence={conf} out of bounds"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_entry_rules_returns_zero_zero():
    """Empty rules list → (0.0, 0.0)."""
    assert score_signal([], {"RSI_14": 20}) == (0.0, 0.0)


def test_bearish_rule_contributes_negative_signal():
    """Satisfied bearish rule → negative signal_value."""
    rules = [{"field": "RSI_14", "operator": ">", "threshold": 70, "direction": "bearish"}]
    signal_value, confidence = score_signal(rules, {"RSI_14": 90})  # 20 above threshold, ref=20 → score 1.0 bearish
    assert signal_value < 0
    assert confidence == pytest.approx(1.0)
