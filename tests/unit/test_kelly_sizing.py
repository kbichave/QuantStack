"""Unit tests for kelly_sizing.py (section-04)."""

import numpy as np
import pytest

from quantstack.core.kelly_sizing import IC_PRIOR, compute_alpha_signals


def _make_candidate(symbol: str, strategy_id: str, signal_value: float) -> dict:
    return {"symbol": symbol, "strategy_id": strategy_id, "signal_value": signal_value}


# ---------------------------------------------------------------------------
# Formula correctness — annualized sigma
# ---------------------------------------------------------------------------

def test_expected_return_uses_annualized_sigma():
    """
    IC=0.05, z=1.0, annualized_sigma=0.30, kelly_fraction=0.5
    → expected_return = 0.05 × 0.5 × 1.0 × 0.30 = 0.0075
    Not 0.138 (which would result from mistakenly using daily sigma ≈0.019).
    """
    candidates = [_make_candidate("AAPL", "strat_a", 1.0)]
    ic_lookup = {"strat_a": 0.05}
    vol_lookup = {"AAPL": 0.30}
    result = compute_alpha_signals(candidates, ic_lookup, vol_lookup, kelly_fraction=0.5)
    assert result[0] == pytest.approx(0.0075, rel=1e-6)


def test_ic_prior_used_when_no_signal_ic_entry():
    """
    IC=None (no history) → uses IC_PRIOR=0.01.
    IC=0.01, z=0.5, annualized_sigma=0.30, kelly_fraction=0.5
    → 0.01 × 0.5 × 0.5 × 0.30 = 0.00075
    """
    candidates = [_make_candidate("AAPL", "strat_new", 0.5)]
    ic_lookup = {"strat_new": None}
    vol_lookup = {"AAPL": 0.30}
    result = compute_alpha_signals(candidates, ic_lookup, vol_lookup, kelly_fraction=0.5)
    assert result[0] == pytest.approx(0.01 * 0.5 * 0.5 * 0.30, rel=1e-6)


def test_ic_prior_used_when_strategy_not_in_lookup():
    """Strategy key absent from signal_ic_lookup → treated as None → IC prior used."""
    candidates = [_make_candidate("TSLA", "strat_absent", 1.0)]
    ic_lookup = {}  # strat_absent not present
    vol_lookup = {"TSLA": 0.50}
    result = compute_alpha_signals(candidates, ic_lookup, vol_lookup, kelly_fraction=0.5)
    assert result[0] == pytest.approx(IC_PRIOR * 0.5 * 1.0 * 0.50, rel=1e-6)


# ---------------------------------------------------------------------------
# Signal value behaviour
# ---------------------------------------------------------------------------

def test_zero_signal_value_returns_zero_expected_return():
    """signal_value = 0.0 → expected_return = 0.0 regardless of IC or vol."""
    candidates = [_make_candidate("AAPL", "strat_a", 0.0)]
    ic_lookup = {"strat_a": 0.1}
    vol_lookup = {"AAPL": 0.40}
    result = compute_alpha_signals(candidates, ic_lookup, vol_lookup)
    assert result[0] == pytest.approx(0.0, abs=1e-12)


def test_negative_signal_value_produces_negative_expected_return():
    """z=-0.5, IC=0.05, sigma=0.30, kelly_fraction=0.5 → negative expected return."""
    candidates = [_make_candidate("AAPL", "strat_a", -0.5)]
    ic_lookup = {"strat_a": 0.05}
    vol_lookup = {"AAPL": 0.30}
    result = compute_alpha_signals(candidates, ic_lookup, vol_lookup, kelly_fraction=0.5)
    assert result[0] < 0


# ---------------------------------------------------------------------------
# Array alignment and length
# ---------------------------------------------------------------------------

def test_output_length_equals_candidates_length():
    """3 candidates → output array of length 3."""
    candidates = [
        _make_candidate("A", "strat_a", 0.8),
        _make_candidate("B", "strat_b", 0.5),
        _make_candidate("C", "strat_c", -0.3),
    ]
    ic_lookup = {"strat_a": 0.05, "strat_b": 0.03, "strat_c": 0.04}
    vol_lookup = {"A": 0.25, "B": 0.35, "C": 0.20}
    result = compute_alpha_signals(candidates, ic_lookup, vol_lookup)
    assert len(result) == 3


def test_output_order_matches_candidates_order():
    """Output[0] corresponds to candidates[0], etc."""
    candidates = [
        _make_candidate("A", "strat", 1.0),
        _make_candidate("B", "strat", 0.0),
        _make_candidate("C", "strat", -1.0),
    ]
    ic_lookup = {"strat": 0.05}
    vol_lookup = {"A": 0.30, "B": 0.30, "C": 0.30}
    result = compute_alpha_signals(candidates, ic_lookup, vol_lookup, kelly_fraction=0.5)
    assert result[0] > 0     # signal_value=1.0 → positive
    assert result[1] == pytest.approx(0.0, abs=1e-12)  # signal_value=0.0 → zero
    assert result[2] < 0     # signal_value=-1.0 → negative


# ---------------------------------------------------------------------------
# kelly_fraction scaling
# ---------------------------------------------------------------------------

def test_kelly_fraction_scales_output_linearly():
    """kelly_fraction=0.25 produces exactly half the output of kelly_fraction=0.5."""
    candidates = [_make_candidate("AAPL", "strat_a", 0.8)]
    ic_lookup = {"strat_a": 0.05}
    vol_lookup = {"AAPL": 0.30}
    r_half = compute_alpha_signals(candidates, ic_lookup, vol_lookup, kelly_fraction=0.5)
    r_quarter = compute_alpha_signals(candidates, ic_lookup, vol_lookup, kelly_fraction=0.25)
    assert r_quarter[0] == pytest.approx(r_half[0] / 2, rel=1e-9)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_candidates_returns_empty_array():
    """Empty candidates list → empty numpy array, no error."""
    result = compute_alpha_signals([], {}, {})
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_missing_vol_uses_floor_of_015():
    """Symbol absent from vol_lookup → uses 0.15 default floor, no raise."""
    candidates = [_make_candidate("XYZ", "strat_a", 1.0)]
    ic_lookup = {"strat_a": 0.05}
    vol_lookup = {}  # XYZ not present
    result = compute_alpha_signals(candidates, ic_lookup, vol_lookup, kelly_fraction=0.5)
    # Expected: IC=0.05 × kelly=0.5 × z=1.0 × vol=0.15
    assert result[0] == pytest.approx(0.05 * 0.5 * 1.0 * 0.15, rel=1e-6)


def test_zero_vol_uses_floor_of_015():
    """vol=0.0 → uses 0.15 floor (not 0.0 which would produce misleading zero return)."""
    candidates = [_make_candidate("XYZ", "strat_a", 1.0)]
    ic_lookup = {"strat_a": 0.05}
    vol_lookup = {"XYZ": 0.0}
    result = compute_alpha_signals(candidates, ic_lookup, vol_lookup, kelly_fraction=0.5)
    assert result[0] == pytest.approx(0.05 * 0.5 * 1.0 * 0.15, rel=1e-6)
