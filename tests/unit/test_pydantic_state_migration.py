"""Tests for Pydantic state migration (section-03).

Verifies that all 4 state classes reject unknown keys, enforce types,
preserve LangGraph reducer annotations, and validate domain invariants.
"""

import operator
from typing import Annotated, get_type_hints

import pytest
from pydantic import ValidationError

from quantstack.graphs.state import (
    ResearchState,
    SupervisorState,
    SymbolValidationState,
    TradingState,
)


# ---------------------------------------------------------------------------
# Rejection tests — extra="forbid"
# ---------------------------------------------------------------------------


def test_trading_state_rejects_unknown_key():
    """TradingState(daly_plan="...") raises ValidationError (typo caught)."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TradingState(daly_plan="typo")


def test_research_state_rejects_unknown_key():
    """ResearchState(hypotheesis="...") raises ValidationError."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ResearchState(hypotheesis="typo")


def test_supervisor_state_rejects_unknown_key():
    """SupervisorState rejects undeclared keys."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SupervisorState(nonexistent_field="bad")


def test_symbol_validation_state_rejects_unknown_key():
    """SymbolValidationState rejects undeclared keys."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SymbolValidationState(nonexistent_field="bad")


# ---------------------------------------------------------------------------
# Type enforcement tests
# ---------------------------------------------------------------------------


def test_trading_state_rejects_wrong_type():
    """cycle_number must be int, not str."""
    with pytest.raises(ValidationError):
        TradingState(cycle_number="not_an_int")


def test_trading_state_rejects_wrong_nested_type():
    """position_reviews must be list, not str."""
    with pytest.raises(ValidationError):
        TradingState(position_reviews="not_a_list")


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


def test_trading_state_accepts_defaults():
    """TradingState() with no args constructs with defaults."""
    state = TradingState()
    assert state.cycle_number == 0
    assert state.regime == ""
    assert state.errors == []
    assert state.decisions == []


def test_research_state_accepts_defaults():
    """ResearchState() constructs with defaults."""
    state = ResearchState()
    assert state.cycle_number == 0
    assert state.errors == []


def test_supervisor_state_accepts_defaults():
    """SupervisorState() constructs with defaults."""
    state = SupervisorState()
    assert state.cycle_number == 0
    assert state.errors == []


def test_symbol_validation_state_accepts_defaults():
    """SymbolValidationState() constructs with defaults."""
    state = SymbolValidationState()
    assert state.validation_results == []
    assert state.errors == []


# ---------------------------------------------------------------------------
# Reducer compatibility tests
# ---------------------------------------------------------------------------


def test_reducer_annotations_preserved():
    """Annotated[list, operator.add] reducers are extractable from Pydantic model."""
    hints = get_type_hints(TradingState, include_extras=True)

    # errors should have operator.add reducer
    errors_hint = hints["errors"]
    assert hasattr(errors_hint, "__metadata__"), "errors must be Annotated"
    assert operator.add in errors_hint.__metadata__

    # decisions too
    decisions_hint = hints["decisions"]
    assert hasattr(decisions_hint, "__metadata__"), "decisions must be Annotated"
    assert operator.add in decisions_hint.__metadata__


def test_research_reducer_annotations_preserved():
    """ResearchState reducers preserved."""
    hints = get_type_hints(ResearchState, include_extras=True)
    errors_hint = hints["errors"]
    assert hasattr(errors_hint, "__metadata__")
    assert operator.add in errors_hint.__metadata__

    vr_hint = hints["validation_results"]
    assert hasattr(vr_hint, "__metadata__")
    assert operator.add in vr_hint.__metadata__


# ---------------------------------------------------------------------------
# Field validator tests
# ---------------------------------------------------------------------------


def test_vol_state_rejects_invalid_value():
    """vol_state must be one of the known volatility states or empty."""
    with pytest.raises(ValidationError, match="vol_state"):
        TradingState(vol_state="invalid_state")


def test_vol_state_accepts_valid_values():
    """vol_state accepts known values."""
    for val in ("", "low", "normal", "high", "extreme"):
        state = TradingState(vol_state=val)
        assert state.vol_state == val


def test_cycle_number_rejects_negative():
    """cycle_number must be >= 0."""
    with pytest.raises(ValidationError, match="cycle_number"):
        TradingState(cycle_number=-1)


# ---------------------------------------------------------------------------
# Model validator tests
# ---------------------------------------------------------------------------


def test_exit_orders_requires_position_reviews():
    """Non-empty exit_orders with empty position_reviews is invalid."""
    with pytest.raises(ValidationError, match="exit_orders"):
        TradingState(
            exit_orders=[{"symbol": "AAPL", "action": "CLOSE"}],
            position_reviews=[],
        )


def test_exit_orders_valid_when_reviews_present():
    """Non-empty exit_orders with non-empty position_reviews is valid."""
    state = TradingState(
        exit_orders=[{"symbol": "AAPL", "action": "CLOSE"}],
        position_reviews=[{"symbol": "AAPL", "action": "CLOSE"}],
    )
    assert len(state.exit_orders) == 1
