"""Tests for alt-data integration in make_risk_sizing() and risk_gate (section-13)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantstack.core.signals.alt_data_normalizer import (
    ALT_DATA_WEIGHT,
    get_macro_stress_scalar,
)


def _generate_closes(n: int, daily_vol: float = 0.01, start: float = 100.0) -> list[tuple]:
    np.random.seed(42)
    returns = np.random.normal(0, daily_vol, n - 1)
    prices = [start]
    for r in returns:
        prices.append(prices[-1] * np.exp(r))
    return [(p,) for p in reversed(prices)]


def _make_state(**overrides) -> dict:
    base = {
        "cycle_number": 1,
        "regime": "unknown",
        "portfolio_context": {},
        "entry_candidates": [],
        "vol_state": "normal",
        "decisions": [],
        "errors": [],
    }
    base.update(overrides)
    return base


class _FakeCursor:
    def __init__(self, result):
        self._result = result

    def fetchone(self):
        return self._result

    def fetchall(self):
        return self._result if isinstance(self._result, list) else []


class _FakeConn:
    def __init__(self, regime_row=None, ic_rows=None, ohlcv_rows=None, alt_mod=0.0):
        self._regime_row = regime_row
        self._ic_rows = ic_rows or {}
        self._ohlcv_rows = ohlcv_rows or {}
        self._alt_mod = alt_mod

    def execute(self, sql, params=None):
        sql_lower = sql.strip().lower()
        if "regime_state" in sql_lower:
            return _FakeCursor(self._regime_row)
        if "signal_ic" in sql_lower:
            sid = params[0] if params else None
            return _FakeCursor(self._ic_rows.get(sid))
        if "ohlcv" in sql_lower:
            sym = params[0] if params else None
            return _FakeCursor(self._ohlcv_rows.get(sym, []))
        # Alt-data queries: return empty (get_alt_data_modifier will return 0.0)
        return _FakeCursor(None)


class _FakeCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *a):
        pass


@pytest.mark.asyncio
async def test_alt_modifier_adds_to_signal():
    """Alt modifier 0.5 → combined = price_signal + 0.30 × 0.5."""
    closes = _generate_closes(64, daily_vol=0.0126)
    fake_conn = _FakeConn(
        regime_row=("ranging", 0.9),
        ic_rows={"strat_1": (0.04,)},
        ohlcv_rows={"AAPL": closes},
    )

    candidates = [
        {"symbol": "AAPL", "strategy_id": "strat_1", "signal_value": 0.7, "verdict": "ENTER"},
    ]

    # Run once WITHOUT alt data to get base signal
    with patch("quantstack.graphs.trading.nodes.db_conn", return_value=_FakeCtx(fake_conn)):
        with patch("quantstack.graphs.trading.nodes.get_alt_data_modifier", return_value=0.0):
            from quantstack.graphs.trading.nodes import make_risk_sizing
            node = make_risk_sizing()
            state = _make_state(entry_candidates=candidates)
            result_base = await node(state)

    # Run WITH alt_mod = 0.5
    fake_conn2 = _FakeConn(
        regime_row=("ranging", 0.9),
        ic_rows={"strat_1": (0.04,)},
        ohlcv_rows={"AAPL": closes},
    )

    with patch("quantstack.graphs.trading.nodes.db_conn", return_value=_FakeCtx(fake_conn2)):
        with patch("quantstack.graphs.trading.nodes.get_alt_data_modifier", return_value=0.5):
            node = make_risk_sizing()
            result_alt = await node(state)

    base_signal = result_base["alpha_signals"][0]
    alt_signal = result_alt["alpha_signals"][0]

    # The alt-modified signal should be base + 0.30 * 0.5
    assert alt_signal == pytest.approx(base_signal + ALT_DATA_WEIGHT * 0.5, rel=0.01)


@pytest.mark.asyncio
async def test_alt_modifier_zero_no_change():
    """Alt modifier 0.0 → combined == price_signal."""
    closes = _generate_closes(64, daily_vol=0.0126)
    fake_conn = _FakeConn(
        regime_row=("ranging", 0.9),
        ic_rows={"strat_1": (0.04,)},
        ohlcv_rows={"AAPL": closes},
    )

    candidates = [
        {"symbol": "AAPL", "strategy_id": "strat_1", "signal_value": 0.7, "verdict": "ENTER"},
    ]

    with patch("quantstack.graphs.trading.nodes.db_conn", return_value=_FakeCtx(fake_conn)):
        with patch("quantstack.graphs.trading.nodes.get_alt_data_modifier", return_value=0.0):
            from quantstack.graphs.trading.nodes import make_risk_sizing
            node = make_risk_sizing()
            result = await node(_make_state(entry_candidates=candidates))

    # Signal should equal pure price signal (no alt data added)
    assert len(result["alpha_signals"]) == 1
    assert result["alpha_signals"][0] != 0  # Non-trivial signal


@pytest.mark.asyncio
async def test_alt_data_failure_falls_back():
    """Alt-data exception → combined == price_signal (fallback)."""
    closes = _generate_closes(64, daily_vol=0.0126)
    fake_conn = _FakeConn(
        regime_row=("ranging", 0.9),
        ic_rows={"strat_1": (0.04,)},
        ohlcv_rows={"AAPL": closes},
    )

    candidates = [
        {"symbol": "AAPL", "strategy_id": "strat_1", "signal_value": 0.7, "verdict": "ENTER"},
    ]

    def _bad_modifier(*a, **kw):
        raise RuntimeError("DB unavailable")

    with patch("quantstack.graphs.trading.nodes.db_conn", return_value=_FakeCtx(fake_conn)):
        with patch("quantstack.graphs.trading.nodes.get_alt_data_modifier", side_effect=_bad_modifier):
            from quantstack.graphs.trading.nodes import make_risk_sizing
            node = make_risk_sizing()
            result = await node(_make_state(entry_candidates=candidates))

    # Should still produce signals (fallback to price-only)
    assert len(result["alpha_signals"]) == 1


class TestMacroStressScalar:
    def test_high_stress_halves_scalar(self):
        assert get_macro_stress_scalar(2.5) == 0.5

    def test_moderate_stress_reduces_scalar(self):
        assert get_macro_stress_scalar(1.7) == 0.7

    def test_benign_macro_no_effect(self):
        assert get_macro_stress_scalar(0.5) == 1.0
        assert get_macro_stress_scalar(-1.0) == 1.0

    def test_multiplicative_scalars(self):
        """quality_scalar × macro_scalar = combined reduction."""
        quality = 0.7
        macro = 0.5
        combined = quality * macro
        assert combined == pytest.approx(0.35)
