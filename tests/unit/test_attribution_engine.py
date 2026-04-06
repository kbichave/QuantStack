"""Unit tests for attribution_engine.py (section-05)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantstack.core.attribution_engine import AttributionRecord, decompose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_date_index(n: int, end: str = "2024-06-01") -> pd.DatetimeIndex:
    end_dt = pd.Timestamp(end)
    return pd.bdate_range(end=end_dt, periods=n)


def _synthetic_returns(
    n: int = 62,
    beta_spy: float = 1.2,
    seed: int = 42,
    noise_scale: float = 0.005,
    rf: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (stock_raw, spy_raw, sector_raw) where sector tracks spy."""
    rng = np.random.default_rng(seed)
    idx = _make_date_index(n)
    spy = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    sector = spy + rng.normal(0, 0.001, n)  # sector closely tracks spy
    stock = beta_spy * spy + rng.normal(rf, noise_scale, n)
    return stock, spy, sector


# ---------------------------------------------------------------------------
# 1. Arithmetic closure
# ---------------------------------------------------------------------------

def test_arithmetic_closure():
    """market_pnl + sector_pnl + alpha_pnl + residual_pnl == total_pnl to 1e-10."""
    stock, spy, sector = _synthetic_returns(n=62, beta_spy=1.2)
    rec = decompose(
        symbol="AAPL",
        strategy_id="strat_test",
        attr_date=stock.index[-1].date(),
        stock_returns=stock,
        spy_returns=spy,
        sector_returns=sector,
        risk_free_rate=0.0,
        position_notional=10_000.0,
        opened_at=date(2024, 1, 2),
    )
    total = rec.market_pnl + rec.sector_pnl + rec.alpha_pnl + rec.residual_pnl
    assert abs(total - rec.total_pnl) < 1e-10


# ---------------------------------------------------------------------------
# 2. Ex-ante betas — regression uses prior window only
# ---------------------------------------------------------------------------

def test_exante_betas_use_prior_window_only():
    """
    Attribution-date return is an outlier. Betas must match fitting on the prior window,
    not the full series.
    """
    n = 62
    stock_normal, spy, sector = _synthetic_returns(n=n, beta_spy=1.2, seed=7)
    stock_outlier = stock_normal.copy()
    stock_outlier.iloc[-1] = 100.0  # extreme outlier on attribution day

    rec_normal = decompose("X", "s", stock_normal.index[-1].date(), stock_normal, spy, sector, 0.0, 1.0, date(2024, 1, 2))
    rec_outlier = decompose("X", "s", stock_outlier.index[-1].date(), stock_outlier, spy, sector, 0.0, 1.0, date(2024, 1, 2))

    # Betas must be equal — the attribution-day outlier must not affect them
    assert rec_normal.beta_market == pytest.approx(rec_outlier.beta_market, rel=1e-9)
    assert rec_normal.beta_sector == pytest.approx(rec_outlier.beta_sector, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Risk-free adjustment changes intercept (alpha)
# ---------------------------------------------------------------------------

def test_risk_free_adjustment_changes_alpha():
    """Passing rf=0.0 vs rf=0.001 (daily) yields a different Jensen's alpha."""
    stock, spy, sector = _synthetic_returns(n=62, beta_spy=1.0, seed=99)
    rec_rf0 = decompose("A", "s", stock.index[-1].date(), stock, spy, sector, 0.0, 1.0, date(2024, 1, 2))
    rec_rf1 = decompose("A", "s", stock.index[-1].date(), stock, spy, sector, 0.001, 1.0, date(2024, 1, 2))
    assert rec_rf0.alpha_pnl != pytest.approx(rec_rf1.alpha_pnl, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. SPY orthogonalization — sector that perfectly tracks SPY → beta_sector ≈ 0
# ---------------------------------------------------------------------------

def test_spy_orthogonalization_sector_tracks_spy():
    """When sector ETF = a + b*SPY (no residual), beta_sector ≈ 0."""
    rng = np.random.default_rng(5)
    n = 62
    idx = _make_date_index(n)
    spy = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    # Sector is exactly a linear function of SPY (plus tiny numerical noise)
    sector = 0.5 + 1.1 * spy
    stock = 1.2 * spy + rng.normal(0, 0.005, n)

    rec = decompose("X", "s", spy.index[-1].date(), stock, spy, sector, 0.0, 1.0, date(2024, 1, 2))
    assert abs(rec.beta_sector) < 0.05  # residual factor carries near-zero signal


# ---------------------------------------------------------------------------
# 5. Insufficient history fallback (< 30 days)
# ---------------------------------------------------------------------------

def test_insufficient_history_returns_conservative_fallback():
    """With 20 days of data (< 30), market_pnl = total_pnl, others = 0.0."""
    n = 20
    idx = _make_date_index(n)
    rng = np.random.default_rng(1)
    stock = pd.Series(rng.normal(0.001, 0.01, n), index=idx)
    spy = pd.Series(rng.normal(0.001, 0.01, n), index=idx)
    sector = pd.Series(rng.normal(0.001, 0.01, n), index=idx)

    rec = decompose("X", "s", stock.index[-1].date(), stock, spy, sector, 0.0, 5_000.0, date(2024, 1, 2))
    expected_total = stock.iloc[-1] * 5_000.0
    assert rec.total_pnl == pytest.approx(expected_total, rel=1e-9)
    assert rec.market_pnl == pytest.approx(rec.total_pnl, rel=1e-9)
    assert rec.sector_pnl == pytest.approx(0.0, abs=1e-10)
    assert rec.alpha_pnl == pytest.approx(0.0, abs=1e-10)
    assert rec.residual_pnl == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 6. Zero-variance sector ETF — no raise, sector_pnl = 0
# ---------------------------------------------------------------------------

def test_zero_variance_sector_no_raise():
    """Constant sector ETF (zero variance) must not raise and sector_pnl must be 0."""
    stock, spy, _ = _synthetic_returns(n=62, beta_spy=1.2)
    sector_const = pd.Series(0.0, index=stock.index)

    rec = decompose("X", "s", stock.index[-1].date(), stock, spy, sector_const, 0.0, 1.0, date(2024, 1, 2))
    assert rec.sector_pnl == pytest.approx(0.0, abs=1e-10)
    # Arithmetic invariant must still hold
    total = rec.market_pnl + rec.sector_pnl + rec.alpha_pnl + rec.residual_pnl
    assert abs(total - rec.total_pnl) < 1e-10


# ---------------------------------------------------------------------------
# 7. holding_day computation
# ---------------------------------------------------------------------------

def test_holding_day_computed_correctly():
    """holding_day = (attr_date - opened_at).days."""
    stock, spy, sector = _synthetic_returns(n=62)
    opened = date(2024, 1, 2)
    attr = stock.index[-1].date()
    rec = decompose("X", "s", attr, stock, spy, sector, 0.0, 1.0, opened)
    assert rec.holding_day == (attr - opened).days


# ---------------------------------------------------------------------------
# 8. residual_pnl is computed by subtraction, not a separate regression
# ---------------------------------------------------------------------------

def test_residual_pnl_computed_by_subtraction():
    """residual_pnl = total_pnl - market_pnl - sector_pnl - alpha_pnl."""
    stock, spy, sector = _synthetic_returns(n=62, beta_spy=0.8)
    rec = decompose("X", "s", stock.index[-1].date(), stock, spy, sector, 0.0, 10_000.0, date(2024, 1, 2))
    expected_residual = rec.total_pnl - rec.market_pnl - rec.sector_pnl - rec.alpha_pnl
    assert rec.residual_pnl == pytest.approx(expected_residual, abs=1e-10)


# ---------------------------------------------------------------------------
# 9. Known-beta recovery (regression accuracy sanity check)
# ---------------------------------------------------------------------------

def test_known_beta_spy_recovery():
    """Stock constructed as 1.2 × SPY + small noise → beta_market ≈ 1.2."""
    rng = np.random.default_rng(100)
    n = 120
    idx = _make_date_index(n)
    spy = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    sector = pd.Series(rng.normal(0.0003, 0.005, n), index=idx)
    stock = 1.2 * spy + rng.normal(0, 0.001, n)  # very low noise → beta should recover well

    rec = decompose("X", "s", spy.index[-1].date(), stock, spy, sector, 0.0, 1.0, date(2024, 1, 2))
    assert rec.beta_market == pytest.approx(1.2, abs=0.05)
