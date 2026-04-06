"""
P&L attribution engine: decomposes realized P&L into market beta, sector beta,
Jensen's alpha, and residual components.

Pure computation module — no database access, no LLM calls, no side effects.
Called by run_attribution() in supervisor/nodes.py (section-08).
"""

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sector name → ETF ticker for orthogonalized factor construction
SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Communication Services": "XLC",
}
_DEFAULT_SECTOR_ETF = "XLK"

# Minimum trading days of history required to fit betas
_MIN_HISTORY = 30


@dataclass
class AttributionRecord:
    date: date
    symbol: str
    strategy_id: str
    total_pnl: float
    market_pnl: float
    sector_pnl: float
    alpha_pnl: float
    residual_pnl: float
    beta_market: float
    beta_sector: float
    sector_etf: str
    holding_day: int


def _conservative_fallback(
    symbol: str,
    strategy_id: str,
    attr_date: date,
    total_pnl: float,
    sector_etf: str,
    holding_day: int,
) -> AttributionRecord:
    """Return a record where all P&L is attributed to market (i.e., unknown)."""
    return AttributionRecord(
        date=attr_date,
        symbol=symbol,
        strategy_id=strategy_id,
        total_pnl=total_pnl,
        market_pnl=total_pnl,
        sector_pnl=0.0,
        alpha_pnl=0.0,
        residual_pnl=0.0,
        beta_market=float("nan"),
        beta_sector=float("nan"),
        sector_etf=sector_etf,
        holding_day=holding_day,
    )


def decompose(
    symbol: str,
    strategy_id: str,
    attr_date: date,
    stock_returns: pd.Series,
    spy_returns: pd.Series,
    sector_returns: pd.Series,
    risk_free_rate: float,
    position_notional: float,
    opened_at: date,
    sector_etf: str = _DEFAULT_SECTOR_ETF,
) -> AttributionRecord:
    """
    Decomposes realized P&L into market beta, sector beta, Jensen's alpha, and residual.

    Args:
        symbol: ticker symbol
        strategy_id: owning strategy
        attr_date: the date being attributed
        stock_returns: series of raw log returns for the stock (60+ days, ending on attr_date)
        spy_returns: SPY raw log returns, same index
        sector_returns: sector ETF raw log returns, same index
        risk_free_rate: daily risk-free rate (subtracted internally before regression)
        position_notional: dollar notional of the position on attr_date
        opened_at: date the position was opened (for holding_day computation)
        sector_etf: sector ETF ticker stored in the record

    Returns:
        AttributionRecord with 4-component P&L decomposition.
        Falls back to conservative record (market_pnl = total_pnl) when:
        - fewer than 30 days of history
        - regression fails with LinAlgError
    """
    holding_day = (attr_date - opened_at).days

    # Attribution-day return (last element)
    stock_today = float(stock_returns.iloc[-1])
    total_pnl = stock_today * position_notional

    if len(stock_returns) < _MIN_HISTORY:
        return _conservative_fallback(symbol, strategy_id, attr_date, total_pnl, sector_etf, holding_day)

    try:
        result = _compute_decomposition(
            stock_returns=stock_returns,
            spy_returns=spy_returns,
            sector_returns=sector_returns,
            risk_free_rate=risk_free_rate,
            position_notional=position_notional,
        )
    except np.linalg.LinAlgError as exc:
        logger.error(
            "LinAlgError in attribution decompose for %s/%s on %s: %s",
            symbol,
            strategy_id,
            attr_date,
            exc,
        )
        return _conservative_fallback(symbol, strategy_id, attr_date, total_pnl, sector_etf, holding_day)

    alpha_pnl, beta_market, beta_sector, market_pnl, sector_pnl = result

    residual_pnl = total_pnl - market_pnl - sector_pnl - alpha_pnl
    assert abs((market_pnl + sector_pnl + alpha_pnl + residual_pnl) - total_pnl) < 1e-10

    return AttributionRecord(
        date=attr_date,
        symbol=symbol,
        strategy_id=strategy_id,
        total_pnl=total_pnl,
        market_pnl=market_pnl,
        sector_pnl=sector_pnl,
        alpha_pnl=alpha_pnl,
        residual_pnl=residual_pnl,
        beta_market=beta_market,
        beta_sector=beta_sector,
        sector_etf=sector_etf,
        holding_day=holding_day,
    )


def _compute_decomposition(
    stock_returns: pd.Series,
    spy_returns: pd.Series,
    sector_returns: pd.Series,
    risk_free_rate: float,
    position_notional: float,
) -> tuple[float, float, float, float, float]:
    """
    Core computation: returns (alpha_pnl, beta_market, beta_sector, market_pnl, sector_pnl).

    Raises np.linalg.LinAlgError on degenerate regression inputs.
    """
    rf = risk_free_rate

    # Excess returns (subtract daily risk-free rate)
    stock_xs = stock_returns - rf
    spy_xs = spy_returns - rf
    sector_xs = sector_returns - rf

    # Split: prior window excludes the attribution day (last row)
    stock_prior = stock_xs.iloc[:-1].values
    spy_prior = spy_xs.iloc[:-1].values
    sector_prior = sector_xs.iloc[:-1].values

    spy_today = float(spy_xs.iloc[-1])
    sector_today = float(sector_xs.iloc[-1])

    n = len(spy_prior)
    ones = np.ones(n)

    # Step 1: SPY-orthogonalize sector factor
    # Regress sector on [const, SPY], extract residuals
    if np.std(sector_prior) == 0.0:
        # Zero-variance sector: orthogonalized factor is zero
        sector_resid_prior = np.zeros(n)
        sector_resid_today = 0.0
    else:
        X_spy_orth = np.column_stack([ones, spy_prior])
        coef_orth, _, _, _ = np.linalg.lstsq(X_spy_orth, sector_prior, rcond=None)
        sector_resid_prior = sector_prior - X_spy_orth @ coef_orth
        # Orthogonalized sector return on attribution day
        sector_resid_today = sector_today - (coef_orth[0] + coef_orth[1] * spy_today)

    # Step 2: Main regression: stock_excess ~ [const, SPY, sector_residuals]
    X_main = np.column_stack([ones, spy_prior, sector_resid_prior])
    coef_main, _, _, _ = np.linalg.lstsq(X_main, stock_prior, rcond=None)
    alpha_daily = float(coef_main[0])
    beta_market = float(coef_main[1])
    beta_sector = float(coef_main[2])

    # Step 3: P&L components using attribution-day returns
    total_pnl = float(stock_returns.iloc[-1]) * position_notional

    market_pnl = beta_market * spy_today * position_notional
    sector_pnl = beta_sector * sector_resid_today * position_notional
    alpha_pnl = alpha_daily * position_notional

    return alpha_pnl, beta_market, beta_sector, market_pnl, sector_pnl
