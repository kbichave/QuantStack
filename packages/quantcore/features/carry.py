# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Carry and macro positioning signals.

Three signal families — all pure computation (no API calls):

1. **EquityCarry** — dividend yield + buyback yield from financial statement data.
   Equity carry = (TTM dividends + TTM buybacks) / market_cap.
   High carry favours longs (income + return of capital); low carry or negative
   FCF flags firms burning cash.

2. **FuturesBasis** — spot vs futures basis as a carry/sentiment proxy.
   Basis = (Futures_price − Spot_price) / Spot_price × 100 (in %).
   Positive basis (contango) = market paying a premium to hold futures.
   Negative basis (backwardation) = spot scarce or high demand.

3. **COTSignals** — CFTC Commitments of Traders (COT) report signals.
   Non-commercial (speculative) net positioning as % of open interest.
   Extreme long → contrarian bearish; extreme short → contrarian bullish.
   The CFTC releases this data freely every Friday as a public CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Equity Carry
# ---------------------------------------------------------------------------


class EquityCarry:
    """
    Equity carry signal: (TTM dividends paid + TTM buybacks) / market cap.

    Inputs come from quarterly cash flow statements and market cap series —
    both available from FD.ai via `get_cash_flow_statements()` and
    `get_financial_metrics()`.

    Parameters
    ----------
    ttm_periods : int
        Number of quarters for trailing twelve months. Default 4.
    high_carry_threshold : float
        Carry yield above which we flag as high carry. Default 0.02 (2%).
    """

    def __init__(self, ttm_periods: int = 4, high_carry_threshold: float = 0.02) -> None:
        self.ttm_periods = ttm_periods
        self.high_carry_threshold = high_carry_threshold

    def compute(self, financials: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        financials : pd.DataFrame
            Must contain columns:
                dividends_paid     – cash dividends paid (positive value; CF statement)
                share_repurchases  – buybacks (positive value; CF statement)
                market_cap         – market capitalisation
            Indexed by period_end or any time index.

        Returns
        -------
        pd.DataFrame with columns:
            ttm_dividends   – trailing 4-quarter dividends
            ttm_buybacks    – trailing 4-quarter buybacks
            total_return_to_shareholders – ttm_dividends + ttm_buybacks
            dividend_yield  – ttm_dividends / market_cap
            buyback_yield   – ttm_buybacks / market_cap
            equity_carry    – (dividends + buybacks) / market_cap
            carry_high      – 1 when equity_carry > high_carry_threshold
        """
        n = self.ttm_periods
        df = financials.copy()

        # Use abs() — some data sources report outflows as negative
        div = df["dividends_paid"].abs()
        buyback = df["share_repurchases"].abs()
        mcap = df["market_cap"].replace(0, np.nan)

        ttm_div     = div.rolling(n).sum()
        ttm_buyback = buyback.rolling(n).sum()
        total_rts   = ttm_div + ttm_buyback

        div_yield    = ttm_div / mcap
        buyback_yield = ttm_buyback / mcap
        carry        = total_rts / mcap

        return pd.DataFrame(
            {
                "ttm_dividends":               ttm_div,
                "ttm_buybacks":                ttm_buyback,
                "total_return_to_shareholders": total_rts,
                "dividend_yield":              div_yield,
                "buyback_yield":               buyback_yield,
                "equity_carry":                carry,
                "carry_high":                  (carry > self.high_carry_threshold).astype(int),
            },
            index=financials.index,
        )


# ---------------------------------------------------------------------------
# Futures Basis
# ---------------------------------------------------------------------------


class FuturesBasis:
    """
    Futures basis (contango/backwardation) as a market regime signal.

    Basis_pct = (futures_close − spot_close) / spot_close × 100

    Positive basis (contango) → market participants expect higher future prices
    or are paying a convenience yield premium.
    Negative basis (backwardation) → spot demand exceeds supply; commodity squeeze.

    For equity index futures (ES/SPY, NQ/QQQ):
    - Sustained positive basis = risk-on positioning
    - Basis collapsing toward zero = liquidity stress / de-risking

    Parameters
    ----------
    roll_period : int
        Days for rolling mean/std of basis (regime context). Default 20.
    """

    def __init__(self, roll_period: int = 20) -> None:
        self.roll_period = roll_period

    def compute(self, futures: pd.Series, spot: pd.Series) -> pd.DataFrame:
        """
        Parameters
        ----------
        futures : pd.Series — futures close prices (e.g. ES1! continuous)
        spot    : pd.Series — spot/ETF close prices (e.g. SPY)
            Must share the same DatetimeIndex.

        Returns
        -------
        pd.DataFrame with columns:
            basis_pct          – (futures - spot) / spot × 100
            basis_roll_mean    – rolling mean of basis_pct
            basis_roll_std     – rolling std of basis_pct
            basis_zscore       – z-score of current basis vs rolling window
            contango           – 1 when basis_pct > 0
            backwardation      – 1 when basis_pct < 0
            basis_extreme_high – 1 when basis_zscore > 2 (unusual premium)
            basis_extreme_low  – 1 when basis_zscore < -2 (unusual discount)
        """
        aligned_spot = spot.reindex(futures.index, method="ffill")
        basis_pct = (futures - aligned_spot) / aligned_spot.replace(0, np.nan) * 100

        roll_mean = basis_pct.rolling(self.roll_period).mean()
        roll_std  = basis_pct.rolling(self.roll_period).std()
        zscore = (basis_pct - roll_mean) / roll_std.replace(0, np.nan)

        return pd.DataFrame(
            {
                "basis_pct":          basis_pct,
                "basis_roll_mean":    roll_mean,
                "basis_roll_std":     roll_std,
                "basis_zscore":       zscore,
                "contango":           (basis_pct > 0).astype(int),
                "backwardation":      (basis_pct < 0).astype(int),
                "basis_extreme_high": (zscore > 2).astype(int),
                "basis_extreme_low":  (zscore < -2).astype(int),
            },
            index=futures.index,
        )


# ---------------------------------------------------------------------------
# COT Signals (CFTC Commitments of Traders)
# ---------------------------------------------------------------------------


class COTSignals:
    """
    CFTC Commitments of Traders (COT) report positioning signals.

    The CFTC publishes a free weekly CSV every Friday at:
    https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm

    Three participant groups:
    - **Commercial** (hedgers): insiders who use futures to hedge real exposure.
      They are typically contra-trend — high commercial long = bearish for price.
    - **Non-commercial** (large speculators): trend-following funds and CTAs.
      Extreme positioning = crowded trade = mean-reversion signal.
    - **Non-reportable** (small speculators): retail. Extreme = contrarian signal.

    Net position = longs − shorts for each group.
    Net% = net / open_interest × 100 (standardised for cross-market comparison).

    Parameters
    ----------
    extreme_threshold : float
        z-score threshold for declaring extreme positioning. Default 2.0.
    roll_window : int
        Weeks for rolling z-score context. Default 52 (1 year of weekly data).
    """

    def __init__(self, extreme_threshold: float = 2.0, roll_window: int = 52) -> None:
        self.extreme_threshold = extreme_threshold
        self.roll_window = roll_window

    def compute(self, cot: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        cot : pd.DataFrame
            Must contain columns (matching CFTC CSV column names or renamed equivalents):
                noncommercial_long    – large spec longs
                noncommercial_short   – large spec shorts
                commercial_long       – commercial longs
                commercial_short      – commercial shorts
                nonreportable_long    – small spec longs
                nonreportable_short   – small spec shorts
                open_interest         – total open interest
            Any datetime index (weekly).

        Returns
        -------
        pd.DataFrame with columns:
            nc_net              – non-commercial net (longs - shorts)
            nc_net_pct          – nc_net / open_interest × 100
            nc_zscore           – rolling z-score of nc_net_pct
            nc_extreme_long     – 1 when nc_zscore > threshold (crowded long → bearish)
            nc_extreme_short    – 1 when nc_zscore < -threshold (crowded short → bullish)
            comm_net            – commercial net
            comm_net_pct        – commercial net % of OI
            sr_net              – small spec net
            cot_sentiment       – composite: nc_net_pct − comm_net_pct (smart vs dumb)
        """
        oi = cot["open_interest"].replace(0, np.nan)

        nc_net = cot["noncommercial_long"] - cot["noncommercial_short"]
        comm_net = cot["commercial_long"] - cot["commercial_short"]
        sr_net = cot["nonreportable_long"] - cot["nonreportable_short"]

        nc_net_pct   = nc_net / oi * 100
        comm_net_pct = comm_net / oi * 100

        roll_mean = nc_net_pct.rolling(self.roll_window).mean()
        roll_std  = nc_net_pct.rolling(self.roll_window).std()
        nc_zscore = (nc_net_pct - roll_mean) / roll_std.replace(0, np.nan)

        # COT sentiment: spec net minus commercial net (smart money flows opposite)
        cot_sentiment = nc_net_pct - comm_net_pct

        return pd.DataFrame(
            {
                "nc_net":           nc_net,
                "nc_net_pct":       nc_net_pct,
                "nc_zscore":        nc_zscore,
                "nc_extreme_long":  (nc_zscore > self.extreme_threshold).astype(int),
                "nc_extreme_short": (nc_zscore < -self.extreme_threshold).astype(int),
                "comm_net":         comm_net,
                "comm_net_pct":     comm_net_pct,
                "sr_net":           sr_net,
                "cot_sentiment":    cot_sentiment,
            },
            index=cot.index,
        )
