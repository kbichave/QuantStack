# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Flow features — insider trades and institutional ownership as daily signals.

Converts sparse event data (insider trades filed irregularly, 13F quarterly)
into daily time series suitable for backtesting and live rule evaluation.

Insider flow: rolling 90-day net shares (buys - sells) → direction label.
Institutional flow: quarter-over-quarter change in total shares held.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

_INSIDER_WINDOW_DAYS = 90
_INSTITUTIONAL_CHANGE_THRESHOLD = 0.02  # 2% change = accumulating/distributing


def compute_insider_flow(
    df: pd.DataFrame,
    insider_trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute rolling insider flow features aligned to a daily OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        insider_trades: DataFrame with columns: transaction_date, transaction_type,
                        shares, price_per_share. Loaded from DataStore.load_insider_trades().

    Returns:
        df with added columns: flow_insider_net_90d, flow_insider_direction.
    """
    result = df.copy()
    result["flow_insider_net_90d"] = 0.0
    result["flow_insider_direction"] = "neutral"

    if insider_trades is None or insider_trades.empty:
        return result

    trades = insider_trades.copy()

    # Normalize transaction_date to datetime
    if "transaction_date" in trades.columns:
        trades["transaction_date"] = pd.to_datetime(trades["transaction_date"])
    else:
        logger.debug("[flow] No transaction_date column in insider trades")
        return result

    # Compute signed shares: buy = positive, sell = negative
    trades["signed_shares"] = trades.apply(_signed_shares, axis=1)

    # Aggregate to daily net
    daily_net = (
        trades.groupby("transaction_date")["signed_shares"]
        .sum()
        .reindex(df.index, fill_value=0.0)
    )

    # Rolling 90-day sum
    rolling_net = daily_net.rolling(window=_INSIDER_WINDOW_DAYS, min_periods=1).sum()
    result["flow_insider_net_90d"] = rolling_net.values

    # Direction label
    result.loc[result["flow_insider_net_90d"] > 0, "flow_insider_direction"] = "buying"
    result.loc[result["flow_insider_net_90d"] < 0, "flow_insider_direction"] = "selling"

    return result


def compute_institutional_flow(
    df: pd.DataFrame,
    ownership: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute institutional ownership flow features.

    Uses quarter-over-quarter change in total shares held across all
    institutional investors. Forward-fills from 13F filing dates.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        ownership: DataFrame with columns: report_date, shares_held, change_pct.
                   Loaded from DataStore.load_institutional_ownership().

    Returns:
        df with added columns: flow_institutional_change_pct, flow_institutional_direction.
    """
    result = df.copy()
    result["flow_institutional_change_pct"] = 0.0
    result["flow_institutional_direction"] = "stable"

    if ownership is None or ownership.empty:
        return result

    own = ownership.copy()

    if "report_date" not in own.columns:
        logger.debug("[flow] No report_date column in institutional ownership")
        return result

    own["report_date"] = pd.to_datetime(own["report_date"])

    # Aggregate: total shares held per report_date across all investors
    quarterly_totals = own.groupby("report_date")["shares_held"].sum().sort_index()

    if len(quarterly_totals) < 2:
        return result

    # Quarter-over-quarter % change
    qoq_change = quarterly_totals.pct_change().dropna()

    # Reindex to daily and forward-fill (13F data is available from filing date)
    daily_change = qoq_change.reindex(df.index, method="ffill").fillna(0.0)
    result["flow_institutional_change_pct"] = (daily_change * 100).round(2).values

    # Direction label
    result.loc[
        result["flow_institutional_change_pct"] > _INSTITUTIONAL_CHANGE_THRESHOLD * 100,
        "flow_institutional_direction",
    ] = "accumulating"
    result.loc[
        result["flow_institutional_change_pct"]
        < -_INSTITUTIONAL_CHANGE_THRESHOLD * 100,
        "flow_institutional_direction",
    ] = "distributing"

    return result


def _signed_shares(row: pd.Series) -> float:
    """Convert an insider trade row to signed shares (buy=positive, sell=negative)."""
    shares = abs(float(row.get("shares", 0)))
    tx_type = str(row.get("transaction_type", "")).lower()

    if any(kw in tx_type for kw in ("purchase", "buy", "acquisition")):
        return shares
    if any(kw in tx_type for kw in ("sale", "sell", "disposition")):
        return -shares
    return 0.0
