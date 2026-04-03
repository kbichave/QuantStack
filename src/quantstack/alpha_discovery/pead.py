"""PEAD (Post-Earnings Announcement Drift) signal generation.

Computes Standardized Unexpected Earnings (SUE) from historical earnings
data and generates entry signals when SUE exceeds a threshold. PEAD is a
well-documented anomaly: stocks with large positive earnings surprises
continue to drift upward for 60+ trading days.

Data source: earnings_calendar table (already populated by EarningsManager).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_sue(
    symbol: str,
    earnings_history: pd.DataFrame,
    min_quarters: int = 4,
) -> float | None:
    """Compute Standardized Unexpected Earnings for the most recent quarter.

    SUE = latest_surprise / std(past_surprises)

    Args:
        symbol: Ticker symbol (for logging).
        earnings_history: DataFrame with columns [report_date, estimate, reported_eps, surprise].
            Must be sorted by report_date ascending. Expects at least `min_quarters + 1` rows.
        min_quarters: Minimum past quarters needed for a reliable std estimate.

    Returns:
        SUE as a float, or None if insufficient data or zero standard deviation.
    """
    if earnings_history is None or len(earnings_history) < min_quarters + 1:
        logger.debug(f"[PEAD] {symbol}: insufficient earnings history ({len(earnings_history) if earnings_history is not None else 0} rows)")
        return None

    surprises = earnings_history["surprise"].dropna()
    if len(surprises) < min_quarters + 1:
        return None

    past_surprises = surprises.iloc[:-1].values
    latest_surprise = surprises.iloc[-1]

    if len(past_surprises) < min_quarters:
        return None

    std = np.std(past_surprises, ddof=1)
    if std < 1e-10:
        logger.debug(f"[PEAD] {symbol}: zero std in past surprises")
        return None

    sue = float(latest_surprise / std)
    return sue


def get_earnings_history(symbol: str, lookback_quarters: int = 8) -> pd.DataFrame:
    """Fetch historical earnings from earnings_calendar table.

    Returns DataFrame with columns: report_date, estimate, reported_eps, surprise.
    Sorted by report_date ascending. Uses db_conn() context manager.
    """
    from quantstack.db import db_conn

    with db_conn() as conn:
        query = """
            SELECT report_date, estimate, reported_eps, surprise
            FROM earnings_calendar
            WHERE symbol = %s
            ORDER BY report_date DESC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(symbol, lookback_quarters))

    if df.empty:
        return df

    return df.sort_values("report_date").reset_index(drop=True)


def generate_pead_signal(
    symbol: str,
    sue: float,
    sue_threshold: float,
    holding_period_days: int,
    meta_label_probability: float | None = None,
) -> dict | None:
    """Produce a PEAD entry signal if SUE exceeds threshold.

    Returns a signal dict compatible with AlphaDiscoveryEngine's entry rule format,
    or None if SUE is below threshold or negative.
    """
    if sue is None or sue < 0 or sue < sue_threshold:
        return None

    # Position sizing
    if meta_label_probability is not None and meta_label_probability > 0:
        sizing_method = "meta_label"
        bet_size = min(1.0, meta_label_probability)
    else:
        sizing_method = "fixed_fractional"
        bet_size = 0.5  # Half-Kelly fallback

    return {
        "symbol": symbol,
        "direction": "long",
        "signal_type": "pead",
        "sue": float(sue),
        "sue_threshold": float(sue_threshold),
        "holding_period_days": holding_period_days,
        "sizing_method": sizing_method,
        "bet_size": bet_size,
    }
