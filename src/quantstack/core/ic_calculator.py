"""
Information Coefficient (IC) calculator for cross-sectional signal evaluation.

Computes Spearman rank IC, rolling ICIR, and IC decay detection.
Pure computation module — no database access, no LLM calls, no side effects.

Called by run_ic_computation() in supervisor/nodes.py (section-10).
"""

import math

import numpy as np
import pandas as pd
from scipy import stats


def compute_cross_sectional_ic(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """
    Computes Spearman rank IC per date across the cross-section of symbols.

    Args:
        signals: DataFrame with columns [symbol, signal_value], index is date.
        forward_returns: DataFrame with columns [symbol, forward_return], same index.

    Returns:
        pd.Series of IC values indexed by date. NaN for dates with fewer than
        5 symbols present in both inputs after alignment.
    """
    # Pivot to wide format: index=date, columns=symbol
    sig_wide = signals.pivot_table(
        index=signals.index, columns="symbol", values="signal_value"
    )
    ret_wide = forward_returns.pivot_table(
        index=forward_returns.index, columns="symbol", values="forward_return"
    )

    # Align on dates and symbols
    sig_wide, ret_wide = sig_wide.align(ret_wide, join="inner")

    ic_values: dict = {}
    for date in sig_wide.index:
        sig_row = sig_wide.loc[date]
        ret_row = ret_wide.loc[date]

        # Drop NaN for either series, then find common symbols
        valid = sig_row.notna() & ret_row.notna()
        n_valid = valid.sum()

        if n_valid < 5:
            ic_values[date] = float("nan")
            continue

        ic, _ = stats.spearmanr(sig_row[valid].values, ret_row[valid].values)
        ic_values[date] = float(ic)

    return pd.Series(ic_values)


def compute_rolling_icir(
    ic_series: pd.Series,
    window: int,
) -> pd.Series:
    """
    Computes rolling ICIR = mean(IC) / std(IC) over the specified window.

    Args:
        ic_series: pd.Series of IC values (NaN values are skipped by rolling).
        window: rolling window size (typically 21 or 63).

    Returns:
        pd.Series of ICIR values. NaN for the first (window-1) periods and
        for any window where std(IC) == 0.
    """
    roll = ic_series.rolling(window, min_periods=window)
    roll_mean = roll.mean()
    roll_std = roll.std()

    icir = roll_mean / roll_std
    # Undefined when std == 0 (constant IC window)
    icir[roll_std == 0] = float("nan")
    return icir


def detect_ic_decay(
    icir_21d: float,
    icir_63d: float,
    repromotion_check: bool = False,
) -> bool:
    """
    Detects IC decay or re-promotion eligibility.

    Decay condition (repromotion_check=False):
        Returns True if icir_21d < 0.3 AND icir_63d < 0.3 (AND condition).
        A strategy must be consistently poor on BOTH windows to be flagged.

    Re-promotion condition (repromotion_check=True):
        Returns True if icir_21d > 0.5 AND icir_63d > 0.5.
        Hysteresis: demote threshold is 0.3, re-promote threshold is 0.5.

    NaN for either input returns False (no action on missing data).
    All comparisons use strict inequalities.
    """
    # Guard against NaN inputs
    if math.isnan(icir_21d) or math.isnan(icir_63d):
        return False

    if repromotion_check:
        return icir_21d > 0.5 and icir_63d > 0.5

    return icir_21d < 0.3 and icir_63d < 0.3
