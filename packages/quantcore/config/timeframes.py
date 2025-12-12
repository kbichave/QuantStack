"""
Timeframe definitions and hierarchy for multi-timeframe analysis.

The hierarchy flows from macro (Weekly) to execution (1H):
    WEEKLY (Macro Regime) → DAILY (Intermediate Trend) → 4H (Swing Context) → 1H (Execution)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List


class Timeframe(Enum):
    """Supported timeframes for analysis."""

    H1 = "1H"  # Execution timeframe
    H4 = "4H"  # Swing context
    D1 = "1D"  # Intermediate trend
    W1 = "1W"  # Macro regime


# Hierarchy from highest (macro) to lowest (execution)
TIMEFRAME_HIERARCHY: List[Timeframe] = [
    Timeframe.W1,
    Timeframe.D1,
    Timeframe.H4,
    Timeframe.H1,
]


@dataclass(frozen=True)
class TimeframeParams:
    """Parameters specific to each timeframe."""

    # EMA periods
    ema_fast: int
    ema_slow: int

    # Momentum
    rsi_period: int
    stoch_k_period: int
    stoch_d_period: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    roc_period: int

    # Volatility
    atr_period: int
    bb_period: int
    bb_std: float
    realized_vol_period: int

    # Volume
    volume_ma_period: int
    obv_period: int

    # Market structure
    swing_lookback: int
    trend_exhaustion_bars: int

    # Z-score thresholds
    zscore_period: int
    zscore_entry_threshold: float
    zscore_exit_threshold: float

    # Trade parameters (ATR multiples)
    tp_atr_multiple: float
    sl_atr_multiple: float
    max_hold_bars: int

    # Pandas resample rule
    resample_rule: str


# Timeframe-specific parameters
TIMEFRAME_PARAMS: Dict[Timeframe, TimeframeParams] = {
    Timeframe.W1: TimeframeParams(
        ema_fast=10,
        ema_slow=20,
        rsi_period=14,
        stoch_k_period=14,
        stoch_d_period=3,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        roc_period=10,
        atr_period=14,
        bb_period=20,
        bb_std=2.0,
        realized_vol_period=20,
        volume_ma_period=20,
        obv_period=20,
        swing_lookback=3,
        trend_exhaustion_bars=4,
        zscore_period=20,
        zscore_entry_threshold=2.0,
        zscore_exit_threshold=0.5,
        tp_atr_multiple=2.5,
        sl_atr_multiple=1.5,
        max_hold_bars=4,
        resample_rule="W-FRI",
    ),
    Timeframe.D1: TimeframeParams(
        ema_fast=20,
        ema_slow=50,
        rsi_period=14,
        stoch_k_period=14,
        stoch_d_period=3,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        roc_period=10,
        atr_period=14,
        bb_period=20,
        bb_std=2.0,
        realized_vol_period=20,
        volume_ma_period=20,
        obv_period=20,
        swing_lookback=5,
        trend_exhaustion_bars=5,
        zscore_period=20,
        zscore_entry_threshold=2.0,
        zscore_exit_threshold=0.5,
        tp_atr_multiple=2.5,
        sl_atr_multiple=1.5,
        max_hold_bars=5,
        resample_rule="D",
    ),
    Timeframe.H4: TimeframeParams(
        ema_fast=20,
        ema_slow=50,
        rsi_period=14,
        stoch_k_period=14,
        stoch_d_period=3,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        roc_period=10,
        atr_period=20,
        bb_period=20,
        bb_std=2.0,
        realized_vol_period=20,
        volume_ma_period=20,
        obv_period=20,
        swing_lookback=5,
        trend_exhaustion_bars=6,
        zscore_period=20,
        zscore_entry_threshold=2.0,
        zscore_exit_threshold=0.5,
        tp_atr_multiple=2.0,
        sl_atr_multiple=1.0,
        max_hold_bars=6,
        resample_rule="4h",
    ),
    Timeframe.H1: TimeframeParams(
        ema_fast=20,
        ema_slow=50,
        rsi_period=14,
        stoch_k_period=14,
        stoch_d_period=3,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        roc_period=10,
        atr_period=20,
        bb_period=20,
        bb_std=2.0,
        realized_vol_period=20,
        volume_ma_period=20,
        obv_period=20,
        swing_lookback=5,
        trend_exhaustion_bars=6,
        zscore_period=20,
        zscore_entry_threshold=2.0,
        zscore_exit_threshold=0.5,
        tp_atr_multiple=1.5,
        sl_atr_multiple=1.0,
        max_hold_bars=6,
        resample_rule="1h",
    ),
}


def get_higher_timeframes(tf: Timeframe) -> List[Timeframe]:
    """Get all timeframes higher than the given one in the hierarchy."""
    tf_idx = TIMEFRAME_HIERARCHY.index(tf)
    return TIMEFRAME_HIERARCHY[:tf_idx]


def get_lower_timeframes(tf: Timeframe) -> List[Timeframe]:
    """Get all timeframes lower than the given one in the hierarchy."""
    tf_idx = TIMEFRAME_HIERARCHY.index(tf)
    return TIMEFRAME_HIERARCHY[tf_idx + 1 :]


def get_next_higher_timeframe(tf: Timeframe) -> Timeframe | None:
    """Get the next higher timeframe, or None if already at highest."""
    tf_idx = TIMEFRAME_HIERARCHY.index(tf)
    if tf_idx == 0:
        return None
    return TIMEFRAME_HIERARCHY[tf_idx - 1]


def get_next_lower_timeframe(tf: Timeframe) -> Timeframe | None:
    """Get the next lower timeframe, or None if already at lowest."""
    tf_idx = TIMEFRAME_HIERARCHY.index(tf)
    if tf_idx == len(TIMEFRAME_HIERARCHY) - 1:
        return None
    return TIMEFRAME_HIERARCHY[tf_idx + 1]
