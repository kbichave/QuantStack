"""
Timeframe definitions and hierarchy for multi-timeframe analysis.

The hierarchy flows from macro (Weekly) to execution (sub-minute):
    WEEKLY (Macro Regime) → DAILY (Intermediate Trend) → 4H (Swing Context)
    → 1H (Execution) → 30M / 15M / 5M (Intraday Scalp) → 1M (Order Flow) → 5S (HFT)
"""

from dataclasses import dataclass
from enum import Enum


class Timeframe(Enum):
    """Supported timeframes for analysis."""

    # Swing / position
    W1 = "1W"  # Macro regime
    D1 = "1D"  # Intermediate trend
    H4 = "4H"  # Swing context
    H1 = "1H"  # Execution (daily trading)

    # Intraday
    M30 = "30M"  # Intraday momentum
    M15 = "15M"  # Intraday structure
    M5 = "5M"  # Scalp setup
    M1 = "1M"  # Order flow / entry timing

    # HFT / tick
    S5 = "5S"  # 5-second bars (high-frequency)


# Hierarchy from highest (macro) to lowest (finest resolution).
# Index 0 = coarsest; higher index = finer.
TIMEFRAME_HIERARCHY: list[Timeframe] = [
    Timeframe.W1,
    Timeframe.D1,
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M30,
    Timeframe.M15,
    Timeframe.M5,
    Timeframe.M1,
    Timeframe.S5,
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
TIMEFRAME_PARAMS: dict[Timeframe, TimeframeParams] = {
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
    # ── Intraday ──────────────────────────────────────────────────────────────
    # Shorter indicator periods reflect the faster mean-reversion at intraday scales.
    # EMA 9/21 is the standard intraday pair; RSI/ATR 9 at M5 and below for responsiveness.
    Timeframe.M30: TimeframeParams(
        ema_fast=9,
        ema_slow=21,
        rsi_period=14,
        stoch_k_period=14,
        stoch_d_period=3,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        roc_period=9,
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
        tp_atr_multiple=1.5,
        sl_atr_multiple=0.75,
        max_hold_bars=8,  # ~4 hours of 30m bars
        resample_rule="30min",
    ),
    Timeframe.M15: TimeframeParams(
        ema_fast=9,
        ema_slow=21,
        rsi_period=14,
        stoch_k_period=14,
        stoch_d_period=3,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        roc_period=9,
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
        tp_atr_multiple=1.5,
        sl_atr_multiple=0.75,
        max_hold_bars=16,  # ~4 hours of 15m bars
        resample_rule="15min",
    ),
    Timeframe.M5: TimeframeParams(
        ema_fast=9,
        ema_slow=21,
        rsi_period=9,
        stoch_k_period=9,
        stoch_d_period=3,
        macd_fast=9,
        macd_slow=21,
        macd_signal=7,
        roc_period=5,
        atr_period=9,
        bb_period=20,
        bb_std=2.0,
        realized_vol_period=20,
        volume_ma_period=20,
        obv_period=20,
        swing_lookback=3,
        trend_exhaustion_bars=4,
        zscore_period=20,
        zscore_entry_threshold=1.5,
        zscore_exit_threshold=0.3,
        tp_atr_multiple=1.5,
        sl_atr_multiple=0.75,
        max_hold_bars=24,  # ~2 hours of 5m bars
        resample_rule="5min",
    ),
    Timeframe.M1: TimeframeParams(
        # Order-flow execution timeframe: fast EMAs, tight stops.
        ema_fast=9,
        ema_slow=21,
        rsi_period=9,
        stoch_k_period=9,
        stoch_d_period=3,
        macd_fast=9,
        macd_slow=21,
        macd_signal=7,
        roc_period=5,
        atr_period=9,
        bb_period=20,
        bb_std=2.0,
        realized_vol_period=20,
        volume_ma_period=20,
        obv_period=20,
        swing_lookback=3,
        trend_exhaustion_bars=3,
        zscore_period=15,
        zscore_entry_threshold=1.5,
        zscore_exit_threshold=0.3,
        tp_atr_multiple=1.0,
        sl_atr_multiple=0.5,
        max_hold_bars=30,  # 30 minutes of 1m bars
        resample_rule="1min",
    ),
    # ── HFT ───────────────────────────────────────────────────────────────────
    Timeframe.S5: TimeframeParams(
        # 5-second bars for microstructure signals.
        # Very short indicator periods — at 5s resolution, a 9-bar EMA spans 45 seconds.
        ema_fast=5,
        ema_slow=13,
        rsi_period=7,
        stoch_k_period=7,
        stoch_d_period=2,
        macd_fast=5,
        macd_slow=13,
        macd_signal=4,
        roc_period=3,
        atr_period=7,
        bb_period=13,
        bb_std=2.0,
        realized_vol_period=12,
        volume_ma_period=12,
        obv_period=12,
        swing_lookback=2,
        trend_exhaustion_bars=2,
        zscore_period=12,
        zscore_entry_threshold=1.5,
        zscore_exit_threshold=0.3,
        tp_atr_multiple=0.75,
        sl_atr_multiple=0.5,
        max_hold_bars=60,  # 5 minutes of 5s bars
        resample_rule="5s",
    ),
}


def get_higher_timeframes(tf: Timeframe) -> list[Timeframe]:
    """Get all timeframes higher than the given one in the hierarchy."""
    tf_idx = TIMEFRAME_HIERARCHY.index(tf)
    return TIMEFRAME_HIERARCHY[:tf_idx]


def get_lower_timeframes(tf: Timeframe) -> list[Timeframe]:
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
