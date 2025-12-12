"""
Event-based trade labeling for ML classification.

Labels each bar based on the outcome of a hypothetical mean-reversion trade:
- Entry at bar close
- TP and SL based on ATR multiples
- Outcome: 1 if TP hit before SL within horizon, 0 otherwise
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.config.timeframes import Timeframe, TIMEFRAME_PARAMS


class TradeOutcome(Enum):
    """Trade outcome classification."""

    WIN = 1  # TP hit before SL
    LOSS = 0  # SL hit before TP
    TIMEOUT = 0  # Neither hit within horizon (counted as loss)


@dataclass
class LabelConfig:
    """Configuration for trade labeling."""

    tp_atr_multiple: float = 1.5
    sl_atr_multiple: float = 1.0
    max_hold_bars: int = 6

    @classmethod
    def from_timeframe(cls, timeframe: Timeframe) -> "LabelConfig":
        """Create config from timeframe parameters."""
        params = TIMEFRAME_PARAMS[timeframe]
        return cls(
            tp_atr_multiple=params.tp_atr_multiple,
            sl_atr_multiple=params.sl_atr_multiple,
            max_hold_bars=params.max_hold_bars,
        )


class EventLabeler:
    """
    Labels bars based on hypothetical trade outcomes.

    For each bar, simulates entering a trade and tracks whether
    TP or SL is hit first within the holding period.
    """

    def __init__(self, config: Optional[LabelConfig] = None):
        """
        Initialize the labeler.

        Args:
            config: Labeling configuration (uses defaults if None)
        """
        self.config = config or LabelConfig()

    def label_long_trades(
        self,
        df: pd.DataFrame,
        atr_column: str = "atr",
    ) -> pd.DataFrame:
        """
        Label long mean-reversion trades.

        Args:
            df: DataFrame with OHLCV and ATR
            atr_column: Name of ATR column

        Returns:
            DataFrame with additional label columns
        """
        result = df.copy()

        if atr_column not in result.columns:
            logger.warning(f"ATR column '{atr_column}' not found, computing it")
            result[atr_column] = self._compute_atr(result)

        # Initialize label columns
        result["label_long"] = np.nan
        result["label_long_bars_to_exit"] = np.nan
        result["label_long_exit_type"] = None
        result["label_long_pnl_pct"] = np.nan

        # Label each bar
        for i in range(len(result) - self.config.max_hold_bars):
            entry_idx = i
            entry_price = result["close"].iloc[entry_idx]
            atr = result[atr_column].iloc[entry_idx]

            if pd.isna(atr) or atr <= 0:
                continue

            tp_price = entry_price + (self.config.tp_atr_multiple * atr)
            sl_price = entry_price - (self.config.sl_atr_multiple * atr)

            # Check future bars
            outcome, bars_held, exit_type, pnl_pct = self._evaluate_long_trade(
                result.iloc[entry_idx + 1 : entry_idx + 1 + self.config.max_hold_bars],
                entry_price,
                tp_price,
                sl_price,
            )

            result.iloc[entry_idx, result.columns.get_loc("label_long")] = outcome.value
            result.iloc[
                entry_idx, result.columns.get_loc("label_long_bars_to_exit")
            ] = bars_held
            result.iloc[entry_idx, result.columns.get_loc("label_long_exit_type")] = (
                exit_type
            )
            result.iloc[entry_idx, result.columns.get_loc("label_long_pnl_pct")] = (
                pnl_pct
            )

        return result

    def label_short_trades(
        self,
        df: pd.DataFrame,
        atr_column: str = "atr",
    ) -> pd.DataFrame:
        """
        Label short mean-reversion trades.

        Args:
            df: DataFrame with OHLCV and ATR
            atr_column: Name of ATR column

        Returns:
            DataFrame with additional label columns
        """
        result = df.copy()

        if atr_column not in result.columns:
            logger.warning(f"ATR column '{atr_column}' not found, computing it")
            result[atr_column] = self._compute_atr(result)

        # Initialize label columns
        result["label_short"] = np.nan
        result["label_short_bars_to_exit"] = np.nan
        result["label_short_exit_type"] = None
        result["label_short_pnl_pct"] = np.nan

        # Label each bar
        for i in range(len(result) - self.config.max_hold_bars):
            entry_idx = i
            entry_price = result["close"].iloc[entry_idx]
            atr = result[atr_column].iloc[entry_idx]

            if pd.isna(atr) or atr <= 0:
                continue

            tp_price = entry_price - (self.config.tp_atr_multiple * atr)
            sl_price = entry_price + (self.config.sl_atr_multiple * atr)

            outcome, bars_held, exit_type, pnl_pct = self._evaluate_short_trade(
                result.iloc[entry_idx + 1 : entry_idx + 1 + self.config.max_hold_bars],
                entry_price,
                tp_price,
                sl_price,
            )

            result.iloc[entry_idx, result.columns.get_loc("label_short")] = (
                outcome.value
            )
            result.iloc[
                entry_idx, result.columns.get_loc("label_short_bars_to_exit")
            ] = bars_held
            result.iloc[entry_idx, result.columns.get_loc("label_short_exit_type")] = (
                exit_type
            )
            result.iloc[entry_idx, result.columns.get_loc("label_short_pnl_pct")] = (
                pnl_pct
            )

        return result

    def label_trades(
        self,
        df: pd.DataFrame,
        atr_column: str = "atr",
    ) -> pd.DataFrame:
        """
        Label both long and short trades.

        Args:
            df: DataFrame with OHLCV and ATR
            atr_column: Name of ATR column

        Returns:
            DataFrame with both long and short labels
        """
        result = self.label_long_trades(df, atr_column)
        result = self.label_short_trades(result, atr_column)
        return result

    def _evaluate_long_trade(
        self,
        future_bars: pd.DataFrame,
        entry_price: float,
        tp_price: float,
        sl_price: float,
    ) -> Tuple[TradeOutcome, int, str, float]:
        """
        Evaluate outcome of a long trade.

        Returns:
            Tuple of (outcome, bars_held, exit_type, pnl_pct)
        """
        for i, (idx, bar) in enumerate(future_bars.iterrows()):
            bars_held = i + 1

            # Check if SL hit (using low)
            if bar["low"] <= sl_price:
                pnl_pct = (sl_price - entry_price) / entry_price * 100
                return TradeOutcome.LOSS, bars_held, "SL", pnl_pct

            # Check if TP hit (using high)
            if bar["high"] >= tp_price:
                pnl_pct = (tp_price - entry_price) / entry_price * 100
                return TradeOutcome.WIN, bars_held, "TP", pnl_pct

        # Timeout - exit at last bar close
        if len(future_bars) > 0:
            exit_price = future_bars["close"].iloc[-1]
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            return TradeOutcome.TIMEOUT, len(future_bars), "TIMEOUT", pnl_pct

        return TradeOutcome.TIMEOUT, 0, "NO_DATA", 0.0

    def _evaluate_short_trade(
        self,
        future_bars: pd.DataFrame,
        entry_price: float,
        tp_price: float,
        sl_price: float,
    ) -> Tuple[TradeOutcome, int, str, float]:
        """
        Evaluate outcome of a short trade.

        Returns:
            Tuple of (outcome, bars_held, exit_type, pnl_pct)
        """
        for i, (idx, bar) in enumerate(future_bars.iterrows()):
            bars_held = i + 1

            # Check if SL hit (using high)
            if bar["high"] >= sl_price:
                pnl_pct = (entry_price - sl_price) / entry_price * 100
                return TradeOutcome.LOSS, bars_held, "SL", pnl_pct

            # Check if TP hit (using low)
            if bar["low"] <= tp_price:
                pnl_pct = (entry_price - tp_price) / entry_price * 100
                return TradeOutcome.WIN, bars_held, "TP", pnl_pct

        # Timeout
        if len(future_bars) > 0:
            exit_price = future_bars["close"].iloc[-1]
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            return TradeOutcome.TIMEOUT, len(future_bars), "TIMEOUT", pnl_pct

        return TradeOutcome.TIMEOUT, 0, "NO_DATA", 0.0

    def _compute_atr(
        self,
        df: pd.DataFrame,
        period: int = 14,
    ) -> pd.Series:
        """Compute ATR if not available."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def get_label_statistics(
        self,
        df: pd.DataFrame,
        label_column: str = "label_long",
    ) -> dict:
        """
        Get statistics for labels.

        Args:
            df: DataFrame with labels
            label_column: Column to analyze

        Returns:
            Dictionary with label statistics
        """
        labels = df[label_column].dropna()

        if len(labels) == 0:
            return {"count": 0}

        stats = {
            "count": len(labels),
            "win_count": int((labels == 1).sum()),
            "loss_count": int((labels == 0).sum()),
            "win_rate": float((labels == 1).mean()),
            "class_balance": float(labels.mean()),
        }

        # Exit type breakdown
        exit_col = label_column.replace("label_", "label_") + "_exit_type"
        if exit_col in df.columns:
            exit_types = df[exit_col].dropna()
            stats["exit_breakdown"] = exit_types.value_counts().to_dict()

        # PnL statistics
        pnl_col = label_column + "_pnl_pct"
        if pnl_col in df.columns:
            pnl = df[pnl_col].dropna()
            stats["avg_pnl_pct"] = float(pnl.mean())
            stats["avg_win_pnl"] = (
                float(pnl[labels == 1].mean()) if (labels == 1).any() else 0.0
            )
            stats["avg_loss_pnl"] = (
                float(pnl[labels == 0].mean()) if (labels == 0).any() else 0.0
            )

        return stats


class MultiTimeframeLabelBuilder:
    """
    Builds labels across multiple timeframes with proper configuration.
    """

    def __init__(self):
        """Initialize multi-TF label builder."""
        self.labelers = {
            tf: EventLabeler(LabelConfig.from_timeframe(tf)) for tf in Timeframe
        }

    def label_all_timeframes(
        self,
        data: dict[Timeframe, pd.DataFrame],
    ) -> dict[Timeframe, pd.DataFrame]:
        """
        Label trades for all timeframes.

        Args:
            data: Dictionary of DataFrames per timeframe

        Returns:
            Dictionary of labeled DataFrames
        """
        result = {}

        for tf, df in data.items():
            if df.empty:
                result[tf] = df
                continue

            logger.info(f"Labeling {tf.value} trades ({len(df)} bars)")
            result[tf] = self.labelers[tf].label_trades(df)

            # Log statistics
            stats = self.labelers[tf].get_label_statistics(result[tf], "label_long")
            logger.info(f"{tf.value} long labels: {stats}")

        return result
