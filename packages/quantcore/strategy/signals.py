"""
Signal generator combining MR rules, filters, and ML predictions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Literal
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.config.timeframes import Timeframe, TIMEFRAME_PARAMS
from quantcore.strategy.rules import MeanReversionRules, EntrySignal
from quantcore.strategy.filters import CombinedFilter, FilterResult
from quantcore.models.predictor import Predictor
from quantcore.hierarchy.cascade import SignalCascade, Signal


@dataclass
class GeneratedSignal:
    """Complete signal with all context."""

    symbol: str
    timestamp: datetime
    direction: Literal["LONG", "SHORT"]

    # Price levels
    entry_price: float
    take_profit: float
    stop_loss: float

    # Confidence metrics
    ml_probability: float
    filter_score: float
    alignment_score: float
    combined_confidence: float

    # Rule details
    mr_signal: EntrySignal = None
    filter_result: FilterResult = None

    # Context
    timeframe: Timeframe = Timeframe.H1
    tf_breakdown: Dict[str, str] = field(default_factory=dict)

    # Risk metrics
    risk_reward_ratio: float = 0.0
    atr: float = 0.0

    def __post_init__(self):
        if self.tf_breakdown is None:
            self.tf_breakdown = {}

        # Calculate R:R
        if self.direction == "LONG":
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit

        if risk > 0:
            self.risk_reward_ratio = reward / risk


class SignalGenerator:
    """
    Generates trading signals by combining:
    1. Mean-reversion rules
    2. RRG + Swing filters
    3. ML probability
    4. Multi-timeframe alignment (via cascade)
    """

    def __init__(
        self,
        timeframe: Timeframe = Timeframe.H1,
        probability_threshold: float = 0.6,
        filter_score_threshold: float = 0.4,
        cascade: Optional[SignalCascade] = None,
        predictor: Optional[Predictor] = None,
    ):
        """
        Initialize signal generator.

        Args:
            timeframe: Execution timeframe
            probability_threshold: Minimum ML probability
            filter_score_threshold: Minimum filter score
            cascade: Optional signal cascade for MTF
            predictor: Optional ML predictor
        """
        self.timeframe = timeframe
        self.probability_threshold = probability_threshold
        self.filter_score_threshold = filter_score_threshold

        # Components
        self.mr_rules = MeanReversionRules(timeframe)
        self.filters = CombinedFilter()
        self.cascade = cascade
        self.predictor = predictor

        self.params = TIMEFRAME_PARAMS[timeframe]

    def generate(
        self,
        symbol: str,
        df: pd.DataFrame,
        mtf_data: Optional[Dict[Timeframe, pd.DataFrame]] = None,
        check_long: bool = True,
        check_short: bool = True,
    ) -> List[GeneratedSignal]:
        """
        Generate signals for current bar.

        Args:
            symbol: Stock symbol
            df: Feature DataFrame for execution timeframe
            mtf_data: Optional multi-timeframe data
            check_long: Whether to check for long signals
            check_short: Whether to check for short signals

        Returns:
            List of generated signals
        """
        if df.empty:
            return []

        signals = []
        current = df.iloc[-1]
        timestamp = (
            df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()
        )

        # Check LONG
        if check_long:
            signal = self._check_direction(symbol, df, "LONG", timestamp, mtf_data)
            if signal:
                signals.append(signal)

        # Check SHORT
        if check_short:
            signal = self._check_direction(symbol, df, "SHORT", timestamp, mtf_data)
            if signal:
                signals.append(signal)

        return signals

    def _check_direction(
        self,
        symbol: str,
        df: pd.DataFrame,
        direction: Literal["LONG", "SHORT"],
        timestamp: datetime,
        mtf_data: Optional[Dict[Timeframe, pd.DataFrame]],
    ) -> Optional[GeneratedSignal]:
        """Check signal for a specific direction."""

        # Step 1: Check MR rules
        mr_signal = self.mr_rules.check_entry(df)

        if not mr_signal.triggered or mr_signal.direction != direction:
            return None

        logger.debug(f"{symbol} {direction} MR rule triggered")

        # Step 2: Apply filters
        filter_result = self.filters.check(df, direction)

        if not filter_result.passed:
            logger.debug(f"{symbol} {direction} filtered: {filter_result.reason}")
            return None

        if filter_result.score < self.filter_score_threshold:
            logger.debug(
                f"{symbol} {direction} filter score too low: {filter_result.score}"
            )
            return None

        # Step 3: Get ML probability (if predictor available)
        ml_probability = 0.5
        if self.predictor:
            try:
                X = df.iloc[[-1]]
                ml_probability = float(self.predictor.predict_proba(X)[0])
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")

        if ml_probability < self.probability_threshold:
            logger.debug(f"{symbol} {direction} ML prob too low: {ml_probability}")
            return None

        # Step 4: Check MTF alignment (if cascade available)
        alignment_score = 1.0
        tf_breakdown = {}

        if self.cascade and mtf_data:
            from quantcore.hierarchy.cascade import CascadeResult

            cascade_result = self.cascade.evaluate(
                symbol, mtf_data, direction, ml_probability
            )

            if not cascade_result.passed:
                logger.debug(
                    f"{symbol} {direction} cascade rejected: "
                    f"{cascade_result.rejection_reason}"
                )
                return None

            if cascade_result.alignment:
                alignment_score = cascade_result.alignment.score

            # Build TF breakdown
            if cascade_result.weekly_ctx:
                tf_breakdown["W1"] = cascade_result.weekly_ctx.regime.value
            if cascade_result.daily_ctx:
                tf_breakdown["D1"] = cascade_result.daily_ctx.direction.value
            if cascade_result.h4_ctx:
                tf_breakdown["H4"] = cascade_result.h4_ctx.phase.value
            tf_breakdown["H1"] = "TRIGGERED"

        # Step 5: Calculate price levels
        current = df.iloc[-1]
        entry_price = current["close"]
        atr = current.get("atr", entry_price * 0.01)

        if direction == "LONG":
            take_profit = entry_price + (self.params.tp_atr_multiple * atr)
            stop_loss = entry_price - (self.params.sl_atr_multiple * atr)
        else:
            take_profit = entry_price - (self.params.tp_atr_multiple * atr)
            stop_loss = entry_price + (self.params.sl_atr_multiple * atr)

        # Calculate combined confidence
        combined_confidence = (
            0.4 * ml_probability + 0.3 * filter_result.score + 0.3 * alignment_score
        )

        return GeneratedSignal(
            symbol=symbol,
            timestamp=timestamp,
            direction=direction,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            ml_probability=ml_probability,
            filter_score=filter_result.score,
            alignment_score=alignment_score,
            combined_confidence=combined_confidence,
            mr_signal=mr_signal,
            filter_result=filter_result,
            timeframe=self.timeframe,
            tf_breakdown=tf_breakdown,
            atr=atr,
        )

    def scan_historical(
        self,
        symbol: str,
        df: pd.DataFrame,
        mtf_data: Optional[Dict[Timeframe, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Scan historical data for signals.

        Args:
            symbol: Stock symbol
            df: Historical feature DataFrame
            mtf_data: Optional MTF data

        Returns:
            DataFrame with signal columns
        """
        result = df.copy()

        # Initialize columns
        result["signal"] = 0
        result["signal_direction"] = "NONE"
        result["signal_confidence"] = 0.0
        result["signal_entry"] = np.nan
        result["signal_tp"] = np.nan
        result["signal_sl"] = np.nan

        for i in range(20, len(result)):  # Skip warmup period
            subset = result.iloc[: i + 1]

            # Generate signals
            signals = self.generate(
                symbol, subset, mtf_data, check_long=True, check_short=True
            )

            for sig in signals:
                idx = result.index[i]
                result.loc[idx, "signal"] = 1
                result.loc[idx, "signal_direction"] = sig.direction
                result.loc[idx, "signal_confidence"] = sig.combined_confidence
                result.loc[idx, "signal_entry"] = sig.entry_price
                result.loc[idx, "signal_tp"] = sig.take_profit
                result.loc[idx, "signal_sl"] = sig.stop_loss

        return result
