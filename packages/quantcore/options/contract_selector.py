"""
Rule-based contract selection for options trading.

Implements the hybrid architecture from the plan:
- Direction from RL agent
- Contract selection is deterministic based on regime + vol
- Earnings gating
- Expiry policy
"""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger

from quantcore.options.models import (
    OptionContract,
    OptionType,
    VerticalSpread,
    OptionsPosition,
    OptionLeg,
)


class VolRegime(Enum):
    """Volatility regime based on IV rank."""

    LOW = "LOW"  # IV_RANK < 30
    MEDIUM = "MEDIUM"  # 30 <= IV_RANK <= 70
    HIGH = "HIGH"  # IV_RANK > 70


class TrendRegime(Enum):
    """Trend regime from higher timeframe."""

    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"


class Direction(Enum):
    """Trading direction from RL agent."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class ContractSelectionResult:
    """Result of contract selection."""

    selected: bool
    position: Optional[OptionsPosition] = None
    reason: str = ""
    structure_type: str = ""  # "LONG_CALL", "CALL_SPREAD", etc.


class ContractSelector:
    """
    Rule-based contract selection.

    Selection logic (from plan):
    - Long direction + Low Vol -> Call debit spread
    - Long direction + High Vol -> Long call
    - Short direction + Low Vol -> Put debit spread
    - Short direction + High Vol -> Long put
    - Expiry: 20-45 DTE, prefer monthly
    - Earnings gating: Block short premium N days before earnings
    """

    # IV Rank thresholds
    LOW_VOL_THRESHOLD = 30
    HIGH_VOL_THRESHOLD = 70

    # Expiry policy
    MIN_DTE = 20
    MAX_DTE = 45
    PREFERRED_DTE = 30

    # Earnings gating
    EARNINGS_BLOCK_DAYS = 5  # No new short premium
    EARNINGS_REDUCE_DAYS = 2  # Reduce exposure

    # Delta targets
    LONG_DELTA_TARGET = 0.40  # For directional long options
    SPREAD_WIDTH_ATR_MULTIPLE = 1.0  # Spread width as multiple of ATR

    def __init__(
        self,
        min_dte: int = 20,
        max_dte: int = 45,
        earnings_block_days: int = 5,
    ):
        """
        Initialize contract selector.

        Args:
            min_dte: Minimum days to expiry
            max_dte: Maximum days to expiry
            earnings_block_days: Days before earnings to block short premium
        """
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.earnings_block_days = earnings_block_days

    def classify_vol_regime(self, iv_rank: float) -> VolRegime:
        """Classify volatility regime from IV rank."""
        if iv_rank < self.LOW_VOL_THRESHOLD:
            return VolRegime.LOW
        elif iv_rank > self.HIGH_VOL_THRESHOLD:
            return VolRegime.HIGH
        else:
            return VolRegime.MEDIUM

    def select_contract(
        self,
        direction: Direction,
        confidence: float,
        vol_regime: VolRegime,
        trend_regime: TrendRegime,
        options_chain: pd.DataFrame,
        underlying_price: float,
        atr: float,
        days_to_earnings: Optional[int] = None,
        position_id: str = "",
        symbol: str = "",
    ) -> ContractSelectionResult:
        """
        Select contract based on rules.

        Args:
            direction: Trading direction (LONG/SHORT/FLAT)
            confidence: Confidence from RL agent [-1, +1]
            vol_regime: Current volatility regime
            trend_regime: Current trend regime
            options_chain: DataFrame with available options
            underlying_price: Current underlying price
            atr: ATR for spread width calculation
            days_to_earnings: Days until next earnings (None if unknown)
            position_id: ID for the new position
            symbol: Ticker symbol for per-ticker config lookup

        Returns:
            ContractSelectionResult with selected position or rejection reason
        """
        # Flat direction - no trade
        if direction == Direction.FLAT:
            return ContractSelectionResult(
                selected=False,
                reason="Direction is FLAT",
            )

        # Low confidence - no trade
        if abs(confidence) < 0.2:
            return ContractSelectionResult(
                selected=False,
                reason=f"Confidence too low: {confidence:.2f}",
            )

        # Check earnings gate - block trades near earnings
        earnings_allowed, earnings_reason = self.check_earnings_gate(
            days_to_earnings=days_to_earnings,
            is_short_premium=False,  # Debit spreads and long options are OK
        )

        if not earnings_allowed:
            return ContractSelectionResult(
                selected=False,
                reason=f"Earnings gate: {earnings_reason}",
            )

        # Log earnings warning if close
        if days_to_earnings is not None and days_to_earnings <= 7:
            logger.warning(
                f"{symbol}: {days_to_earnings} days to earnings - reduced size recommended"
            )

        # Check trend regime restrictions
        if trend_regime == TrendRegime.BEAR and direction == Direction.LONG:
            # In bear regime, reduce long exposure or use puts instead
            logger.debug("Bear regime - considering protective structure")

        # Filter options by DTE
        valid_options = self._filter_by_dte(options_chain)
        if valid_options.empty:
            return ContractSelectionResult(
                selected=False,
                reason=f"No options in DTE range {self.min_dte}-{self.max_dte}",
            )

        # Determine structure based on direction and vol
        if direction == Direction.LONG:
            result = self._select_long_structure(
                vol_regime=vol_regime,
                trend_regime=trend_regime,
                options_chain=valid_options,
                underlying_price=underlying_price,
                atr=atr,
                days_to_earnings=days_to_earnings,
                position_id=position_id,
            )
        else:  # SHORT direction
            result = self._select_short_structure(
                vol_regime=vol_regime,
                trend_regime=trend_regime,
                options_chain=valid_options,
                underlying_price=underlying_price,
                atr=atr,
                days_to_earnings=days_to_earnings,
                position_id=position_id,
            )

        return result

    def _filter_by_dte(self, options_chain: pd.DataFrame) -> pd.DataFrame:
        """Filter options by DTE range."""
        if options_chain.empty:
            return options_chain

        today = date.today()

        df = options_chain.copy()
        if "expiry" in df.columns:
            df["dte"] = (pd.to_datetime(df["expiry"]).dt.date - today).apply(
                lambda x: x.days
            )
            df = df[(df["dte"] >= self.min_dte) & (df["dte"] <= self.max_dte)]

        return df

    def _select_long_structure(
        self,
        vol_regime: VolRegime,
        trend_regime: TrendRegime,
        options_chain: pd.DataFrame,
        underlying_price: float,
        atr: float,
        days_to_earnings: Optional[int],
        position_id: str,
    ) -> ContractSelectionResult:
        """Select structure for bullish direction."""

        # In BEAR regime, we might want protective puts instead
        if trend_regime == TrendRegime.BEAR:
            return ContractSelectionResult(
                selected=False,
                reason="Long direction blocked in BEAR regime",
            )

        if vol_regime == VolRegime.LOW:
            # Low vol -> Call debit spread (defined risk)
            return self._build_call_spread(
                options_chain=options_chain,
                underlying_price=underlying_price,
                spread_width=atr * self.SPREAD_WIDTH_ATR_MULTIPLE,
                position_id=position_id,
            )
        else:
            # High/Medium vol -> Long call
            return self._build_long_call(
                options_chain=options_chain,
                underlying_price=underlying_price,
                target_delta=self.LONG_DELTA_TARGET,
                position_id=position_id,
            )

    def _select_short_structure(
        self,
        vol_regime: VolRegime,
        trend_regime: TrendRegime,
        options_chain: pd.DataFrame,
        underlying_price: float,
        atr: float,
        days_to_earnings: Optional[int],
        position_id: str,
    ) -> ContractSelectionResult:
        """Select structure for bearish direction."""

        # In BULL regime with short direction, be cautious
        if trend_regime == TrendRegime.BULL:
            logger.debug("Short direction in BULL regime - using defined risk")

        if vol_regime == VolRegime.LOW:
            # Low vol -> Put debit spread (defined risk)
            return self._build_put_spread(
                options_chain=options_chain,
                underlying_price=underlying_price,
                spread_width=atr * self.SPREAD_WIDTH_ATR_MULTIPLE,
                position_id=position_id,
            )
        else:
            # High vol -> Long put
            return self._build_long_put(
                options_chain=options_chain,
                underlying_price=underlying_price,
                target_delta=-self.LONG_DELTA_TARGET,
                position_id=position_id,
            )

    def _build_long_call(
        self,
        options_chain: pd.DataFrame,
        underlying_price: float,
        target_delta: float,
        position_id: str,
    ) -> ContractSelectionResult:
        """Build long call position."""
        calls = options_chain[options_chain["option_type"].str.upper() == "CALL"]

        if calls.empty:
            return ContractSelectionResult(
                selected=False,
                reason="No calls available",
            )

        # Find call closest to target delta
        if "delta" in calls.columns:
            calls = calls.copy()
            calls["delta_diff"] = abs(calls["delta"] - target_delta)
            best_call = calls.loc[calls["delta_diff"].idxmin()]
        else:
            # Fall back to ATM
            calls = calls.copy()
            calls["strike_diff"] = abs(calls["strike"] - underlying_price)
            best_call = calls.loc[calls["strike_diff"].idxmin()]

        contract = OptionContract.from_dict(best_call.to_dict())

        leg = OptionLeg(
            contract=contract,
            quantity=1,
            entry_price=contract.mid,
        )

        position = OptionsPosition(
            position_id=position_id,
            underlying=contract.underlying,
            legs=[leg],
        )

        return ContractSelectionResult(
            selected=True,
            position=position,
            structure_type="LONG_CALL",
            reason=f"Long {contract.strike} call, {contract.days_to_expiry} DTE",
        )

    def _build_long_put(
        self,
        options_chain: pd.DataFrame,
        underlying_price: float,
        target_delta: float,
        position_id: str,
    ) -> ContractSelectionResult:
        """Build long put position."""
        puts = options_chain[options_chain["option_type"].str.upper() == "PUT"]

        if puts.empty:
            return ContractSelectionResult(
                selected=False,
                reason="No puts available",
            )

        # Find put closest to target delta
        if "delta" in puts.columns:
            puts = puts.copy()
            puts["delta_diff"] = abs(puts["delta"] - target_delta)
            best_put = puts.loc[puts["delta_diff"].idxmin()]
        else:
            # Fall back to ATM
            puts = puts.copy()
            puts["strike_diff"] = abs(puts["strike"] - underlying_price)
            best_put = puts.loc[puts["strike_diff"].idxmin()]

        contract = OptionContract.from_dict(best_put.to_dict())

        leg = OptionLeg(
            contract=contract,
            quantity=1,
            entry_price=contract.mid,
        )

        position = OptionsPosition(
            position_id=position_id,
            underlying=contract.underlying,
            legs=[leg],
        )

        return ContractSelectionResult(
            selected=True,
            position=position,
            structure_type="LONG_PUT",
            reason=f"Long {contract.strike} put, {contract.days_to_expiry} DTE",
        )

    def _build_call_spread(
        self,
        options_chain: pd.DataFrame,
        underlying_price: float,
        spread_width: float,
        position_id: str,
    ) -> ContractSelectionResult:
        """Build call debit spread (bull call spread)."""
        calls = options_chain[options_chain["option_type"].str.upper() == "CALL"]

        if len(calls) < 2:
            return ContractSelectionResult(
                selected=False,
                reason="Not enough calls for spread",
            )

        # Select expiry with most liquidity
        if "volume" in calls.columns:
            best_expiry = calls.groupby("expiry")["volume"].sum().idxmax()
            calls = calls[calls["expiry"] == best_expiry]
        else:
            calls = calls[calls["expiry"] == calls["expiry"].iloc[0]]

        # Find long strike (ATM or slightly OTM)
        calls = calls.copy()
        calls["strike_diff"] = abs(calls["strike"] - underlying_price)
        long_call = calls.loc[calls["strike_diff"].idxmin()]
        long_strike = long_call["strike"]

        # Find short strike (long_strike + spread_width)
        target_short_strike = long_strike + spread_width
        calls["short_diff"] = abs(calls["strike"] - target_short_strike)

        # Ensure short strike is higher than long strike
        higher_strikes = calls[calls["strike"] > long_strike]
        if higher_strikes.empty:
            return ContractSelectionResult(
                selected=False,
                reason="No strikes available for short leg",
            )

        short_call = higher_strikes.loc[higher_strikes["short_diff"].idxmin()]

        long_contract = OptionContract.from_dict(long_call.to_dict())
        short_contract = OptionContract.from_dict(short_call.to_dict())

        long_leg = OptionLeg(
            contract=long_contract,
            quantity=1,
            entry_price=long_contract.mid,
        )

        short_leg = OptionLeg(
            contract=short_contract,
            quantity=-1,
            entry_price=short_contract.mid,
        )

        position = OptionsPosition(
            position_id=position_id,
            underlying=long_contract.underlying,
            legs=[long_leg, short_leg],
        )

        return ContractSelectionResult(
            selected=True,
            position=position,
            structure_type="CALL_DEBIT_SPREAD",
            reason=f"Call spread {long_contract.strike}/{short_contract.strike}, {long_contract.days_to_expiry} DTE",
        )

    def _build_put_spread(
        self,
        options_chain: pd.DataFrame,
        underlying_price: float,
        spread_width: float,
        position_id: str,
    ) -> ContractSelectionResult:
        """Build put debit spread (bear put spread)."""
        puts = options_chain[options_chain["option_type"].str.upper() == "PUT"]

        if len(puts) < 2:
            return ContractSelectionResult(
                selected=False,
                reason="Not enough puts for spread",
            )

        # Select expiry with most liquidity
        if "volume" in puts.columns:
            best_expiry = puts.groupby("expiry")["volume"].sum().idxmax()
            puts = puts[puts["expiry"] == best_expiry]
        else:
            puts = puts[puts["expiry"] == puts["expiry"].iloc[0]]

        # Find long strike (ATM or slightly OTM)
        puts = puts.copy()
        puts["strike_diff"] = abs(puts["strike"] - underlying_price)
        long_put = puts.loc[puts["strike_diff"].idxmin()]
        long_strike = long_put["strike"]

        # Find short strike (long_strike - spread_width)
        target_short_strike = long_strike - spread_width
        puts["short_diff"] = abs(puts["strike"] - target_short_strike)

        # Ensure short strike is lower than long strike
        lower_strikes = puts[puts["strike"] < long_strike]
        if lower_strikes.empty:
            return ContractSelectionResult(
                selected=False,
                reason="No strikes available for short leg",
            )

        short_put = lower_strikes.loc[lower_strikes["short_diff"].idxmin()]

        long_contract = OptionContract.from_dict(long_put.to_dict())
        short_contract = OptionContract.from_dict(short_put.to_dict())

        long_leg = OptionLeg(
            contract=long_contract,
            quantity=1,
            entry_price=long_contract.mid,
        )

        short_leg = OptionLeg(
            contract=short_contract,
            quantity=-1,
            entry_price=short_contract.mid,
        )

        position = OptionsPosition(
            position_id=position_id,
            underlying=long_contract.underlying,
            legs=[long_leg, short_leg],
        )

        return ContractSelectionResult(
            selected=True,
            position=position,
            structure_type="PUT_DEBIT_SPREAD",
            reason=f"Put spread {long_contract.strike}/{short_contract.strike}, {long_contract.days_to_expiry} DTE",
        )

    def check_earnings_gate(
        self,
        days_to_earnings: Optional[int],
        is_short_premium: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if trade is blocked by earnings gate.

        Args:
            days_to_earnings: Days until next earnings
            is_short_premium: True if position involves selling premium

        Returns:
            Tuple of (allowed, reason)
        """
        if days_to_earnings is None:
            return True, "Earnings date unknown"

        if days_to_earnings <= self.earnings_block_days:
            if is_short_premium:
                return (
                    False,
                    f"Short premium blocked: {days_to_earnings} days to earnings",
                )

            if days_to_earnings <= self.EARNINGS_REDUCE_DAYS:
                return True, f"Reduce exposure: {days_to_earnings} days to earnings"

        return True, "OK"
