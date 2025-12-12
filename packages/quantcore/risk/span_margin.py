"""
SPAN-Like Margin Calculation for Options Portfolios.

Implements a simplified version of CME's SPAN (Standard Portfolio
Analysis of Risk) margin methodology.

SPAN calculates margin by:
1. Scanning risk: Maximum portfolio loss across 16 price/vol scenarios
2. Inter-spread credits: Reduction for offsetting positions
3. Delivery margin: Additional margin for near-expiry ITM options
4. Net option value: Credit for option value in portfolio

This implementation is suitable for backtesting and risk management,
not for actual margin calculation with brokers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from quantcore.options.models import OptionsPosition, OptionType, OptionLeg
from quantcore.options.pricing import black_scholes_price


class MarginTier(Enum):
    """Margin requirement tier based on position type."""

    LONG_OPTION = "long_option"  # Just the premium paid
    COVERED_CALL = "covered_call"  # Reduced margin
    CASH_SECURED_PUT = "cash_secured_put"  # Strike * 100
    NAKED_OPTION = "naked_option"  # Full SPAN calculation
    SPREAD = "spread"  # Width of spread
    STRADDLE = "straddle"  # Naked margin on both sides


@dataclass
class MarginBreakdown:
    """Detailed margin calculation breakdown."""

    scan_risk: float  # Maximum loss from 16 scenarios
    inter_spread_credit: float  # Credit for offsetting positions
    delivery_margin: float  # Additional for near-expiry ITM
    short_option_minimum: float  # Minimum for short options
    net_option_value: float  # Premium credit
    total_margin: float  # Final margin requirement

    # Component details
    scenario_losses: List[float] = field(default_factory=list)
    worst_scenario: str = ""
    margin_tier: MarginTier = MarginTier.LONG_OPTION


@dataclass
class SPANScenario:
    """A single SPAN scenario for risk scanning."""

    name: str
    price_move_pct: float
    vol_move_pct: float


class SPANMarginCalculator:
    """
    SPAN-style margin calculation for options portfolios.

    Uses a 16-scenario scanning methodology similar to CME SPAN:
    - Price moves: -3σ to +3σ in thirds
    - Vol moves: -1σ to +1σ
    - Extreme scenarios with price moves > 3σ

    Key concepts:
    - Scanning Risk: Max loss across all scenarios
    - Inter-commodity Spread Credit: Reduces margin for offsetting positions
    - Intra-commodity Spread Charge: Additional charge for calendar spreads
    - Delivery Margin: For near-expiry options
    """

    # SPAN risk scenarios (price_fraction, vol_fraction) of scan range
    # 16 standard scenarios
    SCAN_SCENARIOS = [
        SPANScenario("unchanged", 0.0, 0.0),
        SPANScenario("up_1/3", 1 / 3, 0.0),
        SPANScenario("down_1/3", -1 / 3, 0.0),
        SPANScenario("up_2/3", 2 / 3, 0.0),
        SPANScenario("down_2/3", -2 / 3, 0.0),
        SPANScenario("up_3/3", 1.0, 0.0),
        SPANScenario("down_3/3", -1.0, 0.0),
        SPANScenario("up_1/3_vol_up", 1 / 3, 1.0),
        SPANScenario("up_1/3_vol_down", 1 / 3, -1.0),
        SPANScenario("down_1/3_vol_up", -1 / 3, 1.0),
        SPANScenario("down_1/3_vol_down", -1 / 3, -1.0),
        SPANScenario("up_2/3_vol_up", 2 / 3, 1.0),
        SPANScenario("up_2/3_vol_down", 2 / 3, -1.0),
        SPANScenario("down_2/3_vol_up", -2 / 3, 1.0),
        SPANScenario("down_2/3_vol_down", -2 / 3, -1.0),
        # Extreme moves (cover most of scan risk)
        SPANScenario("extreme_up", 2.0, 0.0),
        SPANScenario("extreme_down", -2.0, 0.0),
    ]

    # Default parameters
    DEFAULT_PRICE_SCAN_RANGE = 0.15  # 15% of underlying price
    DEFAULT_VOL_SCAN_RANGE = 0.25  # 25% of current IV
    EXTREME_MOVE_COVERAGE = 0.35  # 35% for extreme scenarios

    # Minimum margin percentages
    MIN_SHORT_OPTION_MARGIN_PCT = 0.10  # 10% of underlying for naked shorts
    MIN_LONG_OPTION_MARGIN_PCT = 0.0  # Long options = premium paid

    def __init__(
        self,
        price_scan_range: float = 0.15,
        vol_scan_range: float = 0.25,
        risk_free_rate: float = 0.05,
    ):
        """
        Initialize SPAN margin calculator.

        Args:
            price_scan_range: Price scan range as fraction (e.g., 0.15 = 15%)
            vol_scan_range: Vol scan range as fraction
            risk_free_rate: Risk-free rate for option repricing
        """
        self.price_scan_range = price_scan_range
        self.vol_scan_range = vol_scan_range
        self.risk_free_rate = risk_free_rate

    def calculate_scan_risk(
        self,
        position: OptionsPosition,
        underlying_price: float,
        current_iv: float,
        days_to_expiry: Optional[int] = None,
    ) -> Tuple[float, List[float], str]:
        """
        Calculate SPAN scanning risk for a position.

        Scanning risk is the maximum potential loss across
        all SPAN scenarios.

        Args:
            position: Options position to evaluate
            underlying_price: Current underlying price
            current_iv: Current implied volatility
            days_to_expiry: Override DTE (uses contract DTE if None)

        Returns:
            Tuple of (max_loss, scenario_losses, worst_scenario_name)
        """
        scenario_losses = []
        worst_loss = 0.0
        worst_scenario = "unchanged"

        # Calculate current position value
        current_value = self._calculate_position_value(
            position, underlying_price, current_iv, days_to_expiry
        )

        for scenario in self.SCAN_SCENARIOS:
            # Calculate stressed parameters
            if "extreme" in scenario.name:
                price_move = scenario.price_move_pct * self.EXTREME_MOVE_COVERAGE
            else:
                price_move = scenario.price_move_pct * self.price_scan_range

            vol_move = scenario.vol_move_pct * self.vol_scan_range

            stressed_price = underlying_price * (1 + price_move)
            stressed_iv = current_iv * (1 + vol_move)
            stressed_iv = max(0.05, min(2.0, stressed_iv))  # Clip to reasonable range

            # Calculate position value under stress
            stressed_value = self._calculate_position_value(
                position, stressed_price, stressed_iv, days_to_expiry
            )

            # Loss is negative P&L
            loss = current_value - stressed_value
            scenario_losses.append(loss)

            if loss > worst_loss:
                worst_loss = loss
                worst_scenario = scenario.name

        return worst_loss, scenario_losses, worst_scenario

    def _calculate_position_value(
        self,
        position: OptionsPosition,
        spot: float,
        iv: float,
        days_to_expiry: Optional[int] = None,
    ) -> float:
        """Calculate total position value."""
        total = 0.0

        for leg in position.legs:
            contract = leg.contract
            dte = (
                days_to_expiry
                if days_to_expiry is not None
                else contract.days_to_expiry
            )
            T = max(0.001, dte / 365)

            price = black_scholes_price(
                S=spot,
                K=contract.strike,
                T=T,
                r=self.risk_free_rate,
                sigma=iv,
                option_type=contract.option_type,
            )

            # Value per contract (x100 multiplier)
            leg_value = price * 100 * leg.quantity
            if not leg.is_long:
                leg_value = -leg_value

            total += leg_value

        return total

    def calculate_inter_spread_credit(
        self,
        positions: Dict[str, OptionsPosition],
    ) -> float:
        """
        Calculate credit for offsetting positions across symbols.

        Highly correlated underlyings (like QQQ and SPY) get margin
        reduction when positions offset each other.

        Args:
            positions: Dict of symbol -> position

        Returns:
            Credit amount to reduce margin
        """
        # Simplified: give credit for long/short pairs in same sector
        total_credit = 0.0

        # Calculate net delta exposure per position
        net_deltas = {}
        for symbol, position in positions.items():
            # Approximate delta: sum of leg deltas
            delta = 0.0
            for leg in position.legs:
                contract = leg.contract
                # Rough delta estimate
                if contract.option_type == OptionType.CALL:
                    leg_delta = 0.5 if leg.is_long else -0.5
                else:
                    leg_delta = -0.5 if leg.is_long else 0.5
                delta += leg_delta * leg.quantity

            net_deltas[symbol] = delta

        # Find offsetting pairs
        symbols = list(positions.keys())
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                d1 = net_deltas[symbols[i]]
                d2 = net_deltas[symbols[j]]

                # If deltas offset (opposite signs), give credit
                if d1 * d2 < 0:
                    offset_amount = min(abs(d1), abs(d2))
                    # 30% credit for offsetting exposure
                    credit = offset_amount * 100 * 0.30
                    total_credit += credit

        return total_credit

    def calculate_delivery_margin(
        self,
        position: OptionsPosition,
        underlying_price: float,
        days_to_expiry_threshold: int = 5,
    ) -> float:
        """
        Calculate additional margin for near-expiry ITM options.

        Options close to expiry with high probability of exercise
        require additional margin for potential delivery.

        Args:
            position: Options position
            underlying_price: Current price
            days_to_expiry_threshold: Days threshold for delivery margin

        Returns:
            Additional delivery margin
        """
        delivery_margin = 0.0

        for leg in position.legs:
            contract = leg.contract

            # Only short positions and near expiry
            if leg.is_long:
                continue

            if contract.days_to_expiry > days_to_expiry_threshold:
                continue

            # Check if ITM
            is_itm = False
            if contract.option_type == OptionType.CALL:
                is_itm = underlying_price > contract.strike
            else:
                is_itm = underlying_price < contract.strike

            if is_itm:
                # Delivery margin = intrinsic value
                if contract.option_type == OptionType.CALL:
                    intrinsic = underlying_price - contract.strike
                else:
                    intrinsic = contract.strike - underlying_price

                delivery_margin += intrinsic * 100 * leg.quantity

        return delivery_margin

    def calculate_short_option_minimum(
        self,
        position: OptionsPosition,
        underlying_price: float,
    ) -> float:
        """
        Calculate minimum margin for short options.

        Even if scanning risk is low, short options require
        a minimum margin based on underlying value.

        Args:
            position: Options position
            underlying_price: Current price

        Returns:
            Minimum margin for short options
        """
        minimum = 0.0

        for leg in position.legs:
            if leg.is_long:
                continue

            # Minimum: 10% of underlying value per short contract
            min_per_contract = underlying_price * 100 * self.MIN_SHORT_OPTION_MARGIN_PCT
            minimum += min_per_contract * leg.quantity

        return minimum

    def calculate_margin(
        self,
        position: OptionsPosition,
        underlying_price: float,
        current_iv: float,
        days_to_expiry: Optional[int] = None,
    ) -> MarginBreakdown:
        """
        Calculate comprehensive SPAN-style margin for a position.

        Args:
            position: Options position
            underlying_price: Current underlying price
            current_iv: Current implied volatility
            days_to_expiry: Override DTE

        Returns:
            MarginBreakdown with all components
        """
        # Determine margin tier
        tier = self._determine_margin_tier(position)

        # Calculate scanning risk
        scan_risk, scenario_losses, worst_scenario = self.calculate_scan_risk(
            position, underlying_price, current_iv, days_to_expiry
        )

        # Calculate delivery margin
        delivery_margin = self.calculate_delivery_margin(position, underlying_price)

        # Short option minimum
        short_min = self.calculate_short_option_minimum(position, underlying_price)

        # Net option value (premium credit for long options)
        net_value = self._calculate_position_value(
            position, underlying_price, current_iv, days_to_expiry
        )

        # No inter-spread credit for single position
        inter_spread = 0.0

        # Calculate total margin based on tier
        if tier == MarginTier.LONG_OPTION:
            # Long options: just the premium paid (absolute value)
            total = max(0, -net_value)
        elif tier == MarginTier.SPREAD:
            # Spreads: max loss (width of spread)
            total = max(scan_risk, 0)
        else:
            # Naked options and complex positions: full SPAN
            total = max(
                scan_risk + delivery_margin,
                short_min,
            )
            # Apply premium credit (but can't go negative)
            if net_value > 0:
                total = max(total - net_value * 0.5, short_min)

        return MarginBreakdown(
            scan_risk=scan_risk,
            inter_spread_credit=inter_spread,
            delivery_margin=delivery_margin,
            short_option_minimum=short_min,
            net_option_value=net_value,
            total_margin=total,
            scenario_losses=scenario_losses,
            worst_scenario=worst_scenario,
            margin_tier=tier,
        )

    def calculate_portfolio_margin(
        self,
        positions: Dict[str, OptionsPosition],
        spot_prices: Dict[str, float],
        ivs: Dict[str, float],
    ) -> Dict[str, MarginBreakdown]:
        """
        Calculate margin for entire portfolio with inter-spread credits.

        Args:
            positions: Dict of symbol -> position
            spot_prices: Dict of symbol -> current price
            ivs: Dict of symbol -> current IV

        Returns:
            Dict of symbol -> MarginBreakdown
        """
        results = {}
        total_before_credit = 0.0

        # Calculate margin per position
        for symbol, position in positions.items():
            spot = spot_prices.get(symbol, 100.0)
            iv = ivs.get(symbol, 0.30)

            breakdown = self.calculate_margin(position, spot, iv)
            results[symbol] = breakdown
            total_before_credit += breakdown.total_margin

        # Calculate inter-spread credit
        inter_credit = self.calculate_inter_spread_credit(positions)

        # Apply credit proportionally
        if total_before_credit > 0:
            credit_ratio = min(inter_credit / total_before_credit, 0.30)
            for symbol in results:
                results[symbol].inter_spread_credit = (
                    results[symbol].total_margin * credit_ratio
                )
                results[symbol].total_margin *= 1 - credit_ratio

        return results

    def _determine_margin_tier(self, position: OptionsPosition) -> MarginTier:
        """Determine the margin tier based on position structure."""
        n_long = sum(1 for leg in position.legs if leg.is_long)
        n_short = sum(1 for leg in position.legs if not leg.is_long)

        if n_short == 0:
            return MarginTier.LONG_OPTION
        elif n_long > 0 and n_short > 0:
            # Could be spread or covered position
            return MarginTier.SPREAD
        else:
            return MarginTier.NAKED_OPTION

    def get_portfolio_summary(
        self,
        margins: Dict[str, MarginBreakdown],
    ) -> Dict[str, float]:
        """
        Get summary statistics for portfolio margin.

        Args:
            margins: Dict of symbol -> MarginBreakdown

        Returns:
            Summary statistics
        """
        total_margin = sum(m.total_margin for m in margins.values())
        total_scan_risk = sum(m.scan_risk for m in margins.values())
        total_credit = sum(m.inter_spread_credit for m in margins.values())
        total_delivery = sum(m.delivery_margin for m in margins.values())
        total_net_value = sum(m.net_option_value for m in margins.values())

        return {
            "total_margin": total_margin,
            "total_scan_risk": total_scan_risk,
            "total_inter_spread_credit": total_credit,
            "total_delivery_margin": total_delivery,
            "total_net_option_value": total_net_value,
            "num_positions": len(margins),
            "avg_margin_per_position": total_margin / len(margins) if margins else 0,
        }


# Convenience function
def calculate_span_margin(
    position: OptionsPosition,
    spot: float,
    iv: float,
    price_scan_range: float = 0.15,
    vol_scan_range: float = 0.25,
) -> float:
    """
    Quick SPAN margin calculation for a single position.

    Args:
        position: Options position
        spot: Current underlying price
        iv: Current implied volatility
        price_scan_range: Price scan range (default 15%)
        vol_scan_range: Vol scan range (default 25%)

    Returns:
        Margin requirement in dollars
    """
    calc = SPANMarginCalculator(
        price_scan_range=price_scan_range,
        vol_scan_range=vol_scan_range,
    )
    breakdown = calc.calculate_margin(position, spot, iv)
    return breakdown.total_margin
