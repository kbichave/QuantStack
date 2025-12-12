"""
Options Slippage Model with Bid-Ask Spread and Market Impact.

Provides realistic execution cost estimation for options trading:
- Bid-ask spread crossing cost
- Market impact (square-root model)
- Liquidity-dependent urgency premium
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np
from loguru import logger


class OrderType(Enum):
    """Order type affecting execution."""

    MARKET = "market"
    LIMIT = "limit"
    LIMIT_IOC = "limit_ioc"  # Immediate-or-cancel


class ExecutionUrgency(Enum):
    """Execution urgency level."""

    LOW = 1  # Patient, use limit orders
    MEDIUM = 2  # Normal, cross some spread
    HIGH = 3  # Urgent, cross full spread
    CRITICAL = 4  # Emergency, pay extra


@dataclass
class SlippageEstimate:
    """Detailed slippage breakdown."""

    mid_price: float
    execution_price: float
    spread_cost: float
    market_impact: float
    urgency_premium: float
    total_slippage: float
    total_slippage_bps: float


class SpreadBasedSlippage:
    """
    Realistic slippage model using actual bid-ask spreads.

    Components:
    1. Spread crossing: Pay half the spread to execute
    2. Market impact: Large orders move the market (square-root law)
    3. Urgency premium: Time-sensitive orders pay more

    The square-root market impact model:
        impact = sigma * sqrt(order_size / ADV) * sign(order)

    This is based on the Almgren-Chriss framework widely used in
    institutional trading.
    """

    # Default parameters (calibrated to typical options markets)
    DEFAULT_IMPACT_COEFFICIENT = 0.1  # Market impact multiplier
    DEFAULT_URGENCY_MULTIPLIER = {
        ExecutionUrgency.LOW: 0.0,
        ExecutionUrgency.MEDIUM: 0.5,
        ExecutionUrgency.HIGH: 1.0,
        ExecutionUrgency.CRITICAL: 2.0,
    }

    # Minimum spread assumptions when bid-ask not available
    MIN_SPREAD_PCT = 0.01  # 1% minimum spread
    MAX_SPREAD_PCT = 0.20  # 20% maximum spread

    def __init__(
        self,
        impact_coefficient: float = 0.1,
        base_volatility: float = 0.30,
    ):
        """
        Initialize slippage model.

        Args:
            impact_coefficient: Market impact multiplier (higher = more impact)
            base_volatility: Base volatility for impact calculation
        """
        self.impact_coefficient = impact_coefficient
        self.base_volatility = base_volatility

    def estimate_execution_price(
        self,
        mid: float,
        bid: Optional[float],
        ask: Optional[float],
        volume: int,
        open_interest: int,
        order_size: int,
        is_buy: bool,
        urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM,
        volatility: Optional[float] = None,
    ) -> SlippageEstimate:
        """
        Estimate execution price with full slippage breakdown.

        Args:
            mid: Mid price of the option
            bid: Bid price (or None to estimate)
            ask: Ask price (or None to estimate)
            volume: Average daily option volume
            open_interest: Open interest
            order_size: Number of contracts to trade
            is_buy: True if buying, False if selling
            urgency: Execution urgency level
            volatility: Option implied volatility (uses base if None)

        Returns:
            SlippageEstimate with price and breakdown
        """
        # Estimate bid-ask if not provided
        if bid is None or ask is None:
            bid, ask = self._estimate_spread(mid, volume, open_interest)

        # Validate and clip spread
        spread = ask - bid
        spread_pct = spread / mid if mid > 0 else self.MIN_SPREAD_PCT
        spread_pct = np.clip(spread_pct, self.MIN_SPREAD_PCT, self.MAX_SPREAD_PCT)

        # Recalculate with clipped spread
        half_spread = mid * spread_pct / 2

        # 1. Spread crossing cost
        spread_cost = half_spread

        # 2. Market impact (square-root model)
        sigma = volatility if volatility else self.base_volatility
        adv = max(volume, 100)  # Avoid division by zero

        # Impact = sigma * sqrt(size / ADV) * impact_coefficient
        size_ratio = order_size / adv
        market_impact = sigma * mid * np.sqrt(size_ratio) * self.impact_coefficient

        # Cap impact at 5% of mid price
        market_impact = min(market_impact, mid * 0.05)

        # 3. Urgency premium
        urgency_mult = self.DEFAULT_URGENCY_MULTIPLIER.get(urgency, 0.5)
        urgency_premium = spread_cost * urgency_mult

        # Total slippage
        total_slippage = spread_cost + market_impact + urgency_premium

        # Execution price
        if is_buy:
            execution_price = mid + total_slippage
        else:
            execution_price = mid - total_slippage

        # Slippage in basis points
        total_slippage_bps = (total_slippage / mid * 10000) if mid > 0 else 0

        return SlippageEstimate(
            mid_price=mid,
            execution_price=execution_price,
            spread_cost=spread_cost,
            market_impact=market_impact,
            urgency_premium=urgency_premium,
            total_slippage=total_slippage,
            total_slippage_bps=total_slippage_bps,
        )

    def _estimate_spread(
        self,
        mid: float,
        volume: int,
        open_interest: int,
    ) -> Tuple[float, float]:
        """
        Estimate bid-ask spread when not available.

        Uses a model based on price level, volume, and open interest.
        """
        # Base spread as percentage of price
        if mid < 0.50:
            base_spread_pct = 0.15  # Very cheap options: wide spreads
        elif mid < 1.00:
            base_spread_pct = 0.10
        elif mid < 5.00:
            base_spread_pct = 0.05
        elif mid < 20.00:
            base_spread_pct = 0.03
        else:
            base_spread_pct = 0.02  # Expensive options: tighter spreads

        # Adjust for liquidity
        liquidity_score = self._calculate_liquidity_score(volume, open_interest)

        # Lower spread for more liquid options
        liquidity_adjustment = 1.0 - 0.5 * liquidity_score
        adjusted_spread_pct = base_spread_pct * liquidity_adjustment

        # Clip to reasonable range
        adjusted_spread_pct = np.clip(
            adjusted_spread_pct,
            self.MIN_SPREAD_PCT,
            self.MAX_SPREAD_PCT,
        )

        half_spread = mid * adjusted_spread_pct / 2
        return mid - half_spread, mid + half_spread

    def _calculate_liquidity_score(
        self,
        volume: int,
        open_interest: int,
    ) -> float:
        """
        Calculate liquidity score from 0 (illiquid) to 1 (very liquid).

        Based on volume and open interest thresholds from options markets.
        """
        # Volume score
        if volume >= 10000:
            vol_score = 1.0
        elif volume >= 1000:
            vol_score = 0.7
        elif volume >= 100:
            vol_score = 0.4
        elif volume >= 10:
            vol_score = 0.2
        else:
            vol_score = 0.0

        # Open interest score
        if open_interest >= 50000:
            oi_score = 1.0
        elif open_interest >= 10000:
            oi_score = 0.7
        elif open_interest >= 1000:
            oi_score = 0.4
        elif open_interest >= 100:
            oi_score = 0.2
        else:
            oi_score = 0.0

        # Combined score (weight volume slightly higher)
        return 0.6 * vol_score + 0.4 * oi_score

    def quick_slippage_bps(
        self,
        mid: float,
        volume: int,
        order_size: int,
        is_buy: bool,
    ) -> float:
        """
        Quick slippage estimate in basis points.

        Useful for backtesting where you just need a number.
        """
        estimate = self.estimate_execution_price(
            mid=mid,
            bid=None,
            ask=None,
            volume=volume,
            open_interest=volume * 5,  # Rough estimate
            order_size=order_size,
            is_buy=is_buy,
            urgency=ExecutionUrgency.MEDIUM,
        )
        return estimate.total_slippage_bps


class ParametricSlippage:
    """
    Simple parametric slippage model for backtesting.

    Uses fixed percentages based on price tier and order size.
    Faster than the full model but less accurate.
    """

    # Slippage tiers by option price
    SLIPPAGE_TIERS = {
        0.50: 0.10,  # 10% for options < $0.50
        1.00: 0.06,  # 6% for $0.50-$1.00
        5.00: 0.03,  # 3% for $1-$5
        20.00: 0.015,  # 1.5% for $5-$20
        float("inf"): 0.01,  # 1% for > $20
    }

    # Size multipliers
    SIZE_MULTIPLIERS = {
        10: 1.0,  # 1-10 contracts: normal
        50: 1.2,  # 11-50: 20% extra
        100: 1.5,  # 51-100: 50% extra
        500: 2.0,  # 101-500: 100% extra
        float("inf"): 3.0,  # > 500: 200% extra
    }

    def estimate_slippage_pct(
        self,
        price: float,
        order_size: int,
    ) -> float:
        """
        Estimate slippage as percentage of price.

        Args:
            price: Option mid price
            order_size: Number of contracts

        Returns:
            Slippage as decimal (e.g., 0.02 for 2%)
        """
        # Get base slippage for price tier
        base_slippage = 0.01
        for threshold, slippage in self.SLIPPAGE_TIERS.items():
            if price <= threshold:
                base_slippage = slippage
                break

        # Get size multiplier
        size_mult = 1.0
        for threshold, mult in self.SIZE_MULTIPLIERS.items():
            if order_size <= threshold:
                size_mult = mult
                break

        return base_slippage * size_mult

    def apply_slippage(
        self,
        mid_price: float,
        order_size: int,
        is_buy: bool,
    ) -> float:
        """
        Apply slippage to get execution price.

        Args:
            mid_price: Mid price
            order_size: Number of contracts
            is_buy: True if buying

        Returns:
            Execution price after slippage
        """
        slippage_pct = self.estimate_slippage_pct(mid_price, order_size)

        if is_buy:
            return mid_price * (1 + slippage_pct)
        else:
            return mid_price * (1 - slippage_pct)


# Default instances for convenience
default_slippage = SpreadBasedSlippage()
parametric_slippage = ParametricSlippage()
