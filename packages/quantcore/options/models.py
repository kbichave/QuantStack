"""
Options domain models.

Core dataclasses for options trading:
- OptionContract: Single option contract
- OptionLeg: Contract with quantity and direction
- OptionsPosition: Multi-leg position
- VerticalSpread: Common spread structure
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import List, Literal, Optional
import pandas as pd


class OptionType(Enum):
    """Option type."""

    CALL = "CALL"
    PUT = "PUT"


class PositionSide(Enum):
    """Position side."""

    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class OptionContract:
    """
    Single option contract.

    Represents market data and Greeks for one contract.
    """

    # Identity
    contract_id: str
    underlying: str
    expiry: date
    strike: float
    option_type: OptionType

    # Market data
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0

    # Greeks (from AlphaVantage or calculated)
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Metadata
    data_timestamp: Optional[datetime] = None

    @property
    def mid(self) -> float:
        """Mid price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid."""
        mid = self.mid
        if mid > 0:
            return self.spread / mid
        return 0.0

    @property
    def days_to_expiry(self) -> int:
        """Days until expiration."""
        today = date.today()
        return (self.expiry - today).days

    @property
    def is_call(self) -> bool:
        """Check if call option."""
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        """Check if put option."""
        return self.option_type == OptionType.PUT

    @property
    def is_itm(self) -> bool:
        """Check if in-the-money (requires underlying price)."""
        # This is a simplified check - delta > 0.5 for calls, < -0.5 for puts
        if self.is_call:
            return self.delta > 0.5
        else:
            return self.delta < -0.5

    def intrinsic_value(self, underlying_price: float) -> float:
        """Calculate intrinsic value."""
        if self.is_call:
            return max(0, underlying_price - self.strike)
        else:
            return max(0, self.strike - underlying_price)

    def time_value(self, underlying_price: float) -> float:
        """Calculate time value (extrinsic value)."""
        return max(0, self.mid - self.intrinsic_value(underlying_price))

    @classmethod
    def from_dict(cls, data: dict) -> "OptionContract":
        """Create from dictionary (e.g., from API response)."""
        return cls(
            contract_id=data.get("contract_id", ""),
            underlying=data.get("underlying", ""),
            expiry=(
                pd.to_datetime(data.get("expiry")).date()
                if data.get("expiry")
                else date.today()
            ),
            strike=float(data.get("strike", 0)),
            option_type=OptionType(data.get("option_type", "CALL").upper()),
            bid=float(data.get("bid", 0)),
            ask=float(data.get("ask", 0)),
            last=float(data.get("last", 0)),
            volume=int(data.get("volume", 0)),
            open_interest=int(data.get("open_interest", 0)),
            iv=float(data.get("iv", 0)),
            delta=float(data.get("delta", 0)),
            gamma=float(data.get("gamma", 0)),
            theta=float(data.get("theta", 0)),
            vega=float(data.get("vega", 0)),
            rho=float(data.get("rho", 0)),
        )


@dataclass
class OptionLeg:
    """
    Option leg in a position.

    Combines a contract with quantity and direction.
    """

    contract: OptionContract
    quantity: int  # Positive for long, negative for short
    entry_price: float = 0.0
    entry_timestamp: Optional[datetime] = None

    @property
    def side(self) -> PositionSide:
        """Position side."""
        return PositionSide.LONG if self.quantity > 0 else PositionSide.SHORT

    @property
    def is_long(self) -> bool:
        """Check if long position."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if short position."""
        return self.quantity < 0

    @property
    def notional_value(self) -> float:
        """Notional value (mid * quantity * 100)."""
        return abs(self.quantity) * self.contract.mid * 100

    @property
    def delta_exposure(self) -> float:
        """Delta exposure."""
        return self.quantity * self.contract.delta * 100

    @property
    def gamma_exposure(self) -> float:
        """Gamma exposure."""
        return self.quantity * self.contract.gamma * 100

    @property
    def theta_exposure(self) -> float:
        """Theta exposure (daily decay)."""
        return self.quantity * self.contract.theta * 100

    @property
    def vega_exposure(self) -> float:
        """Vega exposure."""
        return self.quantity * self.contract.vega * 100

    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.entry_price == 0:
            return 0.0
        current_price = self.contract.mid
        pnl_per_contract = (current_price - self.entry_price) * 100
        return self.quantity * pnl_per_contract

    def max_loss(self) -> float:
        """Calculate maximum possible loss."""
        if self.is_long:
            # Long option: max loss is premium paid
            return abs(self.quantity) * self.entry_price * 100
        else:
            # Short option: theoretically unlimited for calls, strike - premium for puts
            if self.contract.is_call:
                return float("inf")
            else:
                return (
                    abs(self.quantity) * (self.contract.strike - self.entry_price) * 100
                )


@dataclass
class OptionsPosition:
    """
    Multi-leg options position.

    Supports any combination of options legs.
    """

    position_id: str
    underlying: str
    legs: List[OptionLeg] = field(default_factory=list)
    entry_timestamp: Optional[datetime] = None
    notes: str = ""

    @property
    def net_delta(self) -> float:
        """Net delta exposure."""
        return sum(leg.delta_exposure for leg in self.legs)

    @property
    def net_gamma(self) -> float:
        """Net gamma exposure."""
        return sum(leg.gamma_exposure for leg in self.legs)

    @property
    def net_theta(self) -> float:
        """Net theta (daily decay)."""
        return sum(leg.theta_exposure for leg in self.legs)

    @property
    def net_vega(self) -> float:
        """Net vega exposure."""
        return sum(leg.vega_exposure for leg in self.legs)

    @property
    def total_premium(self) -> float:
        """Total premium paid (positive) or received (negative)."""
        return sum(leg.quantity * leg.entry_price * 100 for leg in self.legs)

    @property
    def num_legs(self) -> int:
        """Number of legs."""
        return len(self.legs)

    @property
    def earliest_expiry(self) -> Optional[date]:
        """Earliest expiration date."""
        if not self.legs:
            return None
        return min(leg.contract.expiry for leg in self.legs)

    @property
    def days_to_earliest_expiry(self) -> Optional[int]:
        """Days to earliest expiration."""
        expiry = self.earliest_expiry
        if expiry:
            return (expiry - date.today()).days
        return None

    def unrealized_pnl(self) -> float:
        """Total unrealized P&L."""
        return sum(leg.unrealized_pnl() for leg in self.legs)

    def max_loss(self) -> float:
        """Maximum loss for defined-risk positions."""
        # For spreads, max loss is typically the width minus premium received
        # This is a simplified calculation
        total = 0.0
        for leg in self.legs:
            loss = leg.max_loss()
            if loss == float("inf"):
                return float("inf")
            total += loss
        return total

    def is_defined_risk(self) -> bool:
        """Check if position has defined (limited) risk."""
        return self.max_loss() != float("inf")

    def add_leg(self, leg: OptionLeg) -> None:
        """Add a leg to the position."""
        self.legs.append(leg)

    def close(self, exit_prices: dict) -> float:
        """
        Close position and return realized P&L.

        Args:
            exit_prices: Dictionary of contract_id -> exit_price

        Returns:
            Realized P&L
        """
        total_pnl = 0.0
        for leg in self.legs:
            exit_price = exit_prices.get(leg.contract.contract_id, leg.contract.mid)
            pnl = (exit_price - leg.entry_price) * leg.quantity * 100
            total_pnl += pnl
        return total_pnl


@dataclass
class VerticalSpread:
    """
    Vertical spread (same expiry, different strikes).

    Types:
    - Bull Call Spread: Buy lower strike call, sell higher strike call
    - Bear Put Spread: Buy higher strike put, sell lower strike put
    - Bull Put Spread (credit): Sell higher strike put, buy lower strike put
    - Bear Call Spread (credit): Sell lower strike call, buy higher strike call
    """

    underlying: str
    expiry: date
    long_strike: float
    short_strike: float
    option_type: OptionType
    quantity: int = 1

    # Market data for each leg
    long_contract: Optional[OptionContract] = None
    short_contract: Optional[OptionContract] = None

    @property
    def width(self) -> float:
        """Spread width (absolute difference in strikes)."""
        return abs(self.long_strike - self.short_strike)

    @property
    def is_debit(self) -> bool:
        """Check if debit spread (pay premium)."""
        if self.option_type == OptionType.CALL:
            return self.long_strike < self.short_strike  # Bull call spread
        else:
            return self.long_strike > self.short_strike  # Bear put spread

    @property
    def is_credit(self) -> bool:
        """Check if credit spread (receive premium)."""
        return not self.is_debit

    @property
    def max_profit(self) -> float:
        """Maximum profit."""
        if not self.long_contract or not self.short_contract:
            return 0.0

        net_premium = self.short_contract.mid - self.long_contract.mid

        if self.is_credit:
            return net_premium * self.quantity * 100
        else:
            return (self.width - abs(net_premium)) * self.quantity * 100

    @property
    def max_loss(self) -> float:
        """Maximum loss."""
        if not self.long_contract or not self.short_contract:
            return self.width * self.quantity * 100

        net_premium = abs(self.short_contract.mid - self.long_contract.mid)

        if self.is_debit:
            return net_premium * self.quantity * 100
        else:
            return (self.width - net_premium) * self.quantity * 100

    @property
    def net_delta(self) -> float:
        """Net delta."""
        if not self.long_contract or not self.short_contract:
            return 0.0
        return (
            (self.long_contract.delta - self.short_contract.delta) * self.quantity * 100
        )

    @property
    def net_theta(self) -> float:
        """Net theta."""
        if not self.long_contract or not self.short_contract:
            return 0.0
        return (
            (self.long_contract.theta - self.short_contract.theta) * self.quantity * 100
        )

    @property
    def breakeven(self) -> float:
        """Breakeven price."""
        if not self.long_contract or not self.short_contract:
            return 0.0

        net_premium = abs(self.long_contract.mid - self.short_contract.mid)

        if self.option_type == OptionType.CALL:
            if self.is_debit:
                return self.long_strike + net_premium
            else:
                return self.short_strike + net_premium
        else:  # PUT
            if self.is_debit:
                return self.long_strike - net_premium
            else:
                return self.short_strike - net_premium

    def to_position(self, position_id: str) -> OptionsPosition:
        """Convert to OptionsPosition."""
        if not self.long_contract or not self.short_contract:
            raise ValueError("Contracts not set")

        long_leg = OptionLeg(
            contract=self.long_contract,
            quantity=self.quantity,
            entry_price=self.long_contract.mid,
        )

        short_leg = OptionLeg(
            contract=self.short_contract,
            quantity=-self.quantity,
            entry_price=self.short_contract.mid,
        )

        return OptionsPosition(
            position_id=position_id,
            underlying=self.underlying,
            legs=[long_leg, short_leg],
            entry_timestamp=datetime.now(),
        )
