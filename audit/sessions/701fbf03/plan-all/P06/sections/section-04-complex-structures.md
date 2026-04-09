# Section 04 -- Complex Option Structures

## Description

Create a `StructureBuilder` system that can construct, validate, and compute payoff profiles for multi-leg option structures. This replaces ad-hoc leg assembly with a typed, validated builder pattern. The `StructureType` enum classifies positions for downstream use in risk gating, P&L attribution, and portfolio Greeks aggregation.

## Files to Create

| File | Purpose |
|------|---------|
| `src/quantstack/core/options/structures.py` | `StructureType` enum, `StructureBuilder` class, payoff computation |

## Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/core/options/models.py` | Add `structure_type: StructureType` field to `OptionsPosition` |
| `src/quantstack/core/options/__init__.py` | Export new symbols |

## What to Implement

### 1. `StructureType` enum

```python
from enum import Enum

class StructureType(str, Enum):
    SINGLE_LEG = "single_leg"
    VERTICAL_SPREAD = "vertical_spread"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    CALENDAR = "calendar"
    DIAGONAL = "diagonal"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    RATIO_SPREAD = "ratio_spread"
```

Using `str, Enum` so the value serializes directly to the DB `structure_type TEXT` column and JSON without conversion.

### 2. Add `structure_type` to `OptionsPosition`

In `models.py`, add to the `OptionsPosition` dataclass:

```python
from quantstack.core.options.structures import StructureType

@dataclass
class OptionsPosition:
    position_id: str
    underlying: str
    legs: list[OptionLeg] = field(default_factory=list)
    structure_type: StructureType = StructureType.SINGLE_LEG  # NEW
    entry_timestamp: datetime | None = None
    notes: str = ""
```

### 3. `StructureBuilder` class

```python
from dataclasses import dataclass
from quantstack.core.options.models import (
    OptionContract, OptionLeg, OptionType, OptionsPosition,
)

@dataclass
class LiquidityFilter:
    """Minimum liquidity requirements for contract selection."""
    max_bid_ask_spread_pct: float = 0.10   # 10% of mid
    min_open_interest: int = 10
    min_volume: int = 0

class StructureBuilder:
    """
    Builds multi-leg option structures from an options chain.

    Each build method:
    1. Selects strikes from the chain based on delta/moneyness targets
    2. Validates liquidity (bid-ask spread, open interest)
    3. Constructs OptionLeg list with correct quantities and directions
    4. Returns an OptionsPosition with the correct StructureType
    """

    def __init__(self, chain: list[OptionContract], liquidity: LiquidityFilter | None = None):
        self.chain = chain
        self.liquidity = liquidity or LiquidityFilter()
        self._liquid_contracts = self._filter_liquid(chain)

    def _filter_liquid(self, contracts: list[OptionContract]) -> list[OptionContract]:
        """Filter contracts meeting liquidity thresholds."""
        result = []
        for c in contracts:
            if c.open_interest < self.liquidity.min_open_interest:
                continue
            if c.volume < self.liquidity.min_volume:
                continue
            if c.mid > 0 and c.spread_pct > self.liquidity.max_bid_ask_spread_pct:
                continue
            result.append(c)
        return result

    def _find_nearest_strike(self, contracts: list[OptionContract],
                              target_delta: float | None = None,
                              target_strike: float | None = None) -> OptionContract | None:
        """Find contract nearest to target delta or strike."""
        if target_strike is not None:
            return min(contracts, key=lambda c: abs(c.strike - target_strike), default=None)
        if target_delta is not None:
            return min(contracts, key=lambda c: abs(abs(c.delta) - abs(target_delta)), default=None)
        return None

    def build_iron_condor(self, expiry_days: int, put_short_delta: float = -0.20,
                          call_short_delta: float = 0.20, wing_width: float = 5.0,
                          quantity: int = 1) -> OptionsPosition:
        """
        Iron condor: sell OTM put + OTM call, buy further OTM put + call for protection.

        Legs:
        1. Buy put  at (put_short_strike - wing_width)   [long, protection]
        2. Sell put at put_short_delta                     [short, premium]
        3. Sell call at call_short_delta                   [short, premium]
        4. Buy call at (call_short_strike + wing_width)   [long, protection]
        """
        # ... (strike selection, liquidity validation, leg construction)
        # Returns OptionsPosition with structure_type=StructureType.IRON_CONDOR

    def build_butterfly(self, center_strike: float, wing_width: float,
                        option_type: str = "call", quantity: int = 1) -> OptionsPosition:
        """
        Butterfly: buy 1 lower, sell 2 center, buy 1 upper.
        Max profit at center strike at expiry. Defined risk.
        """

    def build_calendar(self, strike: float, near_expiry_days: int,
                       far_expiry_days: int, option_type: str = "call",
                       quantity: int = 1) -> OptionsPosition:
        """
        Calendar spread: sell near-term, buy far-term at same strike.
        Profits from near-term theta decay and/or IV expansion in far leg.
        """

    def build_straddle(self, atm_strike: float, quantity: int = 1,
                       direction: str = "long") -> OptionsPosition:
        """
        Straddle: buy/sell ATM call + ATM put at same strike and expiry.
        Long: profits from large moves. Short: profits from time decay.
        """

    def build_strangle(self, put_strike: float, call_strike: float,
                       quantity: int = 1, direction: str = "long") -> OptionsPosition:
        """
        Strangle: buy/sell OTM call + OTM put at different strikes, same expiry.
        Wider breakevens than straddle, lower cost.
        """
```

### 4. Payoff computation methods on `OptionsPosition`

Add these methods to `OptionsPosition` in `models.py`:

```python
import numpy as np

def compute_payoff_at_expiry(self, spot_range: list[float]) -> list[tuple[float, float]]:
    """
    Compute position P&L at expiry for each spot price in range.

    Returns list of (spot, pnl) tuples. PnL includes entry premium.
    """
    results = []
    for spot in spot_range:
        pnl = 0.0
        for leg in self.legs:
            if leg.contract.is_call:
                intrinsic = max(0, spot - leg.contract.strike)
            else:
                intrinsic = max(0, leg.contract.strike - spot)
            leg_pnl = (intrinsic - leg.entry_price) * leg.quantity * 100
            pnl += leg_pnl
        results.append((spot, round(pnl, 2)))
    return results

def max_profit(self) -> float:
    """Max profit across a wide spot range at expiry."""
    if not self.legs:
        return 0.0
    strikes = [leg.contract.strike for leg in self.legs]
    center = sum(strikes) / len(strikes)
    spot_range = [center * (1 + i/100) for i in range(-50, 51)]
    payoffs = self.compute_payoff_at_expiry(spot_range)
    return max(pnl for _, pnl in payoffs)

def max_loss_computed(self) -> float:
    """Max loss across a wide spot range at expiry (negative value)."""
    if not self.legs:
        return 0.0
    strikes = [leg.contract.strike for leg in self.legs]
    center = sum(strikes) / len(strikes)
    spot_range = [center * (1 + i/100) for i in range(-50, 51)]
    payoffs = self.compute_payoff_at_expiry(spot_range)
    return min(pnl for _, pnl in payoffs)

def breakeven_points(self) -> list[float]:
    """Spot prices where P&L crosses zero at expiry."""
    if not self.legs:
        return []
    strikes = [leg.contract.strike for leg in self.legs]
    center = sum(strikes) / len(strikes)
    spot_range = [center * (1 + i/200) for i in range(-100, 101)]  # 0.5% steps
    payoffs = self.compute_payoff_at_expiry(spot_range)
    crossings = []
    for i in range(1, len(payoffs)):
        prev_spot, prev_pnl = payoffs[i-1]
        curr_spot, curr_pnl = payoffs[i]
        if prev_pnl * curr_pnl < 0:  # sign change
            # Linear interpolation
            frac = abs(prev_pnl) / (abs(prev_pnl) + abs(curr_pnl))
            be = prev_spot + frac * (curr_spot - prev_spot)
            crossings.append(round(be, 2))
    return crossings
```

**Note**: `max_loss_computed()` is named differently from the existing `max_loss()` method to avoid breaking existing callers. The existing `max_loss()` method sums per-leg max losses and is a simpler approximation. The new method computes the actual payoff-based max loss. Future cleanup can deprecate the old method.

## Tests to Write

File: `tests/unit/options/test_structures.py`

1. **test_structure_type_enum_values** -- All 9 types have correct string values matching DB column expectations.
2. **test_iron_condor_has_four_legs** -- Builder produces 4 legs with correct long/short directions.
3. **test_iron_condor_max_loss_equals_wing_width_minus_credit** -- Known strikes + premiums. Verify max loss = wing_width * 100 - net_credit.
4. **test_iron_condor_max_profit_equals_net_credit** -- At center of range, profit = total premium received.
5. **test_butterfly_has_three_strikes** -- 3 distinct strikes, center sold 2x, wings bought 1x each.
6. **test_butterfly_max_profit_at_center** -- Payoff at center strike is maximum.
7. **test_calendar_different_expiries** -- Near and far legs have different expiry dates.
8. **test_straddle_symmetric_breakevens** -- Long straddle breakevens equidistant from strike.
9. **test_strangle_wider_breakevens_than_straddle** -- At same cost, strangle breakevens are wider.
10. **test_liquidity_filter_rejects_wide_spread** -- Contract with 15% spread rejected when filter is 10%.
11. **test_liquidity_filter_rejects_low_oi** -- Contract with OI=5 rejected when min is 10.
12. **test_payoff_at_expiry_call_spread** -- Bull call spread: known strikes, known premiums. Verify payoff at 5 spot prices matches manual calculation.
13. **test_breakeven_points_straddle** -- Long straddle with known premium. Verify 2 breakeven points.
14. **test_no_contracts_available** -- Empty chain. Builder should raise `ValueError` or return empty position with warning.

## Edge Cases

- **Insufficient liquid contracts**: Builder selects from liquid-filtered chain. If no contract meets liquidity thresholds for a required leg, raise `ValueError` with a message specifying which leg could not be filled. Do not silently pick illiquid contracts.
- **Non-standard strike spacing**: Some symbols have $1 strikes, others $2.50 or $5. Wing width must snap to the nearest available strike, not assume uniform spacing.
- **Calendar spread with same expiry**: If near and far expiry resolve to the same date, this is not a calendar -- it is a vertical or straddle. Validate `near_expiry != far_expiry`.
- **Ratio spread asymmetry**: Ratio spreads have unequal quantities (e.g., buy 1 sell 2). The quantity imbalance creates undefined risk on one side. `is_defined_risk()` must return `False` for ratio spreads.
- **Payoff computation for American options**: The payoff-at-expiry computation is only exact for European options. For American options, early exercise value is not captured. Document this limitation; it is acceptable for screening purposes.
- **OptionsPosition backward compatibility**: Adding `structure_type` with a default value means existing code that constructs `OptionsPosition` without this field continues to work.
