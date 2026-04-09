# Section 06 -- Hedging Engine

## Description

Build a hedging engine that monitors portfolio Greeks exposure and generates neutralizing orders when thresholds are breached. The initial implementation supports delta hedging via underlying shares. The architecture uses a strategy pattern to allow additional hedging strategies (gamma hedging via options, vega hedging) in the future without modifying the engine.

## Files to Create

| File | Purpose |
|------|---------|
| `src/quantstack/core/options/hedging.py` | `HedgingStrategy` ABC, `DeltaHedgingStrategy`, `HedgingEngine` |

## Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/execution/execution_monitor.py` | Integrate hedging check into periodic monitoring loop |
| `src/quantstack/execution/trade_service.py` | Ensure hedge orders route through existing order submission |
| `src/quantstack/config/feedback_flags.py` | Add `delta_hedging_enabled()` flag |

## What to Implement

### 1. `HedgingStrategy` ABC

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from quantstack.core.options.models import PortfolioGreeks
from quantstack.execution.paper_broker import OrderRequest


@dataclass
class HedgeOrder:
    """A hedge order to be submitted."""
    symbol: str
    quantity: int           # positive = buy, negative = sell
    order_type: str = "market"
    reason: str = ""
    strategy_tag: str = "delta_hedge"


class HedgingStrategy(ABC):
    """Base class for portfolio hedging strategies."""

    @abstractmethod
    def evaluate(self, greeks: PortfolioGreeks) -> list[HedgeOrder]:
        """
        Evaluate current Greeks and return hedge orders if thresholds breached.
        Returns empty list if no hedging needed.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        ...
```

### 2. `DeltaHedgingStrategy`

```python
from loguru import logger

@dataclass
class DeltaHedgingStrategy(HedgingStrategy):
    """
    Threshold-based delta hedging using underlying shares.

    When |portfolio_delta_dollars| exceeds the threshold, generate a share order
    to neutralize the delta. Orders are sized to bring delta to zero and rounded
    to the nearest whole share.

    Example:
      Portfolio delta = +$3,200 (long delta)
      AAPL spot = $180
      Hedge: sell round(3200 / 180) = sell 18 shares of AAPL
      (This is simplified -- in practice, delta is per-symbol)
    """
    threshold_dollars: float = 500.0    # minimum delta to trigger hedge
    max_hedge_notional: float = 50000.0 # cap on single hedge order notional

    def name(self) -> str:
        return "delta_hedge"

    def evaluate(self, greeks: PortfolioGreeks) -> list[HedgeOrder]:
        orders = []

        for symbol, breakdown in greeks.per_symbol.items():
            delta_d = breakdown.delta_dollars
            if abs(delta_d) < self.threshold_dollars:
                continue

            # Need to sell shares to offset positive delta (or buy for negative)
            # delta_dollars = delta * spot * qty * 100
            # To neutralize: sell delta_dollars / spot shares
            spot = self._get_spot(symbol)
            if spot <= 0:
                logger.warning(f"DeltaHedge: no spot for {symbol}, skipping")
                continue

            shares_to_trade = -round(delta_d / spot)  # negative = sell to offset long delta
            if shares_to_trade == 0:
                continue

            # Cap notional
            notional = abs(shares_to_trade * spot)
            if notional > self.max_hedge_notional:
                capped_shares = int(self.max_hedge_notional / spot) * (1 if shares_to_trade > 0 else -1)
                logger.warning(
                    f"DeltaHedge: capping {symbol} from {shares_to_trade} to {capped_shares} shares "
                    f"(notional cap ${self.max_hedge_notional:,.0f})"
                )
                shares_to_trade = capped_shares

            orders.append(HedgeOrder(
                symbol=symbol,
                quantity=shares_to_trade,
                order_type="market",
                reason=f"Delta hedge: delta_dollars=${delta_d:,.0f}, selling {shares_to_trade} shares",
            ))

        return orders

    def _get_spot(self, symbol: str) -> float:
        """Fetch current spot price. Implementation uses cached quotes."""
        # Use the same quote source as portfolio Greeks computation
        from quantstack.data.providers import DataProviderRegistry
        try:
            import asyncio
            registry = DataProviderRegistry()
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context; use the cache or sync fallback
                from quantstack.shared.cache import get_cached_quote
                return get_cached_quote(symbol) or 0.0
            quote = loop.run_until_complete(registry.get_quote(symbol))
            return quote.get("price", 0)
        except Exception:
            return 0.0
```

### 3. `HedgingEngine`

```python
@dataclass
class HedgingEngine:
    """
    Coordinates multiple hedging strategies and manages rebalance intervals.

    Prevents over-hedging by enforcing a minimum interval between hedge executions
    per symbol. Combines orders from all strategies before submission.
    """
    strategies: list[HedgingStrategy] = field(default_factory=list)
    rebalance_interval_minutes: int = 30
    _last_hedge_time: dict[str, datetime] = field(default_factory=dict)

    def should_hedge(self, symbol: str) -> bool:
        """Check if enough time has passed since last hedge for this symbol."""
        last = self._last_hedge_time.get(symbol)
        if last is None:
            return True
        elapsed = (datetime.utcnow() - last).total_seconds() / 60
        return elapsed >= self.rebalance_interval_minutes

    def evaluate_and_execute(self, greeks: PortfolioGreeks) -> list[HedgeOrder]:
        """
        Run all strategies, filter by rebalance interval, execute orders.

        Returns list of executed orders for audit trail.
        """
        from quantstack.config.feedback_flags import delta_hedging_enabled

        if not delta_hedging_enabled():
            return []

        all_orders = []
        for strategy in self.strategies:
            orders = strategy.evaluate(greeks)
            all_orders.extend(orders)

        # Filter by rebalance interval
        executable = []
        for order in all_orders:
            if not self.should_hedge(order.symbol):
                logger.info(
                    f"Hedging: skipping {order.symbol} -- last hedge within "
                    f"{self.rebalance_interval_minutes}min"
                )
                continue
            executable.append(order)

        # Execute via trade_service
        executed = []
        for order in executable:
            success = self._submit_order(order)
            if success:
                self._last_hedge_time[order.symbol] = datetime.utcnow()
                executed.append(order)

        return executed

    def _submit_order(self, order: HedgeOrder) -> bool:
        """Submit hedge order through trade_service."""
        try:
            from quantstack.execution.trade_service import submit_order
            submit_order(
                symbol=order.symbol,
                quantity=order.quantity,
                order_type=order.order_type,
                tag=order.strategy_tag,
                reason=order.reason,
            )
            logger.info(f"Hedge order submitted: {order.symbol} qty={order.quantity}")
            return True
        except Exception as e:
            logger.error(f"Hedge order failed: {order.symbol} qty={order.quantity} err={e}")
            return False
```

### 4. Feedback flag in `feedback_flags.py`

```python
def delta_hedging_enabled() -> bool:
    """P06: Automatic delta hedging of options portfolio."""
    return _flag("FEEDBACK_DELTA_HEDGING")
```

Default: `False` (off). Must be explicitly enabled via `FEEDBACK_DELTA_HEDGING=true`.

### 5. Integration with execution monitor

In the periodic check cycle of `execution_monitor.py`:

```python
from quantstack.core.options.hedging import HedgingEngine, DeltaHedgingStrategy
from quantstack.core.options.engine import compute_portfolio_greeks

# Initialize once at monitor startup
_hedging_engine = HedgingEngine(
    strategies=[DeltaHedgingStrategy(threshold_dollars=500.0)],
    rebalance_interval_minutes=30,
)

# In periodic check:
greeks = compute_portfolio_greeks()
hedge_orders = _hedging_engine.evaluate_and_execute(greeks)
if hedge_orders:
    logger.info(f"Executed {len(hedge_orders)} hedge orders")
```

## Tests to Write

File: `tests/unit/options/test_hedging.py`

1. **test_delta_below_threshold_no_orders** -- Portfolio delta = $400, threshold = $500. Verify empty order list.
2. **test_delta_above_threshold_generates_order** -- Portfolio delta = $3,000 on AAPL, spot = $180. Verify sell order for ~17 shares.
3. **test_share_rounding** -- Delta requires 16.7 shares. Verify rounded to 17 (nearest whole share).
4. **test_negative_delta_buys_shares** -- Portfolio delta = -$2,000. Verify buy order generated.
5. **test_notional_cap_limits_order** -- Delta = $100,000, spot = $50, cap = $50,000. Verify order capped at 1,000 shares (not 2,000).
6. **test_rebalance_interval_blocks_rehedge** -- Hedge at t=0, attempt at t=15min (interval=30). Verify blocked. Attempt at t=31min: allowed.
7. **test_rebalance_interval_per_symbol** -- AAPL hedged at t=0, MSFT never hedged. At t=15min, AAPL blocked but MSFT allowed.
8. **test_hedging_disabled_flag_returns_empty** -- `FEEDBACK_DELTA_HEDGING=false`. Verify no orders even with large delta.
9. **test_multiple_symbols_independent_orders** -- AAPL delta=$1,000, MSFT delta=$2,000. Verify 2 separate orders.
10. **test_zero_spot_price_skipped** -- Symbol with spot=0. Verify skipped with warning, no division by zero.
11. **test_order_submission_failure_does_not_update_last_hedge_time** -- Mock `submit_order` to raise. Verify `_last_hedge_time` not updated.
12. **test_engine_combines_multiple_strategies** -- Two strategies both generating orders. Verify all combined.

## Edge Cases

- **Race condition on rebalance interval**: The `_last_hedge_time` dict is in-memory and per-process. If multiple monitor instances run (unlikely but possible during deploys), they could double-hedge. Mitigation: use a DB-based "last hedge" timestamp if multi-instance deployment becomes real.
- **Partial fills**: A hedge order for 17 shares might fill 10. The next cycle will see residual delta and re-hedge the remaining 7 shares. This is correct behavior -- no special handling needed.
- **Hedging increases position count**: Each hedge order creates an equity position. This adds to portfolio complexity. The supervisor should track hedge positions separately via the `strategy_tag = "delta_hedge"`.
- **Spot price staleness**: If cached quotes are 15 minutes old, the hedge quantity may be mis-sized. Acceptable for delta hedging (delta is relatively stable over 15 minutes for most positions). Would be a problem for gamma hedging.
- **Hedge orders go through risk gate**: Hedge orders submitted via `trade_service.submit_order` will pass through `risk_gate.py`. Ensure the risk gate does not reject hedges as "new positions" -- they should be tagged and recognized as hedges.
- **Market closed**: Hedge orders submitted outside market hours will queue. This is fine for the next open but may execute at a different price. Consider checking market hours before submitting.
