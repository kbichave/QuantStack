# P06 Implementation Plan: Options Desk Upgrade

## 1. Background

QuantStack has a mature options core (pricing engine with multiple backends, IV surface construction, Black-Scholes Greeks, contract selector, SPAN margin) but the agent-facing tools are mostly stubbed and key operational capabilities (auto-hedging, pin risk, portfolio Greeks) are missing. This phase wires the existing core into usable tools and adds the operational layer needed for a professional options desk.

## 2. Anti-Goals

- **Do NOT add QuantLib-Python** — the vollib + financepy + scipy stack handles everything P06 needs. QuantLib adds C++ build complexity for marginal benefit.
- **Do NOT implement gamma scalping or theta harvesting** — those are P08 (Options Market-Making). P06 builds the hedging engine interface; P08 adds strategies.
- **Do NOT implement auto-rolling** — rolling requires strategy decisions (new strikes, new expiry). Pin risk triggers alerts + auto-close only.
- **Do NOT build SABR/SVI vol surface fitting** — bilinear interpolation on the existing IVSurface class is sufficient for P06. SABR adapter exists but doesn't need to be wired in.
- **Do NOT implement live streaming Greeks** — compute Greeks on-demand per cycle, not real-time streaming.

## 3. Wire Stubbed Tools

### 3.1 price_option

Wire to existing `price_option_dispatch()` in `core/options/engine.py`:

```python
def price_option(spot, strike, time_to_expiry, volatility, option_type) -> str:
    """Call price_option_dispatch with vollib backend."""
```

The dispatch function already handles European pricing with fallback to internal BS. Add `risk_free_rate` and `dividend_yield` optional params.

### 3.2 compute_implied_vol

Wire to `implied_vol_vollib()` from `core/options/adapters/vollib_adapter.py` or `implied_volatility()` from `core/options/pricing.py`:

```python
def compute_implied_vol(market_price, spot, strike, time_to_expiry, option_type) -> str:
    """Invert BS to get IV from market price."""
```

Use Brent's method solver (already implemented in pricing.py). Handle edge cases: deep ITM/OTM where IV is undefined.

### 3.3 get_iv_surface

Wire to `IVSurface` from `core/options/iv_surface.py`:

```python
def get_iv_surface(symbol) -> str:
    """Construct IV surface from current options chain data."""
```

Steps: fetch options chain → extract IVPoints → construct IVSurface → return metrics (ATM IV, skew, term structure, grid).

### 3.4 analyze_option_structure

New logic combining IVSurface metrics with strategy recommendations:

```python
def analyze_option_structure(symbol) -> str:
    """Analyze vol surface and recommend structures."""
```

Logic: construct IV surface → compute skew, term structure → map to strategy recommendations (high skew → put spread, inverted term → calendar, high IV rank → sell premium).

### 3.5 score_trade_structure

New logic for multi-leg P&L analysis:

```python
def score_trade_structure(symbol, legs) -> str:
    """Score a multi-leg options structure on risk/reward."""
```

Compute: max profit, max loss, breakeven points, probability of profit (from IV surface), Greeks exposure, net premium. Return composite score.

### 3.6 simulate_trade_outcome

New scenario stress testing:

```python
def simulate_trade_outcome(symbol, legs, scenarios) -> str:
    """Stress test a multi-leg trade under price/vol scenarios."""
```

Default scenarios: ±5%, ±10%, ±20% price moves crossed with ±10%, ±25% IV changes. Use BS repricing with shifted inputs.

## 4. Hedging Engine

### 4.1 Module: `src/quantstack/core/options/hedging.py`

```python
class HedgingStrategy(ABC):
    """Base class for hedging strategies. P06 implements delta hedging; P08 adds gamma/theta."""
    @abstractmethod
    def compute_hedge_orders(self, portfolio_greeks: PortfolioGreeks, market_data: MarketData) -> list[HedgeOrder]: ...

class DeltaHedgingStrategy(HedgingStrategy):
    """Threshold-based delta hedging via underlying shares."""
    def __init__(self, delta_threshold: float = 500.0, rebalance_interval_minutes: int = 30): ...

class HedgingEngine:
    """Orchestrates hedging across strategies."""
    def __init__(self, strategies: list[HedgingStrategy]): ...
    def evaluate(self, portfolio_greeks: PortfolioGreeks, market_data: MarketData) -> list[HedgeOrder]: ...
    def execute(self, orders: list[HedgeOrder]) -> list[FillResult]: ...
```

### 4.2 Delta Hedging Logic

1. Compute portfolio delta across all options positions (sum of position_delta × quantity × multiplier)
2. If |portfolio_delta_dollars| > threshold (configurable, default $500): generate share buy/sell order to neutralize
3. Round to nearest 1 share (fractional not supported for hedging)
4. Execute through existing `trade_service.py`
5. Log hedge action with pre/post delta

### 4.3 Scheduling

Hedging evaluation runs at configurable interval (default: every 30 minutes during market hours). Triggered by execution_monitor's periodic check.

## 5. Pin Risk & Expiration Management

### 5.1 Pin Risk Detection

Add to `src/quantstack/execution/execution_monitor.py`:

```python
def check_pin_risk(self, positions: list[Position]) -> list[PinRiskAlert]:
    """Flag positions with DTE < 3 AND spot within 2% of strike."""
```

For each short option position:
- If DTE < 3 AND abs(spot - strike) / spot < 0.02: pin risk
- If DTE < 1 AND short option: critical assignment risk

### 5.2 Actions

1. **Alert**: Insert into `system_alerts` table with severity=HIGH
2. **Auto-close**: If `pin_risk_auto_close_enabled()` flag is True (default True) AND DTE < 1: submit close order
3. **Log**: Full context — symbol, strike, DTE, spot, distance_to_strike_pct

### 5.3 Feature Flag

`pin_risk_auto_close_enabled()` in `feedback_flags.py`. Default True. Kill switch via `FEEDBACK_PIN_RISK_AUTO_CLOSE`.

## 6. Complex Structure Support

### 6.1 Structure Types

Add `StructureType` enum to `core/options/models.py`:

```python
class StructureType(Enum):
    SINGLE_LEG = "single_leg"
    VERTICAL_SPREAD = "vertical_spread"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    CALENDAR_SPREAD = "calendar"
    DIAGONAL_SPREAD = "diagonal"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    RATIO_SPREAD = "ratio_spread"
```

### 6.2 StructureBuilder

New `src/quantstack/core/options/structures.py`:

```python
class StructureBuilder:
    """Build multi-leg option structures from strategy intent."""
    
    def build_iron_condor(self, chain, spot, target_dte, wing_width) -> OptionsPosition: ...
    def build_butterfly(self, chain, spot, target_dte, wing_width) -> OptionsPosition: ...
    def build_calendar(self, chain, strike, front_dte, back_dte) -> OptionsPosition: ...
    def build_straddle(self, chain, spot, target_dte) -> OptionsPosition: ...
    def build_strangle(self, chain, spot, target_dte, width) -> OptionsPosition: ...
```

Each builder: selects strikes from chain data, validates liquidity (bid-ask spread < threshold), creates OptionLeg list, returns OptionsPosition with `structure_type` set.

### 6.3 P&L Profile

Each OptionsPosition structure type provides:

```python
def compute_payoff_at_expiry(self, spot_range: np.ndarray) -> np.ndarray: ...
def max_profit(self) -> float: ...
def max_loss(self) -> float: ...
def breakeven_points(self) -> list[float]: ...
```

## 7. Portfolio Greeks Aggregation

### 7.1 PortfolioGreeks Model

```python
@dataclass
class PortfolioGreeks:
    total_delta: float          # $ delta exposure
    total_gamma: float          # $ gamma per 1% move
    total_theta: float          # daily theta decay $
    total_vega: float           # $ vega per 1% IV change
    total_rho: float
    per_symbol: dict[str, PositionGreeks]
    per_strategy: dict[str, PositionGreeks]
    snapshot_time: datetime
```

### 7.2 Aggregation

New function `compute_portfolio_greeks()` in `core/options/engine.py`:

1. Query all open options positions
2. For each: compute current Greeks using spot price and current IV
3. Aggregate per-position → per-symbol → per-strategy → portfolio total
4. Convert to dollar terms (delta × spot × quantity × multiplier)

### 7.3 Greeks History

New `portfolio_greeks_history` table: snapshot per cycle (symbol_greeks JSONB, strategy_greeks JSONB, portfolio_greeks JSONB, timestamp).

### 7.4 P&L Attribution

Daily job `compute_greek_pnl_attribution()`:

```
daily_pnl = Δdelta × underlying_move + 0.5 × gamma × move² + theta × 1_day + vega × Δiv + residual
```

Store in `options_pnl_attribution` table: (date, symbol, delta_pnl, gamma_pnl, theta_pnl, vega_pnl, unexplained_pnl).

## 8. Schema Migrations

1. `portfolio_greeks_history` — portfolio-level Greeks snapshots
2. `options_pnl_attribution` — daily P&L by Greek
3. Add `structure_type` column to positions table for options

## 9. Testing Strategy

### Unit Tests
- Tool wiring: each tool returns valid JSON for standard inputs
- Hedging engine: threshold triggers, order generation, delta neutralization
- Pin risk: detection at various DTE/strike distances
- StructureBuilder: iron condor/butterfly/calendar construction from mock chain
- Portfolio Greeks: aggregation math correctness
- P&L attribution: decomposition matches known scenarios

### Edge Cases
- IV surface with sparse data (few strikes/expiries) → graceful degradation
- Delta hedge with fractional shares → round correctly
- Pin risk with zero DTE → immediate close
- Complex structure with illiquid legs → reject (bid-ask spread check)
