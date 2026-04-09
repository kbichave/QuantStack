# P02: Execution Reality

**Objective:** Transform phantom execution into real algorithmic trading. Implement TWAP/VWAP child orders, Greeks-aware risk gate, intraday circuit breakers, TCA feedback loop, and liquidity modeling.

**Scope:** execution/, core/risk/, paper_broker.py, order_lifecycle.py

**Depends on:** None

**Enables:** P08 (Options Market-Making), P12 (Multi-Asset)

**Effort estimate:** 1-2 weeks

---

## What Changes

### 2.1 Real Execution Algorithms (QS-E1)

**Problem:** TWAP/VWAP selected by `order_lifecycle.py:455-478` but paper broker executes everything as a single fill.

**Implementation:**
```python
# New: src/quantstack/execution/algos/twap.py
class TWAPAlgorithm:
    """Time-Weighted Average Price: split order into N child orders over duration."""
    def generate_child_orders(self, parent: Order, duration_minutes: int) -> list[ChildOrder]:
        n_slices = max(1, duration_minutes // 2)  # 2-min intervals
        child_qty = parent.quantity // n_slices
        # Jitter ±10% to avoid predictable pattern
        
# New: src/quantstack/execution/algos/vwap.py  
class VWAPAlgorithm:
    """Volume-Weighted: distribute based on historical volume profile."""
    def generate_child_orders(self, parent: Order, profile: VolumeProfile) -> list[ChildOrder]:
        # Weight by 30-min volume buckets from historical data
```

**Paper broker changes:**
- Simulate fills against historical bar data with realistic participation constraints
- Add market impact: `impact_bps = k * sqrt(order_pct_adv)` where k calibrated per market cap tier
- Time-of-day spread adjustment: wider at open/close

**Files:**
- New: `src/quantstack/execution/algos/twap.py`
- New: `src/quantstack/execution/algos/vwap.py`
- `src/quantstack/execution/paper_broker.py` — simulate child order fills with realistic impact
- `src/quantstack/execution/order_lifecycle.py` — dispatch to algo engine

### 2.2 Options Greeks in Risk Gate (QS-E3)

**Problem:** Risk gate checks DTE + premium only. No delta/gamma/vega/theta limits.

**Implementation:** Wire existing `core/risk/options_risk.py` (444 lines, already implemented) into `risk_gate.py`.

**Limits to enforce:**
| Greek | Limit | Why |
|-------|-------|-----|
| Portfolio Delta | ±$5K per 1% move | Directional exposure cap |
| Portfolio Gamma | $2K per 1% squared | Convexity exposure cap |
| Portfolio Vega | $1K per 1 vol point | Vol exposure cap |
| Daily Theta | -$200/day max | Time decay budget |
| Pin Risk | Block if DTE < 3 AND within 2% of strike | Assignment risk |

**Files:**
- `src/quantstack/execution/risk_gate.py` — add Greeks checks in options path
- `src/quantstack/core/risk/options_risk.py` — already exists, ensure portfolio aggregation works

### 2.3 Intraday Circuit Breaker (QS-E5)

**Problem:** Only daily loss limit (-2%) after realized losses. No unrealized P&L monitoring.

**Implementation:**
```python
# New thresholds in execution_monitor:
UNREALIZED_THRESHOLDS = {
    -0.015: "HALT_NEW_ENTRIES",     # -1.5% unrealized → stop entering
    -0.025: "SYSTEMATIC_EXIT",       # -2.5% → begin systematic exit
    -0.050: "EMERGENCY_LIQUIDATION", # -5% → emergency liquidate all
}

# Velocity check:
if (pnl_change_5min < -0.01):  # -1% in 5 minutes
    trigger("VELOCITY_HALT")
```

**Files:**
- `src/quantstack/execution/execution_monitor.py` — add unrealized P&L tracking + velocity
- `src/quantstack/execution/risk_gate.py` — add circuit breaker state check

### 2.4 TCA Feedback Loop (QS-E6)

**Problem:** Pre-trade cost estimates don't calibrate from realized costs.

**Implementation:**
- After every fill: EWMA update of Almgren-Chriss parameters
- `forecast = 0.9 * old_forecast + 0.1 * realized_cost`
- Per-symbol, per-time-of-day bucket
- Until 50 trades accumulated: use 2x conservative multiplier

**Files:**
- `src/quantstack/execution/tca_recalibration.py` — add EWMA online update
- `src/quantstack/hooks/trade_hooks.py` — trigger TCA recalibration on fill

### 2.5 Liquidity Model (QS-E4)

**Problem:** Only `daily_volume < min_daily_volume` check.

**Implementation:**
```python
# New: src/quantstack/execution/liquidity_model.py
class LiquidityModel:
    def estimate_depth(self, symbol: str, time_of_day: datetime) -> DepthEstimate:
        """Estimate available depth from historical volume profile."""
        # Stressed liquidity = depth * 0.3 (crisis haircut)
    
    def pre_trade_check(self, order: Order) -> LiquidityVerdict:
        """If order_size > depth * 0.1: scale down or reject."""
```

**Files:**
- New: `src/quantstack/execution/liquidity_model.py`
- `src/quantstack/execution/risk_gate.py` — add liquidity check pre-trade

### 2.6 Best Execution Audit Trail (QS-E7)

**What:** Add `execution_audit` table with NBBO reference, fill venue, algo rationale.

**Files:**
- `src/quantstack/db.py` — add `execution_audit` table
- `src/quantstack/execution/order_lifecycle.py` — populate on every fill

## Tests

| Test | What It Verifies |
|------|-----------------|
| `test_twap_generates_children` | TWAP splits 1000-share order into 10 child orders |
| `test_vwap_weights_by_volume` | Child orders weighted by historical volume profile |
| `test_greeks_blocks_large_gamma` | Short straddle blocked when portfolio gamma exceeded |
| `test_circuit_breaker_halts` | -2.5% unrealized → systematic exit triggered |
| `test_tca_ewma_updates` | Cost estimate changes after realized fill |
| `test_liquidity_rejects_illiquid` | Large order in illiquid stock rejected |

## Acceptance Criteria

1. Paper broker fills TWAP orders over time (not instant single fill)
2. Portfolio Greeks computed and enforced pre-trade
3. Intraday circuit breaker triggers on -1.5% unrealized
4. TCA cost estimates change after each fill (EWMA)
5. Execution audit trail populated with NBBO + venue for every fill

## Risk

| Risk | Severity | Mitigation |
|------|----------|-----------|
| TWAP child orders overwhelm order management | Medium | Rate limit to 1 child/2min |
| Greeks computation slow for large portfolios | Low | Cache Greeks, recompute every 5 min |
| Circuit breaker triggers too often in volatile markets | Medium | Calibrate thresholds from historical vol |

## References

- CTO Audit: QS-E1 through QS-E7
- See `../gaps/tier1-2-critical-gaps.md`
