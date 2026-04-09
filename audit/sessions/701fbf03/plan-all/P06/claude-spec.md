# P06 Spec: Options Desk Upgrade

## Context
QuantStack has substantial options infrastructure: pricing engine with vollib/financepy backends, IV surface construction, Black-Scholes Greeks, contract selector, and SPAN margin. However, 6 of 8 options tools are stubbed, there's no auto-hedging, no pin risk management, no complex structure definitions, and no portfolio-level Greeks aggregation.

## What Already Exists
- `core/options/engine.py` — price_option_dispatch, compute_greeks_dispatch (multi-backend)
- `core/options/iv_surface.py` — IVSurface with interpolation, ATM IV, skew, term structure
- `core/options/pricing.py` — Black-Scholes price, Greeks, IV solver
- `core/options/models.py` — OptionContract, OptionLeg, OptionsPosition
- `core/options/contract_selector.py` — Rule-based contract selection
- `core/risk/span_margin.py` — SPAN margin calculation
- `alpha_discovery/vol_arb.py` — Vol arb strategy

## What This Phase Must Deliver

### 1. Wire 6 Stubbed Tools
Connect existing core functions to the 6 stubbed tools in options_tools.py. Mostly wiring, minimal new logic.

### 2. Hedging Engine
New `core/options/hedging.py` with threshold-based delta hedging. Interface supports pluggable strategies for P08 gamma scalping / theta harvesting.

### 3. Pin Risk & Expiration Management
Detection in execution_monitor.py. Alert + auto-close for short options with DTE < 3 within 2% of strike.

### 4. Complex Structure Support
StructureBuilder that constructs iron condors, butterflies, calendars, diagonals, straddles/strangles from strategy intent. Extends OptionsPosition model.

### 5. Portfolio Greeks Aggregation
Per-position, per-strategy, portfolio-wide Greeks rollup. Greeks history table for time-series. Daily P&L attribution by Greek.

## Constraints
- No QuantLib dependency — existing vollib + financepy + scipy stack is sufficient
- Hedging engine interface must be pluggable for P08
- Pin risk auto-close is flag-gated (kill switch)
- All new features have unit tests
