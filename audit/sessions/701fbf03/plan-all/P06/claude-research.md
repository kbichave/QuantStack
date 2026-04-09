# P06 Codebase Research: Options Desk Upgrade

## Current Implementation State

### Options Tools (options_tools.py)
- `fetch_options_chain` — **IMPLEMENTED** (calls DataProviderRegistry)
- `compute_greeks` — **IMPLEMENTED** (calls compute_greeks_dispatch from core/options/engine.py)
- `price_option` — **STUBBED** (returns error)
- `compute_implied_vol` — **STUBBED**
- `analyze_option_structure` — **STUBBED**
- `get_iv_surface` — **STUBBED**
- `score_trade_structure` — **STUBBED**
- `simulate_trade_outcome` — **STUBBED** (additional tool beyond P06 spec)

### Core Options Module (core/options/)
Well-developed with:
- `engine.py` — Pricing dispatch with vollib, financepy, internal backends
- `iv_surface.py` — Full `IVSurface` class with bilinear interpolation on log-moneyness × sqrt(T), ATM IV, skew, term structure extraction
- `pricing.py` — Black-Scholes price, Greeks, IV solver (Brent's method)
- `models.py` — OptionType, OptionContract, OptionLeg, OptionsPosition dataclasses
- `contract_selector.py` — Rule-based contract selection by regime + vol
- `slippage.py` — Options slippage model
- `adapters/` — vollib_adapter.py, financepy_adapter.py, pysabr_adapter.py, quantsbin_adapter.py

### Risk Module (core/risk/)
- `options_risk.py` — Exists (need to check contents)
- `span_margin.py` — SPAN margin calculation exists
- `controls.py` — Risk controls
- `stress_testing.py` — Stress testing

### Strategy Module
- `core/strategy/options_rules.py` — Options strategy rules
- `core/strategy/options_ml.py` — ML-based options strategy
- `alpha_discovery/vol_arb.py` — Vol arb module exists
- `alpha_discovery/earnings_catalyst_options.py` — Earnings options

### What's NOT Done
1. **Tool wiring** — 6 stubs need to call existing core functions (price_option_dispatch, IVSurface, etc.)
2. **Auto-hedging** — No hedging module exists (grep for "auto_hedge", "delta_hedge" = 0 results)
3. **Pin risk** — No pin risk detection
4. **Expiration management** — No auto-close/roll near expiration
5. **Complex structures** — No iron condor/butterfly/calendar definitions as structured types
6. **Greeks portfolio aggregation** — No portfolio-level Greeks rollup
7. **Greeks P&L attribution** — No decomposition of P&L by Greek contribution

## Testing
- No dedicated options tests found
- Core module has `core/tests/` but nothing options-specific
