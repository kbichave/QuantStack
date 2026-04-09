# P06: Options Desk Upgrade

**Objective:** Evolve from basic directional options to a professional-grade options desk with vol surface modeling, Greeks monitoring, hedging automation, and complex structure support.

**Scope:** core/risk/options_risk.py, execution/, tools/langchain/options_tools.py

**Depends on:** None (P02 Greeks integration helps but not blocking)

**Enables:** P08 (Options Market-Making)

**Effort estimate:** 1-2 weeks

---

## What Changes

### 6.1 Vol Surface Modeling
- Build IV surface from options chain data (strike × DTE matrix)
- Track skew (25-delta risk reversal), term structure (1M vs 3M ATM)
- Detect vol regime: low/normal/high/extreme from VIX + realized vol
- Use vol surface for more accurate options pricing

**Packages:** `py_vollib` (existing), `QuantLib-Python` (add for surface fitting)

### 6.2 Greeks Monitoring Dashboard
- Real-time portfolio Greeks aggregation (delta, gamma, theta, vega, rho)
- Greeks time-series tracking — how do Greeks evolve as DTE approaches?
- Greeks P&L attribution — how much P&L from delta vs gamma vs theta vs vega?

### 6.3 Dynamic Hedging Automation
- Auto-hedge portfolio delta when |delta_exposure| > threshold
- Gamma scalping: buy options for long gamma, hedge delta, profit from realized > implied
- Theta harvesting: short options when IV rank > 50th percentile, hedge delta

### 6.4 Complex Structure Support
Currently supports: long calls/puts, vertical spreads
Add: iron condors, butterflies, calendars, diagonals, straddles/strangles, ratio spreads

### 6.5 Pin Risk & Expiration Management
- Flag positions with DTE < 3 AND within 2% of strike
- Auto-close or roll positions approaching expiration
- Assignment risk calculation for short options

### 6.6 Implement Stubbed Options Tools
Implement the 6 stubbed options tools:
- `compute_greeks` — portfolio Greeks aggregation
- `price_option` — Black-Scholes + SABR
- `compute_implied_vol` — IV from market prices
- `analyze_option_structure` — structure selection engine
- `get_iv_surface` — vol surface construction
- `score_trade_structure` — risk/reward scoring

## Files to Create/Modify

| File | Change |
|------|--------|
| New: `src/quantstack/core/options/vol_surface.py` | IV surface fitting |
| New: `src/quantstack/core/options/hedging.py` | Auto-hedge delta, gamma scalping |
| New: `src/quantstack/core/options/structures.py` | Complex structure definitions |
| `src/quantstack/tools/langchain/options_tools.py` | Implement 6 stubbed tools |
| `src/quantstack/execution/execution_monitor.py` | Pin risk, expiration management |

## Acceptance Criteria

1. Vol surface constructed from options chain data
2. Portfolio Greeks computed and displayed in real-time
3. Auto-delta-hedge triggers when exposure exceeds threshold
4. Complex structures (iron condor, butterfly, calendar) supported
5. Pin risk flagged for DTE < 3 near strike
6. All 6 options tools functional (not stubbed)
