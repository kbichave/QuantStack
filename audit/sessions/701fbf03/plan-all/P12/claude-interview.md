# P12 Self-Interview: Multi-Asset Expansion

## Q1: Why futures first instead of crypto?
**A:** Futures provide the highest diversification benefit with the lowest implementation complexity. IBKR is already a planned execution path (referenced in codebase), so the broker adapter exists conceptually. Futures markets are well-regulated with predictable behavior. Crypto is second because it adds 24/7 coverage but requires a separate exchange integration (Binance) and has higher operational risk.

## Q2: How does the AssetClass abstraction prevent per-asset spaghetti code?
**A:** The AssetClass ABC defines a uniform interface: `get_data_providers()`, `get_risk_model()`, `get_signal_collectors()`, `get_execution_adapter()`, `get_trading_hours()`, `get_position_limits()`. Each asset class implements this interface. The signal engine, risk gate, and execution pipeline call the interface — they don't know which asset class they're working with. This prevents per-asset special-casing in the core loop.

## Q3: How do you handle the different margin requirements across asset classes?
**A:** Each AssetClass implementation returns its own `RiskModel` which includes margin calculation. For equities: Reg-T margin. For futures: SPAN margin (initial + maintenance). For crypto: no margin (fully funded positions). The risk gate aggregates total margin utilization across all asset classes against the portfolio-level limit.

## Q4: What cross-asset signals provide actual alpha vs just correlation tracking?
**A:** Four cross-asset signals with evidence of predictive value: (a) equity-bond correlation regime — when correlation flips positive, it signals macro stress → reduce risk; (b) commodity-equity lead/lag — oil price changes predict energy sector returns by 1-3 days; (c) FX carry trade indicator — carry unwind signals risk-off → reduce equity exposure; (d) crypto-equity correlation — when correlation spikes, crypto loses its diversification benefit → reduce crypto weight.

## Q5: How does 24/7 crypto trading work with the existing scheduler?
**A:** Extend the scheduler with mode awareness. Crypto trading runs in ALL modes (market hours, extended, overnight, weekend). Equity/futures only trade during their respective market hours. The scheduler checks `asset_class.get_trading_hours()` for each asset before triggering cycles. This requires the P15 operating modes to be crypto-aware.

## Q6: What's the maximum portfolio allocation to crypto?
**A:** 10% of total portfolio notional. Within that: max 3% per single crypto position (BTC can be 3%, ETH 3%, SOL 2%). This reflects higher vol — crypto daily vol is ~3-5x equity vol, so position sizes must be proportionally smaller to maintain risk parity across asset classes.

## Q7: How do you handle the IBKR MCP import error that's noted in the codebase?
**A:** The IBKR MCP import error is a pre-existing issue (noted in CLAUDE.md Key Facts). P12 should fix this as part of the futures adapter implementation — either lazy import the IBKR module or properly integrate the IBKR TWS API. The futures adapter depends on this being resolved.

## Q8: What schema changes are needed for multi-asset positions?
**A:** Two changes: (a) add `asset_class` column to `positions` table (enum: equity, options, futures, crypto, forex); (b) new `asset_class_config` table with per-class settings (enabled, position_limit_pct, instruments JSONB, margin_type). All existing equity positions get `asset_class='equity'` via migration default.
