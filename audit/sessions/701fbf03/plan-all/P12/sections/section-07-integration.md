# Section 07: Integration

## Objective

Wire all P12 components together: register asset classes at startup, connect the trading graph to multi-asset execution paths, update the scheduler for 24/7 crypto and 23h futures, and ensure end-to-end flow works from signal collection through risk gate to order execution.

**Depends on:** section-04-cross-asset-signals, section-05-risk-gate-multi-asset, section-06-schema-migrations

## Files to Modify

### `src/quantstack/asset_classes/__init__.py`
Add a `bootstrap_asset_classes()` function that:
1. Reads `asset_class_config` from DB
2. For each enabled class, instantiates the corresponding `AssetClass` implementation
3. Registers it in the asset class registry
4. Logs which asset classes are active

Called once at system startup (from `start.sh` / Docker entrypoint).

### `scripts/scheduler.py`
Update the scheduler to support multi-asset trading schedules:
- Equity: existing schedule (market hours Mon-Fri)
- Futures: 18:00-17:00 ET Sun-Fri (23h/day)
- Crypto: 24/7 continuous
- Each asset class gets its own scheduling loop based on `TradingSchedule`
- Use `asyncio.gather()` to run multiple schedule loops concurrently

### `src/quantstack/graphs/trading/nodes.py`
Update entry scanning and position monitoring nodes:
- Entry scanner should iterate over all enabled asset classes, not just equity
- For each asset class, use its specific signal collectors and data providers
- Pass `asset_class` parameter to `RiskGate.check()` calls
- Route approved orders to the correct execution adapter

### `src/quantstack/execution/trade_service.py`
Update the trade service to:
- Accept `asset_class` parameter in trade execution functions
- Look up the correct `BrokerAdapter` from the asset class registry
- Include `asset_class`, `contract_multiplier`, and `margin_required` when persisting positions
- Log asset class in all trade audit entries

### `src/quantstack/execution/broker_factory.py`
Extend broker factory to return the appropriate broker based on asset class:
- `EQUITY` / `OPTIONS` -> Alpaca (existing)
- `FUTURES` -> IBKR adapter
- `CRYPTO` -> Binance adapter
- Fall back to paper broker if asset class adapter not configured

### `src/quantstack/graphs/supervisor/` (relevant nodes)
Update supervisor health monitoring to:
- Check health per asset class
- Report per-asset-class P&L in health dashboard
- Detect and report when an asset class is halted due to daily loss limit

### `src/quantstack/config/feedback_flags.py`
Ensure `MULTI_ASSET_ENABLED` flag gates all new code paths. When False, system behaves identically to pre-P12.

## Files to Create

### `src/quantstack/asset_classes/bootstrap.py`
Startup bootstrap logic (separate from `__init__.py` for testability):

```python
def bootstrap_asset_classes(db_url: str | None = None) -> dict[AssetClassType, AssetClass]:
    """Read config from DB, instantiate enabled asset classes, register them."""
```

Mapping from config `class_name` string to implementation class:
- `"equity"` -> existing (no AssetClass wrapper needed yet)
- `"futures"` -> `FuturesAssetClass`
- `"crypto"` -> `CryptoAssetClass`
- Others -> skip with warning (not yet implemented)

## Implementation Details

1. **Feature flag is the master switch.** If `MULTI_ASSET_ENABLED=False`, `bootstrap_asset_classes()` returns empty dict and no new code paths activate. This is the safe default.
2. The scheduler must not spawn crypto/futures loops when those asset classes are disabled. Check the registry, not just the flag.
3. In the trading graph, the entry scanner should process asset classes sequentially (not in parallel) to avoid overwhelming the signal engine. Each asset class scan should complete before the next begins.
4. Broker factory should use a strategy pattern: try asset-class-specific adapter first, then fall back to paper broker. Never fail silently — log clearly when falling back.
5. The supervisor should aggregate per-asset-class health into a single dashboard view. Add a `multi_asset_status` field to the health check output.
6. All integration points must handle the case where an asset class was enabled at startup but its external dependency (IBKR, Binance) becomes unavailable. Circuit-break at the adapter level, not at the graph level.

## Test Requirements

### `tests/integration/test_multi_asset_bootstrap.py`
- With `MULTI_ASSET_ENABLED=False` -> no asset classes registered
- With futures enabled in DB -> `FuturesAssetClass` registered
- With crypto enabled in DB -> `CryptoAssetClass` registered
- Missing DB config -> graceful degradation (log error, continue with equity only)

### `tests/integration/test_multi_asset_e2e.py`
- Mock IBKR + Binance adapters
- Signal collection -> risk gate -> order submission for futures
- Signal collection -> risk gate -> order submission for crypto
- Risk gate rejection -> no order submitted (verify no leakage)
- Asset class halted mid-session -> no new orders for that class, existing positions monitored

### `tests/integration/test_scheduler_multi_asset.py`
- Scheduler spawns correct number of loops for enabled asset classes
- Crypto loop runs outside equity market hours
- Futures loop respects 23h schedule
- Disabled asset class -> no loop spawned

## Acceptance Criteria

- [ ] `bootstrap_asset_classes()` correctly reads DB config and registers implementations
- [ ] `MULTI_ASSET_ENABLED=False` results in zero behavioral change from pre-P12
- [ ] Scheduler handles multi-asset trading schedules concurrently
- [ ] Trading graph entry scanner processes all enabled asset classes
- [ ] Trade service routes to correct broker adapter per asset class
- [ ] Broker factory falls back to paper broker with clear logging
- [ ] Supervisor reports per-asset-class health
- [ ] Circuit breaker: external dependency failure halts that asset class only
- [ ] All integration tests pass
