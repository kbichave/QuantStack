# P06 TDD Plan: Options Desk Upgrade

Testing framework: pytest. Test location: `tests/unit/options/`.

## Section 3: Wire Stubbed Tools

```python
# Test: price_option returns valid JSON with price for standard call
# Test: price_option returns valid JSON with price for standard put
# Test: price_option handles zero time_to_expiry (at expiry) gracefully
# Test: compute_implied_vol returns IV for ATM option
# Test: compute_implied_vol returns error for deep OTM where IV undefined
# Test: get_iv_surface returns surface metrics for symbol with chain data
# Test: get_iv_surface returns error when no chain data available
# Test: analyze_option_structure returns strategy recommendations
# Test: score_trade_structure computes max_profit/max_loss for vertical spread
# Test: score_trade_structure computes breakeven for iron condor
# Test: simulate_trade_outcome returns P&L for price shock scenarios
# Test: simulate_trade_outcome returns P&L for vol shock scenarios
```

## Section 4: Hedging Engine

```python
# Test: DeltaHedgingStrategy generates buy order when portfolio delta < -threshold
# Test: DeltaHedgingStrategy generates sell order when portfolio delta > +threshold
# Test: DeltaHedgingStrategy generates no order when delta within threshold
# Test: DeltaHedgingStrategy rounds shares to nearest integer
# Test: HedgingEngine evaluates multiple strategies and combines orders
# Test: HedgingEngine respects rebalance_interval (no hedge if too recent)
```

## Section 5: Pin Risk & Expiration

```python
# Test: check_pin_risk flags position with DTE=2, spot within 1% of strike
# Test: check_pin_risk does not flag position with DTE=10
# Test: check_pin_risk does not flag position with spot 5% from strike
# Test: auto-close triggered for DTE<1 short option when flag enabled
# Test: auto-close NOT triggered when flag disabled
# Test: pin risk alert written to system_alerts table
```

## Section 6: Complex Structures

```python
# Test: build_iron_condor creates 4-leg position with correct strikes
# Test: build_butterfly creates 3-leg position with correct ratios
# Test: build_calendar creates 2-leg position with different expiries
# Test: build_straddle creates ATM call+put at same strike
# Test: build_strangle creates OTM call+put at different strikes
# Test: StructureBuilder rejects legs with wide bid-ask spread
# Test: compute_payoff_at_expiry correct for vertical spread
# Test: max_profit and max_loss correct for iron condor
# Test: breakeven_points correct for straddle
```

## Section 7: Portfolio Greeks

```python
# Test: compute_portfolio_greeks aggregates delta across positions
# Test: compute_portfolio_greeks converts to dollar terms correctly
# Test: per_symbol aggregation groups by symbol
# Test: per_strategy aggregation groups by strategy_id
# Test: Greek P&L attribution decomposes known scenario correctly
# Test: Greeks history snapshot written to DB
```
