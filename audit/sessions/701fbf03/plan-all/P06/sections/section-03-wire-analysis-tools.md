# Section 03 -- Wire Analysis Tools (get_iv_surface, analyze_option_structure, score_trade_structure, simulate_trade_outcome)

## Description

Wire the four remaining stubbed analysis tools in `src/quantstack/tools/langchain/options_tools.py`. These tools give agents the ability to analyze volatility surfaces, recommend strategies based on skew/term structure, score multi-leg trade proposals, and stress-test trades under price/vol scenarios.

## Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/tools/langchain/options_tools.py` | Replace stub bodies of 4 tools |

## What to Implement

### 1. `get_iv_surface`

Existing signature (line ~107): takes `symbol: str`, returns JSON string.

Implementation:

```python
async def get_iv_surface(symbol: ...) -> str:
    try:
        from quantstack.data.providers import DataProviderRegistry
        from quantstack.core.options.iv_surface import IVPoint, IVSurface

        registry = DataProviderRegistry()
        chain = await registry.get_options_chain(symbol=symbol, expiry_min_days=7, expiry_max_days=90)

        # Get current spot price
        quote = await registry.get_quote(symbol)
        spot = quote.get("price") or quote.get("last", 0)
        if spot <= 0:
            return json.dumps({"error": f"Could not retrieve spot price for {symbol}"})

        # Extract IVPoints from chain
        iv_points = []
        for expiry_group in chain.get("calls", []) + chain.get("puts", []):
            for contract in expiry_group if isinstance(expiry_group, list) else [expiry_group]:
                iv = contract.get("iv") or contract.get("implied_volatility", 0)
                if iv > 0:
                    iv_points.append(IVPoint(
                        strike=float(contract["strike"]),
                        expiry_days=int(contract.get("dte", 30)),
                        iv=float(iv),
                        delta=contract.get("delta"),
                        option_type=contract.get("option_type", "call"),
                    ))

        if len(iv_points) < 5:
            return json.dumps({"error": "Insufficient IV data points", "count": len(iv_points)})

        surface = IVSurface(spot_price=spot)
        surface.fit(iv_points)
        metrics = surface.compute_metrics()

        return json.dumps({
            "symbol": symbol,
            "spot": spot,
            "metrics": {
                "atm_iv_30d": metrics.atm_iv_30d,
                "atm_iv_60d": metrics.atm_iv_60d,
                "atm_iv_90d": metrics.atm_iv_90d,
                "skew_25d_30d": metrics.skew_25d_30d,
                "skew_25d_60d": metrics.skew_25d_60d,
                "term_structure_slope": metrics.term_structure_slope,
                "vol_of_vol": metrics.vol_of_vol,
            },
            "point_count": len(iv_points),
        }, default=str)
    except Exception as e:
        logger.error(f"get_iv_surface({symbol}) failed: {e}")
        return json.dumps({"error": str(e), "symbol": symbol})
```

Key points:
- Fetches chain via `DataProviderRegistry` (same path as `fetch_options_chain`).
- Constructs `IVPoint` list, builds `IVSurface`, calls `compute_metrics()`.
- Returns `IVSurfaceMetrics` fields as flat JSON for agent consumption.
- Minimum 5 IV data points required; below that, surface interpolation is unreliable.

### 2. `analyze_option_structure`

Existing signature (line ~98): takes `symbol: str`, returns JSON string.

Implementation builds on `get_iv_surface` output and maps surface characteristics to strategy recommendations:

```python
async def analyze_option_structure(symbol: ...) -> str:
    try:
        # Reuse get_iv_surface logic (extract into shared helper to avoid duplication)
        surface_json = await get_iv_surface.ainvoke({"symbol": symbol})
        surface_data = json.loads(surface_json)

        if "error" in surface_data:
            return surface_json  # Propagate upstream error

        metrics = surface_data["metrics"]
        recommendations = []

        skew_30d = metrics["skew_25d_30d"]
        term_slope = metrics["term_structure_slope"]
        atm_iv = metrics["atm_iv_30d"]

        # Strategy mapping rules
        if skew_30d > 0.05:
            recommendations.append({
                "strategy": "put_spread",
                "rationale": f"Elevated put skew ({skew_30d:.3f}) -- sell rich puts via vertical spread",
                "confidence": "medium",
            })
        if skew_30d < -0.02:
            recommendations.append({
                "strategy": "call_spread",
                "rationale": f"Call skew premium ({skew_30d:.3f}) -- sell rich calls via vertical spread",
                "confidence": "medium",
            })
        if term_slope < -0.02:
            recommendations.append({
                "strategy": "calendar_spread",
                "rationale": f"Inverted term structure ({term_slope:.3f}) -- buy far, sell near",
                "confidence": "medium",
            })
        if atm_iv > 0.40:
            recommendations.append({
                "strategy": "sell_premium",
                "rationale": f"High ATM IV ({atm_iv:.1%}) -- sell straddle/strangle or iron condor",
                "confidence": "high" if atm_iv > 0.60 else "medium",
            })
        if atm_iv < 0.15:
            recommendations.append({
                "strategy": "buy_premium",
                "rationale": f"Low ATM IV ({atm_iv:.1%}) -- buy straddle/strangle ahead of catalyst",
                "confidence": "low",
            })

        if not recommendations:
            recommendations.append({
                "strategy": "neutral",
                "rationale": "No strong skew or term structure signal -- no clear edge",
                "confidence": "low",
            })

        return json.dumps({
            "symbol": symbol,
            "surface_metrics": metrics,
            "recommendations": recommendations,
        }, default=str)
    except Exception as e:
        logger.error(f"analyze_option_structure({symbol}) failed: {e}")
        return json.dumps({"error": str(e), "symbol": symbol})
```

### 3. `score_trade_structure`

Existing signature (line ~116): takes `symbol: str` and `legs: list[dict]`.

Implementation:

```python
async def score_trade_structure(symbol: ..., legs: ...) -> str:
    try:
        from quantstack.core.options.engine import price_option_dispatch, compute_greeks_dispatch

        # Build position from legs
        total_premium = 0.0
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        payoff_at_strikes = []

        for leg in legs:
            strike = float(leg["strike"])
            tte = float(leg.get("time_to_expiry", leg.get("tte", 30 / 365)))
            opt_type = leg.get("option_type", "call")
            action = leg.get("action", "buy")
            qty = int(leg.get("quantity", 1))
            vol = float(leg.get("volatility", leg.get("iv", 0.30)))
            direction = 1 if action == "buy" else -1

            price_result = price_option_dispatch(
                spot=float(leg.get("spot", 0)),  # caller must provide or we fetch
                strike=strike, time_to_expiry=tte, vol=vol, option_type=opt_type,
            )
            greeks_result = compute_greeks_dispatch(
                spot=float(leg.get("spot", 0)), strike=strike,
                time_to_expiry=tte, volatility=vol, option_type=opt_type,
            )

            leg_price = price_result.get("price", 0) * direction * qty * 100
            total_premium += leg_price
            total_delta += greeks_result.get("delta", 0) * direction * qty * 100
            total_gamma += greeks_result.get("gamma", 0) * direction * qty * 100
            total_theta += greeks_result.get("theta", 0) * direction * qty * 100
            total_vega += greeks_result.get("vega", 0) * direction * qty * 100

        # Compute payoff at expiry across a range of spot prices
        # (simplified: use strikes +/- 20% in 1% steps)
        spot_ref = float(legs[0].get("spot", legs[0]["strike"]))
        spot_range = [spot_ref * (1 + pct / 100) for pct in range(-20, 21)]
        max_profit = float("-inf")
        max_loss = float("inf")
        breakevens = []

        prev_pnl = None
        for s in spot_range:
            pnl = 0.0
            for leg in legs:
                strike = float(leg["strike"])
                opt_type = leg.get("option_type", "call")
                action = leg.get("action", "buy")
                qty = int(leg.get("quantity", 1))
                vol = float(leg.get("volatility", leg.get("iv", 0.30)))
                direction = 1 if action == "buy" else -1
                entry_px = price_option_dispatch(
                    spot=float(leg.get("spot", spot_ref)), strike=strike,
                    time_to_expiry=float(leg.get("time_to_expiry", 30/365)),
                    vol=vol, option_type=opt_type,
                ).get("price", 0)
                # Intrinsic at expiry
                if opt_type == "call":
                    expiry_value = max(0, s - strike)
                else:
                    expiry_value = max(0, strike - s)
                leg_pnl = (expiry_value - entry_px) * direction * qty * 100
                pnl += leg_pnl

            max_profit = max(max_profit, pnl)
            max_loss = min(max_loss, pnl)
            if prev_pnl is not None and prev_pnl * pnl < 0:
                breakevens.append(round(s, 2))
            prev_pnl = pnl

        # Composite score: reward/risk ratio weighted by prob of profit (simplified)
        risk = abs(max_loss) if max_loss < 0 else 1.0
        reward_risk = max_profit / risk if risk > 0 else 0
        pop_estimate = sum(1 for s in spot_range if ... ) / len(spot_range)  # placeholder

        return json.dumps({
            "symbol": symbol,
            "net_premium": round(total_premium, 2),
            "greeks": {"delta": round(total_delta, 4), "gamma": round(total_gamma, 4),
                       "theta": round(total_theta, 4), "vega": round(total_vega, 4)},
            "max_profit": round(max_profit, 2),
            "max_loss": round(max_loss, 2),
            "breakeven_points": breakevens,
            "reward_risk_ratio": round(reward_risk, 2),
            "composite_score": ...,  # see implementation note below
        }, default=str)
    except Exception as e:
        logger.error(f"score_trade_structure({symbol}) failed: {e}")
        return json.dumps({"error": str(e), "symbol": symbol})
```

**Implementation note on composite score**: Compute as `0.4 * normalized_rr + 0.3 * pop_estimate + 0.2 * (1 - abs(delta)/100) + 0.1 * (theta > 0)`. Normalize reward/risk to [0,1] by capping at 5.0. This gives a 0-1 score favoring good risk/reward, high probability, low directional bias, and positive theta.

### 4. `simulate_trade_outcome`

Existing signature (line ~127): takes `symbol`, `legs`, optional `scenarios`.

Implementation:

```python
async def simulate_trade_outcome(symbol: ..., legs: ..., scenarios: ... = None) -> str:
    try:
        from quantstack.core.options.engine import price_option_dispatch

        # Default scenarios: price moves x IV moves
        if scenarios is None:
            price_moves = [-20, -10, -5, 0, 5, 10, 20]
            iv_changes = [-0.25, -0.10, 0, 0.10, 0.25]
            scenarios = [
                {"price_move_pct": pm, "iv_change": ivc}
                for pm in price_moves for ivc in iv_changes
            ]

        spot_ref = float(legs[0].get("spot", legs[0]["strike"]))
        results = []

        for scenario in scenarios:
            pm_pct = scenario.get("price_move_pct", 0)
            iv_chg = scenario.get("iv_change", 0)
            dte_shift = scenario.get("days_forward", 0)
            new_spot = spot_ref * (1 + pm_pct / 100)

            scenario_pnl = 0.0
            for leg in legs:
                strike = float(leg["strike"])
                tte = float(leg.get("time_to_expiry", 30/365))
                vol = float(leg.get("volatility", leg.get("iv", 0.30)))
                opt_type = leg.get("option_type", "call")
                action = leg.get("action", "buy")
                qty = int(leg.get("quantity", 1))
                direction = 1 if action == "buy" else -1

                # Entry price (at current spot/vol)
                entry = price_option_dispatch(
                    spot=spot_ref, strike=strike, time_to_expiry=tte,
                    vol=vol, option_type=opt_type,
                ).get("price", 0)

                # Scenario price (shifted spot, shifted vol, reduced time)
                new_tte = max(tte - dte_shift / 365, 1 / 365)  # floor at 1 day
                new_vol = max(vol + iv_chg, 0.01)  # floor at 1%
                scenario_px = price_option_dispatch(
                    spot=new_spot, strike=strike, time_to_expiry=new_tte,
                    vol=new_vol, option_type=opt_type,
                ).get("price", 0)

                leg_pnl = (scenario_px - entry) * direction * qty * 100
                scenario_pnl += leg_pnl

            results.append({
                "price_move_pct": pm_pct,
                "iv_change": iv_chg,
                "days_forward": dte_shift,
                "new_spot": round(new_spot, 2),
                "pnl": round(scenario_pnl, 2),
            })

        return json.dumps({
            "symbol": symbol,
            "base_spot": spot_ref,
            "scenarios": results,
            "best_case": max(results, key=lambda r: r["pnl"]),
            "worst_case": min(results, key=lambda r: r["pnl"]),
        }, default=str)
    except Exception as e:
        logger.error(f"simulate_trade_outcome({symbol}) failed: {e}")
        return json.dumps({"error": str(e), "symbol": symbol})
```

Default scenario grid: 7 price moves x 5 IV changes = 35 scenarios. Each reprices all legs via `price_option_dispatch`, computing full BS price (not just intrinsic). This captures gamma and vega effects that intrinsic-only payoff diagrams miss.

## Tests to Write

File: `tests/unit/options/test_options_tools.py` (analysis subset)

1. **test_get_iv_surface_returns_metrics** -- Mock `DataProviderRegistry` to return a synthetic chain with known IVs. Verify all `IVSurfaceMetrics` fields present in response.
2. **test_get_iv_surface_insufficient_data** -- Return chain with < 5 IV points. Verify error message.
3. **test_analyze_option_structure_high_skew** -- Mock surface with `skew_25d_30d=0.08`. Verify `put_spread` recommendation present.
4. **test_analyze_option_structure_inverted_term** -- Mock surface with `term_structure_slope=-0.05`. Verify `calendar_spread` recommendation.
5. **test_analyze_option_structure_high_iv** -- Mock ATM IV = 0.65. Verify `sell_premium` recommendation with `confidence=high`.
6. **test_score_trade_structure_vertical_spread** -- Bull call spread (buy 100C, sell 105C). Verify max_loss = debit paid, max_profit = width - debit.
7. **test_score_trade_structure_iron_condor** -- 4-leg IC. Verify defined risk, max_loss = wing width - net credit.
8. **test_simulate_default_scenarios_count** -- No custom scenarios. Verify 35 results (7x5 grid).
9. **test_simulate_custom_scenario** -- Single scenario `{price_move_pct: 10, iv_change: -0.05}`. Verify single result with correct new_spot.
10. **test_simulate_zero_tte_floor** -- Scenario with `days_forward` exceeding remaining DTE. Verify TTE floored at 1/365, not zero or negative.

## Edge Cases

- **Empty options chain**: `get_iv_surface` returns `< 5 points` error. `analyze_option_structure` propagates the error. Agent must handle gracefully.
- **Zero spot price from quote**: Guard against division by zero in log-moneyness calculation. Return error if `spot <= 0`.
- **Stale IV data**: Chain data from AlphaVantage is EOD. Intraday IV surface will be stale. Consider adding a `data_staleness_warning` field if chain timestamp > 1 hour old.
- **score_trade_structure with missing `spot` in legs**: Must either fetch spot from quote or require it. Current design expects `spot` in each leg dict -- add validation and a clear error if missing.
- **simulate_trade_outcome with extreme moves**: `price_move_pct=100` (doubling) is valid. `price_move_pct=-100` (zero spot) would break BS. Clamp `new_spot` to `max(new_spot, 0.01)`.
- **IV change driving vol negative**: `iv_change=-0.50` when current vol is 0.20 would give -0.30. Floor at 0.01 (1%) to prevent BS NaN.
