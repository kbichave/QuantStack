# Section 02 -- Wire Pricing Tools (price_option, compute_implied_vol)

## Description

Two tools in `src/quantstack/tools/langchain/options_tools.py` currently return `{"error": "Tool pending implementation"}`. Wire them to the existing pricing engine so agents can price options and extract implied volatility during research and trade evaluation.

## Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/tools/langchain/options_tools.py` | Replace stub bodies of `price_option` and `compute_implied_vol` with real implementations |

## What to Implement

### 1. `price_option` tool

The tool signature already exists (lines ~72-94 of `options_tools.py`). Replace the stub body with:

```python
@tool
async def price_option(
    spot: Annotated[float, Field(description="...")],
    strike: Annotated[float, Field(description="...")],
    time_to_expiry: Annotated[float, Field(description="...")],
    volatility: Annotated[float, Field(description="...")],
    option_type: Annotated[str, Field(description="...")] = "call",
    risk_free_rate: Annotated[float, Field(description="Annualized risk-free rate, e.g. 0.05 for 5%")] = 0.05,
    dividend_yield: Annotated[float, Field(description="Continuous dividend yield, e.g. 0.02 for 2%")] = 0.0,
) -> str:
    """...(keep existing docstring)..."""
    try:
        from quantstack.core.options.engine import price_option_dispatch

        result = price_option_dispatch(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            vol=volatility,
            rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )
        return json.dumps(result, default=str)
    except Exception as e:
        logger.error(f"price_option failed: spot={spot}, strike={strike}, T={time_to_expiry}, vol={volatility}, err={e}")
        return json.dumps({"error": str(e)})
```

Key changes vs. the stub:
- Add `risk_free_rate` and `dividend_yield` parameters (both with sensible defaults).
- Call `price_option_dispatch()` from `core/options/engine.py`, which already handles backend routing (vollib / financepy / internal).
- Log input parameters on failure for debugging.

### 2. `compute_implied_vol` tool

The tool signature already exists (lines ~80-94). Replace the stub body with:

```python
@tool
async def compute_implied_vol(
    market_price: Annotated[float, Field(description="...")],
    spot: Annotated[float, Field(description="...")],
    strike: Annotated[float, Field(description="...")],
    time_to_expiry: Annotated[float, Field(description="...")],
    option_type: Annotated[str, Field(description="...")] = "call",
    risk_free_rate: Annotated[float, Field(description="Annualized risk-free rate")] = 0.05,
    dividend_yield: Annotated[float, Field(description="Continuous dividend yield")] = 0.0,
) -> str:
    """...(keep existing docstring)..."""
    try:
        from quantstack.core.options.adapters.vollib_adapter import implied_vol_vollib

        iv = implied_vol_vollib(
            market_price=market_price,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            rate=risk_free_rate,
            option_type=option_type,
        )
        return json.dumps({"implied_volatility": iv, "method": "vollib"})
    except Exception:
        # Fallback: internal Newton solver
        try:
            from quantstack.core.options.pricing import implied_volatility

            iv = implied_volatility(
                market_price=market_price,
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
            )
            return json.dumps({"implied_volatility": iv, "method": "internal_newton"})
        except Exception as e:
            logger.error(
                f"compute_implied_vol failed: mkt={market_price}, S={spot}, K={strike}, "
                f"T={time_to_expiry}, type={option_type}, err={e}"
            )
            return json.dumps({
                "error": str(e),
                "implied_volatility": None,
                "note": "IV undefined -- contract may be deep ITM/OTM or near-zero time value",
            })
```

Key design decisions:
- **Primary path**: `implied_vol_vollib()` -- fast, uses `py_vollib` analytical solver.
- **Fallback path**: `implied_volatility()` from `core/options/pricing.py` -- internal Newton/Brent solver. Activated when vollib raises (e.g., negative time value, extreme moneyness).
- **Graceful degradation**: When both solvers fail, return `null` IV with an explanatory note rather than crashing. Deep ITM options where extrinsic value is near zero, and deep OTM options where the market price is below the minimum BS price, legitimately have undefined IV.

### Parameter additions to existing signatures

Add `risk_free_rate` (default 0.05) and `dividend_yield` (default 0.0) parameters to both tool signatures. These were missing from the original stubs. The `compute_implied_vol` stub already has `spot`, `strike`, `time_to_expiry`, `market_price`, and `option_type` -- just add the two new params at the end with defaults.

## Tests to Write

File: `tests/unit/options/test_options_tools.py` (pricing subset)

1. **test_price_option_call_returns_valid_json** -- ATM call (spot=100, strike=100, T=0.25, vol=0.3). Verify JSON has a numeric `price` field > 0.
2. **test_price_option_put_returns_valid_json** -- Same but `option_type="put"`. Verify put price > 0 and put-call parity approximately holds.
3. **test_price_option_zero_time** -- `time_to_expiry=0`. Should return intrinsic value (no time value). Verify ITM call returns `max(0, S-K)`.
4. **test_price_option_deep_otm** -- strike=200, spot=100 call. Price should be near zero but non-negative.
5. **test_compute_iv_roundtrip** -- Price an option with known vol=0.30, feed that price into `compute_implied_vol`, verify recovered IV is within 0.001 of 0.30.
6. **test_compute_iv_deep_itm_fallback** -- Deep ITM option where vollib may fail. Verify fallback path returns IV or graceful null.
7. **test_compute_iv_zero_market_price** -- market_price=0. Should return error/null IV, not raise.
8. **test_compute_iv_negative_time_value** -- market_price < intrinsic value (possible with stale quotes). Should handle gracefully.

## Edge Cases

- **time_to_expiry = 0 or negative**: `price_option_dispatch` should clamp to intrinsic value. Verify the engine handles this; if not, add a guard in the tool.
- **volatility = 0**: BS formula degenerates. The dispatch should return intrinsic value. Add a guard: `if vol <= 0: return intrinsic`.
- **Negative market_price for IV**: Reject immediately with a clear error rather than passing to the solver.
- **Very high IV (> 5.0 / 500%)**: vollib may raise. The fallback solver should handle or cap at a reasonable maximum (e.g., 10.0).
- **Dividend yield > risk-free rate**: Valid for high-yield stocks. BS-Merton handles this correctly but early exercise becomes optimal for American calls -- log a warning if exercise_style is not considered.
