# Section 09 -- Unit Tests

## Description

Comprehensive test suite for all P06 Options Desk components. Tests are organized by subsystem in `tests/unit/options/`. Each test file is self-contained with its own fixtures and mocks. Tests use `pytest` with `pytest-asyncio` for async tool tests. All external dependencies (DB, broker, data providers) are mocked.

## Files to Create

| File | Purpose |
|------|---------|
| `tests/unit/options/__init__.py` | Package marker |
| `tests/unit/options/test_options_tools.py` | All 6 wired tools (price_option, compute_implied_vol, get_iv_surface, analyze_option_structure, score_trade_structure, simulate_trade_outcome) |
| `tests/unit/options/test_hedging.py` | DeltaHedgingStrategy, HedgingEngine, rebalance intervals |
| `tests/unit/options/test_pin_risk.py` | Pin risk detection, alerting, auto-close |
| `tests/unit/options/test_structures.py` | StructureType, StructureBuilder, payoff computation |
| `tests/unit/options/test_portfolio_greeks.py` | PortfolioGreeks aggregation, dollar conversion, DB roundtrip |
| `tests/unit/options/test_pnl_attribution.py` | Greek P&L decomposition, upsert, date handling |

## Test File: `test_options_tools.py`

Tests for all 6 wired tools. Each tool returns a JSON string; tests parse and validate structure.

```python
"""Tests for options_tools.py -- all 6 wired LangGraph tools."""

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Fixtures
@pytest.fixture
def mock_price_dispatch():
    """Mock price_option_dispatch to return deterministic price."""
    with patch("quantstack.core.options.engine.price_option_dispatch") as mock:
        mock.return_value = {"price": 5.25, "backend": "vollib", "model": "black_scholes"}
        yield mock

@pytest.fixture
def mock_iv_vollib():
    """Mock implied_vol_vollib to return known IV."""
    with patch("quantstack.core.options.adapters.vollib_adapter.implied_vol_vollib") as mock:
        mock.return_value = 0.30
        yield mock

@pytest.fixture
def mock_chain():
    """Mock DataProviderRegistry.get_options_chain with synthetic chain."""
    # ... returns calls/puts with known strikes, IVs, Greeks
```

### Tests

1. **test_price_option_call_returns_valid_json** -- Invoke `price_option` with ATM params. Parse JSON. Assert `price` > 0, `backend` present.
2. **test_price_option_put_returns_valid_json** -- Same for puts. Assert price > 0.
3. **test_price_option_zero_time_returns_intrinsic** -- T=0, ITM call. Assert price == max(0, S-K).
4. **test_price_option_deep_otm_near_zero** -- K >> S. Assert 0 <= price < 0.10.
5. **test_price_option_includes_risk_free_rate** -- Verify `rate` param passed to dispatch.
6. **test_price_option_includes_dividend_yield** -- Verify `dividend_yield` param passed to dispatch.
7. **test_compute_iv_roundtrip** -- Price with vol=0.30, feed price to compute_implied_vol. Assert |recovered_iv - 0.30| < 0.001.
8. **test_compute_iv_vollib_fallback_to_internal** -- Mock vollib to raise, internal solver returns 0.28. Assert method == "internal_newton".
9. **test_compute_iv_both_fail_returns_null** -- Both solvers fail. Assert `implied_volatility` is None, `note` field present.
10. **test_compute_iv_zero_market_price** -- market_price=0. Assert error or null IV, no exception.
11. **test_get_iv_surface_returns_metrics** -- Mock chain. Assert all IVSurfaceMetrics fields present.
12. **test_get_iv_surface_insufficient_data** -- Chain with 3 points. Assert error message about insufficient data.
13. **test_analyze_option_structure_high_skew** -- Mock surface with skew=0.08. Assert `put_spread` in recommendations.
14. **test_analyze_option_structure_inverted_term** -- term_slope=-0.05. Assert `calendar_spread` recommended.
15. **test_analyze_option_structure_high_iv** -- ATM IV=0.65. Assert `sell_premium` with confidence=high.
16. **test_analyze_option_structure_no_signal** -- Flat surface. Assert `neutral` recommendation.
17. **test_score_trade_structure_vertical_spread** -- Bull call spread. Assert max_loss = debit, max_profit = width - debit.
18. **test_score_trade_structure_iron_condor** -- 4-leg IC. Assert defined risk, reward_risk_ratio > 0.
19. **test_score_trade_structure_returns_greeks** -- Assert delta, gamma, theta, vega in response.
20. **test_simulate_default_scenarios_count** -- No custom scenarios. Assert 35 results (7 price x 5 IV).
21. **test_simulate_custom_scenario** -- Single scenario. Assert 1 result with correct new_spot.
22. **test_simulate_tte_floor** -- days_forward > remaining DTE. Assert TTE >= 1/365.
23. **test_simulate_iv_floor** -- iv_change makes vol negative. Assert vol floored at 0.01.

## Test File: `test_hedging.py`

```python
"""Tests for hedging engine -- DeltaHedgingStrategy and HedgingEngine."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from quantstack.core.options.hedging import (
    DeltaHedgingStrategy, HedgingEngine, HedgeOrder,
)
from quantstack.core.options.models import PortfolioGreeks, GreeksBreakdown
```

### Tests

1. **test_delta_below_threshold_no_orders** -- delta=$400, threshold=$500. Assert empty list.
2. **test_delta_above_threshold_generates_sell** -- delta=$3000, spot=$180. Assert sell ~17 shares.
3. **test_negative_delta_generates_buy** -- delta=-$2000, spot=$100. Assert buy 20 shares.
4. **test_share_rounding_up** -- delta=$1670, spot=$100. Assert 17 shares (round(16.7)).
5. **test_share_rounding_down** -- delta=$1640, spot=$100. Assert 16 shares (round(16.4)).
6. **test_notional_cap** -- delta=$100000, spot=$50, cap=$50000. Assert 1000 shares max.
7. **test_rebalance_interval_blocks** -- Last hedge 15min ago, interval=30min. Assert blocked.
8. **test_rebalance_interval_allows_after_expiry** -- Last hedge 31min ago. Assert allowed.
9. **test_rebalance_per_symbol** -- AAPL hedged recently, MSFT never. MSFT allowed, AAPL blocked.
10. **test_flag_disabled_no_execution** -- `FEEDBACK_DELTA_HEDGING=false`. Assert empty.
11. **test_zero_spot_skipped** -- spot=0 for a symbol. Assert no order, no crash.
12. **test_submission_failure_no_time_update** -- Mock submit to raise. Assert _last_hedge_time unchanged.
13. **test_multiple_strategies_combined** -- Two strategies return orders. Assert all appear in result.
14. **test_zero_delta_no_order** -- delta exactly $0. Assert no order generated.

## Test File: `test_pin_risk.py`

```python
"""Tests for pin risk detection and auto-close."""

import pytest
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
```

### Tests

1. **test_no_short_options** -- Only long calls. Assert empty flagged list.
2. **test_dte_5_not_flagged** -- Short put, DTE=5, near strike. Not flagged.
3. **test_dte_2_near_strike_flagged** -- Short call, DTE=2, 1% from strike. Flagged.
4. **test_dte_2_far_from_strike_not_flagged** -- Short put, DTE=2, 5% away. Not flagged.
5. **test_exact_at_strike_flagged** -- spot == strike. Distance=0%. Flagged.
6. **test_boundary_distance_not_flagged** -- Distance exactly 2.0%. `>=` threshold, NOT flagged.
7. **test_alert_inserted_in_db** -- Mock db_conn. Verify INSERT with severity=HIGH.
8. **test_auto_close_dte_0_enabled** -- DTE=0, flag=true. Verify close order submitted.
9. **test_auto_close_dte_0_disabled** -- DTE=0, flag=false. Alert only, no close order.
10. **test_auto_close_dte_2_not_triggered** -- DTE=2, flag=true. Alert only (DTE >= 1).
11. **test_close_failure_second_alert** -- Close raises. Verify second alert with error.
12. **test_multiple_positions_partial_flag** -- 3 positions, 1 has pin risk. Assert 1 flagged.
13. **test_position_status_set_to_closing** -- After auto-close, verify UPDATE to status='closing'.

## Test File: `test_structures.py`

```python
"""Tests for StructureType, StructureBuilder, payoff computation."""

import pytest
from datetime import date

from quantstack.core.options.structures import StructureType, StructureBuilder, LiquidityFilter
from quantstack.core.options.models import OptionContract, OptionType, OptionsPosition
```

### Tests

1. **test_enum_string_values** -- All 9 StructureType values match expected strings.
2. **test_enum_serializes_to_string** -- `StructureType.IRON_CONDOR.value == "iron_condor"`.
3. **test_iron_condor_four_legs** -- Build IC. Assert 4 legs, 2 long, 2 short.
4. **test_iron_condor_strikes_ordered** -- Long put < short put < short call < long call.
5. **test_iron_condor_max_loss** -- Known premiums. max_loss = wing_width*100 - net_credit.
6. **test_iron_condor_max_profit** -- max_profit = net_credit.
7. **test_butterfly_three_unique_strikes** -- 3 strikes, center sold 2x.
8. **test_butterfly_max_profit_at_center** -- Payoff peaks at center strike.
9. **test_butterfly_max_loss_at_wings** -- Payoff at wing strikes equals -(net debit).
10. **test_calendar_different_expiries** -- Near and far legs have different dates.
11. **test_calendar_same_strike** -- Both legs at same strike.
12. **test_straddle_two_legs_same_strike** -- 1 call + 1 put at same strike.
13. **test_straddle_breakevens_symmetric** -- Breakevens equidistant from ATM strike.
14. **test_strangle_wider_breakevens** -- Breakevens wider than equivalent straddle.
15. **test_liquidity_rejects_wide_spread** -- 15% spread, 10% threshold. Rejected.
16. **test_liquidity_rejects_low_oi** -- OI=5, min=10. Rejected.
17. **test_payoff_bull_call_spread** -- Buy 100C, sell 105C. Verify payoff at S=90,100,102.5,105,110.
18. **test_breakeven_points_count** -- Straddle: 2 breakevens. Butterfly: 2 breakevens.
19. **test_empty_chain_raises** -- No contracts. Assert ValueError.
20. **test_options_position_default_structure_type** -- New OptionsPosition has structure_type=SINGLE_LEG.

## Test File: `test_portfolio_greeks.py`

```python
"""Tests for portfolio-level Greeks aggregation."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from quantstack.core.options.models import PortfolioGreeks, GreeksBreakdown
```

### Tests

1. **test_empty_portfolio_all_zeros** -- No positions. All totals == 0.
2. **test_single_long_call_delta** -- delta=0.5, spot=100, qty=1. delta_dollars=5000.
3. **test_short_put_positive_delta** -- delta=-0.3, qty=-1. delta_dollars = -0.3 * 100 * (-1) * 100 = +3000.
4. **test_gamma_dollar_formula** -- gamma=0.05, spot=100, qty=1. gamma_dollars = 0.5*0.05*10000*100*0.01 = 250.
5. **test_theta_negative_for_long** -- theta=-0.05, qty=1. theta_dollars = -5.0.
6. **test_multi_position_sums** -- 3 positions. per_symbol totals == sum of individual.
7. **test_per_strategy_isolation** -- 2 strategies. Each accumulates independently.
8. **test_portfolio_total_equals_sum_of_symbols** -- total_delta == sum(per_symbol[s].delta_dollars).
9. **test_to_db_row_json** -- Verify JSON structure of symbol_greeks and strategy_greeks.
10. **test_store_retrieve_roundtrip** -- Insert snapshot, query back, verify all fields.
11. **test_missing_spot_skipped** -- One symbol no spot. Others still computed.
12. **test_fallback_iv_30pct** -- No IV data. Verify 0.30 used (check via greeks output).

## Test File: `test_pnl_attribution.py`

```python
"""Tests for Greek P&L attribution."""

import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch
```

### Tests

1. **test_delta_pnl_2pct_move** -- delta=$5000, 2% up. delta_pnl=$100.
2. **test_gamma_pnl_3pct_move** -- gamma=$250 (per 1%), 3% move. gamma_pnl = 250 * 9 * 0.01 = $22.50.
3. **test_theta_pnl_one_day** -- theta=-$50. theta_pnl=-$50.
4. **test_vega_pnl_iv_up** -- vega=$200, iv_change=+0.05. vega_pnl=$10.
5. **test_unexplained_residual** -- actual=$150, attributed=$120. unexplained=$30.
6. **test_no_greeks_snapshot_skips** -- No snapshot for date. Returns None for symbol.
7. **test_upsert_overwrites** -- Insert, then recompute. Verify updated values.
8. **test_weekend_skips_to_friday** -- Monday attribution. prev_date = Friday.
9. **test_zero_price_change_flat** -- price_change=0. delta_pnl=0, gamma_pnl=0, theta still computed.
10. **test_multiple_symbols** -- 3 symbols. Each independent attribution row.
11. **test_missing_price_data** -- No daily_prices row. price_change=0, pnl mostly unexplained.
12. **test_negative_gamma_large_move** -- Short gamma, 5% move. Large negative gamma_pnl.

## Shared Test Infrastructure

### `tests/unit/options/conftest.py`

```python
"""Shared fixtures for options tests."""

import pytest
from datetime import date, datetime
from quantstack.core.options.models import OptionContract, OptionType, OptionLeg

@pytest.fixture
def sample_call_contract():
    return OptionContract(
        contract_id="AAPL240419C00180000",
        underlying="AAPL",
        expiry=date(2026, 4, 19),
        strike=180.0,
        option_type=OptionType.CALL,
        bid=5.00, ask=5.50, last=5.25,
        volume=1200, open_interest=5000,
        iv=0.30, delta=0.50, gamma=0.03,
        theta=-0.05, vega=0.15, rho=0.02,
    )

@pytest.fixture
def sample_put_contract():
    return OptionContract(
        contract_id="AAPL240419P00180000",
        underlying="AAPL",
        expiry=date(2026, 4, 19),
        strike=180.0,
        option_type=OptionType.PUT,
        bid=4.80, ask=5.30, last=5.05,
        volume=800, open_interest=3000,
        iv=0.32, delta=-0.50, gamma=0.03,
        theta=-0.04, vega=0.14, rho=-0.02,
    )

@pytest.fixture
def sample_chain(sample_call_contract, sample_put_contract):
    """Synthetic chain with 5 strikes for calls and puts."""
    contracts = []
    for offset in [-10, -5, 0, 5, 10]:
        for opt_type, base in [(OptionType.CALL, sample_call_contract), (OptionType.PUT, sample_put_contract)]:
            c = OptionContract(
                contract_id=f"{base.underlying}_{base.strike + offset}_{opt_type.value}",
                underlying=base.underlying,
                expiry=base.expiry,
                strike=base.strike + offset,
                option_type=opt_type,
                bid=max(0.10, base.bid - offset * 0.3),
                ask=max(0.20, base.ask - offset * 0.3),
                iv=base.iv + offset * 0.002,
                delta=base.delta - offset * 0.02,
                gamma=base.gamma,
                theta=base.theta,
                vega=base.vega,
                volume=base.volume,
                open_interest=base.open_interest,
            )
            contracts.append(c)
    return contracts

@pytest.fixture
def mock_db_conn():
    """Mock db_conn context manager."""
    from unittest.mock import MagicMock, patch
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []
    mock_conn.execute.return_value.fetchone.return_value = None
    with patch("quantstack.db.db_conn") as mock:
        mock.return_value.__enter__ = lambda s: mock_conn
        mock.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_conn
```

## Edge Cases Covered Across All Test Files

- **Null/empty inputs**: Empty chain, no positions, zero prices, null Greeks
- **Boundary values**: Exact thresholds (2% distance, $500 delta, 30-min interval)
- **Sign conventions**: Long vs short, calls vs puts, positive vs negative delta
- **Fallback paths**: IV solver fallback, missing spot prices, stale data
- **Concurrency**: Idempotent schema creation, upsert semantics
- **Time boundaries**: DTE=0, T=0, weekends, market hours
- **Mathematical edge cases**: Zero division guards, negative vol floors, deep ITM/OTM
