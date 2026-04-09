# P12 TDD Plan: Multi-Asset Expansion

## 1. Asset Class Base Framework

```python
# tests/unit/asset_classes/test_base.py

class TestAssetClassInterface:
    def test_get_data_providers_returns_list(self):
        """Each AssetClass implementation returns a non-empty list of DataProviders."""

    def test_get_risk_model_returns_valid_model(self):
        """Each AssetClass returns a RiskModel with required methods."""

    def test_get_signal_collectors_returns_list(self):
        """Each AssetClass returns collectors specific to that asset class."""

    def test_get_execution_adapter_returns_adapter(self):
        """Each AssetClass returns a BrokerAdapter for order routing."""

    def test_get_trading_hours_returns_schedule(self):
        """Each AssetClass returns a TradingSchedule with open/close times."""

    def test_get_position_limits_returns_limits(self):
        """Each AssetClass returns PositionLimits with max notional and per-position caps."""


class TestAssetClassEdgeCases:
    def test_disabled_asset_class_not_tradeable(self):
        """Asset class with enabled=False in config is excluded from trading."""

    def test_unknown_instrument_rejected(self):
        """Instrument not in asset class's instruments list is rejected."""
```

## 2. Futures Asset Class

```python
# tests/unit/asset_classes/test_futures.py

class TestFuturesAssetClass:
    def test_supported_instruments(self):
        """Futures class supports ES, NQ, CL, GC, ZN."""

    def test_data_provider_ibkr(self):
        """Futures data provider includes IBKR historical adapter."""

    def test_trading_schedule_23h(self):
        """Futures schedule is 23h/day Mon-Fri (with 1h maintenance break)."""

    def test_margin_model_span(self):
        """Risk model uses SPAN margin calculation."""

    def test_notional_limits_enforced(self):
        """Position sizing respects notional limits for each contract."""


class TestFuturesSignalCollectors:
    def test_contango_backwardation_signal(self):
        """Contango -> bearish carry signal; backwardation -> bullish."""

    def test_cot_positioning_signal(self):
        """Large speculator net long -> crowding risk flagged."""

    def test_roll_yield_computation(self):
        """Roll yield correctly computed from front-month vs next-month prices."""


class TestFuturesEdgeCases:
    def test_contract_rollover_handled(self):
        """Near-expiry contract triggers roll to next month."""

    def test_limit_move_day(self):
        """Price lock-limit day handled gracefully (no fills possible)."""

    def test_ibkr_connection_failure(self):
        """IBKR data unavailable -> circuit breaker, no crash."""
```

## 3. Crypto Asset Class

```python
# tests/unit/asset_classes/test_crypto.py

class TestCryptoAssetClass:
    def test_supported_instruments(self):
        """Crypto class supports BTC, ETH, SOL."""

    def test_data_provider_binance(self):
        """Crypto data provider includes Binance REST adapter."""

    def test_trading_schedule_24_7(self):
        """Crypto schedule is 24/7 with no market close."""

    def test_position_limit_2_to_3_pct(self):
        """Max per-position allocation is 2-3% (higher vol -> tighter cap)."""

    def test_tighter_stops_than_equity(self):
        """Stop-loss thresholds are tighter than equity defaults."""


class TestCryptoSignalCollectors:
    def test_funding_rate_signal(self):
        """Positive funding rate -> crowded longs; negative -> crowded shorts."""

    def test_on_chain_metrics_signal(self):
        """On-chain data (e.g., exchange inflows) produces directional signal."""

    def test_social_sentiment_signal(self):
        """Social sentiment score contributes to signal with appropriate weight."""


class TestCryptoEdgeCases:
    def test_exchange_maintenance_downtime(self):
        """Binance maintenance window -> graceful pause, no error."""

    def test_extreme_volatility_circuit_breaker(self):
        """>20% intraday move triggers position freeze until review."""

    def test_stablecoin_depeg_risk(self):
        """Trading pair with depegging stablecoin flagged as elevated risk."""
```

## 4. Cross-Asset Signals

```python
# tests/unit/asset_classes/test_cross_asset_signals.py

class TestCrossAssetSignals:
    def test_equity_bond_correlation_regime(self):
        """Equity-bond correlation computed and regime classified (positive/negative correlation)."""

    def test_commodity_equity_lead_lag(self):
        """Commodity price moves detected as leading/lagging equity moves."""

    def test_fx_carry_trade_indicator(self):
        """FX carry signal computed from interest rate differentials."""

    def test_crypto_equity_correlation(self):
        """Crypto-equity correlation tracked and regime classified."""


class TestCrossAssetEdgeCases:
    def test_insufficient_history_for_correlation(self):
        """Fewer than minimum data points -> correlation returns None."""

    def test_asset_class_disabled_excluded_from_cross_signals(self):
        """Disabled asset class data excluded from cross-asset computations."""

    def test_timezone_alignment(self):
        """Cross-asset data aligned by UTC timestamp despite different trading hours."""
```

## 5. Risk Gate Multi-Asset Extensions

```python
# tests/unit/execution/test_risk_gate_multi_asset.py

class TestMultiAssetRiskGate:
    def test_per_asset_class_position_limits(self):
        """Each asset class has independent position limit enforced."""

    def test_cross_asset_correlation_exposure(self):
        """Correlated positions across asset classes flagged if combined exposure too high."""

    def test_margin_requirements_per_class(self):
        """Different margin models applied per asset class (SPAN for futures, Reg-T for equity)."""

    def test_total_portfolio_notional_limit(self):
        """Total notional across all asset classes cannot exceed portfolio-level cap."""

    def test_crypto_position_size_scaled_for_vol(self):
        """Crypto position sizing adjusted downward for higher volatility."""


class TestMultiAssetRiskGateEdgeCases:
    def test_single_asset_class_active(self):
        """Risk gate works correctly when only equity is enabled."""

    def test_all_asset_classes_at_limit(self):
        """New trade rejected when every asset class is at its position limit."""

    def test_margin_call_scenario(self):
        """Margin shortfall triggers position reduction recommendations."""

    def test_overnight_risk_for_futures(self):
        """Overnight gap risk for futures positions factored into risk checks."""
```

## 6. Schema and Config

```python
# tests/unit/asset_classes/test_asset_class_config.py

class TestAssetClassConfig:
    def test_config_loaded_from_db(self):
        """asset_class_config table drives which classes are enabled."""

    def test_instruments_jsonb_parsed(self):
        """Instruments JSONB column correctly parsed into list of symbols."""

    def test_position_limit_pct_applied(self):
        """position_limit_pct from config used in risk gate calculations."""

    def test_positions_table_has_asset_class(self):
        """Positions table includes asset_class column for multi-asset queries."""

    def test_filter_positions_by_asset_class(self):
        """Can query positions filtered by asset_class value."""
```
