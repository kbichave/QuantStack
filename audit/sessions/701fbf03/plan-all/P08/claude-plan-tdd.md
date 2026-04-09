# P08 TDD Plan: Options Market-Making

## 1. Vol Arb Engine

```python
# tests/unit/strategy/test_vol_arb_engine.py

class TestVolArbSignal:
    def test_sell_signal_when_iv_overpriced(self):
        """IV > realized_vol by threshold -> short vol signal."""

    def test_buy_signal_when_iv_underpriced(self):
        """IV < realized_vol by threshold -> long vol signal."""

    def test_no_signal_below_iv_rank_threshold(self):
        """IV rank < 50th percentile -> no signal regardless of IV vs realized spread."""

    def test_signal_strength_scales_with_divergence(self):
        """Larger IV-RV divergence produces stronger signal value."""

    def test_exit_on_iv_mean_reversion(self):
        """Signal flips to exit when IV converges toward realized vol."""

    def test_exit_on_time_decay_profit_target(self):
        """Exit triggered when theta profit exceeds configurable target."""


class TestVolArbEdgeCases:
    def test_sparse_iv_surface_skips_symbol(self):
        """When IV surface has < minimum data points, symbol is skipped with no signal."""

    def test_zero_realized_vol_does_not_divide_by_zero(self):
        """Realized vol of zero handled gracefully (skip or cap)."""

    def test_nan_iv_values_filtered(self):
        """NaN in IV data does not propagate into signal computation."""
```

## 2. Dispersion Trading

```python
# tests/unit/strategy/test_dispersion.py

class TestDispersionSignal:
    def test_entry_when_implied_corr_exceeds_realized(self):
        """Implied correlation > realized correlation -> dispersion entry signal."""

    def test_no_entry_when_correlation_gap_below_threshold(self):
        """Small correlation gap -> no trade."""

    def test_correlation_computation_correctness(self):
        """Implied and realized correlation computed from component/index vols match expected values."""

    def test_delta_neutral_hedge_ratios(self):
        """Hedge ratios for index short + component longs produce near-zero net delta."""


class TestDispersionEdgeCases:
    def test_missing_component_data_skips_trade(self):
        """If any component lacks options data, dispersion trade is skipped."""

    def test_single_component_dominates_index(self):
        """High concentration in one component flagged as elevated risk."""

    def test_correlation_spike_exit(self):
        """Left-tail correlation spike triggers exit logic."""
```

## 3. Gamma Scalping

```python
# tests/unit/strategy/test_gamma_scalping.py

class TestGammaScalping:
    def test_hedge_triggered_on_underlying_move(self):
        """0.5% underlying move triggers delta re-hedge."""

    def test_hedge_triggered_on_time_interval(self):
        """30-min interval triggers delta re-hedge regardless of move."""

    def test_profitable_when_realized_vol_exceeds_implied(self):
        """Simulated scenario: realized > implied -> positive P&L after hedges."""

    def test_unprofitable_when_theta_dominates(self):
        """Simulated scenario: realized < implied -> negative P&L from theta bleed."""

    def test_auto_exit_on_theta_bleed_threshold(self):
        """Position closed when cumulative theta loss exceeds gamma profit by threshold."""

    def test_pnl_decomposition_gamma_vs_theta(self):
        """P&L correctly attributed to gamma profit and theta cost components."""
```

## 4. Iron Condor Harvesting

```python
# tests/unit/strategy/test_iron_condor.py

class TestIronCondorEntry:
    def test_entry_in_ranging_regime_high_iv(self):
        """Ranging regime + IV rank > 50 -> condor entry signal."""

    def test_no_entry_in_trending_regime(self):
        """Trending regime -> no condor entry."""

    def test_strike_selection_otm(self):
        """Short strikes are OTM by configurable delta (e.g. 0.16 delta)."""

    def test_defined_risk_spread_width(self):
        """Long strikes provide defined max loss per side."""


class TestIronCondorManagement:
    def test_close_at_50_pct_profit(self):
        """Position closed when 50% of max credit captured."""

    def test_roll_tested_side(self):
        """When short strike breached, tested side rolled out/up/down."""

    def test_max_loss_capped(self):
        """Loss cannot exceed spread width minus credit received."""
```

## 5. Hedging Engine Extensions

```python
# tests/unit/strategy/test_hedging_extensions.py

class TestGammaHedging:
    def test_reduces_portfolio_gamma_below_threshold(self):
        """After gamma hedge, net portfolio gamma < target threshold."""

    def test_minimizes_cost(self):
        """Hedge selects cheaper option combinations over expensive ones."""

    def test_no_action_when_gamma_within_tolerance(self):
        """Portfolio gamma already below threshold -> no hedge trades."""


class TestVegaHedging:
    def test_reduces_portfolio_vega_below_threshold(self):
        """After vega hedge, net portfolio vega < target threshold."""

    def test_uses_calendar_spread_adjustment(self):
        """Vega hedge trades options at different expiries."""

    def test_no_action_when_vega_within_tolerance(self):
        """Portfolio vega already below threshold -> no hedge trades."""
```

## 6. Market-Making Agent

```python
# tests/unit/agents/test_options_market_maker.py

class TestMarketMakerNode:
    def test_strategy_selection_trending_regime(self):
        """Trending regime -> vol arb strategy selected."""

    def test_strategy_selection_ranging_regime(self):
        """Ranging regime -> iron condor strategy selected."""

    def test_trade_proposals_pass_through_risk_gate(self):
        """All generated proposals are submitted to risk gate before execution."""

    def test_risk_limit_max_vega(self):
        """Trade rejected when portfolio vega would exceed $5,000 limit."""

    def test_risk_limit_single_strike_concentration(self):
        """Trade rejected when single-strike exposure would exceed 20%."""

    def test_existing_position_management(self):
        """Agent identifies positions needing management (roll, close, hedge)."""


class TestMarketMakerEdgeCases:
    def test_no_iv_surface_available(self):
        """Symbol with no IV surface data -> skipped gracefully."""

    def test_all_symbols_below_vol_threshold(self):
        """No mispricings found -> no trade proposals generated (not an error)."""
```

## 7. P&L Attribution

```python
# tests/unit/strategy/test_vol_pnl_attribution.py

class TestVolPnLAttribution:
    def test_strategy_level_decomposition(self):
        """P&L attributed per-strategy across delta, gamma, theta, vega."""

    def test_vol_pnl_realized_vs_implied(self):
        """Vol P&L = realized_vol_profit - implied_vol_cost computed correctly."""

    def test_attribution_sums_to_total(self):
        """Sum of all strategy-level P&L equals total portfolio P&L."""
```
