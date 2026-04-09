# P11 TDD Plan: Alternative Data Sources

## 1. Congressional Trades Collector

```python
# tests/unit/signal_engine/collectors/test_congressional.py

class TestCongressionalCollector:
    def test_net_buy_produces_bullish_signal(self):
        """Net congressional buys with significant $ volume -> positive signal."""

    def test_net_sell_produces_bearish_signal(self):
        """Net congressional sells -> negative signal."""

    def test_signal_scales_with_dollar_amount(self):
        """Larger trade sizes produce stronger signal magnitude."""

    def test_multiple_members_aggregated(self):
        """Trades from multiple congress members aggregated correctly."""

    def test_collect_returns_standard_dict(self):
        """collect(symbol) returns dict with signal_value, confidence, source fields."""


class TestCongressionalEdgeCases:
    def test_no_trades_for_symbol(self):
        """Symbol with no congressional activity returns neutral signal."""

    def test_api_returns_empty_response(self):
        """Empty API response -> neutral signal, no error."""

    def test_stale_filings_discounted(self):
        """Filings older than 45 days weighted less than recent ones."""

    def test_api_rate_limit_handled(self):
        """429 response triggers backoff, does not crash collector."""

    def test_api_down_graceful_degradation(self):
        """API failure returns None/skip, does not block signal synthesis."""
```

## 2. Web Traffic Collector

```python
# tests/unit/signal_engine/collectors/test_web_traffic.py

class TestWebTrafficCollector:
    def test_traffic_growth_above_avg_bullish(self):
        """3-month traffic growth > market average -> positive signal."""

    def test_traffic_decline_bearish(self):
        """Declining traffic -> negative signal."""

    def test_engagement_metrics_factor_in(self):
        """Higher time-on-site and pages/visit boost signal strength."""

    def test_sector_filter_applied(self):
        """Non-e-commerce/SaaS companies filtered out (signal = None)."""

    def test_monthly_update_frequency(self):
        """Collector returns cached value if called within same month."""


class TestWebTrafficEdgeCases:
    def test_symbol_not_covered(self):
        """Symbol without web traffic data returns None (skip)."""

    def test_zero_baseline_traffic(self):
        """Company with near-zero baseline traffic -> growth calc does not divide by zero."""

    def test_api_error_graceful_skip(self):
        """SimilarWeb API error -> returns None, logs warning."""
```

## 3. Job Postings Collector

```python
# tests/unit/signal_engine/collectors/test_job_postings.py

class TestJobPostingsCollector:
    def test_hiring_surge_bullish(self):
        """>20% YoY job posting increase -> positive signal."""

    def test_layoff_signal_bearish(self):
        """Sharp decline in postings -> negative signal."""

    def test_role_type_weighting(self):
        """Engineering roles weighted higher than admin roles for growth signal."""

    def test_slow_signal_lead_time(self):
        """Signal labeled with 3-6 month lead time metadata."""


class TestJobPostingsEdgeCases:
    def test_no_historical_baseline(self):
        """New company with no prior year data -> no signal (skip)."""

    def test_seasonal_hiring_normalized(self):
        """Seasonal patterns (e.g. holiday retail) do not produce false signals."""

    def test_api_unavailable(self):
        """Data source down -> returns None, no crash."""
```

## 4. Patent Collector

```python
# tests/unit/signal_engine/collectors/test_patents.py

class TestPatentCollector:
    def test_patent_acceleration_bullish(self):
        """Increasing patent filing rate -> positive long-term signal."""

    def test_high_citation_score_boosts_signal(self):
        """Patents with high citation counts produce stronger signal."""

    def test_technology_category_tagged(self):
        """Signal metadata includes patent technology category."""

    def test_very_slow_signal_metadata(self):
        """Signal labeled with 6-12 month lead time."""


class TestPatentEdgeCases:
    def test_no_patents_for_symbol(self):
        """Company with zero patent activity -> neutral signal."""

    def test_uspto_api_error(self):
        """USPTO API failure -> returns None, logs warning."""

    def test_company_name_to_assignee_mapping(self):
        """Ticker correctly mapped to USPTO assignee name for lookup."""
```

## 5. Signal Engine Integration

```python
# tests/unit/signal_engine/test_alt_data_integration.py

class TestAltDataSynthesisIntegration:
    def test_collector_registered_in_engine(self):
        """Each alt data collector registered in signal engine's collector list."""

    def test_initial_weight_is_5_pct(self):
        """Alt data collectors start with 0.05 weight in synthesis."""

    def test_ic_tracked_from_day_one(self):
        """IC observations recorded for every alt data signal emission."""

    def test_alt_data_contributes_to_symbol_brief(self):
        """SymbolBrief includes alt data signal values when available."""

    def test_missing_alt_data_does_not_block_synthesis(self):
        """Synthesis completes normally when alt data collector returns None."""

    def test_weight_adjustment_by_ic(self):
        """Collector weight increases if IC proves consistently positive."""
```

## 6. Data Provider Integration

```python
# tests/unit/signal_engine/test_alt_data_providers.py

class TestAltDataProviderPattern:
    def test_rate_limiting_enforced(self):
        """API calls respect shared token bucket rate limits."""

    def test_circuit_breaker_on_consecutive_failures(self):
        """After N consecutive API failures, circuit breaker opens."""

    def test_circuit_breaker_recovery(self):
        """After cooldown period, circuit breaker allows retry."""

    def test_freshness_tracked(self):
        """Each successful fetch updates data_freshness table entry."""

    def test_stale_data_flagged(self):
        """Data older than expected refresh interval flagged as stale."""
```
