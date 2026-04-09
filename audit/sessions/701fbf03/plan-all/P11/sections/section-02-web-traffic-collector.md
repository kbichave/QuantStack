# Section 02: Web Traffic Collector

## Objective

Build a signal collector that ingests web traffic data (via SimilarWeb API) and produces a directional signal for companies where web traffic correlates with revenue — primarily e-commerce, SaaS, and ad-supported businesses. Expected IC 0.03-0.05 with monthly update cadence.

## Files to Create

### `src/quantstack/signal_engine/collectors/web_traffic.py`

New collector module following the `async collect_*(symbol, store) -> dict` pattern.

**Implementation details:**

1. **Sector filter** — `WEB_TRAFFIC_SECTORS`: set of GICS sector/industry codes where web traffic is a revenue proxy. Includes: "Internet & Direct Marketing Retail", "Application Software", "Interactive Media & Services", "Internet Services & Infrastructure". Symbols outside these sectors return `{}` immediately (no API call wasted).

2. **Domain mapping** — `TICKER_TO_DOMAIN: dict[str, str]`: maps tickers to primary domains. E.g., `{"AMZN": "amazon.com", "SHOP": "shopify.com"}`. For unmapped tickers, attempt to derive from company name in DataStore (`store.get_company_overview(symbol)`). If no domain found, return `{}`.

3. **API client** — HTTP client for SimilarWeb API (`https://api.similarweb.com/v1/website/{domain}/total-traffic-and-engagement/visits`). Requires `SIMILARWEB_API_KEY` env var. 8-second timeout.

4. **Signal computation** — `compute_web_traffic_signal(current: dict, historical: list[dict]) -> dict[str, Any]`:
   - Compute `traffic_growth_3m`: (current_visits - visits_3m_ago) / visits_3m_ago.
   - Compute `traffic_growth_yoy`: year-over-year change.
   - Compute `engagement_delta`: change in avg_visit_duration and pages_per_visit.
   - `web_traffic_signal_score` in [-1.0, 1.0]: `tanh(traffic_growth_3m * 5) * 0.5 + tanh(engagement_delta * 3) * 0.3 + tanh(traffic_growth_yoy * 2) * 0.2`.
   - Confidence = 0.6 if only visits data, 0.8 if engagement data also present.
   - Return dict: `web_traffic_signal_score`, `traffic_growth_3m`, `traffic_growth_yoy`, `engagement_delta`, `visits_current`, `confidence`.

5. **Entry point** — `async def collect_web_traffic(symbol: str, store: DataStore) -> dict[str, Any]`:
   - Check sector eligibility first (cheap, no I/O).
   - Check freshness: `check_freshness(symbol, "web_traffic", max_days=35)` (monthly data + buffer).
   - Fetch current + 12 months historical from SimilarWeb.
   - On failure, return `{}`.

## Files to Modify

### `src/quantstack/signal_engine/staleness.py`

Add entry to `STALENESS_THRESHOLDS`:
```python
"web_traffic": 35,
```

## Test Requirements

### `tests/unit/signal_engine/test_web_traffic_collector.py`

1. **test_compute_signal_growing_traffic** — 30% 3m growth → positive score.
2. **test_compute_signal_declining_traffic** — -20% 3m decline → negative score.
3. **test_compute_signal_flat** — 0% growth → near-zero score.
4. **test_engagement_boost** — Traffic flat but engagement up → mild positive.
5. **test_sector_filter_ecommerce** — AMZN (eligible) → proceeds to API call.
6. **test_sector_filter_industrial** — CAT (ineligible) → returns `{}` without API call.
7. **test_domain_mapping_known** — AMZN → "amazon.com".
8. **test_domain_mapping_unknown** — Unmapped ticker with no company overview → returns `{}`.
9. **test_api_failure** — Mock httpx raising timeout → returns `{}`.
10. **test_missing_api_key** — No `SIMILARWEB_API_KEY` → returns `{}`.

## Acceptance Criteria

- [ ] `collect_web_traffic(symbol, store)` returns signal dict or `{}` on failure/ineligibility.
- [ ] Sector filter prevents unnecessary API calls for non-web-revenue companies.
- [ ] Domain resolution works for known tickers and gracefully handles unknowns.
- [ ] Signal computation is pure and independently testable.
- [ ] All 10 unit tests pass.
- [ ] Staleness threshold registered at 35 days.
