# Section 04: Patent Filings Collector

## Objective

Build a signal collector that ingests USPTO patent filing data (via PatentsView API, free) and produces a long-term R&D strength signal. Expected IC 0.01-0.03 with a 6-12 month lead time — the slowest signal in the alt-data suite. Useful as a confirmatory factor for tech and pharma holdings.

## Files to Create

### `src/quantstack/signal_engine/collectors/patents.py`

New collector module following the `async collect_*(symbol, store) -> dict` pattern.

**Implementation details:**

1. **API client** — HTTP client for USPTO PatentsView API (`https://api.patentsview.org/patents/query`). No API key required (free, public). Query by assignee organization name (mapped from ticker). 10-second timeout (USPTO can be slow).

2. **Assignee mapping** — `TICKER_TO_ASSIGNEE: dict[str, str]`: maps tickers to USPTO assignee names. E.g., `{"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation"}`. For unmapped tickers, attempt fuzzy match using company name from DataStore. If no match, return `{}`.

3. **Patent metrics** — `compute_patent_metrics(patents: list[dict]) -> dict[str, Any]`:
   - `patent_count_12m`: grants in last 12 months.
   - `patent_count_prior_12m`: grants in the 12 months before that.
   - `patent_acceleration`: (patent_count_12m - patent_count_prior_12m) / max(patent_count_prior_12m, 1).
   - `citation_score`: average forward citations per patent (higher = more impactful).
   - `technology_categories`: Counter of CPC (Cooperative Patent Classification) codes → identifies R&D focus areas.
   - `top_category`: most frequent CPC section (e.g., "G06" = computing, "A61" = medical).

4. **Signal computation** — `compute_patent_signal(metrics: dict) -> dict[str, Any]`:
   - `patent_signal_score` in [-1.0, 1.0]: `tanh(patent_acceleration) * 0.4 + tanh(citation_score / 10) * 0.3 + tanh(patent_count_12m / 100) * 0.3`.
   - Negative acceleration (fewer patents than prior year) → bearish lean but mild (R&D slowdown).
   - Confidence: 0.3 (very slow signal, lowest IC in the suite).
   - Return dict: `patent_signal_score`, `patent_count_12m`, `patent_acceleration`, `citation_score`, `top_category`, `technology_categories`, `confidence`.

5. **Entry point** — `async def collect_patents(symbol: str, store: DataStore) -> dict[str, Any]`:
   - Check freshness: `check_freshness(symbol, "patent_filings", max_days=35)` (monthly cadence).
   - Resolve assignee name from ticker.
   - Query PatentsView for last 24 months of grants.
   - On failure, return `{}`.

6. **Sector relevance check** — Only compute for sectors where patent activity is meaningful: Technology, Healthcare, Industrials, Materials. For Financials, Utilities, Real Estate → return `{}` immediately.

## Files to Modify

### `src/quantstack/signal_engine/staleness.py`

Add entry to `STALENESS_THRESHOLDS`:
```python
"patent_filings": 35,
```

## Test Requirements

### `tests/unit/signal_engine/test_patent_collector.py`

1. **test_compute_metrics_basic** — 50 patents in 12m, 30 prior → acceleration ~0.67.
2. **test_compute_metrics_no_prior** — 0 prior patents → acceleration uses max(prior, 1) denominator.
3. **test_compute_signal_accelerating** — Rising patent count → positive score.
4. **test_compute_signal_decelerating** — Declining patent count → mild negative score.
5. **test_compute_signal_zero_patents** — No patents → score 0.0, confidence 0.0.
6. **test_citation_score** — High citation average → score boosted.
7. **test_technology_categories** — Patents with CPC codes → correct category counts.
8. **test_sector_filter_tech** — AAPL (Technology) → proceeds to API.
9. **test_sector_filter_financial** — JPM (Financials) → returns `{}` immediately.
10. **test_assignee_mapping** — Known ticker → correct assignee name.
11. **test_api_failure** — Mock httpx timeout → returns `{}`.

## Acceptance Criteria

- [ ] `collect_patents(symbol, store)` returns signal dict or `{}` on failure/ineligibility.
- [ ] No API key required (USPTO PatentsView is free and public).
- [ ] Sector filter prevents API calls for irrelevant sectors (Financials, Utilities, Real Estate).
- [ ] Patent acceleration denominator avoids division by zero.
- [ ] Confidence never exceeds 0.3 (slowest, lowest-IC signal).
- [ ] Signal computation is pure and independently testable.
- [ ] All 11 unit tests pass.
- [ ] Staleness threshold registered at 35 days.
