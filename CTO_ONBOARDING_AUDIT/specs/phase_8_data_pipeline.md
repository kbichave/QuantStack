# Phase 8: Data Pipeline Hardening — Deep Plan Spec

**Timeline:** Week 4-6 (parallel with Phases 4-5)
**Effort:** 14-17 days
**Gate:** Signal cache fresh. Data providers redundant. Intel sources live.

---

## Context

This spec is part of the QuantStack CTO Onboarding Audit implementation plan (164 findings, overall grade C-). Phase 8 hardens what's already good. The signal engine is QuantStack's crown jewel — 16 concurrent collectors, 2-6s wall-clock, fault-tolerant, regime-adaptive synthesis weights. The data pipeline is idempotent with rate limiting. But several single points of failure and cache invalidation gaps would cause silent degradation in 24/7 operation.

**Full audit reference:** [`CTO_ONBOARDING_AUDIT/`](../README.md)
**Primary audit section:** [`09_DATA_SIGNALS.md`](../09_DATA_SIGNALS.md)

---

## Objective

Ensure signal cache freshness, add staleness rejection to collectors, build data provider redundancy (eliminate AV as single point of failure), fix drift detection timing, and get intel sources operational.

---

## Items

### 8.1 Signal Cache Auto-Invalidation

- **Finding:** CTO DC1 | **Severity:** CRITICAL | **Effort:** 1 day
- **Audit section:** [`09_DATA_SIGNALS.md` §8.1](../09_DATA_SIGNALS.md)
- **Problem:** Signal cache has 1-hour TTL, but intraday data refreshes every 5 minutes. Trading decision could use 55-minute-old SignalBrief while fresh data sits in DB unused. `cache.invalidate(symbol)` exists but never called after data refresh.
- **Fix:** Hook `cache.invalidate(symbol)` into `scheduled_refresh.py` after each intraday refresh cycle.
- **Key files:** `src/quantstack/data/scheduled_refresh.py`, signal cache
- **Acceptance criteria:**
  - [ ] Data refresh triggers signal cache invalidation for affected symbols
  - [ ] No signal brief older than most recent data refresh is served

### 8.2 Staleness Rejection in Collectors

- **Finding:** CTO DC3 | **Severity:** CRITICAL | **Effort:** 2 days
- **Audit section:** [`09_DATA_SIGNALS.md` §8.2](../09_DATA_SIGNALS.md)
- **Problem:** Data validator flags stale data, but signal engine doesn't check freshness before running collectors. Collector can compute RSI on 3-week-old data and return confident signal.
- **Fix:** Each collector checks `data_metadata.last_timestamp` before computing. Max staleness: Technical/Volume=2 trading days, Options=3 days, Sentiment=7 days, Fundamentals=90 days, Macro=45 days. Return `{}` when stale.
- **Key files:** All 16 signal collectors
- **Acceptance criteria:**
  - [ ] Every collector checks data freshness before computing
  - [ ] Stale collectors return `{}` and penalize confidence
  - [ ] Log when collectors skip due to stale data

### 8.3 AV Redundancy (FRED, EDGAR)

- **Finding:** CTO DC2 | **Severity:** HIGH | **Effort:** 5-7 days
- **Audit section:** [`09_DATA_SIGNALS.md` §8.3](../09_DATA_SIGNALS.md)
- **Problem:** AV is sole source for options chains, earnings, macro indicators, news sentiment, fundamentals, insider/institutional data. Only OHLCV has fallback (Alpaca IEX). If AV down, 12 of 14 acquisition phases fail silently.
- **Fix:** Add secondary providers:
  - OHLCV → Alpaca (already exists)
  - Macro indicators → FRED API (free)
  - Fundamentals + Earnings → SEC EDGAR (free)
  - Options chains → CBOE/Polygon ($99-$199/mo) when budget allows
  - Alert on 3+ consecutive phase failures
- **Key files:** Data acquisition pipeline, provider abstraction layer
- **Acceptance criteria:**
  - [ ] FRED API configured for macro indicators
  - [ ] SEC EDGAR configured for fundamentals and earnings
  - [ ] Alert fires on 3+ consecutive failures for any data source
  - [ ] System continues operating when AV is down (degraded but not dead)

### 8.4 Drift Detection Pre-Cache

- **Finding:** CTO DH1 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`09_DATA_SIGNALS.md` §8.4](../09_DATA_SIGNALS.md)
- **Problem:** PSI drift check runs AFTER brief synthesized and cached. If CRITICAL drift, stale brief still served for up to 1 hour.
- **Fix:** Move drift check to pre-cache-store. CRITICAL drift → don't cache (or cache with `low_confidence=True`), penalize confidence by 0.30, publish `DRIFT_CRITICAL` event.
- **Key files:** Signal engine caching logic, drift detection
- **Acceptance criteria:**
  - [ ] Drift check runs before caching
  - [ ] CRITICAL drift prevents normal caching
  - [ ] Event published for supervisor awareness

### 8.5 Web Search Configuration

- **Finding:** DO-5 | **Severity:** HIGH | **Effort:** 0.5 day
- **Audit section:** [`09_DATA_SIGNALS.md` §8.10](../09_DATA_SIGNALS.md)
- **Problem:** Web search returns "not configured — set SEARCH_API_KEY". `market_intel` agent falls back to stale training data.
- **Fix:** Configure Tavily or Brave API key ($20/mo).
- **Key files:** Web search tool configuration, `.env`
- **Acceptance criteria:**
  - [ ] Web search functional with valid API key
  - [ ] Market intel agent receives current news

### 8.6 SEC Filings Population

- **Finding:** DO-5 | **Severity:** HIGH | **Effort:** 2 days
- **Audit section:** [`09_DATA_SIGNALS.md` §8.10](../09_DATA_SIGNALS.md)
- **Problem:** SEC filings table exists but never populated. Insider trading table exists but empty. No earnings transcripts tool. No analyst ratings tool.
- **Fix:** Implement EDGAR scraper (free API) for 10-K, 10-Q, 8-K filings. Populate insider trading from SEC Form 4.
- **Key files:** Data acquisition pipeline, new EDGAR scraper
- **Acceptance criteria:**
  - [ ] SEC filings populated for all universe symbols
  - [ ] Insider trading data populated from Form 4

### 8.7 OHLCV Partitioning

- **Finding:** CTO (MEDIUM) | **Severity:** MEDIUM | **Effort:** 2 days
- **Audit section:** [`09_DATA_SIGNALS.md` §8.11](../09_DATA_SIGNALS.md)
- **Problem:** `ohlcv` table uses composite PK `(symbol, timeframe, timestamp)`. With 50 symbols × 3 timeframes × years of data → hundreds of millions of rows. No partitioning.
- **Fix:** Partition by symbol (hash) or time range (monthly) using PostgreSQL native partitioning.
- **Key files:** Database schema, migration
- **Acceptance criteria:**
  - [ ] OHLCV table partitioned
  - [ ] Query performance validated at scale
  - [ ] Migration path documented

### 8.8 Options Refresh Expansion

- **Finding:** CTO DH2 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`09_DATA_SIGNALS.md`](../09_DATA_SIGNALS.md)
- **Problem:** Options data refreshed only for actively held positions. No pre-trade options scanning for universe symbols.
- **Fix:** Add configurable options refresh for watchlist symbols, not just current holdings.
- **Key files:** Options data refresh scheduler
- **Acceptance criteria:**
  - [ ] Options data refreshed for watchlist symbols
  - [ ] Pre-trade options analysis has fresh data

---

## Dependencies

- **Runs parallel with:** Phase 4 (Agent Architecture) and Phase 5 (Cost Optimization)
- **No hard dependencies on other phases** (data pipeline is foundational)
- **8.3 (FRED/EDGAR) is independent** and can start immediately

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| 8.3: EDGAR API has rate limits and HTML parsing complexity | Start with structured XBRL data (machine-readable). Fall back to HTML parsing for older filings. |
| 8.7: OHLCV partitioning requires table recreation | Use `pg_partman` for online migration. Test on backup first. |
| 8.2: Staleness rejection may cause signal gaps | Return `{}` (no signal) is better than fake confident signal on stale data. Synthesis redistributes weight. |

---

## Validation Plan

1. **Cache invalidation (8.1):** Refresh AAPL data → verify next signal request triggers cache miss → new SignalBrief generated.
2. **Staleness (8.2):** Set AAPL OHLCV last_timestamp to 5 days ago → verify technical collector returns `{}`.
3. **AV redundancy (8.3):** Block AV API → verify FRED returns macro data, EDGAR returns fundamentals.
4. **Drift detection (8.4):** Inject CRITICAL drift → verify brief NOT cached, confidence penalized.
5. **Web search (8.5):** Ask market_intel about today's news → verify current results returned.
