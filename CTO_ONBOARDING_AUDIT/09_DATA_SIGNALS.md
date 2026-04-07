# 08 — Data Pipeline & Signal Engine: Keep the Foundation Solid

**Priority:** P2
**Timeline:** Week 4-6
**Gate:** Signal cache auto-invalidated. Data staleness rejected. Secondary data providers configured.

---

## Why This Section Matters

The signal engine is QuantStack's crown jewel — 16 concurrent collectors, 2-6s wall-clock, fault-tolerant, regime-adaptive synthesis weights. The data pipeline is idempotent with rate limiting. These work well. But several single points of failure and cache invalidation gaps would cause silent degradation in 24/7 operation. The fixes here are about hardening what's already good.

---

## 8.1 Signal Cache Auto-Invalidation

**Finding ID:** CTO DC1
**Severity:** CRITICAL
**Effort:** 1 day

### The Problem

Signal cache has a 1-hour TTL, but intraday data refreshes every 5 minutes. A trading decision could be based on a 55-minute-old SignalBrief while fresh data sits in the DB unused. Manual `cache.invalidate(symbol)` exists but is never called after data refresh.

### The Fix

Hook `cache.invalidate(symbol)` into `scheduled_refresh.py` after each intraday refresh cycle. When fresh data arrives, stale signals are automatically evicted.

### Acceptance Criteria

- [ ] Data refresh triggers signal cache invalidation for affected symbols
- [ ] No signal brief older than the most recent data refresh is served
- [ ] Verified: fresh data → cache miss → new SignalBrief generated

---

## 8.2 Data Staleness Rejection in Collectors

**Finding ID:** CTO DC3
**Severity:** CRITICAL
**Effort:** 2 days

### The Problem

The data validator flags stale data, but the signal engine doesn't check freshness before running collectors. A collector can compute RSI on 3-week-old data and return a confident signal.

### The Fix

Each collector checks `data_metadata.last_timestamp` before computing:

| Collector Type | Max Staleness | On Stale Data |
|---------------|--------------|---------------|
| Technical (price-based) | 2 trading days | Return `{}` (no signal) |
| Volume | 2 trading days | Return `{}` |
| Options flow | 3 trading days | Return `{}` |
| Sentiment | 7 days | Return `{}` |
| Fundamentals | 90 days | Return `{}` |
| Macro | 45 days | Return `{}` |

### Acceptance Criteria

- [ ] Every collector checks data freshness before computing
- [ ] Stale collectors return `{}` and penalize confidence
- [ ] Log when collectors skip due to stale data

---

## 8.3 Alpha Vantage Redundancy

**Finding ID:** CTO DC2
**Severity:** HIGH
**Effort:** 5-7 days

### The Problem

AV is the sole source for: options chains, earnings, macro indicators, news sentiment, fundamentals, insider/institutional data. Only OHLCV has a fallback (Alpaca IEX). If AV goes down, 12 of 14 acquisition phases fail silently.

### The Fix

Add secondary providers for critical data types:

| Data Type | Primary | Secondary | Cost |
|----------|---------|-----------|------|
| OHLCV | Alpha Vantage | Alpaca (already exists) | Free |
| Options chains | Alpha Vantage | CBOE/Polygon | $99-$199/mo |
| Macro indicators | Alpha Vantage | FRED API | Free |
| Fundamentals | Alpha Vantage | SEC EDGAR | Free |
| Earnings | Alpha Vantage | SEC EDGAR (8-K filings) | Free |
| News/Sentiment | AV + Groq | Reddit/Stocktwits (already partial) | Free |

Additionally: alert when >3 consecutive phase failures for any data type.

### Acceptance Criteria

- [ ] FRED API configured for macro indicators
- [ ] SEC EDGAR configured for fundamentals and earnings
- [ ] Alert fires on 3+ consecutive failures for any data source
- [ ] System continues operating when AV is down (degraded but not dead)

---

## 8.4 Drift Detection Pre-Cache

**Finding ID:** CTO DH1
**Severity:** HIGH
**Effort:** 1 day

### The Problem

PSI drift check runs AFTER the brief is already synthesized and cached. If drift is CRITICAL, the stale brief is still served for up to 1 hour.

### The Fix

Move drift check to pre-cache-store. If CRITICAL drift detected:
- Don't cache the brief (or cache with `low_confidence=True` tag)
- Penalize confidence by 0.30
- Publish `DRIFT_CRITICAL` event to EventBus

### Acceptance Criteria

- [ ] Drift check runs before caching
- [ ] CRITICAL drift prevents normal caching
- [ ] Event published for supervisor awareness

---

## 8.5 Sentiment Fallback Fix

**Finding ID:** CTO DH3
**Severity:** HIGH
**Effort:** 0.5 day

### The Problem

When no headlines are available or Groq times out, sentiment returns 0.5 (neutral). This is safe but misleading — a fake neutral signal carries the same weight as a real one in synthesis.

### The Fix

Return `{}` instead of fake neutral when data is unavailable. Let synthesis redistribute the weight to other active collectors.

### Acceptance Criteria

- [ ] Unavailable sentiment returns `{}`, not 0.5
- [ ] Synthesis correctly redistributes weight when sentiment absent

---

## 8.6 Signal Correlation and Redundancy Tracking

**Finding ID:** QS-S5
**Severity:** CRITICAL
**Effort:** 2 days

### The Problem

22 collectors run independently with static weights. Technical RSI and ML direction signals are often >0.7 correlated. No correlation matrix computed. System claims 22 independent signals — effective independent count may be 10-12.

### The Fix

| Step | Action |
|------|--------|
| 1 | Weekly: compute pairwise signal correlation matrix |
| 2 | If `corr(signal_A, signal_B) > 0.7`: halve weight of weaker signal |
| 3 | Report effective signal count = eigenvalues > 0.1 of correlation matrix |
| 4 | Store correlation matrix for trend analysis |

### Acceptance Criteria

- [ ] Weekly signal correlation matrix computed and stored
- [ ] Highly correlated signals have reduced weight
- [ ] Effective independent signal count reported (expect 10-15 from 22)

---

## 8.7 Conflicting Signal Resolution

**Finding ID:** QS-S9
**Severity:** HIGH
**Effort:** 1 day

### The Problem

When technical says "bullish" but ML says "bearish" and sentiment says "neutral," the system computes a weighted average — landing at "slightly bullish" or "slightly bearish." It then trades on this weak, conflicting signal.

### The Fix

Add conflict detection: `if max_signal - min_signal > 0.5: flag as "conflicting"`. When conflicting: cap conviction at 0.3 regardless of weighted average. Or: skip trade entirely and log.

### Acceptance Criteria

- [ ] Signal conflict detected when max-min spread > 0.5
- [ ] Conflicting signals cap conviction at 0.3 (or skip trade)
- [ ] Conflict events logged for analysis

---

## 8.8 Regime Detection Enhancement

**Finding ID:** QS-S7
**Severity:** HIGH
**Effort:** 3 days

### The Problem

HMM regime model: 3 states, minimum 120 bars (4 months for new symbols), no transition detection. Most losses occur during regime transitions — when the model is most uncertain.

### The Fix

| Step | Action |
|------|--------|
| 1 | Add transition probability output from HMM |
| 2 | When P(transition) > 0.3: reduce all signal weights by 50%, halve position sizes |
| 3 | Add vol-conditioned sub-regimes (trending_up_low_vol vs trending_up_high_vol) |
| 4 | During 2-3 day transition window: paper-only for new entries |

### Acceptance Criteria

- [ ] Regime transition probability computed and available to risk gate
- [ ] High transition probability automatically reduces sizing
- [ ] Vol-conditioned sub-regimes implemented

---

## 8.9 Conviction Scaling Calibration

**Finding ID:** QS-S8
**Severity:** HIGH
**Effort:** 2 days

### The Problem

Conviction adjustments are additive and fixed: ADX > 25 = +0.10, HMM stability > 0.8 = +0.05, weekly-daily conflict = -0.15. These magnitudes are not empirically calibrated. A +0.10 bonus on 0.15 base conviction is a 67% increase; on 0.85 it's an 11% increase.

### The Fix

Replace additive adjustments with multiplicative factors:

```
adjusted_conviction = base_conviction * adx_factor * stability_factor * conflict_factor
```

Calibrate factors quarterly from realized signal-to-return performance (requires IC tracking from Section 02).

### Acceptance Criteria

- [ ] Multiplicative factors replace additive adjustments
- [ ] Factors calibrated from historical performance data
- [ ] Quarterly recalibration scheduled

---

## 8.10 Web Search and Intel Sources

**Finding ID:** DO-5
**Severity:** HIGH
**Effort:** 2-3 days

### The Problem

| Intel Source | Status |
|-------------|--------|
| Web search (breaking news) | **BROKEN** — returns "not configured — set SEARCH_API_KEY" |
| SEC filings (10-K, 10-Q, 8-K) | **EMPTY** — table exists, never populated |
| Insider trading data | **EMPTY** — table exists, never populated |
| Earnings transcripts | **NO TOOL** |
| Analyst ratings | **NO TOOL** |

The `market_intel` agent tries web search, gets error, falls back to stale training data.

### The Fix

| Source | Action | Cost |
|--------|--------|------|
| Web search | Configure Tavily or Brave API key | $20/mo |
| SEC filings | Implement EDGAR scraper (free API) | Free |
| Insider trading | Populate from SEC Form 4 via EDGAR | Free |

### Acceptance Criteria

- [ ] Web search functional with valid API key
- [ ] SEC filings populated for all universe symbols
- [ ] Market intel agent receives current news, not stale data

---

## 8.11 OHLCV Table Partitioning

**Finding ID:** CTO (MEDIUM)
**Severity:** MEDIUM
**Effort:** 2 days

### The Problem

`ohlcv` table uses composite PK `(symbol, timeframe, timestamp)`. With 50 symbols x 3 timeframes x years of data, this grows to hundreds of millions of rows. No partitioning strategy.

### The Fix

Partition by symbol (hash) or by time range (monthly). PostgreSQL native partitioning.

### Acceptance Criteria

- [ ] OHLCV table partitioned
- [ ] Query performance validated at scale
- [ ] Migration path from unpartitioned → partitioned documented

---

## 8.12 Macro Indicators Lag by 30+ Days

**Finding ID:** CTO DH4
**Severity:** HIGH
**Effort:** 1 day

### The Problem

GDP, CPI, unemployment data is monthly at best. The macro collector may use 6-week-old data. For regime detection this matters less, but for rate-sensitive strategies (bonds, REITs) it's a blind spot.

### The Fix

Add FRED API as secondary source (free). FRED publishes economic indicators same-day. Add freshness flag: if macro data > 45 days old, flag in SignalBrief as `macro_stale=True`.

### Acceptance Criteria

- [ ] FRED API configured for critical macro indicators
- [ ] Macro staleness flagged in SignalBrief
- [ ] Rate-sensitive strategy decisions account for macro freshness

---

## 8.13 PostgreSQL Connection Pool Sizing

**Finding ID:** CTO DH5
**Severity:** HIGH
**Effort:** 0.5 day

### The Problem

Default pool size 20. With 3 concurrent graph loops, each running multiple agents with DB queries, plus data acquisition and streaming persistence, 20 connections may not be enough. Under load, queries will block waiting for connections.

### The Fix

Monitor pool utilization via `pg_stat_activity`. Consider raising to 50+ or using PgBouncer for connection pooling at scale.

### Acceptance Criteria

- [ ] Pool utilization monitored
- [ ] Pool size increased if utilization > 80%
- [ ] PgBouncer evaluated for multi-container setup

---

## Summary: Data & Signals Delivery

| # | Item | Effort | Priority |
|---|------|--------|----------|
| 8.1 | Signal cache auto-invalidation | 1 day | CRITICAL |
| 8.2 | Staleness rejection in collectors | 2 days | CRITICAL |
| 8.3 | AV redundancy | 5-7 days | HIGH |
| 8.4 | Drift detection pre-cache | 1 day | HIGH |
| 8.5 | Sentiment fallback fix | 0.5 day | HIGH |
| 8.6 | Signal correlation tracking | 2 days | CRITICAL |
| 8.7 | Conflicting signal resolution | 1 day | HIGH |
| 8.8 | Regime detection enhancement | 3 days | HIGH |
| 8.9 | Conviction calibration | 2 days | HIGH |
| 8.10 | Web search + intel sources | 2-3 days | HIGH |
| 8.11 | OHLCV partitioning | 2 days | MEDIUM |
| 8.12 | Macro indicator freshness (FRED) | 1 day | HIGH |
| 8.13 | Connection pool sizing | 0.5 day | HIGH |

**Total estimated effort: 24-30 engineering days.**
