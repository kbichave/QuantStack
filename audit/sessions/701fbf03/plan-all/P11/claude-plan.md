# P11 Implementation Plan: Alternative Data Sources

## 1. Background

P07 (data architecture) provides the multi-provider fallback chain and data freshness infrastructure. P11 adds non-traditional data signals: congressional trades, web traffic, job postings, and patent filings. Each becomes a signal collector integrated into the existing signal engine.

## 2. Anti-Goals

- **Do NOT add expensive data sources** — credit card ($1K+/mo) and satellite ($500+/mo) are deferred
- **Do NOT build scrapers** — use APIs only (Quiver Quantitative, USPTO, SimilarWeb)
- **Do NOT bypass IC tracking** — every new collector has IC tracked from day 1
- **Do NOT overweight alt data** — initial weight = 0.05 (5%), increase only if IC proves positive

## 3. Priority Order (by cost/IC ratio)

1. **Congressional trades** (Quiver Quantitative, free) — IC 0.03-0.06, HIGH priority
2. **Web traffic** (SimilarWeb API, $100-500/mo) — IC 0.03-0.05, HIGH priority
3. **Job postings** (Indeed/LinkedIn scraping alternative: Thinknum, or free Glassdoor API) — IC 0.02-0.04, MEDIUM
4. **Patent filings** (USPTO API, free) — IC 0.01-0.03, LOW

## 4. Congressional Trades Collector

New `src/quantstack/signal_engine/collectors/congressional.py`:
- Source: Quiver Quantitative API (free tier)
- Signal: Congressional insider buys/sells with $ amount and timing
- Logic: Net buy count × average $ size → bullish signal, net sell → bearish
- Lead time: 5-30 days
- Update frequency: daily (filings are published within 45 days)

## 5. Web Traffic Collector

New `src/quantstack/signal_engine/collectors/web_traffic.py`:
- Source: SimilarWeb API
- Signal: Monthly unique visitors delta, engagement metrics (time on site, pages/visit)
- Logic: traffic_growth_3m > market_avg → bullish for e-commerce/SaaS
- Sector filter: only apply to companies where web traffic correlates with revenue
- Update frequency: monthly

## 6. Job Postings Collector

New `src/quantstack/signal_engine/collectors/job_postings.py`:
- Source: Thinknum API or free alternative
- Signal: Job posting count delta by role type (engineering = growth, sales = expansion)
- Logic: hiring_surge (>20% YoY) → bullish, layoff_signal → bearish
- Lead time: 3-6 months (slow signal)
- Update frequency: weekly

## 7. Patent Collector

New `src/quantstack/signal_engine/collectors/patents.py`:
- Source: USPTO API (PatentsView)
- Signal: Patent count, citation score, technology category
- Logic: patent_acceleration → R&D strength → long-term bullish
- Lead time: 6-12 months (very slow)
- Update frequency: monthly

## 8. Integration

Each collector follows existing collector pattern:
- Implements `collect(symbol, ...)` → returns dict with signal fields
- Registered in signal engine's collector list
- Gets weight in synthesis (initial 0.05, adjusted by IC)
- IC tracked automatically by P01 infrastructure

## 9. Data Provider Integration

Each collector's API adapter follows P07 DataProvider pattern:
- Rate limiting via shared token bucket
- Circuit breaker on consecutive failures
- Freshness tracking in data_freshness table

## 10. Testing

- Each collector: mock API response → verify signal computation
- IC tracking: verify collector observations recorded
- Synthesis integration: verify alt data signals contribute to SymbolBrief
- Edge cases: API down → graceful degradation, symbol not covered → skip
