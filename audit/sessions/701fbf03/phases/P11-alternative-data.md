# P11: Alternative Data Sources

**Objective:** Add non-traditional data signals: web traffic, job postings, patent filings, satellite/shipping, credit card proxies, and congressional trades.

**Scope:** data/providers/, signal_engine/collectors/

**Depends on:** P07 (data architecture)

**Effort estimate:** 2 weeks

---

## What Changes

### 11.1 Web Traffic & App Downloads
- **Source:** SimilarWeb API or SEMrush
- **Signal:** Website traffic trends predict revenue surprises (e-commerce, SaaS)
- **Collector:** `web_traffic_collector.py` — monthly traffic delta, engagement metrics
- **IC expectation:** 0.03-0.05 for e-commerce/SaaS companies

### 11.2 Job Postings
- **Source:** Indeed/LinkedIn scraping or Thinknum API
- **Signal:** Hiring surge → expansion → future revenue growth. Layoffs → contraction.
- **Collector:** `job_postings_collector.py` — job count delta, role types (engineering vs sales)
- **IC expectation:** 0.02-0.04 (slow signal, 3-6 month lead)

### 11.3 Patent Filings
- **Source:** USPTO API (free), Google Patents
- **Signal:** Patent activity predicts R&D output, future competitive position
- **Collector:** `patent_collector.py` — patent count, citation score
- **IC expectation:** 0.01-0.03 (very slow signal, 6-12 month lead)

### 11.4 Congressional & Insider Trades (Enhanced)
- **Source:** Capitol Trades API, Quiver Quantitative
- **Signal:** Congressional insider trades (STOCK Act data)
- **Collector:** Enhance existing `insider_signals.py` with congressional data
- **IC expectation:** 0.03-0.06 (short-term, 5-30 day lead)

### 11.5 Shipping & Satellite Data
- **Source:** MarineTraffic (AIS data), Planet Labs (satellite)
- **Signal:** Port congestion, factory activity, retail parking lots
- **Collector:** `shipping_collector.py` — trade flow volumes
- **IC expectation:** 0.02-0.04 (macro signal, 1-3 month lead)
- **Cost consideration:** Satellite data is expensive ($500+/mo) — defer unless alpha is proven

### 11.6 Credit Card Proxy Data
- **Source:** Affinity Solutions, Second Measure (now Bloomberg)
- **Signal:** Consumer spending trends predict retail/restaurant revenue
- **Cost:** $1K-5K/mo for decent coverage — may be too expensive for <$100K
- **Alternative:** Bloomberg Terminal (if available) or publicly-scraped proxies

## Build-vs-Buy
| Source | Cost | IC Potential | Priority |
|--------|------|-------------|----------|
| Congressional trades | Free (Quiver) | 0.03-0.06 | HIGH |
| Web traffic | $100-500/mo | 0.03-0.05 | HIGH |
| Job postings | Free (scraping) | 0.02-0.04 | MEDIUM |
| Patents | Free (USPTO) | 0.01-0.03 | LOW |
| Shipping/AIS | $200-1K/mo | 0.02-0.04 | LOW |
| Credit card | $1K-5K/mo | 0.04-0.08 | DEFER (cost too high for <$100K) |

## Acceptance Criteria

1. At least 3 new alt data collectors integrated into signal engine
2. Each new collector has IC tracked from day 1 (P01 dependency)
3. Alt data signals contribute to SignalBrief with proper weighting
4. Cost per signal justified by IC contribution
