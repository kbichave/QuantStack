# P11 Spec: Alternative Data Sources

## Deliverables

### D1: Congressional Trades Collector
- Quiver Quantitative API integration
- Signal: net buy count × avg $ size → directional signal
- Daily batch update, 45-day filing lag accounted for
- IC tracking from day 1

### D2: Web Traffic Collector
- SimilarWeb API integration
- Signal: traffic growth vs market avg → bullish for e-commerce/SaaS
- Sector filter: Consumer Discretionary, Comm Services, Info Tech only
- Monthly update frequency

### D3: Job Postings Collector
- Thinknum API or free alternative
- Signal: hiring surge (>20% YoY) → bullish, layoff signal → bearish
- Weekly update, 3-6 month lead time
- Role type weighting (engineering > sales > admin)

### D4: Patent Filings Collector
- USPTO PatentsView API
- Signal: patent count acceleration, citation score
- Monthly update, 6-12 month lead time
- Very low expected IC (0.01-0.03), value is in uncorrelated diversification

### D5: Data Provider Integration
- Each collector's API adapter follows P07 DataProvider pattern
- Rate limiting, circuit breaker, freshness tracking
- Symbol mapping table for cross-referencing identifiers

### D6: Synthesis Integration
- All collectors registered in signal engine
- Initial weight = 0.05 each, adjusted by IC
- Graceful degradation when API unavailable

## Dependencies
- P07 (Data Architecture): DataProvider pattern, freshness tracking
- P05 (Signal Synthesis): IC-driven weight adjustment
