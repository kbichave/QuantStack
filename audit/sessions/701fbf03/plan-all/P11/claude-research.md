# P11 Research: Alternative Data Sources

## Codebase Research

### What Exists
- **Signal engine collectors**: `src/quantstack/signal_engine/collectors/` — trend, rsi, macd, bb, sentiment, ml, flow, regime, etc.
- **Collector pattern**: each implements `collect(symbol, ...)` → dict, registered in synthesis
- **IC tracking**: automatic IC attribution for all collectors (P01/P05)
- **Data providers**: multi-provider fallback chain (P07 plan), rate limiting, circuit breakers
- **Signal synthesis**: weight profiles per collector, regime-conditional

### What's Needed (Gaps)
1. **Congressional trades collector**: No Quiver Quantitative integration — free API, high IC potential
2. **Web traffic collector**: No SimilarWeb integration — paid API ($100-500/mo)
3. **Job postings collector**: No employment data integration — Thinknum or free alternative
4. **Patent filings collector**: No USPTO API integration — free, low IC but novel
5. **API adapters**: Each needs rate limiting, circuit breaker, freshness tracking per P07 DataProvider pattern

## Domain Research

### Congressional Trading Signal
- Quiver Quantitative provides congressional trade data (STOCK Act disclosures)
- Signal strength: members with finance committee seats trade ~6% better than market
- Latency: filings published within 45 days — signal is slow but meaningful
- Free tier available, sufficient for daily batch processing

### Web Traffic as Earnings Predictor
- SimilarWeb API provides monthly unique visitors, engagement metrics
- Strong predictor for e-commerce/SaaS companies (revenue correlated with traffic)
- Less useful for B2B, financial, industrial companies — needs sector filter
- Monthly update frequency limits signal freshness

### Alternative Free Sources
- SEC EDGAR: insider trading (Form 4) — free, similar signal to congressional
- USPTO PatentsView: patent data — completely free API
- BLS.gov: employment data — free but aggregated (not company-level)
