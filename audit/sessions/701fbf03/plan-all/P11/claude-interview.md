# P11 Self-Interview: Alternative Data Sources

## Q1: How do you handle the 45-day lag in congressional trade disclosures?
**A:** Accept it as a feature, not a bug. The signal is slow but meaningful — congressional trades predict 60-90 day returns, so 45-day disclosure delay still leaves 15-45 days of predictive value. The collector marks signals with disclosure_date and trade_date separately. IC is computed against forward returns from disclosure_date, not trade_date.

## Q2: What happens when the Quiver Quantitative API goes down or changes?
**A:** Standard P07 DataProvider pattern: circuit breaker after 3 consecutive failures, exponential backoff retry. If the API is down for >24h, the congressional collector returns no signal (neutral), and synthesis continues with remaining collectors. The IC for the collector drops (no observations), which naturally reduces its weight.

## Q3: Which sectors should the web traffic signal apply to?
**A:** Sector filter: ONLY apply to companies where web traffic proxies revenue. Include: Consumer Discretionary (e-commerce), Communication Services (social/media), Information Technology (SaaS). Exclude: Financials, Industrials, Utilities, Energy, Materials, Real Estate, Healthcare (traffic ≠ revenue). The sector mapping is configurable.

## Q4: How do you prevent the alt data signals from degrading overall synthesis quality?
**A:** Initial weight = 0.05 (5%) for each new collector — small enough to not move the needle if the signal is noise. IC tracking kicks in immediately. If IC is consistently below 0.02 after 30 days, the weight floors to near-zero automatically via the P05 IC-driven weight system. No manual intervention needed.

## Q5: What's the data freshness expectation for each collector?
**A:** Congressional: daily update (batch overnight), stale after 48h. Web traffic: monthly update, stale after 45 days. Job postings: weekly update, stale after 14 days. Patents: monthly update, stale after 45 days. Each tracked in data_freshness table (P07).

## Q6: How do patent filings translate to a tradeable signal?
**A:** Patent acceleration = YoY increase in patent filings for a company. This indicates R&D investment intensity. It's a very slow signal (6-12 month lead time) with low IC (0.01-0.03), but it's uncorrelated with other signals — the diversification benefit matters. Weight it appropriately low and let IC tracking decide if it earns more weight.

## Q7: How do you handle symbol mapping between alt data sources and the universe?
**A:** Each API uses different identifiers. Congressional data uses ticker symbols (need mapping for dual-listed). SimilarWeb uses domain names (need company→domain mapping table). USPTO uses assignee names (need company→patent assignee fuzzy matching). A symbol_mapping table in Postgres handles the cross-reference.

## Q8: What's the cost structure and is it within budget?
**A:** Congressional (Quiver): free. Patents (USPTO): free. Job postings (Thinknum): $200-500/mo or use free Glassdoor API. Web traffic (SimilarWeb): $100-500/mo. Total: $300-1000/mo. Given the potential IC uplift and diversification benefit, this is acceptable. Start with free sources (congressional + patents), add paid sources after validating the pattern works.
