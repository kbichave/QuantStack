# P07 Self-Interview: Data Architecture Evolution

## Q1: Which secondary providers are realistic for <$100K capital?
**Q:** The spec lists Polygon, Yahoo, FMP, FRED, CBOE. Which are actually free/cheap enough?
**A:** Free tier: Yahoo Finance (yfinance, unlimited but unreliable), FRED (free, 120/min), SEC EDGAR (free, 10/sec), Alpaca (free with brokerage account). Cheap: Polygon ($29/mo basic, historical data). FMP ($14/mo for fundamentals). CBOE options data is expensive ($500+/mo) — defer. Priority: Polygon for OHLCV fallback, Yahoo as last resort, FMP for fundamentals fallback.

## Q2: How aggressive should the db.py decomposition be?
**Q:** The spec says split into schema.py, migrations.py, queries.py, connection.py. Is this a big-bang rewrite or incremental?
**A:** Incremental strangler fig. Start by extracting `connection.py` (pool setup, `db_conn()` context manager) since it's referenced everywhere. Then extract `migrations.py` (all CREATE TABLE IF NOT EXISTS blocks). Keep the rest in db.py for now. Do NOT move queries out — they're deeply interleaved with business logic in other modules.

## Q3: Point-in-time — which tables need available_date?
**Q:** Adding available_date to fundamentals. Which specific tables?
**A:** `financial_statements` (quarterly reports), `earnings_calendar` (earnings dates), `insider_trades` (SEC Form 4 filing date), `institutional_holdings` (13F filing date). The key insight: `available_date` = when the data became publicly known, NOT when it was recorded in our DB. For SEC filings, this is the filing date. For earnings, it's the release date.

## Q4: OHLCV partitioning strategy?
**Q:** Partition by what? The spec says (timeframe, timestamp). Monthly or yearly?
**A:** Partition by `timeframe` (list partition: '1d', '1h', '5m') then by `timestamp` (range, monthly for intraday, yearly for daily). This matches access patterns: queries almost always filter on timeframe first, then time range. Monthly partitions for 5-min data keep each partition at ~50K rows (50 symbols × 78 bars/day × 20 trading days). Yearly for daily is fine (~12.5K rows per partition).

## Q5: Data staleness — how aggressive during market hours?
**Q:** The 8-hour threshold is too loose. What should it be?
**A:** Tiered: 30 minutes during market hours (09:30-16:00 ET), 8 hours after-hours, 24 hours weekends. When data is stale during market hours, auto-disable the dependent signal collectors rather than trading on stale data. Alert via EventBus.

## Q6: Schema migration versioning — how complex?
**Q:** Full migration framework (like Alembic) or simple version tracking?
**A:** Simple `schema_migrations` table with (migration_name, applied_at, checksum). Each migration is a named function. On startup, check which migrations have run, execute new ones in order. No rollback support (we use idempotent DDL, so re-running is safe). No Alembic — it's overkill for this pattern.

## Q7: What does P11 (Alt Data) need from P07?
**Q:** P07 enables P11. What's the interface contract?
**A:** P11 needs: (1) multi-provider registration pattern so new providers can be added without modifying core code, (2) provider-agnostic rate limiting, (3) data freshness checking per provider. The data fetcher should support a `DataProvider` ABC that new alt data sources implement.

## Q8: Yahoo Finance reliability concerns?
**Q:** yfinance is known for breaking. Should we rely on it?
**A:** Use as last-resort fallback only. Never primary. Cache aggressively (24h for daily OHLCV, 1h for intraday). Rate limit to 1 req/sec. If yfinance breaks, the system should still function on AV + Polygon. It's free insurance, not a dependency.
