# Implementation Plan: Phase 8 — Data Pipeline Hardening

## 1. Context and Motivation

QuantStack is an autonomous trading system built on LangGraph. Its signal engine runs 22+ collectors concurrently (technical, sentiment, macro, options flow, etc.), synthesizes them into a `SignalBrief`, and caches the result for downstream trading decisions. The data pipeline feeds these collectors through a 14-phase acquisition system backed by Alpha Vantage (AV).

A CTO audit identified several hardening gaps that would cause silent degradation in 24/7 operation:

- **Stale cache serving:** The signal cache has a 1-hour TTL, but intraday data refreshes every 5 minutes. A trading decision could use a 55-minute-old SignalBrief.
- **No freshness checks in collectors:** Collectors compute on whatever data exists, even if weeks old. A technical indicator on 3-week-old prices returns a confident but meaningless signal.
- **Single provider dependency:** Alpha Vantage is the sole source for 12 of 14 data acquisition phases. An AV outage makes the system nearly blind.
- **Drift detection timing bug:** Feature drift is detected after caching, so a CRITICAL drift brief still gets served at full confidence for up to an hour.
- **OHLCV table growth:** 7.6M rows with no partitioning. 79% is 5-minute data, growing ~100K rows/day.
- **Empty SEC tables:** Insider trading and SEC filings tables exist but were never populated.

This plan addresses 7 items (8.1-8.4, 8.6-8.8). Item 8.5 (Web Search Configuration) is excluded per stakeholder decision.

---

## 2. Signal Cache Auto-Invalidation (Item 8.1)

### Problem

`src/quantstack/signal_engine/cache.py` implements a TTLCache with a configurable TTL (default 3600s). The `invalidate(symbol)` method exists but is never called after intraday data refreshes. The only bulk invalidation is `cache.clear()` during EOD refresh. This means the 5-minute intraday refresh cycle writes fresh data to the database, but the signal cache continues serving the old SignalBrief until TTL expiry.

### Approach

The fix is surgical: after `scheduled_refresh.py` completes a refresh cycle for a set of symbols, call `cache.invalidate(symbol)` for each refreshed symbol. This requires importing the cache module into the scheduled refresh module and calling invalidation after each successful data write.

**Intraday refresh flow (modified):**
1. Bulk quotes fetched for universe → invalidate all refreshed symbols
2. 5-min OHLCV fetched for watched symbols → invalidate those symbols
3. News sentiment fetched for top 10 → invalidate those symbols

**EOD refresh flow:** Already calls `cache.clear()` — no change needed.

### Design Decisions

- Invalidate per-symbol (not `clear()`) during intraday to avoid thundering herd on all symbols simultaneously.
- Invalidation happens after the DB write succeeds, not before. If the write fails, the cached brief (even if somewhat stale) is still the best available data.
- Add cache stats logging after each refresh cycle for observability: `logger.info(f"Cache invalidated {len(symbols)} symbols. Stats: {cache.stats()}")`
- **Known limitation:** A narrow race condition exists where the signal engine starts computing a brief, the refresh cycle invalidates the cache, and then the signal engine caches its (now slightly stale) result. This window is seconds-long and strictly better than the current 55-minute staleness window. Accepted as-is.

### Files to Modify

- `src/quantstack/data/scheduled_refresh.py` — add `cache.invalidate(symbol)` calls after data writes
- `src/quantstack/signal_engine/cache.py` — no changes needed (API already exists)

---

## 3. Staleness Rejection in Collectors (Item 8.2)

### Problem

The 22 signal collectors read from a local DataStore and compute signals without checking whether the underlying data is current. The `data_metadata` table tracks `last_timestamp` per (symbol, timeframe), but collectors don't consult it. A collector computing RSI on data that hasn't been updated in three weeks will return a confident signal that means nothing.

### Approach

Introduce a shared freshness-checking helper in a new module `src/quantstack/signal_engine/staleness.py`. Each collector calls this helper before computing, passing its required freshness threshold. If data is too stale, the collector returns `{}` (empty dict), which the synthesis engine already handles by redistributing weight.

The per-collector decorator pattern was chosen over a centralized engine-level check because:
1. Different collectors depend on different data types and timeframes
2. Some collectors (e.g., `macro.py`) read from `macro_indicators` rather than `ohlcv`, so the freshness check target varies
3. Self-documenting: reading a collector's source tells you its freshness requirements

### Staleness Helper Design

Create `src/quantstack/signal_engine/staleness.py` with a single function:

```python
def check_freshness(store: DataStore, symbol: str, table: str, max_days: int) -> bool:
    """Check if the most recent data for symbol in table is within max_days of now.
    
    Returns True if fresh enough. Returns False (and logs) if stale or missing.
    Uses data_metadata.last_timestamp. All data sources must have data_metadata rows
    populated during acquisition (prerequisite: extend acquisition pipeline to update
    data_metadata for macro_indicators, news_sentiment, options_chains, insider_trades, etc.).
    """
```

The `table` parameter allows collectors to specify which data source they depend on (e.g., `"ohlcv"` for technical, `"macro_indicators"` for macro, `"news_sentiment"` for sentiment).

**Prerequisite:** Extend the acquisition pipeline to populate `data_metadata` rows for ALL data sources (not just OHLCV). Currently only OHLCV phases update `data_metadata`. Add `data_metadata` upserts to macro, news, options, insider, institutional, and other acquisition phases. This eliminates expensive fallback `MAX()` queries and ensures consistent freshness tracking across all data types.

**Performance consideration:** With `data_metadata` populated universally, `check_freshness()` is a single indexed lookup per call. At 22 collectors x 50 symbols = 1,100 calls per signal engine cycle, this adds ~100ms overhead (negligible vs the 2-6s collector runtime). If overhead exceeds 2s in benchmarks, batch all freshness checks into a single query at the engine level before dispatching to collectors.

### Staleness Thresholds

Each collector type has a maximum acceptable data age, calibrated to the data's natural update frequency:

| Category | Collectors | Max Days | Rationale |
|----------|-----------|----------|-----------|
| Price-derived | technical, volume, regime, statarb, risk | 4 | 2 trading days = up to 4 calendar days (covers 3-day weekends) |
| Options-derived | options_flow, options_flow_collector, put_call_ratio | 3 | Options chains update daily |
| News/sentiment | sentiment, sentiment_alphavantage, social_sentiment | 7 | News has a longer relevance window |
| Fundamentals | fundamentals, quality, earnings_momentum | 90 | Quarterly data |
| Macro/commodity | macro, commodity, cross_asset | 45 | Monthly indicators |
| Insider/flow | insider_signals, flow | 30 | Reported quarterly but some real-time |
| Short interest | short_interest | 14 | Bi-monthly FINRA updates |
| Sector | sector | 7 | Weekly rotation signals |
| Events | events | 30 | Calendar-based |
| EWF | ewf_collector | 7 | External forecast freshness |
| ML | ml_signal | N/A | Model-dependent, skip staleness check |

### Implementation Strategy

For each of the 22 collectors (excluding ml_signal):
1. Add a call to `check_freshness()` at the top of the collector's main function, before any computation
2. If stale, log a warning with the collector name, symbol, actual data age, and threshold
3. Return `{}` immediately

This is a mechanical change across all collector files. The shared helper keeps the per-file diff small (2-3 lines per collector).

### Files to Create/Modify

- New: `src/quantstack/signal_engine/staleness.py`
- Modified: All 22 collector files in `src/quantstack/signal_engine/collectors/` (except `ml_signal.py`)

---

## 4. Data Provider Redundancy — FRED and EDGAR (Item 8.3)

### Problem

Alpha Vantage is the sole data provider for nearly everything. The 14-phase acquisition pipeline (`acquisition_pipeline.py`) calls `AlphaVantageClient` methods directly. If AV goes down or exhausts its rate limit, the system loses macro data, fundamentals, insider data, and more. Only OHLCV has a documented Alpaca fallback.

### Architecture: DataProvider ABC + Provider Registry

Introduce a formal provider abstraction layer. This is the largest structural change in Phase 8.

**Directory structure:**
```
src/quantstack/data/providers/
    __init__.py
    base.py           # DataProvider ABC
    registry.py        # ProviderRegistry (best-source routing)
    alpha_vantage.py   # Wraps existing fetcher.py
    fred.py            # FRED API via fredapi
    edgar.py           # SEC EDGAR via edgartools
```

### DataProvider ABC (`base.py`)

The abstract base class defines methods per data category. Not every provider implements every method — the registry handles routing.

```python
class DataProvider(ABC):
    """Abstract data provider interface.
    
    Only name() is abstract. All fetch methods have default implementations
    that raise NotImplementedError, allowing each provider to implement
    only the data types it supports.
    """
    
    @abstractmethod
    def name(self) -> str: ...
    
    def fetch_ohlcv_daily(self, symbol: str) -> pd.DataFrame | None: ...
    def fetch_macro_indicator(self, indicator: str) -> pd.DataFrame | None: ...
    def fetch_fundamentals(self, symbol: str) -> dict | None: ...
    def fetch_insider_transactions(self, symbol: str) -> pd.DataFrame | None: ...
    def fetch_institutional_holdings(self, symbol: str) -> pd.DataFrame | None: ...
    def fetch_earnings_history(self, symbol: str) -> pd.DataFrame | None: ...
    def fetch_options_chain(self, symbol: str, date: str) -> pd.DataFrame | None: ...
    def fetch_sec_filings(self, symbol: str, form_types: list[str]) -> pd.DataFrame | None: ...
```

**Return value semantics:**
- `NotImplementedError` — provider doesn't support this data type (registry skips to next provider, does NOT count as failure)
- `None` — provider tried but found no data for this symbol (counts as "no data", not a failure)
- Empty DataFrame — provider returned successfully but with no rows
- Populated DataFrame — success

Default implementations of all fetch methods raise `NotImplementedError`. Only `name()` is `@abstractmethod`. This means FRED only needs to implement `fetch_macro_indicator`, EDGAR only implements `fetch_insider_transactions`/`fetch_fundamentals`/etc., and the AV adapter implements everything.

**Provider initialization validation:** Each provider validates its configuration at `__init__` time. If `FRED_API_KEY` is not set, the FRED provider raises `ConfigurationError` immediately — not on the first `fetch()` call. The registry catches `ConfigurationError` during provider registration and logs a warning, excluding misconfigured optional providers from routing.

### Provider Registry (`registry.py`)

The registry maps (data_type) → ordered list of providers. It tries the primary provider first, falls back to secondary on failure, and tracks consecutive failures per provider/data-type pair.

**Best-source routing table** (configured in registry, not hardcoded per call site):

| Data Type | Primary | Fallback(s) |
|-----------|---------|-------------|
| `ohlcv_daily` | AlphaVantage | Alpaca |
| `ohlcv_intraday` | AlphaVantage | Alpaca |
| `macro_indicator` | AlphaVantage | FRED |
| `fundamentals` | AlphaVantage | EDGAR |
| `earnings_history` | AlphaVantage | EDGAR |
| `insider_transactions` | EDGAR | AlphaVantage |
| `institutional_holdings` | EDGAR | AlphaVantage |
| `options_chain` | AlphaVantage | (none) |
| `news_sentiment` | AlphaVantage | (none) |
| `sec_filings` | EDGAR | (none) |
| `commodities` | AlphaVantage | FRED (partial) |

**Initial routing note:** Macro indicators start with AV as primary and FRED as fallback (strangler fig pattern). After FRED proves stable over 2+ weeks of production operation, swap FRED to primary via configuration. This avoids making a new, untested dependency the primary source on day one.

The `fetch()` method on the registry:
1. Look up the provider chain for the requested data type
2. **Circuit breaker check:** If the primary provider has `consecutive_failures >= 3 AND last_failure_at > now() - 10 minutes`, skip directly to fallback (avoids paying timeout penalty on known-broken providers)
3. Call the primary provider (or fallback if circuit breaker tripped)
4. If it returns `None` or raises, log the failure and try next provider
5. Track consecutive failures per provider/data-type in the DB
6. If `consecutive_failures >= 3`: insert alert into `system_events` table
7. On success: reset `consecutive_failures` to 0

### FRED Provider (`fred.py`)

Uses `fredapi` library (pin to specific version). Requires `FRED_API_KEY` environment variable. 120 req/min rate limit (generous — daily macro refresh needs ~15 calls).

**Key series mapping** (FRED series ID → QuantStack `macro_indicators.indicator` name):

| FRED Series | QuantStack Indicator | Notes |
|-------------|---------------------|-------|
| `DGS10` | `TREASURY_YIELD_10Y` | Daily, maps to existing AV indicator |
| `DGS2` | `TREASURY_YIELD_2Y` | Daily |
| `T10Y2Y` | `YIELD_CURVE_SPREAD` | Pre-computed spread |
| `FEDFUNDS` | `FED_FUNDS_RATE` | Monthly effective rate |
| `CPIAUCSL` | `CPI` | Monthly, seasonally adjusted |
| `UNRATE` | `UNEMPLOYMENT` | Monthly |
| `GDP` | `REAL_GDP` | Quarterly |
| `BAMLH0A0HYM2` | `HIGH_YIELD_OAS` | Daily credit spread |
| `ICSA` | `INITIAL_CLAIMS` | Weekly |

The FRED provider normalizes data into the same `(indicator, date, value)` format that AV uses for `macro_indicators`, so existing downstream code (macro collector, cross-asset collector) works unchanged.

### EDGAR Provider (`edgar.py`)

Uses `edgartools` library (pin to specific version; add startup health check that verifies a known CIK resolves). No API key needed. 10 req/sec rate limit. Requires `User-Agent` header with company name and contact email (SEC policy).

**EDGAR MVP milestone:** To unblock Item 8.6 early, implement EDGAR in two phases:
1. **MVP (Days 5-6):** CIK resolution + Form 4 insider parsing. This unblocks 8.6.
2. **Full (Day 7):** XBRL financials + 13F institutional holdings.

**Capabilities:**
- XBRL Company Facts → financial statements (income, balance sheet, cash flow)
- Form 4 filings → insider transactions
- 13F filings → institutional holdings
- Filing metadata → sec_filings table (new, see Item 8.6)

**CIK Resolution:** `edgartools` handles ticker → CIK mapping internally. Cache the mapping to avoid repeated lookups.

**Data normalization:** EDGAR returns data in different schemas than AV. The EDGAR provider normalizes into the existing table schemas:
- Insider transactions → `insider_trades` table format (ticker, transaction_date, owner_name, transaction_type, shares, price_per_share)
- Institutional holdings → `institutional_ownership` table format
- Financial statements → `financial_statements` table format

### Alpha Vantage Provider (`alpha_vantage.py`)

Wraps the existing `AlphaVantageClient` in `fetcher.py` behind the `DataProvider` interface. This is a thin adapter — all existing fetcher methods remain, just called through the ABC interface.

### Failure Tracking and Alerts

**New table: `data_provider_failures`**
```sql
CREATE TABLE data_provider_failures (
    provider VARCHAR NOT NULL,
    data_type VARCHAR NOT NULL,
    consecutive_failures INTEGER DEFAULT 0,
    last_failure_at TIMESTAMP,
    last_error TEXT,
    PRIMARY KEY (provider, data_type)
)
```

On each provider call:
- Success → reset `consecutive_failures` to 0
- Failure → increment, update `last_failure_at` and `last_error`
- If `consecutive_failures >= 3` → insert into existing `alerts` or `system_events` table

The supervisor graph already monitors system health tables, so it will pick up these alerts automatically.

### Integration with Acquisition Pipeline

Modify `acquisition_pipeline.py` to use the provider registry instead of calling `AlphaVantageClient` directly. Each phase calls `registry.fetch(data_type, symbol)` instead of `client.fetch_*(symbol)`.

The existing rate limiting in `AlphaVantageClient` stays — the AV provider wraps it. FRED and EDGAR providers implement their own rate limiting (FRED: 120/min via `fredapi`'s built-in; EDGAR: simple 10/sec throttle).

**Structured observability:** Every registry `fetch()` call logs structured fields: `provider_name`, `data_type`, `symbol`, `latency_ms`, `success` (bool), `fallback_used` (bool), `circuit_breaker_tripped` (bool). This enables querying provider health and latency distribution from logs.

### Environment Variables

```bash
FRED_API_KEY=           # Required for FRED provider (free, register at fred.stlouisfed.org)
EDGAR_USER_AGENT=       # Required by SEC: "CompanyName admin@email.com"
```

### Files to Create/Modify

- New: `src/quantstack/data/providers/__init__.py`
- New: `src/quantstack/data/providers/base.py` — DataProvider ABC
- New: `src/quantstack/data/providers/registry.py` — routing + failure tracking
- New: `src/quantstack/data/providers/alpha_vantage.py` — adapter around existing fetcher
- New: `src/quantstack/data/providers/fred.py` — FRED implementation
- New: `src/quantstack/data/providers/edgar.py` — EDGAR implementation
- Modified: `src/quantstack/data/acquisition_pipeline.py` — use registry
- Modified: `src/quantstack/data/scheduled_refresh.py` — use registry
- Modified: `src/quantstack/data/_schema.py` — add `data_provider_failures` table
- Modified: `.env.example` — add `FRED_API_KEY`, `EDGAR_USER_AGENT`

---

## 5. Drift Detection Pre-Cache (Item 8.4)

### Problem

In `src/quantstack/signal_engine/engine.py` (lines 131-171), the drift check runs after the SignalBrief is synthesized. The brief is then cached with `_cache_put(symbol, brief)` regardless of drift severity. A CRITICAL drift sets `brief.drift_warning = True` but the brief still sits in cache at full confidence for up to 1 hour.

### Approach

Reorder the code in `engine.py` so that drift detection runs before the cache write. Based on drift severity, adjust both the confidence and the cache TTL:

**Drift → cache behavior mapping:**
| Drift Severity | Confidence Penalty | Cache TTL | Event |
|---------------|-------------------|-----------|-------|
| NONE | 0 | Default (3600s) | None |
| WARNING | -0.10 | Half (1800s) | Log only |
| CRITICAL | -0.30 | Short (300s / 5 min) | Insert `DRIFT_CRITICAL` into `system_events` |

### Cache TTL Override

The current `cache.put(symbol, brief)` uses a fixed TTL. The underlying `TTLCache` in `src/quantstack/shared/cache.py` stores entries as `(value, timestamp)` tuples with a single `self._ttl`. To support per-entry TTL:

1. Modify `TTLCache` in `shared/cache.py` to store `(value, timestamp, entry_ttl)` tuples
2. `set()` gains an optional `ttl` parameter; when `None`, falls back to `self._ttl`
3. `get()` checks expiry against the entry's own TTL
4. This is backward-compatible — all existing consumers that don't pass `ttl` get the same behavior

**Blast radius note:** `TTLCache` is shared infrastructure (also used by IC output cache). The change is additive (new optional parameter) but add a regression test verifying existing consumers aren't affected.

Then `signal_engine/cache.py`'s `put()` passes the TTL through:

```python
def put(self, symbol: str, brief: SignalBrief, ttl: int | None = None) -> None:
    """Store brief with optional per-entry TTL override."""
```

### Engine Code Flow (Modified)

The modified flow in `engine.py`:
1. Run all 22 collectors concurrently (unchanged)
2. Synthesize SignalBrief from collector outputs (unchanged)
3. **Run drift detection** on the synthesized brief
4. **Apply confidence penalty** based on drift severity
5. **Determine cache TTL** based on drift severity
6. Cache the brief with the determined TTL
7. Return the brief

### DRIFT_CRITICAL Event

When drift severity is CRITICAL, insert an event that the supervisor graph can detect:

```sql
INSERT INTO system_events (event_type, symbol, severity, details, created_at)
VALUES ('DRIFT_CRITICAL', :symbol, 'critical', :drift_report_json, NOW())
```

The supervisor graph already queries `system_events` for actionable items. No additional wiring needed.

### Files to Modify

- `src/quantstack/shared/cache.py` — add per-entry TTL support to `TTLCache`
- `src/quantstack/signal_engine/engine.py` — reorder drift check, apply confidence penalty, pass TTL
- `src/quantstack/signal_engine/cache.py` — pass `ttl` parameter through to TTLCache

---

## 6. SEC Filings Population (Item 8.6)

### Problem

The `sec_filings` table (if it exists) and `insider_trades` table are empty. The EDGAR provider built in Item 8.3 gives us the fetching capability. This item wires it into the acquisition pipeline.

### Approach

Add two new acquisition phases to the pipeline:

**Phase 15: SEC Filings** — Fetch 10-K, 10-Q, and 8-K filing metadata for all universe symbols. Store in a new `sec_filings` table. This provides filing dates, accession numbers, and document URLs for downstream analysis.

**Phase 16: EDGAR Insider/Institutional** — Fetch Form 4 (insider trading) and 13F (institutional holdings) data from EDGAR. Normalize and upsert into existing `insider_trades` and `institutional_ownership` tables.

### New Table: sec_filings

```python
@dataclass
class SECFiling:
    accession_number: str    # PK
    symbol: str
    form_type: str           # '10-K', '10-Q', '8-K'
    filing_date: date
    period_of_report: date | None
    primary_doc_url: str | None
    fetched_at: datetime
```

### Data Normalization

EDGAR returns different field names than what the existing tables expect. The EDGAR provider (`providers/edgar.py`) handles the mapping:

- EDGAR `transactionDate` → `insider_trades.transaction_date`
- EDGAR `ownerName` → `insider_trades.owner_name`
- EDGAR `transactionShares` → `insider_trades.shares`
- EDGAR `pricePerShare` → `insider_trades.price_per_share`
- EDGAR `isDirector`/`isOfficer`/`isTenPercentOwner` → derive `transaction_type` as 'buy' or 'sell' based on `transactionAcquiredDisposedCode`

Upsert into existing tables using `ON CONFLICT DO UPDATE` to maintain idempotency.

### Freshness Checks

- SEC filings: Fetch if no filing exists for the current quarter (check `filing_date > 90 days ago`)
- Insider transactions (Form 4): Fetch if `fetched_at` is older than 7 days
- Institutional holdings (13F): Fetch if `fetched_at` is older than 90 days (quarterly filings)

### Rate Limiting

EDGAR allows 10 req/sec. With 50 universe symbols:
- SEC filings: ~50 calls (1 per symbol) → 5 seconds
- Form 4: ~50 calls → 5 seconds
- 13F: ~50 calls → 5 seconds
- Total: ~15 seconds for the full universe

### Files to Create/Modify

- Modified: `src/quantstack/data/_schema.py` — add `sec_filings` table DDL
- Modified: `src/quantstack/data/acquisition_pipeline.py` — add phases 15 and 16
- The EDGAR provider (`providers/edgar.py`) is created in Item 8.3

---

## 7. OHLCV Partitioning (Item 8.7)

### Problem

The `ohlcv` table has 7.6M rows with composite PK `(symbol, timeframe, timestamp)`. Distribution: 6M rows are 5-minute bars (79%), 1.2M hourly (16%), 400K daily (5%). Growing by ~100K rows/day. No partitioning means every query scans the full index.

### Approach: Monthly Range Partitioning

Partition by `timestamp` using monthly ranges. This is the natural choice because:
1. Most queries filter by time range (`WHERE timestamp >= X`)
2. The partition key (`timestamp`) is already in the PK, satisfying PostgreSQL's constraint
3. Monthly granularity yields ~36 partitions for 3 years of data — well within the planning efficiency sweet spot

### Partition Layout

```
ohlcv (parent, partitioned)
├── ohlcv_2024_01  (Jan 2024)
├── ohlcv_2024_02  (Feb 2024)
├── ...
├── ohlcv_2026_04  (Apr 2026, current)
├── ohlcv_2026_05  (May 2026, pre-created)
├── ohlcv_2026_06  (Jun 2026, pre-created)
├── ohlcv_2026_07  (Jul 2026, pre-created)
├── ohlcv_2026_08  (Aug 2026, pre-created)
└── ohlcv_default  (catch-all for out-of-range data)
```

Pre-create 4 months of future partitions. Use a startup hook in `db.py` to create the next month's partition if it doesn't exist (4 lines of SQL). No external extension dependency.

### Migration Strategy (Expand-Contract, Weekend Window)

The migration runs during a weekend maintenance window with Docker services stopped.

**Migration script (`scripts/migrations/partition_ohlcv.py`):**

1. **Pre-flight:** Verify row count, take a full backup, validate no active connections
2. **Create new partitioned table:** Same schema as existing `ohlcv`, declared as `PARTITION BY RANGE (timestamp)`
3. **Create monthly partitions:** One per month from the earliest data to 4 months ahead
4. **Migrate data in monthly batches:** `INSERT INTO ohlcv_new SELECT * FROM ohlcv WHERE timestamp >= X AND timestamp < Y` — commit after each month to avoid long transactions
5. **Verify:** Compare row counts between old and new tables
6. **Atomic swap:** `ALTER TABLE ohlcv RENAME TO ohlcv_old; ALTER TABLE ohlcv_new RENAME TO ohlcv;`
7. **Create indexes:** Recreate any indexes on the new partitioned table (they auto-propagate to partitions)
8. **Validate:** Run sample queries, verify EXPLAIN shows partition pruning
9. **Cleanup:** Drop `ohlcv_old` after validation period (keep for 1 week)

**Estimated time:** 7.6M rows migrated in monthly batches → ~5-10 minutes total for the data copy. The full process including verification takes ~15-20 minutes.

### Partition Maintenance

No pg_partman dependency. Instead, add a lightweight `ensure_ohlcv_partitions()` function to `db.py` that:
1. Checks if the next 4 months of partitions exist
2. Creates any missing ones
3. Called during application startup (idempotent)

This is ~10 lines of SQL and eliminates an external extension dependency.

### Application Code Impact

**Zero application code changes required.** The partitioned table has the same name, schema, and PK as the original. All existing queries work unchanged. The only difference is PostgreSQL's internal query planner now prunes irrelevant partitions.

### Rollback

If the migration fails:
1. `ALTER TABLE ohlcv RENAME TO ohlcv_failed; ALTER TABLE ohlcv_old RENAME TO ohlcv;`
2. Restart Docker services
3. Investigate and fix before retrying

### Files to Create/Modify

- New: `scripts/migrations/partition_ohlcv.py` — full migration script
- Modified: `src/quantstack/data/_schema.py` — update DDL to create partitioned table (for fresh installs)
- Possibly modified: `src/quantstack/db.py` — add partition maintenance utility

---

## 8. Options Refresh Expansion (Item 8.8)

### Problem

The audit flagged that options data was refreshed only for actively held positions. Research shows the current implementation already refreshes for watched symbols + top 30 universe symbols. However, the cap of 30 is hardcoded and may miss important symbols during high-activity periods.

### Approach

Make the options refresh scope configurable and add pre-trade refresh capability:

1. **Configurable top-N:** Extract the hardcoded `30` into an environment variable `OPTIONS_REFRESH_TOP_N` (default: 30).
2. **Strategy-aware refresh:** Add symbols from active strategies that use options-related signals (options_flow, put_call_ratio collectors) to the refresh list.
3. **Pre-trade refresh:** When a symbol enters the trading pipeline (identified by the entry scanner), trigger an options refresh if the data is older than the current trading day.

### Configuration

```bash
OPTIONS_REFRESH_TOP_N=30    # Number of top universe symbols to refresh (default: 30)
```

### Pre-Trade Refresh Flow

The trading graph's entry scanner evaluates symbols before generating trade ideas. Before options-dependent analysis, check if the symbol has today's options chain. If not, fetch it inline (with a timeout to avoid blocking the trading pipeline).

This is a lightweight addition to the existing entry scanning node — not a separate background process.

### Files to Modify

- `src/quantstack/data/scheduled_refresh.py` — extract hardcoded 30, add strategy-aware symbol selection
- Possibly: trading graph entry scanner node — add pre-trade options freshness check

---

## 9. Execution Order and Dependencies

```
Week 1 (parallel start):
  ├── 8.1 Signal Cache Auto-Invalidation (1 day)
  ├── 8.2 Staleness Rejection (2 days)
  ├── 8.4 Drift Detection Pre-Cache (1 day) — no dependency on 8.1, different files
  └── 8.3 Provider Abstraction + FRED + EDGAR (5-7 days)
       ├── Days 1-2: DataProvider ABC, registry, AV adapter
       ├── Days 3-4: FRED provider + macro indicator mapping
       ├── Days 5-6: EDGAR MVP (CIK + Form 4) — unblocks 8.6
       └── Day 7: EDGAR full (XBRL financials + 13F)

Week 2:
  ├── 8.6 SEC Filings Population (2 days, after EDGAR MVP)
  └── 8.8 Options Refresh Expansion (1 day)

Weekend 2:
  └── 8.7 OHLCV Partitioning (migration during weekend window)

Week 3:
  └── Integration testing + validation plan execution
```

### Critical Path

8.3 (Provider Redundancy) is the critical path at 5-7 days. By parallelizing 8.1, 8.2, and 8.4 in Week 1, and defining an EDGAR MVP milestone to unblock 8.6 early, we maximize throughput. The EDGAR MVP (CIK + Form 4 only) ships by Day 6, enabling 8.6 to start while full EDGAR (XBRL/13F) continues.

---

## 10. Validation Plan

Each item has specific acceptance tests (detailed in the TDD plan). The high-level validation flow:

1. **Cache invalidation (8.1):** Refresh AAPL data → verify next signal request triggers cache miss → new SignalBrief generated with fresh data.
2. **Staleness (8.2):** Set AAPL OHLCV `last_timestamp` to 5 days ago → verify technical collector returns `{}` → verify synthesis redistributes weight.
3. **AV redundancy (8.3):** Block AV API (mock failure) → verify FRED returns macro data → verify EDGAR returns fundamentals and insider data → verify alert fires after 3 failures.
4. **Drift detection (8.4):** Inject CRITICAL drift data → verify brief cached with 5-min TTL → verify confidence penalized by 0.30 → verify `DRIFT_CRITICAL` event in system_events.
5. **SEC filings (8.6):** Run EDGAR acquisition phase → verify sec_filings populated → verify insider_trades populated from Form 4.
6. **OHLCV partitioning (8.7):** Run migration on backup → verify row counts match → verify EXPLAIN shows partition pruning → verify sample queries return correct results.
7. **Options refresh (8.8):** Set `OPTIONS_REFRESH_TOP_N=50` → verify 50 symbols refreshed → verify pre-trade refresh triggers for stale options data.
8. **All-collectors-stale (8.2 edge case):** Set ALL data sources to stale → verify synthesis handles zero-contributor case gracefully (returns low-confidence brief or explicit "no data" response, not division-by-zero).
9. **Staleness performance benchmark (8.2):** Measure signal engine latency before and after staleness checks across 50 symbols. Verify overhead < 2s.

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| EDGAR XBRL gaps for older filings | Medium | Low | Skip pre-2009 filings. XBRL coverage is good for recent years. |
| FRED API key issues | Low | Medium | Free registration. 120 req/min is far above our needs (~15 calls/day). |
| edgartools library breaks (SEC endpoint changes) | Medium | Medium | Pin version. Startup health check. Fall back to direct SEC HTTP calls for critical Form 4 data. |
| Staleness rejection reduces signal coverage | Expected | Acceptable | By design. `{}` is better than confident-but-wrong. Synthesis handles it. |
| Provider registry adds debugging complexity | Low | Medium | Comprehensive per-call logging. Provider health metrics in supervisor. |
| OHLCV migration data corruption | Low | Critical | Full backup before migration. Row count validation. 1-week rollback window. |
| Rate limit contention between AV + FRED + EDGAR | Low | Low | Different APIs with independent rate limits. No shared quota. |
