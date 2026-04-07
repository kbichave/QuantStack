# Synthesized Specification: Phase 8 — Data Pipeline Hardening

## Overview

Phase 8 hardens QuantStack's data pipeline against silent degradation in 24/7 operation. The signal engine (22 concurrent collectors, 2-6s wall-clock, fault-tolerant) is the system's crown jewel but has several single points of failure: stale cache serving, no per-collector freshness checks, Alpha Vantage as sole provider for 12 of 14 acquisition phases, and a 7.6M-row OHLCV table with no partitioning.

**Scope:** 7 items (8.1-8.4, 8.6-8.8). Item 8.5 (Web Search) excluded per stakeholder decision.

**Gate criteria:** Signal cache fresh. Data providers redundant. Intel sources live.

---

## Item 8.1: Signal Cache Auto-Invalidation

**Problem:** Signal cache has 1-hour TTL (`SIGNAL_ENGINE_CACHE_TTL=3600`). Intraday data refreshes every 5 minutes via `scheduled_refresh.py`. Trading decisions could use a 55-minute-old SignalBrief while fresh data sits in DB unused. `cache.invalidate(symbol)` exists in `src/quantstack/signal_engine/cache.py` but is never called after data refresh.

**Solution:** Hook `cache.invalidate(symbol)` into `scheduled_refresh.py` after each intraday refresh cycle completes for a symbol. After EOD refresh, call `cache.clear()` (already happens).

**Key files:**
- `src/quantstack/data/scheduled_refresh.py` — add invalidation calls after data writes
- `src/quantstack/signal_engine/cache.py` — existing `invalidate(symbol)` and `clear()` methods

**Acceptance criteria:**
- Data refresh triggers signal cache invalidation for affected symbols
- No signal brief older than most recent data refresh is served
- Cache stats logged for observability

---

## Item 8.2: Staleness Rejection in Collectors

**Problem:** Data validator flags stale data, but collectors don't check freshness before computing. A collector can compute RSI on 3-week-old data and return a confident signal.

**Architecture decision:** Per-collector decorator pattern. Each collector declares its own `max_staleness` threshold. A shared helper function validates freshness using `data_metadata.last_timestamp`.

**Staleness thresholds (from spec):**
| Collector Type | Max Staleness |
|---------------|---------------|
| Technical, Volume | 2 trading days |
| Options Flow, Put/Call Ratio | 3 days |
| Sentiment, Social Sentiment, News | 7 days |
| Fundamentals, Quality, Earnings Momentum | 90 days |
| Macro, Commodity, Cross-Asset | 45 days |
| Insider Signals, Flow (institutional) | 30 days |
| Short Interest | 14 days |
| Regime | 2 trading days (uses OHLCV) |
| ML Signal | No staleness check (model-dependent) |
| StatArb | 2 trading days |
| Sector | 7 days |
| Events | 30 days |
| EWF | 7 days (age_hours field already exists) |

**Implementation pattern:**
```python
# Shared helper in signal_engine/staleness.py
def check_freshness(store: DataStore, symbol: str, timeframe: str, max_days: int) -> bool:
    """Returns True if data is fresh enough to use."""
    meta = store.get_metadata(symbol, timeframe)
    if meta is None or meta.last_timestamp is None:
        return False
    age = (now() - meta.last_timestamp).days
    return age <= max_days

# Per-collector usage
async def collect_technical(symbol: str, store: DataStore) -> dict:
    if not check_freshness(store, symbol, "D1", max_days=2):
        logger.warning(f"[technical] {symbol}: stale data, skipping")
        return {}
    # ... existing logic
```

**Key files:**
- New: `src/quantstack/signal_engine/staleness.py` — shared freshness helper
- All 22 collectors in `src/quantstack/signal_engine/collectors/`

**Acceptance criteria:**
- Every collector checks data freshness before computing
- Stale collectors return `{}` (synthesis redistributes weight automatically)
- Log when collectors skip due to stale data (with symbol, collector name, data age)

---

## Item 8.3: Data Provider Redundancy (FRED, EDGAR)

**Problem:** AV is sole source for options chains, earnings, macro indicators, news sentiment, fundamentals, insider/institutional data. Only OHLCV has fallback (Alpaca IEX). If AV goes down, 12 of 14 acquisition phases fail silently.

**Architecture decision:** ABC DataProvider interface with a provider registry. Best-source routing per data type — use the most authoritative source for each data category, with fallback to alternatives.

**Provider routing:**

| Data Type | Primary Source | Fallback | Rationale |
|-----------|---------------|----------|-----------|
| OHLCV (daily, intraday) | Alpha Vantage | Alpaca IEX | AV has full history; Alpaca for intraday |
| Macro indicators (GDP, yields, CPI, unemployment) | **FRED** | Alpha Vantage | FRED is the authoritative source (Federal Reserve data) |
| Fundamentals (income stmt, balance sheet, cash flow) | Alpha Vantage | **SEC EDGAR (XBRL)** | AV has cleaner API; EDGAR for redundancy |
| Earnings history | Alpha Vantage | SEC EDGAR | AV has structured EPS data |
| Insider transactions | **SEC EDGAR (Form 4)** | Alpha Vantage | EDGAR is the authoritative source (SEC filings) |
| Institutional holdings (13F) | **SEC EDGAR** | Alpha Vantage | EDGAR is the authoritative source |
| Company overview | Alpha Vantage | SEC EDGAR (XBRL) | AV has consolidated profile |
| SEC filings (10-K, 10-Q, 8-K) | **SEC EDGAR** | N/A (new capability) | Only available from SEC |
| Options chains | Alpha Vantage | N/A (future: CBOE/Polygon) | No free alternative yet |
| News sentiment | Alpha Vantage | N/A | AV-specific scoring |
| Commodities, Forex | Alpha Vantage | **FRED** | FRED has gold, copper; AV has broader set |

**New dependencies:**
- `fredapi` — Python FRED client (free API key required, 120 req/min)
- `edgartools` — Python SEC EDGAR client (no API key, 10 req/sec, User-Agent required)

**DataProvider ABC:**
```python
class DataProvider(ABC):
    @abstractmethod
    def fetch_macro_indicators(self, indicators: list[str]) -> dict[str, pd.DataFrame]: ...
    @abstractmethod
    def fetch_fundamentals(self, symbol: str) -> dict: ...
    @abstractmethod
    def fetch_insider_transactions(self, symbol: str) -> pd.DataFrame: ...
    # ... per data type
```

**Alert on failures:** Track consecutive failures per provider per data type in `system_state` table. Insert alert into `alerts` table on 3+ consecutive failures. Supervisor graph monitors alerts table and triggers self-healing.

**Key files:**
- New: `src/quantstack/data/providers/base.py` — DataProvider ABC
- New: `src/quantstack/data/providers/alpha_vantage.py` — wraps existing fetcher.py
- New: `src/quantstack/data/providers/fred.py` — FRED implementation
- New: `src/quantstack/data/providers/edgar.py` — SEC EDGAR implementation
- New: `src/quantstack/data/providers/registry.py` — best-source routing logic
- Modified: `src/quantstack/data/acquisition_pipeline.py` — use provider registry
- Modified: `src/quantstack/data/scheduled_refresh.py` — use provider registry
- New: DB table `data_provider_failures` for consecutive failure tracking

**Acceptance criteria:**
- FRED API configured for macro indicators (yields, GDP, CPI, unemployment, etc.)
- SEC EDGAR configured for fundamentals (XBRL) and insider data (Form 4)
- EDGAR data normalized into existing tables (insider_trades, institutional_ownership, financial_statements)
- Alert fires on 3+ consecutive failures for any provider/data-type pair
- System continues operating when AV is down (degraded but not dead)

---

## Item 8.4: Drift Detection Pre-Cache

**Problem:** PSI drift check runs AFTER brief is synthesized and cached (engine.py lines 131-171). CRITICAL drift sets `drift_warning=True` but brief is still cached with full TTL and no confidence penalty.

**Architecture decision:** Cache with short TTL (5-10 min) + degraded confidence (-0.30) on CRITICAL drift.

**Solution:**
1. Move drift check to BEFORE `_cache_put()` in engine.py
2. On CRITICAL drift:
   - Penalize overall confidence by 0.30
   - Cache with reduced TTL (5 minutes instead of default 3600s)
   - Publish `DRIFT_CRITICAL` event to `system_events` table
   - Set `brief.drift_warning = True` (existing)
3. On WARNING drift:
   - Penalize confidence by 0.10
   - Cache with half TTL (30 minutes)
   - Log warning but no event

**Key files:**
- `src/quantstack/signal_engine/engine.py` — reorder drift check before cache write
- `src/quantstack/signal_engine/cache.py` — support per-entry TTL override

**Acceptance criteria:**
- Drift check runs before caching
- CRITICAL drift → 5-min TTL, -0.30 confidence, DRIFT_CRITICAL event
- WARNING drift → 30-min TTL, -0.10 confidence
- Event published for supervisor awareness

---

## Item 8.6: SEC Filings Population

**Problem:** SEC filings table exists but never populated. Insider trading table exists but empty (from AV). No earnings transcripts or analyst ratings tools.

**Solution:** Use `edgartools` library to populate from SEC EDGAR (free, 10 req/sec):
- **10-K, 10-Q, 8-K filings:** New `sec_filings` table (or populate existing if schema matches)
- **Form 4 insider trading:** Normalize into existing `insider_trades` table
- **Institutional holdings (13F):** Normalize into existing `institutional_ownership` table

**Schema normalization:** EDGAR data maps to existing column schemas. Key mapping:
- EDGAR `transactionDate` → `insider_trades.transaction_date`
- EDGAR `transactionShares` → `insider_trades.shares`
- EDGAR `pricePerShare` → `insider_trades.price_per_share`
- EDGAR `isDirector/isOfficer/isTenPercentOwner` → derive `transaction_type`

**SEC filings table (new):**
```sql
CREATE TABLE sec_filings (
    accession_number VARCHAR NOT NULL PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    form_type VARCHAR NOT NULL,  -- '10-K', '10-Q', '8-K'
    filing_date DATE NOT NULL,
    period_of_report DATE,
    primary_doc_url VARCHAR,
    summary TEXT,  -- extracted key sections
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Key files:**
- `src/quantstack/data/providers/edgar.py` — EDGAR fetcher (shared with 8.3)
- `src/quantstack/data/_schema.py` — add sec_filings table
- `src/quantstack/data/acquisition_pipeline.py` — add SEC filings phase

**Acceptance criteria:**
- SEC filings (10-K, 10-Q, 8-K) populated for all universe symbols
- Insider trading data populated from Form 4 into existing table
- Data normalized to existing schema (single source of truth)

---

## Item 8.7: OHLCV Partitioning

**Problem:** `ohlcv` table has 7.6M rows with composite PK `(symbol, timeframe, timestamp)`. No partitioning. Distribution: 79% 5-min bars, 16% hourly, 5% daily. Growing by ~100K rows/day with 50 symbols.

**Solution:** Monthly range partitioning on `timestamp` using PostgreSQL native partitioning + pg_partman for automated partition management.

**Migration strategy (expand-contract):**
1. Create new partitioned table with identical schema
2. Create monthly partitions covering all existing data (oldest to current + 4 months ahead)
3. Migrate data in monthly batches (COMMIT between batches)
4. Verify row counts match
5. Atomic rename swap during **weekend maintenance window** (Docker services stopped)
6. Drop old table after validation

**pg_partman config:**
```sql
SELECT partman.create_parent(
    p_parent_table := 'public.ohlcv',
    p_control := 'timestamp',
    p_interval := '1 month',
    p_premake := 4,
    p_default_table := true
);
```

**Key files:**
- New: migration script `scripts/migrations/partition_ohlcv.py`
- `src/quantstack/data/_schema.py` — update DDL for partitioned table
- `docker-compose.yml` — may need pg_partman extension

**Acceptance criteria:**
- OHLCV table partitioned by month
- Query performance validated (EXPLAIN shows partition pruning)
- Migration script tested on backup first
- Rollback procedure documented

---

## Item 8.8: Options Refresh Expansion

**Problem (per audit):** Options data refreshed only for actively held positions.

**Current state (from research):** Already broader than audit states — EOD refresh fetches options for watched symbols + top 30 universe symbols. However, the cap of 30 may miss important symbols during high-activity periods.

**Solution:** Make the refresh scope configurable:
- Default: watched symbols + top N universe symbols (configurable via `OPTIONS_REFRESH_TOP_N`, default 30)
- Add watchlist-based refresh: symbols from active strategies that use options signals
- Pre-trade scanning: refresh options for any symbol entering the trading pipeline

**Key files:**
- `src/quantstack/data/scheduled_refresh.py` — configurable options refresh scope

**Acceptance criteria:**
- Options data refreshed for configurable number of universe symbols
- Pre-trade options analysis has fresh data for any symbol under evaluation
- Rate limit budget respected (options calls are expensive)

---

## Dependencies & Ordering

```
8.1 (Cache Invalidation) ──────────────── Independent, start first (1 day)
8.2 (Staleness Rejection) ─────────────── Independent, start first (2 days)
8.3 (Provider Redundancy) ─────────────── Independent, longest item (5-7 days)
  └─ 8.6 (SEC Filings) ────────────────── Depends on 8.3's EDGAR provider (2 days)
8.4 (Drift Pre-Cache) ─────────────────── Depends on 8.1 (cache changes) (1 day)
8.7 (OHLCV Partitioning) ──────────────── Independent, weekend migration (2 days)
8.8 (Options Refresh) ─────────────────── Independent (1 day)
```

**Recommended execution order:**
1. **Week 1:** 8.1 + 8.2 + 8.3 (parallel start)
2. **Week 2:** 8.3 continued + 8.4 + 8.6 (after EDGAR provider ready)
3. **Weekend 2:** 8.7 (migration during weekend window)
4. **Week 3:** 8.8 + integration testing + validation

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| EDGAR HTML parsing complexity for older filings | 8.6 delays | Start with XBRL (structured, machine-readable). Skip pre-2009 filings. |
| FRED API key quota exceeded | Macro data gaps | 120 req/min is generous; daily macro refresh needs ~15 calls. Not a real risk. |
| pg_partman not available in Docker PostgreSQL | 8.7 blocked | Use native partitioning DDL without pg_partman. Manual partition management. |
| Staleness rejection causes signal gaps | Reduced signal coverage | `{}` return is correct — synthesis redistributes weight. Better than fake confidence on stale data. |
| Provider registry adds complexity | Harder debugging | Comprehensive logging per provider call. Provider health dashboard in supervisor. |
| OHLCV migration data loss | Catastrophic | Backup before migration. Validate row counts. Keep old table until confidence period passes. |
