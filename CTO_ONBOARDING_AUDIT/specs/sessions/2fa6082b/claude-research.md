# Research Findings: Phase 8 Data Pipeline Hardening

---

## Part 1: Codebase Research

### 1. Signal Engine & Collectors

**Location:** `src/quantstack/signal_engine/`

The signal engine has **22 collectors** (not 16 as the audit states), all in `src/quantstack/signal_engine/collectors/`:

| # | File | Domain |
|---|------|--------|
| 1 | technical.py | Momentum (MACD, RSI), volatility (BB, ATR), advanced (Supertrend, Ichimoku, HMA) |
| 2 | regime.py | HMM-primary regime classifier with rule-based fallback |
| 3 | volume.py | Volume profile, OBV, A/D |
| 4 | risk.py | VaR, stop loss levels, liquidity risk |
| 5 | events.py | Earnings, splits, dividends |
| 6 | fundamentals.py | P/E, ROE, debt/equity, growth metrics |
| 7 | sentiment.py | News sentiment (deprecated) |
| 8 | sentiment_alphavantage.py | AV news + Groq LLM reasoning |
| 9 | macro.py | Fed funds, yields, unemployment, inflation, VIX |
| 10 | sector.py | Sector momentum, rotation signals |
| 11 | flow.py | Institutional/insider flows |
| 12 | cross_asset.py | Risk-on/off via SPY/VIX, bonds, commodities |
| 13 | quality.py | Earnings quality, accrual ratio, cash conversion |
| 14 | ml_signal.py | Trained model predictions |
| 15 | statarb.py | Pairs trading signals |
| 16 | options_flow.py | Dealer positioning: GEX, gamma flip, DEX, max pain |
| 17 | options_flow_collector.py | Async wrapper around options_flow |
| 18 | social_sentiment.py | Reddit + Stocktwits |
| 19 | insider_signals.py | Insider transaction clustering |
| 20 | short_interest.py | Short float %, short squeeze metrics |
| 21 | put_call_ratio.py | PCR trend, contrarian signals |
| 22 | earnings_momentum.py | Consecutive beats/misses, PEAD drift |
| 23 | commodity.py | Gold/silver ratio, copper/gold, USD strength |
| 24 | ewf_collector.py | Elliott Wave Forecast |

**Collector Architecture:**
- All async-wrapped sync functions (`asyncio.to_thread`)
- Return empty dict `{}` on failure (graceful degradation)
- Read from local DataStore (no API calls during collection)
- Per-collector 10-second timeout in SignalEngine (`engine.py` lines 73, 250)
- No base class — each collector is standalone
- 22 collectors run concurrently via `asyncio.gather()`

**Existing Freshness Checks (minimal):**
- `technical.py:82-88`: Checks if daily_df has >= 60 bars
- `regime.py:68-75`: Returns "unknown" regime if < 60 bars
- **No staleness checks based on `data_metadata.last_timestamp`** — this is the gap item 8.2 addresses

### 2. Signal Cache

**Location:** `src/quantstack/signal_engine/cache.py`

```python
_ttl = int(os.environ.get("SIGNAL_ENGINE_CACHE_TTL", "3600"))  # 1 hour default
_enabled = os.environ.get("SIGNAL_ENGINE_CACHE_ENABLED", "true").lower() == "true"
_cache = TTLCache(ttl_seconds=_ttl)
```

**API:**
- `get(symbol)` — returns SignalBrief if fresh
- `put(symbol, brief)` — stores with TTL
- `invalidate(symbol)` — explicit removal
- `clear()` — wipe entire cache
- `stats()` — hits/misses/size

**Usage in engine.py (lines 108-117, 179):**
```python
cached = _cache_get(symbol)
if cached is not None:
    return cached
# ... run collectors ...
_cache_put(symbol, brief)
```

**Key finding:** `cache.invalidate(symbol)` exists but is **never called after data refresh** — confirming item 8.1. `cache.clear()` is called in EOD refresh only.

### 3. Data Pipeline & Scheduled Refresh

**Files:**
- `src/quantstack/data/scheduled_refresh.py` — intraday (5min) + EOD
- `src/quantstack/data/acquisition_pipeline.py` — 14-phase historical acquisition
- `src/quantstack/data/fetcher.py` — AlphaVantageClient

**Intraday Refresh (every 5 min during market hours):**
- Bulk quotes for full universe (100 symbols/call)
- 5-min OHLCV for watched symbols (positions + active strategies, capped at 20)
- News sentiment for top 10 symbols (batched 5/call)

**EOD Refresh (after market close):**
- Daily OHLCV for full universe
- Options chains for watched + top 30 universe symbols
- Company overview for stale symbols (>7 days)
- Earnings calendar (global, 1 call)

**14 Acquisition Phases:**
1. ohlcv_5min, 2. ohlcv_1h, 3. ohlcv_daily, 4. financials, 5. earnings_history, 6. macro, 7. insider, 8. institutional, 9. corporate_actions, 10. options, 11. news, 12. fundamentals, 13. commodities, 14. put_call_ratio

**Rate Limiting:** Sliding window in AlphaVantageClient, 75 req/min premium, priority queue (low/normal/critical), daily quota gate at 50%/80%/100%.

**Provider Architecture:** AlphaVantageClient is the **only** provider. No formal provider abstraction layer. Alpaca fallback mentioned in docstring for OHLCV only.

### 4. Database Schema

**Location:** `src/quantstack/data/_schema.py`

**OHLCV table (the partitioning target):**
```sql
CREATE TABLE ohlcv (
    symbol VARCHAR NOT NULL,
    timeframe VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, timeframe, timestamp)
)
```
- Composite PK: (symbol, timeframe, timestamp)
- Timeframes: "D1", "W1", "M1", "M5", "H1"
- **No partitioning currently**

**data_metadata (freshness tracking):**
```sql
CREATE TABLE data_metadata (
    symbol VARCHAR NOT NULL,
    timeframe VARCHAR NOT NULL,
    first_timestamp TIMESTAMP,
    last_timestamp TIMESTAMP,
    row_count INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, timeframe)
)
```

**Other key tables:** ohlcv_1m, tick_data, options_chains, earnings_calendar, company_overview, financial_statements, insider_trades, institutional_ownership, corporate_actions, news_sentiment, macro_indicators, put_call_ratio

### 5. Drift Detection

**Location:** `src/quantstack/learning/drift_detector.py`

- Two-sample Kolmogorov-Smirnov test (< 1ms, pure numpy)
- Severity: NONE (< 0.10), WARNING (0.10-0.25), CRITICAL (>= 0.25)
- Tracked features: rsi_14, atr_pct, adx_14, bb_pct, volume_ratio, regime_confidence

**Integration with SignalEngine (engine.py lines 131-171):**
```python
drift_report = DriftDetector().check_drift_from_brief(strategy_id=symbol, brief=brief.model_dump())
if drift_report.severity == "CRITICAL":
    brief.drift_warning = True
```

**Key finding for item 8.4:** Drift check runs AFTER brief is synthesized. The `_cache_put(symbol, brief)` call happens after drift check, but the brief is **still cached even with CRITICAL drift** — only `drift_warning=True` flag is set. No confidence penalty, no cache skip.

### 6. Testing Setup

**Framework:** pytest 9.0+

**Structure:**
```
tests/
├── unit/
│   ├── conftest.py              # Shared fixtures (OHLCV generators, settings mocks)
│   ├── signal_engine/           # Collector-level tests (sentiment, insider, short_interest)
│   ├── calibration/
│   ├── execution/
│   ├── graphs/trading/
│   ├── ml/
│   └── ~150+ test files
```

**Key fixtures in conftest.py:**
- OHLCV generators: `make_ohlcv_df`, `make_monotonic_uptrend/downtrend`, `make_flat_market`, etc.
- Settings mocks: `mock_settings`, `patch_get_settings`
- Async support: `run_async` fixture

**Testing gaps relevant to Phase 8:**
- Limited collector-level tests (only 3 of 22+ collectors)
- No integration tests for full signal pipeline
- No tests for drift detector's cache interaction
- No tests for cache invalidation flows

### 7. Web Search / Market Intel

**Location:** `src/quantstack/tools/langchain/intelligence_tools.py`

```python
@tool
async def web_search(query: ...) -> str:
    return json.dumps({
        "error": "Web search not configured — set SEARCH_API_KEY in .env",
        "query": query, "results": [],
    })
```

**Status:** Stub implementation. Not wired to any search API. `market_intel` agent falls back to stale training data.

### 8. Options Data Refresh

**EOD Refresh (scheduled_refresh.py lines 214-230):**
- Refreshes for watched symbols + top 30 universe symbols (capped at 30 total)
- Full chain snapshot (all strikes/expiries for current date)
- Not position-only — already broader than the audit suggests

**Acquisition Pipeline Phase 10:** Historical options chain per symbol, idempotent (skips if already have today's chain).

---

## Part 2: Web Research

### FRED API Integration (Python)

**Overview:** FRED provides 800,000+ economic time series. Free API key required.

**Rate Limits:** 120 requests/minute per API key. No hard daily limit.

**Recommended Library:** `fredapi` (v0.5.2, 1.1k GitHub stars, pandas-native)

```python
from fredapi import Fred
fred = Fred(api_key=os.environ["FRED_API_KEY"])
gdp = fred.get_series("GDP")
# Point-in-time data (avoids look-ahead bias)
cpi_at_date = fred.get_series_as_of_date("CPIAUCSL", "2024-01-15")
```

**Key Series for Trading:**

| Series ID | Name | Frequency | Trading Relevance |
|-----------|------|-----------|-------------------|
| `DGS10` | 10-Year Treasury Yield | Daily | Risk-free rate, equity discount |
| `DGS2` | 2-Year Treasury Yield | Daily | Yield curve slope |
| `T10Y2Y` | 10Y-2Y Spread | Daily | Pre-computed recession predictor |
| `VIXCLS` | VIX | Daily | Volatility regime |
| `FEDFUNDS` | Fed Funds Rate | Monthly | Monetary policy stance |
| `CPIAUCSL` | CPI All Urban | Monthly | Inflation regime |
| `UNRATE` | Unemployment Rate | Monthly | Economic health |
| `GDP` | GDP | Quarterly | Expansion/contraction regime |
| `BAMLH0A0HYM2` | High Yield OAS | Daily | Credit risk appetite |
| `ICSA` | Initial Jobless Claims | Weekly | High-frequency labor signal |

**Sources:** https://github.com/mortada/fredapi, https://fred.stlouisfed.org/docs/api/fred/

### SEC EDGAR API (XBRL + Form 4)

**Rate Limits:** 10 requests/second. **Required User-Agent header:** `"CompanyName AdminEmail"`. No API key needed.

**Recommended Library:** `edgartools` (3,680+ commits, active maintenance, XBRL + Form 4 support)

```python
from edgar import Company, set_identity
set_identity("quantstack admin@example.com")

company = Company("AAPL")
financials = company.get_financials()
balance_sheet = financials.balance_sheet()

# Form 4 insider trading
form4_filings = company.get_filings(form="4")
form4 = form4_filings[0].obj()
transactions = form4.transactions  # DataFrame
```

**XBRL vs HTML Tradeoffs:**
- **XBRL** (structured): Standardized us-gaap taxonomy, cross-company comparable, machine-readable. Thinner coverage pre-2009.
- **HTML** (unstructured): Complete content including MD&A, risk factors. Requires NLP, brittle.
- **Recommendation:** XBRL for quantitative financials. HTML only for qualitative text sections.

**Sources:** https://github.com/dgunning/edgartools, https://data.sec.gov/api/xbrl/

### PostgreSQL Native Partitioning

**Partitioning Strategy for OHLCV:**
- **Range partition by timestamp (monthly)** — primary recommendation
- Partition key MUST be in primary key: `PRIMARY KEY (symbol, timeframe, timestamp)` already satisfies this
- **Start simple** (range by month), add hash sub-partitioning by symbol only if monthly partitions exceed memory

**pg_partman for Automation:**
```sql
CREATE EXTENSION pg_partman SCHEMA partman;
SELECT partman.create_parent(
    p_parent_table := 'public.ohlcv',
    p_control := 'timestamp',
    p_interval := '1 month',
    p_premake := 4,
    p_default_table := true
);
```

**Online Migration Strategy (expand-contract):**
1. Create new partitioned table with same schema
2. Create monthly partitions covering all existing data
3. Migrate data in batches (per-month, COMMIT between batches)
4. Catch up delta writes during migration
5. Atomic rename swap (`ALTER TABLE ... RENAME TO`)
6. Validate row counts, drop old table after confidence period

**Performance Notes:**
- Partition pruning is on by default — queries with `WHERE timestamp >= X` scan only relevant partitions
- Keep partition count in hundreds (not thousands) to avoid planning overhead
- Create indexes concurrently on individual partitions to avoid locking

**Sources:** https://www.postgresql.org/docs/current/ddl-partitioning.html, https://github.com/pgpartman/pg_partman

### Event-Driven Cache Invalidation

**Recommended Pattern: Dual-TTL (Soft + Hard)**
- **Soft TTL:** When expired, trigger async background refresh. Continue serving cached value.
- **Hard TTL:** Absolute expiration. Block until fresh data if soft refresh failed.

**Recommended TTLs for QuantStack:**

| Signal Type | Soft TTL | Hard TTL | Rationale |
|-------------|----------|----------|-----------|
| Technical indicators | 5 min (intraday) / 1 hr (daily) | 15 min / 4 hr | Timeframe-dependent |
| Sentiment | 15 min | 1 hr | News decays quickly |
| Macro regime | 6 hr | 24 hr | Slow-moving, expensive to compute |
| Fundamentals | 24 hr | 7 days | Quarterly changes |

**Cache Stampede Prevention:**
1. **Mutex lock:** Only one request recomputes; others wait or serve stale
2. **Probabilistic Early Expiration (XFetch):** Random pre-TTL refresh to spread load
3. **Request coalescing:** Collapse concurrent requests for same key
4. **External recomputation:** Background worker pre-refreshes (best for scheduled data)

**Event-Driven Invalidation via PostgreSQL LISTEN/NOTIFY:**
```python
# Publisher (after data refresh):
await conn.execute("NOTIFY signal_refresh, 'sentiment:AAPL'")

# Subscriber (cache manager):
await conn.add_listener("signal_refresh", handle_invalidation)
```

No external message broker needed at QuantStack's scale.

**Drift Detection + Cache Integration:**
- CRITICAL drift → don't cache (or cache with `low_confidence=True`)
- Track age distribution of cached entries; alert if >20% beyond soft TTL
- Log value drift events when fresh vs cached values differ by threshold

**Sources:** AWS Builders' Library (caching strategies), XFetch paper (Vattani et al. 2015)
