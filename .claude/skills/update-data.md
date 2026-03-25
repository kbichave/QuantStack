---
name: update-data
description: Audit DB coverage across all universe symbols and fill any data gaps. Runs all 12 acquisition phases to ensure nothing is stale before research or trading.
user_invocable: true
---

# /update-data — Data Coverage Audit & Gap Fill

You are the **Data Operations Operator** for QuantPod. Your job is to ensure the DB has
complete, current data for all universe symbols before research or trading begins.

---

## Step 1: Check What's in the DB

Call `list_stored_symbols` to see which symbols have OHLCV data. Note which universe
symbols are missing or stale (last date > 1 trading day ago).

The universe is defined in `src/quantstack/data/universe.py` — **77 symbols total**:
- **ETFs (26):** SPY QQQ IWM TLT GLD GDX VXX TQQQ SQQQ XLE XLF XLK XLV XLI XLP + XLY XLB XLRE XLU XLC MDY HYG LQD IEF SHY UUP
- **Large-cap equities (35):** AAPL MSFT NVDA AMD AVGO INTC ORCL CRM PLTR UBER GOOGL META AMZN NFLX TSLA HD MCD COST WMT KO JPM BAC GS V MA C UNH JNJ LLY PFE ABBV XOM CVX BA CAT
- **Speculative / emerging (16):** ALAB SMCI NBIS IONQ RGTI QBTS RKLB LUNR JOBY ACHR MSTR MARA RIOT HOOD SOFI RDDT

**To add a new symbol:** add it to `INITIAL_LIQUID_UNIVERSE` in `universe.py`. That is the
only place. All other code (settings, watchlists, acquisition, skills) derives from it automatically.

---

## Step 2: Run the Acquisition

All phases are **idempotent** — existing rows are skipped, only deltas fetched.
Running with no arguments defaults to all 12 phases for all universe symbols.

### Full refresh — all 12 phases, all symbols (run this)
```bash
python scripts/acquire_historical_data.py
```
At 75 req/min (AV premium): ~40 min cold start, ~5-15 min incremental.

### Targeted gap-fill — all phases for specific symbols
```bash
python scripts/acquire_historical_data.py --symbols AAPL NVDA TSLA JPM
```

### Dry-run — estimate API calls without running
```bash
python scripts/acquire_historical_data.py --dry-run
python scripts/acquire_historical_data.py --symbols AAPL NVDA --dry-run
```

---

## Step 3: Verify Coverage

After acquisition, call `list_stored_symbols` again and confirm all symbols are present.

If symbols failed (logged as `fail` in the summary table):
- **API key missing:** `echo $ALPHA_VANTAGE_API_KEY` — must be set
- **Rate limit:** `echo $ALPHA_VANTAGE_RATE_LIMIT` — must be `75` for premium plan
- **Symbol unsupported:** some speculative tickers may have sparse options data — that's expected; OHLCV and fundamentals should still populate

---

## Phase Reference

| Phase | Data | Per-symbol API calls |
|-------|------|---------------------|
| `ohlcv_daily` | Daily OHLCV (20yr history) | 1 |
| `ohlcv_5min` | 5-min bars (24mo, delta-only) | 24 (months) |
| `ohlcv_1h` | 1-hour bars (24mo) | 24 |
| `options` | Options chains | 1 |
| `earnings_history` | EPS history | 1 |
| `news` | News sentiment (30d rolling) | batched (1 per 5 symbols) |
| `insider` | Insider transactions | 1 |
| `fundamentals` | Company overview | 1 |
| `macro` | 9 global macro series | 9 total (not per symbol) |
| `financials` | Income / BS / CF statements | 3 |
| `institutional` | 13F holdings | 1 |
| `corporate_actions` | Dividends + splits | 2 |

---

## Scheduler

The scheduler (`scripts/scheduler.py`) runs **all 12 phases every weekday at 08:00 ET**
automatically. After a cold-start or if a new symbol is added, run `/update-data` manually
to fill the gap immediately without waiting for tomorrow's scheduled run.

```bash
# Check scheduler is running
python scripts/scheduler.py --dry-run

# Trigger a one-off full refresh right now
python scripts/scheduler.py --run-now data_refresh

# Or run the script directly
python scripts/acquire_historical_data.py
```

---

## Adding a New Ticker (the only step required)

1. Open `src/quantstack/data/universe.py`
2. Add to `INITIAL_LIQUID_UNIVERSE`:
   ```python
   "TICKER": UniverseSymbol("TICKER", "Company Name", Sector.TECHNOLOGY),
   ```
3. If it belongs to a named subset (speculative, cross-asset, etc.), add it there too
4. Run `/update-data` to acquire its history immediately

That's it. settings.py, watchlists, scheduler, screener, and this skill all derive from
`INITIAL_LIQUID_UNIVERSE` — nothing else needs to change.

---

## Environment Requirements

```bash
ALPHA_VANTAGE_API_KEY=<premium key>   # Required
ALPHA_VANTAGE_RATE_LIMIT=75           # Required for premium plan (default is 5 = free tier)
```
