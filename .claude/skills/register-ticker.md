---
name: register-ticker
description: Register one or more new tickers into the universe. Looks up metadata from Alpha Vantage, adds to universe.py with a one-line description, acquires all 12 phases of historical data, and checks for stale symbols and upcoming events.
user_invocable: true
---

# /register-ticker — Register New Ticker(s)

You are the **Universe Manager** for QuantPod. Your job is to safely add new tickers
to the trading universe with full metadata and historical data.

**Usage:** `/register-ticker HIMS` or `/register-ticker HIMS CELH SOUN`

---

## Step 0: Normalise Input

Accept one or more symbols. Uppercase, strip whitespace. If none provided, ask the user.

---

## Step 1: Check If Already Registered

Read `src/quantstack/data/universe.py` and check `INITIAL_LIQUID_UNIVERSE` for each symbol.

- **Already there:** skip Steps 2–4 for that symbol; still run Steps 5–7.
- **Not there:** proceed with full registration.

---

## Step 2: Fetch Company Metadata

Call `acquire_historical_data` with `phases=["fundamentals"]` and `symbols=[<ticker>]`.

This fetches the Alpha Vantage OVERVIEW and stores it in `company_overview` (name, sector,
industry, description, market cap, beta, etc.).

Also call `get_company_facts` for market cap context (to help classify the group).

---

## Step 3: Determine Sector, Group, and Description

From the AV overview result:

**Sector mapping** (AV string → `Sector` enum):
| AV string | Enum |
|-----------|------|
| Technology | `Sector.TECHNOLOGY` |
| Health Care | `Sector.HEALTHCARE` |
| Financials | `Sector.FINANCIALS` |
| Consumer Discretionary | `Sector.CONSUMER_DISCRETIONARY` |
| Consumer Staples | `Sector.CONSUMER_STAPLES` |
| Industrials | `Sector.INDUSTRIALS` |
| Energy | `Sector.ENERGY` |
| Materials | `Sector.MATERIALS` |
| Utilities | `Sector.UTILITIES` |
| Real Estate | `Sector.REAL_ESTATE` |
| Communication Services | `Sector.COMMUNICATION` |
| ETF / N/A / blank | `Sector.ETF` with `is_etf=True` |

**Group classification:**
- `speculative` — market cap < $10B OR highly volatile / thematic name
- `large_cap` — market cap > $10B, established business
- `macro_etf` — bond, currency, commodity, or volatility ETF
- `general` — doesn't clearly fit the above

**Description** — one line (≤ 100 chars) summarising what the company does:
- Use the first sentence of the AV `Description` field, truncated to 100 chars
- Or write a cleaner one-liner if the AV text is poor (e.g. "AI-powered telehealth platform")

---

## Step 4: Edit `universe.py`

Add the new ticker to `INITIAL_LIQUID_UNIVERSE`. Insert it under the appropriate section:

```python
# ===== Section Name =====
"TICKER": UniverseSymbol("TICKER", "Full Company Name", Sector.TECHNOLOGY, description="One-liner description"),
```

For ETFs add `is_etf=True`:
```python
"TICKER": UniverseSymbol("TICKER", "ETF Full Name", Sector.ETF, is_etf=True, description="One-liner"),
```

**If group == "speculative":** also add to `SPECULATIVE_SYMBOLS` tuple at the bottom.
**If group == "macro_etf":** add to `CREDIT_ETFS` or `SECTOR_ETFS` as appropriate.

Update the comment count at the top of `INITIAL_LIQUID_UNIVERSE`:
```python
# Target universe: N tickers (...)
```

---

## Step 5: Acquire All Historical Data

Call `acquire_historical_data` with all 12 phases for the new ticker(s):

```
phases: null (= all 12)
symbols: ["TICKER"]
```

This is idempotent — safe if fundamentals already ran in Step 2.

---

## Step 6: Check for Stale Symbols

Call `list_stored_symbols`. Find any symbols where `last_date` < yesterday's date.

Surface them to the user:
```
⚠ Stale symbols detected (last update > 1 trading day ago):
  VXX — last: 2026-03-20
  IONQ — last: 2026-03-18
→ Run /update-data to refresh all, or acquire_historical_data for specific symbols.
```

If ≤ 5 stale symbols and user is watching, offer to refresh immediately.

---

## Step 7: Check for Upcoming Events

Call `get_event_calendar` for the next 7 days.

Flag any upcoming earnings for universe symbols:
```
📅 Upcoming earnings this week:
  NVDA — Thu Mar 27 (estimate: $0.97 EPS)
  AAPL — Fri Mar 28 (estimate: $1.54 EPS)
→ Ensure fresh data before these dates.
```

---

## Step 8: Return Summary

```
✅ Registered: HIMS — "Telehealth platform for health and wellness products"
   Sector: Healthcare | Group: speculative
   Added to: INITIAL_LIQUID_UNIVERSE, SPECULATIVE_SYMBOLS

📊 Data acquired (12 phases):
   ohlcv_daily ✓ | ohlcv_5min ✓ | options ✓ | fundamentals ✓
   news ✓ | insider ✓ | earnings_history ✓ | financials ✓ ...

⚠ Stale: VXX (2026-03-20), IONQ (2026-03-18) — run /update-data
📅 Upcoming: NVDA Thu, AAPL Fri
```

---

## Notes

- The `register_ticker` MCP tool handles the DB/data side programmatically.
  This skill does the same thing PLUS edits `universe.py` so the ticker persists
  across sessions and is picked up by all other tools automatically.
- Always edit `universe.py` — that is the single source of truth. Nothing else needs updating.
- If the ticker is invalid (AV returns no data), stop and tell the user.
