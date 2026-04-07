# Interview Transcript — Phase 8: Data Pipeline Hardening

---

## Q1: Provider Role (FRED/EDGAR vs AV)

**Question:** For item 8.3 (AV Redundancy): Research shows FRED is free (120 req/min) and EDGAR is free (10 req/sec). The macro collector currently pulls 10 indicators from AV. Should FRED fully replace AV for macro data, or serve as a fallback that only activates when AV fails? Same question for EDGAR vs AV for fundamentals.

**Answer:** Best-source routing per data type — use whichever source is more authoritative for each specific data type (e.g., FRED for yields, AV for company data).

---

## Q2: Drift Detection + Cache Behavior

**Question:** For item 8.4 (Drift Detection): Research confirms drift check runs AFTER cache write. The spec says 'penalize confidence by 0.30'. Should CRITICAL drift briefs be completely blocked from cache (force recompute on every request until drift resolves), or cached with degraded confidence and a short TTL?

**Answer:** Cache with short TTL + degraded confidence — cache for 5-10 min with confidence penalty. Balances freshness vs compute cost.

---

## Q3: OHLCV Data Scale

**Question:** For item 8.7 (OHLCV Partitioning): What's the current data volume?

**Answer:** User wasn't sure. Direct DB query showed **7,616,935 rows** — breakdown: 5M rows M5 (5-min), 1.2M H1 (hourly), 400K D1 (daily). Solidly medium-to-large. Partitioning justified.

---

## Q4: Web Search Provider

**Question:** For item 8.5 (Web Search): Tavily ($20/mo, AI-optimized) vs Brave (free tier) vs provider-agnostic interface?

**Answer:** Skip this — item 8.5 is excluded from this plan.

---

## Q5: Staleness Check Architecture

**Question:** For item 8.2 (Staleness Rejection): Centralized in SignalEngine, per-collector decorator, or hybrid?

**Answer:** Per-collector decorator — each collector declares its own max_staleness. More granular and self-documenting.

---

## Q6: EDGAR Schema Strategy

**Question:** For item 8.6 (SEC Filings): Should EDGAR data go into existing tables (normalize), separate tables, or same tables with source column?

**Answer:** Same tables, normalize data — upsert into existing insider_trades/institutional_ownership. Single source of truth.

---

## Q7: Provider Architecture

**Question:** ABC interface + provider registry, standalone fetcher modules, or extend existing pattern?

**Answer:** ABC interface + provider registry — formal DataProvider ABC, each source implements it. Pipeline selects best provider per data type.

---

## Q8: OHLCV Migration Timing

**Question:** The OHLCV partitioning requires briefly pausing writes (~5 seconds) during the final table swap. Weekend stop vs overnight?

**Context clarification:** User initially asked "Does this migration not happen only when we switch providers?" — clarified that partitioning is a database optimization independent of provider changes. It restructures the existing table for better query performance at 7.6M+ rows.

**Answer:** Weekend stop (safest) — stop Docker services, run migration, restart. Takes ~10 minutes total.

---

## Q9: Alert Routing

**Question:** For the alert system when data providers fail 3+ times consecutively: DB table + supervisor (internal) vs external notifications?

**Answer:** DB table + supervisor graph (keep it internal). Supervisor graph detects and self-heals. No external notifications needed.

---

## Summary of Decisions

1. **Provider strategy:** Best-source routing per data type
2. **Provider architecture:** ABC interface + provider registry
3. **Drift + cache:** CRITICAL drift → cache with short TTL (5-10 min) + degraded confidence (-0.30)
4. **Staleness checks:** Per-collector decorator pattern
5. **EDGAR schema:** Normalize into existing tables
6. **OHLCV:** 7.6M rows, monthly range partitioning, weekend maintenance window
7. **Alerts:** DB table + supervisor graph (internal only)
8. **Web search (8.5):** Excluded from this plan
9. **Options refresh (8.8):** Already broader than audit suggests (top 30 symbols)
