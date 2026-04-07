# Section 12: Pre-Trade Risk Gate Additions

## Overview

The risk gate (`src/quantstack/execution/risk_gate.py`) is the single enforcement point for all trade safety. It currently checks per-position limits (size, liquidity, participation rate, holding period, DTE bounds, execution quality, macro stress) and runs continuous monitoring via `monitor()` (position drift, correlation spikes, regime flips, daily P&L proximity). What it lacks are **portfolio-level pre-trade checks**: correlation with existing positions, daily deployment heat budget, and sector concentration.

This section adds three new checks inside `risk_gate.check()`. All three share these properties:

- Added inside `check()`, not as separate layers — the risk gate remains the single enforcement point
- Each produces a `RiskViolation` with a descriptive message on failure
- Each has configurable thresholds (correlation: 0.7, heat: 30%, sector: 40%)
- All fail closed on missing data — missing data is "unknown risk," not "no risk"

**Dependency**: Section 01 (DB migration) must complete first. The `positions` table already has the schema needed (entries are queryable by `opened_at` for heat budget). Sector mapping is already available in `src/quantstack/universe.py` via the `Sector` enum and `UniverseSymbol.sector` field, so no new table is required for the initial implementation.

---

## Tests

All tests belong in `tests/unit/test_risk_gate_pretrade.py`. The existing test patterns use `unittest.mock.MagicMock` / `AsyncMock` and pytest fixtures.

### 12.1 Pre-Trade Correlation Check

```python
# Test: new position with 0.8 correlation to existing → rejected with RiskViolation
#   - Mock portfolio with one existing position (e.g., AAPL)
#   - Mock DataStore.load_ohlcv to return 30-day series where candidate (e.g., MSFT) has 0.8 corr
#   - Call check() for MSFT → assert approved == False, violation.rule == "pretrade_correlation"

# Test: new position with 0.6 correlation → approved
#   - Same setup, but mock returns series with 0.6 corr → assert approved == True

# Test: boundary — 0.7 exactly → rejected (>=)
#   - Construct synthetic series with exactly 0.7 correlation → assert approved == False

# Test: correlation data unavailable (insufficient history) → fail closed, rejected
#   - Mock DataStore.load_ohlcv to return None or DataFrame with <20 rows
#   - Assert approved == False, violation.rule == "pretrade_correlation_data_missing"

# Test: data feed error → fail closed, rejected
#   - Mock DataStore.load_ohlcv to raise Exception
#   - Assert approved == False (not an unhandled crash)

# Test: new symbol with <20 days history → sector proxy correlation used
#   - Mock load_ohlcv returns 10 rows for candidate
#   - Mock universe lookup returns candidate's sector
#   - Mock sector ETF correlation computed instead → verify sector ETF symbol used

# Test: no existing positions → correlation check is a no-op, passes
#   - Empty portfolio → check() proceeds past correlation without rejection
```

### 12.2 Portfolio Heat Budget

```python
# Test: daily notional at 31% of equity → rejected
#   - Mock portfolio equity = 100_000
#   - Mock DB query returns today's entries totaling $31,000 notional
#   - New entry for $0 (just the existing heat) → rejected (already over budget)

# Test: daily notional at 29% → approved
#   - Same setup, today's entries total $29,000 → approved

# Test: boundary — 30% exactly → rejected (>=)
#   - Today's entries = $30,000 exactly → rejected

# Test: cumulative — two entries totaling 31% in same day → second rejected
#   - First entry $15,000 (15%) → approved
#   - Second entry $16,000 (16%, cumulative 31%) → rejected

# Test: system-wide — deployment from another graph service counted in budget
#   - DB query returns entries from any source (not filtered by graph) → all counted

# Test: day rollover → budget resets
#   - Mock DB returns zero for today (yesterday's entries don't count) → approved

# Test: configurable threshold — set to 50%, verify 40% approved
#   - Inject RiskLimits with max_daily_heat_pct=0.50 → $40k on $100k equity → approved
```

### 12.3 Sector Concentration

```python
# Test: adding position would push sector to 41% → rejected
#   - Mock portfolio: 2 tech positions = $39k of $100k equity
#   - New tech position $2k → would be 41% → rejected, rule == "sector_concentration"

# Test: adding position keeps sector at 39% → approved
#   - Same setup, new position $0.5k → 39.5% → still under 40% → approved

# Test: unknown sector → treated as own sector (conservative, no concentration trigger)
#   - Symbol not in universe → no sector match → treated as unique sector → approved
#   - (Will not trigger concentration because it's the only symbol in its "sector")

# Test: configurable threshold — set to 50%, verify 45% approved
#   - Inject RiskLimits with max_sector_concentration_pct=0.50 → 45% in one sector → approved

# Test: sector mapping data missing → fail closed
#   - Universe lookup raises or returns None → assert approved == False
```

---

## Implementation Details

### File: `src/quantstack/execution/risk_gate.py`

#### New fields on `RiskLimits`

Add three new configurable limits to the `RiskLimits` dataclass:

```python
# Pre-trade portfolio checks (v phase-4)
max_pretrade_correlation: float = 0.70      # reject if corr > threshold with any existing position
max_daily_heat_pct: float = 0.30            # max % of equity deployed in new entries per day
max_sector_concentration_pct: float = 0.40  # max % of equity in any single sector
```

Add corresponding `from_env` loaders:

```python
if v := os.getenv("RISK_MAX_PRETRADE_CORRELATION"):
    limits.max_pretrade_correlation = float(v)
if v := os.getenv("RISK_MAX_DAILY_HEAT_PCT"):
    limits.max_daily_heat_pct = float(v)
if v := os.getenv("RISK_MAX_SECTOR_CONCENTRATION_PCT"):
    limits.max_sector_concentration_pct = float(v)
```

#### New private methods on `RiskGate`

Three new methods, each returning a list of `RiskViolation` (empty list means pass):

**`_check_pretrade_correlation(self, symbol: str, current_price: float) -> list[RiskViolation]`**

Logic:
1. Get all existing positions from `self._portfolio.get_positions()`
2. If no existing positions, return `[]` (nothing to correlate against)
3. For each existing position, load 30-day daily returns for both the candidate and the existing symbol from `DataStore`
4. If candidate has fewer than 20 days of data, fall back to sector proxy: look up the candidate's sector from `UniverseSymbol`, find the corresponding sector ETF (e.g., `XLK` for `Technology`), and compute correlation against the ETF instead
5. If correlation data is still unavailable (no sector mapping, no ETF data), **fail closed** — return a `RiskViolation` with rule `"pretrade_correlation_data_missing"`
6. If any pairwise correlation >= `self.limits.max_pretrade_correlation`, return a `RiskViolation` with rule `"pretrade_correlation"` including the correlated pair and the actual correlation value
7. Wrap the entire method in a try/except. On any exception, fail closed with a violation

Sector-to-ETF mapping: use a simple dict within this method mapping `Sector` enum values to sector ETF symbols (e.g., `Sector.TECHNOLOGY: "XLK"`, `Sector.HEALTHCARE: "XLV"`, etc.). The `SECTOR_ETFS` tuple already exists in `universe.py` for reference.

**`_check_heat_budget(self, order_notional: float, equity: float) -> list[RiskViolation]`**

Logic:
1. Query the `positions` table for all entries opened today: `SELECT COALESCE(SUM(quantity * avg_cost), 0) FROM positions WHERE opened_at::date = CURRENT_DATE`
2. This is a **system-wide** query — it sees deployments from all graph services because they share the same PostgreSQL database. Do not use an in-memory accumulator.
3. Compute `(today_notional + order_notional) / equity`
4. If >= `self.limits.max_daily_heat_pct`, return a `RiskViolation` with rule `"daily_heat_budget"` including the actual percentage and the limit
5. The query is cheap (indexed on `opened_at`) and correctness matters more than microseconds saved by caching

**`_check_sector_concentration(self, symbol: str, order_notional: float, equity: float) -> list[RiskViolation]`**

Logic:
1. Look up the candidate symbol's sector from the universe: `INITIAL_LIQUID_UNIVERSE.get(symbol)` to get `UniverseSymbol.sector`
2. If sector is unknown (symbol not in universe), treat as its own unique sector and return `[]` — conservative but won't trigger false concentration
3. Get all existing positions, compute notional per sector by looking up each position's sector from the universe
4. Add the candidate's `order_notional` to its sector's total
5. If `sector_total / equity >= self.limits.max_sector_concentration_pct`, return a `RiskViolation` with rule `"sector_concentration"` including the sector name, actual percentage, and limit
6. If the universe lookup itself fails (exception), fail closed with a violation

#### Integration into `check()`

Insert the three new checks after the existing liquidity and participation checks (after step 6c / macro stress scalar) but **before** the per-symbol position size check (step 7). This placement ensures portfolio-level rejection happens before position-level scaling logic runs.

The checks only apply to new entries (buys that increase exposure), not to position reductions. Add a guard: if `is_reducing` is True, skip the three new checks.

The checks only apply to the equity path. Options already have their own dedicated block. Add a comment noting this and a future TODO for options portfolio-level checks if needed.

Approximate insertion point (between steps 6c and 7 in the current `check()` method):

```python
# -- 6d. Pre-trade portfolio-level checks (skip for position reductions)
if not is_reducing:
    order_notional = quantity * current_price
    equity = snapshot.total_equity or 100_000.0

    # Correlation with existing positions
    corr_violations = self._check_pretrade_correlation(symbol, current_price)
    violations.extend(corr_violations)

    # Daily heat budget
    heat_violations = self._check_heat_budget(order_notional, equity)
    violations.extend(heat_violations)

    # Sector concentration
    sector_violations = self._check_sector_concentration(symbol, order_notional, equity)
    violations.extend(sector_violations)

    if violations:
        for v in violations:
            logger.warning(f"[RISK] PRETRADE VIOLATION [{v.rule}]: {v.description}")
        return RiskVerdict(approved=False, violations=violations)
```

Note: `order_notional` and `equity` are computed here before the existing step 7 block, which also computes them. This is intentional — the existing step 7 block includes forward-testing scalar logic that should not affect portfolio-level checks. The two computations serve different purposes.

#### Sector-to-ETF Mapping

A constant dict mapping `Sector` values to their SPDR sector ETF symbol. Place as a module-level constant near the top of `risk_gate.py`:

```python
SECTOR_ETF_MAP: dict[Sector, str] = {
    Sector.TECHNOLOGY: "XLK",
    Sector.HEALTHCARE: "XLV",
    Sector.FINANCIALS: "XLF",
    Sector.ENERGY: "XLE",
    Sector.CONSUMER_DISCRETIONARY: "XLY",
    Sector.CONSUMER_STAPLES: "XLP",
    Sector.INDUSTRIALS: "XLI",
    Sector.MATERIALS: "XLB",
    Sector.REAL_ESTATE: "XLRE",
    Sector.UTILITIES: "XLU",
    Sector.COMMUNICATION: "XLC",
}
```

Import `Sector` from `quantstack.universe` and `UniverseSymbol` / `INITIAL_LIQUID_UNIVERSE` as needed.

---

## Key Design Decisions

1. **All three checks inside `check()`, not as middleware or decorators.** The risk gate is a single enforcement point by design (CLAUDE.md: "Risk gate is LAW"). Adding checks as external layers would create bypass paths.

2. **Heat budget uses a DB query, not an in-memory counter.** Multiple graph services (research, trading, supervisor) can deploy capital. An in-memory counter only sees its own process. The DB is the shared source of truth. The query cost is negligible compared to LLM calls in the pipeline.

3. **Sector mapping from `universe.py`, not a new DB table.** The universe already maps every symbol to a `Sector` enum. A new `symbol_sectors` table is unnecessary complexity for the initial implementation. If the universe grows beyond what's in `INITIAL_LIQUID_UNIVERSE` (e.g., dynamic symbol discovery), a DB-backed sector lookup can be added later.

4. **Fail-closed on missing data.** All three checks treat data absence as "unknown risk" and reject. This is the correct conservative default for a system that manages real capital. A false rejection costs a missed opportunity; a false approval costs real money.

5. **Correlation check uses the same `DataStore.load_ohlcv` / 30-day rolling approach** that `_check_correlation_spikes` in `monitor()` already uses. This keeps the data access pattern consistent and reuses the same infrastructure. The pre-trade check uses a stricter threshold (0.7) than the monitoring alert (0.8) because it's cheaper to prevent a correlated entry than to unwind one.

6. **Checks run only on the equity path.** The options path already exits `check()` early with its own verdict. Portfolio-level checks for options (delta-adjusted correlation, vega concentration) are a future concern — adding a TODO comment is appropriate here, but the implementation is out of scope for this section.

---

## Dependencies

| Dependency | What's Needed | Status |
|------------|---------------|--------|
| Section 01 (DB migration) | Positions table must exist with `opened_at` column (for heat budget query) | `opened_at` already exists in schema |
| `DataStore` | `load_ohlcv(symbol, Timeframe.D1)` for correlation data | Already used by `_check_correlation_spikes` in `monitor()` |
| `universe.py` | `Sector` enum, `INITIAL_LIQUID_UNIVERSE`, `UniverseSymbol.sector` | Already exists and populated |
| `pg_conn()` | DB connection for heat budget query | Already imported in risk_gate.py |

No blocking dependencies from other sections. This section is parallelizable with sections 05-08, 10-11, and 13.
