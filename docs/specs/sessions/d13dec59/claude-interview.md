# Interview Transcript — Dashboard v2

## Q1: Missing Database Tables

**Question:** The spec references tables that may not exist yet (market_holidays, benchmark_daily, breakthrough_features, risk_snapshots). Should the plan include creating missing tables, or should the dashboard gracefully degrade for missing data?

**Answer:** Both — create if trivial, degrade otherwise. Create missing tables and make sure the other systems use them to populate them.

---

## Q2: Rollout Strategy

**Question:** Should this be built as a phased rollout (e.g., Phase 1 = Textual migration + Overview tab, Phase 2 = remaining tabs), or all 6 tabs in one implementation pass?

**Answer:** All at once.

---

## Q3: Code Location

**Question:** Where should the dashboard v2 code live? Currently it's a monolithic scripts/dashboard.py (796 lines). The v2 will be significantly larger.

**Answer:** `src/quantstack/dashboard/` package. Proper Python package with app.py, widgets/, queries/, etc. Importable and testable.

---

## Q4: Charting Approach

**Question:** The spec mentions Plotext for charts. Research shows it has quirks integrating with Textual (it writes directly to terminal). Should we use custom Unicode rendering (sparklines, block chars) or invest in Plotext integration?

**Answer:** Custom Unicode only. Build sparklines, heatmaps, and bar charts with Unicode block/braille chars. Zero dependency, full Textual compatibility.

---

## Q5: Database Access Pattern

**Question:** The current dashboard runs queries synchronously with psycopg2. Textual is async. How should DB access work?

**Answer:** Thread workers. Use `@work(thread=True)` to run blocking psycopg2 queries in thread pool. Minimal migration effort.

---

## Q6: v1 Compatibility

**Question:** The --simple flag should preserve v1 behavior. Should v1 code remain in scripts/dashboard.py untouched, or be refactored to share query logic with v2?

**Answer:** Replace v1 completely. No --simple flag. v2 replaces v1 entirely.

---

## Q7: Drill-Down Detail Level

**Question:** For drill-down modals (Enter on a position/strategy/signal), how much detail should be shown? The spec lists full SignalBrief (66+ fields) for signals.

**Answer:** Curated summaries. Show a curated subset of the most useful fields. Keep modals concise.

---

## Q8: Write Capabilities

**Question:** Should the dashboard have any write capabilities (e.g., trigger kill switch, force refresh data, retire a strategy), or strictly read-only?

**Answer:** Strictly read-only. Dashboard is an observation tool only. All actions happen through the trading system.
