# Implementation Plan — Phase 9: Missing Roles & Scale

## Background

QuantStack is an autonomous trading system built on three LangGraph StateGraphs (Research, Trading, Supervisor) running as Docker services. The system researches strategies, trains models, executes trades, and learns from outcomes with no human in the loop.

Phase 9 addresses operational gaps identified in a CTO onboarding audit (164 findings, grade C-): missing monitoring agents, incomplete alert lifecycle, and 24/7 readiness. These items are important but not existential — they improve operational completeness and scale readiness.

**All prerequisites are complete:** Phases 1-3 (safety, core wiring, risk) and Phase 5 (hardcoded model string removal) are done.

---

## Section 1: Database Schema — Corporate Actions & System Alerts

### 1.1 Corporate Actions Tables

Two new tables in `src/quantstack/db.py` via `ensure_tables()`:

**`corporate_actions`** — stores raw events from AV and EDGAR

```python
@dataclass
class CorporateAction:
    symbol: str
    event_type: str  # "dividend", "split", "merger", "acquisition", "delisting"
    source: str      # "alpha_vantage", "edgar_8k"
    effective_date: date
    announcement_date: date | None
    raw_payload: dict  # JSONB — full API response for audit trail
    processed: bool    # whether auto-adjustment has been applied
    created_at: datetime
```

Unique constraint on `(symbol, event_type, effective_date, source)` for idempotent inserts.

**`split_adjustments`** — audit trail for cost basis adjustments

```python
@dataclass
class SplitAdjustment:
    symbol: str
    effective_date: date
    split_ratio: float     # e.g., 4.0 for 4:1 split, 0.1 for 1:10 reverse
    old_quantity: float
    new_quantity: float
    old_cost_basis: float  # per share
    new_cost_basis: float  # per share
    applied_at: datetime
```

Unique constraint on `(symbol, effective_date, event_type)` to prevent double-adjustment while allowing multiple event types on the same date.

**Invariant to assert:** `old_quantity * old_cost_basis == new_quantity * new_cost_basis` (total cost unchanged).

### 1.2 System Alerts Table

**`system_alerts`** — separate from equity alerts, for operational/system events

```python
@dataclass
class SystemAlert:
    id: int                # BIGSERIAL PK
    category: str          # risk_breach, service_failure, kill_switch, data_quality, performance_degradation, factor_drift, ack_timeout
    severity: str          # info, warning, critical, emergency
    status: str            # open, acknowledged, escalated, resolved
    source: str            # graph/module that created it
    title: str             # one-line summary
    detail: str            # full context
    metadata: dict         # JSONB — structured context (positions affected, thresholds, etc.)
    acknowledged_by: str | None
    acknowledged_at: datetime | None
    escalated_at: datetime | None
    resolved_at: datetime | None
    resolution: str | None
    created_at: datetime
```

### 1.3 EventBus ACK Columns

Extend the existing `loop_events` table with three new columns:

- `requires_ack BOOLEAN DEFAULT FALSE`
- `expected_ack_by TIMESTAMPTZ` — null if ACK not required
- `acked_at TIMESTAMPTZ` — set when consumer calls `bus.ack()`
- `acked_by TEXT` — which consumer ACKed

New **`dead_letter_events`** table for events that missed their ACK window:

```python
@dataclass
class DeadLetterEvent:
    original_event_id: str
    event_type: str
    source_loop: str
    payload: dict
    published_at: datetime
    expected_ack_by: datetime
    retry_count: int
    dead_lettered_at: datetime
```

### 1.4 Factor Exposure Configuration Table

**`factor_config`** — configurable thresholds and benchmark

```python
@dataclass
class FactorConfig:
    config_key: str   # PK: "beta_drift_threshold", "sector_max_pct", "momentum_crowding_pct", "benchmark_symbol"
    value: str        # stored as string, parsed by caller
    updated_at: datetime
```

Default rows: `beta_drift_threshold=0.3`, `sector_max_pct=40`, `momentum_crowding_pct=70`, `benchmark_symbol=SPY`.

### 1.5 Factor Exposure History Table

**`factor_exposure_history`** — stores per-cycle factor snapshots for trend analysis

```python
@dataclass
class FactorExposureSnapshot:
    portfolio_beta: float
    sector_weights: dict     # JSONB
    style_scores: dict       # JSONB
    momentum_crowding_pct: float
    benchmark_symbol: str
    alerts_triggered: int
    computed_at: datetime
```

### 1.6 Cycle Attribution Table

**`cycle_attribution`** — stores per-cycle P&L decomposition

```python
@dataclass
class CycleAttributionRow:
    cycle_id: str
    graph_cycle_number: int
    total_pnl: float
    factor_contribution: float
    timing_contribution: float
    selection_contribution: float
    cost_contribution: float
    per_position: dict       # JSONB — serialized list of PositionAttribution
    computed_at: datetime
```

### 1.7 TradingState Schema Update

**Critical:** `TradingState` in `src/quantstack/graphs/state.py` uses `extra="forbid"` (Pydantic strict validation). The new attribution node needs a state field declared *before* the node can return data.

Add to `TradingState`:
```python
cycle_attribution: dict = {}  # Populated by attribution_node after reflect
```

This must be done as part of schema work, not deferred to Section 4.

---

## Section 2: Corporate Actions Monitor (Item 9.1)

### 2.1 Data Collectors

New module: `src/quantstack/data/corporate_actions.py`

**Alpha Vantage collector:**

```python
async def fetch_av_dividends(symbol: str) -> list[CorporateAction]:
    """Fetch dividend history from AV DIVIDENDS endpoint. Returns parsed CorporateAction list."""

async def fetch_av_splits(symbol: str) -> list[CorporateAction]:
    """Fetch split history from AV SPLITS endpoint. Returns parsed CorporateAction list."""
```

Both functions use the existing AV rate limiter (`src/quantstack/data/fetcher.py` patterns). Poll daily for all universe symbols. Deduplicate on insert using the unique constraint.

**EDGAR 8-K collector:**

```python
async def fetch_edgar_8k_events(symbol: str, cik: str) -> list[CorporateAction]:
    """Fetch recent 8-K filings from EDGAR submissions API.
    
    Parses items 1.01 (M&A signing), 2.01 (acquisition completion), 
    3.03 (rights modification), 5.01 (change in control).
    
    Requires edgartools library. SEC rate limit: 10 req/s with User-Agent header.
    """
```

Add `edgartools` to `pyproject.toml` dependencies.

**CIK→Symbol Mapping:** Use SEC's `company_tickers.json` endpoint (`https://www.sec.gov/files/company_tickers.json`) to build the mapping at startup. Cache locally. Update weekly via supervisor scheduled task. If a ticker is not found in the mapping, log a warning and skip EDGAR for that symbol (AV coverage remains).

### 2.2 Split Auto-Adjustment

```python
async def apply_split_adjustment(symbol: str, split_ratio: float, effective_date: date) -> SplitAdjustment | None:
    """Auto-adjust cost basis and quantity for a stock split.
    
    1. Check split_adjustments table — skip if already applied (idempotent)
    2. Check if broker (Alpaca) has already adjusted the position — compare broker qty vs DB qty
       If broker already adjusted, sync DB to broker state and skip manual adjustment
    3. Read current position from portfolio_state
    3. Compute: new_qty = old_qty * split_ratio, new_cost = old_cost / split_ratio
    4. Assert invariant: total_cost unchanged
    5. Update position in DB
    6. Write split_adjustments audit row
    7. Create system alert: 'Split applied: {symbol} {ratio}'
    """
```

For reverse splits (ratio < 1): handle fractional shares by rounding down and noting cash-out amount in metadata.

### 2.3 M&A Thesis Flagging

When an 8-K item 1.01 or 2.01 is detected for a held symbol:
1. Create a system alert with category `thesis_review`, severity `critical`
2. Include: filing date, item type, brief description from filing
3. The supervisor graph's health check node will pick up critical alerts and flag for review

### 2.4 Scheduled Job

Add a new function to `src/quantstack/data/scheduled_refresh.py`:

```python
async def refresh_corporate_actions(symbols: list[str]) -> dict:
    """Daily corporate actions check for all holdings.
    
    Called by supervisor graph's scheduled_tasks node.
    1. Fetch AV dividends + splits for all symbols
    2. Fetch EDGAR 8-K events for all symbols  
    3. Insert new events (dedup on unique constraint)
    4. Auto-apply any unprocessed splits
    5. Flag M&A events as thesis reviews
    6. Return summary: {new_dividends, new_splits, new_ma_events, splits_applied}
    """
```

Rate limiting: AV calls capped at 75/min (existing limiter). EDGAR at 10 req/s. For a 50-symbol universe, total daily calls: ~100 AV + ~50 EDGAR = well within limits.

---

## Section 3: Factor Exposure Monitor (Item 9.2)

### 3.1 Factor Computation Module

New module: `src/quantstack/risk/factor_exposure.py`

```python
@dataclass
class FactorExposure:
    portfolio_beta: float
    sector_weights: dict[str, float]  # sector → pct
    top_sector: str
    top_sector_pct: float
    style_scores: dict[str, float]    # momentum, value, growth, quality → score
    momentum_crowding_pct: float
    computed_at: datetime

async def compute_factor_exposure(positions: list, benchmark_symbol: str) -> FactorExposure:
    """Compute portfolio factor exposure against benchmark.
    
    Beta: regress portfolio daily returns against benchmark over trailing 60 days.
    Sector weights: sum position notional by GICS sector / total notional.
    Style scores: aggregate per-position style factors (momentum = 12M-1M return, 
    value = P/E rank, growth = revenue growth rank, quality = ROA rank).
    Momentum crowding: % of portfolio in top-momentum quintile.
    """
```

### 3.2 Drift Alert Logic

```python
async def check_factor_drift(exposure: FactorExposure, config: dict[str, str]) -> list[SystemAlert]:
    """Check factor exposure against configurable thresholds.
    
    Reads thresholds from factor_config table.
    Returns list of system alerts for any threshold breaches.
    Alert categories: factor_drift.
    """
```

Thresholds read from `factor_config` table at runtime — no code change needed to adjust.

### 3.3 Integration Point

Called by the supervisor graph's `health_check` node every cycle. Results stored in a `factor_exposure_history` table for trend analysis. Dashboard widgets (Section 8) display current exposure.

---

## Section 4: Performance Attribution Node (Item 9.3)

### 4.1 Attribution Engine

New module: `src/quantstack/performance/attribution.py`

```python
@dataclass
class CycleAttribution:
    cycle_id: str
    total_pnl: float
    factor_contribution: float    # market + sector + style
    timing_contribution: float    # entry/exit quality vs VWAP
    selection_contribution: float # stock-specific alpha (residual)
    cost_contribution: float      # slippage + commission
    per_position: list[PositionAttribution]
    computed_at: datetime

async def compute_cycle_attribution(
    positions: list, 
    fills: list, 
    benchmark_returns: float,
    sector_returns: dict[str, float]
) -> CycleAttribution:
    """Decompose cycle P&L into four components.
    
    Factor: sum of (position_weight * benchmark_return) + sector residual.
    Timing: compare actual entry/exit prices to cycle VWAP.
    Selection: residual P&L after removing factor and timing.
    Cost: realized slippage (fill price vs mid) + commissions.
    
    All four components must sum to total_pnl (accounting identity).
    """
```

### 4.2 Trading Graph Node

Add `attribution` node to the trading graph in `src/quantstack/graphs/trading/graph.py`:

- Insert after `reflect` node, before END
- Deterministic (no LLM call) — pure computation
- Reads: current positions, recent fills, benchmark data from state
- Writes: `CycleAttribution` to `cycle_attribution` DB table and updates graph state

Stores results in `cycle_attribution` table (defined in Section 1.6).

### 4.3 Node Implementation Pattern

Follow the existing trading graph node pattern:
1. Function signature: `async def attribution_node(state: TradingState) -> dict`
2. Reads positions and fills from state (already available after `reflect`)
3. Fetches benchmark returns for the cycle period
4. Calls `compute_cycle_attribution()`
5. Stores result in DB via `db_conn()` context manager
6. Returns updated state with attribution summary

---

## Section 5: System-Level Alert Lifecycle (Item 9.4)

### 5.1 Alert Tools

Five new LangChain tools in `src/quantstack/tools/langchain/system_alert_tools.py`:

```python
@tool
async def create_system_alert(category: str, severity: str, title: str, detail: str, metadata: dict | None = None) -> str:
    """Create a new system-level alert. Returns alert ID."""

@tool
async def acknowledge_alert(alert_id: int, agent_name: str) -> str:
    """Mark alert as being investigated. Sets status to 'acknowledged'."""

@tool  
async def escalate_alert(alert_id: int, reason: str) -> str:
    """Bump severity one level and set status to 'escalated'."""

@tool
async def resolve_alert(alert_id: int, resolution: str) -> str:
    """Close alert with resolution notes. Sets status to 'resolved'."""

@tool
async def query_system_alerts(
    severity: str | None = None, 
    status: str | None = None, 
    category: str | None = None,
    since_hours: int = 24
) -> str:
    """Query system alerts with filters. Returns formatted alert list."""
```

### 5.2 Registration

Add all five tools to `TOOL_REGISTRY` in `src/quantstack/tools/registry.py`. Bind to supervisor graph agents in `src/quantstack/graphs/supervisor/config/agents.yaml` — specifically the health_monitor and diagnostician agents.

### 5.3 Internal Helper

For programmatic alert creation (from non-LLM code like risk gate, kill switch, corporate actions):

```python
# src/quantstack/tools/functions/system_alerts.py
async def emit_system_alert(category: str, severity: str, title: str, detail: str, metadata: dict | None = None) -> int:
    """Direct DB insert for system alerts from deterministic code paths.
    
    Used by: risk_gate, kill_switch, corporate_actions, factor_exposure, event_bus_monitor.
    NOT a LangChain tool — called directly by Python code.
    """
```

This avoids routing through the LLM for programmatic alerts while sharing the same DB table.

---

## Section 6: Dashboard Alert Integration (Item 9.5)

### 6.1 TUI Dashboard — Alerts Widget

New widget in `src/quantstack/tui/widgets/alerts_widget.py`:

- Inherits from `RefreshableWidget` (existing base class)
- Displays recent system alerts in a table: severity (color-coded), title, status, age
- Refresh tier: T1 (highest priority — always refreshes)
- Placed on the Overview tab
- Query: `SELECT * FROM system_alerts WHERE status != 'resolved' ORDER BY severity DESC, created_at DESC LIMIT 20`

Color mapping: emergency=red bold, critical=red, warning=yellow, info=dim.

### 6.2 Web Dashboard — Alerts Pane

Extend `src/quantstack/dashboard/app.py`:

**New API endpoint:**
```python
@app.get("/api/alerts")
async def get_alerts(status: str = "open", limit: int = 20) -> list[dict]:
    """Return recent system alerts for dashboard display."""
```

**Frontend changes:** Add a collapsible alerts banner at the top of the 2x2 grid. Critical/emergency alerts show as a persistent red banner. Lower severity alerts in a collapsible panel. SSE stream extended to include alert events.

### 6.3 Event Publishing

Extend `src/quantstack/dashboard/events.py` to publish alert events alongside agent events. When `emit_system_alert()` is called, also call `publish_event()` with event_type `"system_alert"` so the SSE stream picks it up in real-time.

### 6.4 Discord TODO

Add a TODO comment in `emit_system_alert()`:
```python
# TODO(kbichave): Add Discord webhook notification for CRITICAL/EMERGENCY alerts.
# Trigger: when DISCORD_WEBHOOK_URL env var is set. See Phase 9 spec item 9.5 for
# webhook patterns (rate limits, batching, embed formatting).
```

---

## Section 7: EventBus ACK Pattern (Item 9.6)

### 7.1 EventBus Extensions

Modify `src/quantstack/coordination/event_bus.py`:

**New constant:** Define which event types require ACK:

```python
ACK_REQUIRED_EVENTS: set[EventType] = {
    EventType.RISK_WARNING,
    EventType.RISK_ENTRY_HALT,
    EventType.RISK_LIQUIDATION,
    EventType.RISK_EMERGENCY,
    EventType.IC_DECAY,
    EventType.REGIME_CHANGE,
    EventType.MODEL_DEGRADATION,
}
```

**Modified `publish()` method:** When event type is in `ACK_REQUIRED_EVENTS`, set `requires_ack=True` and `expected_ack_by = now + ACK_TIMEOUT_SECONDS`. Use a fixed 600-second timeout for all risk events (the publisher doesn't know the consumer's cycle interval — they're different graphs). This is ~2x the supervisor cycle and ~2x the trading cycle, giving adequate buffer.

**Migration note:** Existing `loop_events` rows will have `requires_ack=NULL` after the column addition. The ACK monitor query filters on `requires_ack=TRUE`, so NULL rows are safely excluded. No backfill needed — the 7-day TTL will age out old rows naturally.

**New `ack()` method:**

```python
def ack(self, event_id: str, consumer_id: str) -> None:
    """Acknowledge receipt and processing of an event.
    
    Sets acked_at and acked_by on the event row.
    Idempotent — re-acking is a no-op.
    """
```

**Modified `poll()` method:** After returning events to consumer, the consumer is responsible for calling `ack()` after processing each event that has `requires_ack=True`.

### 7.2 ACK Monitor

New function added to the supervisor graph's `health_check` node:

```python
async def check_missed_acks() -> list[SystemAlert]:
    """Query for events where requires_ack=True AND acked_at IS NULL AND expected_ack_by < now().
    
    Escalation tiers (based on how many cycles overdue):
    - 1 cycle overdue: retry (re-publish event)
    - 3 cycles overdue: warning system alert
    - 5 cycles overdue: move to dead_letter_events + CRITICAL system alert
    
    Returns list of system alerts created.
    """
```

This runs every supervisor cycle (300s). The happy path (events ACKed on time) adds zero latency — ACK monitoring is purely background.

### 7.3 Consumer-Side Pattern

In each graph runner's event processing loop, after processing a polled event:

```python
events = bus.poll(consumer_id="trading_graph", event_types=risk_events)
for event in events:
    process_event(event)
    if event.requires_ack:
        bus.ack(event.event_id, consumer_id="trading_graph")
```

This is a small change to each runner's event handling code.

---

## Section 8: Multi-Mode 24/7 Operation (Item 9.7)

### 8.1 Mode Detection

Extend `src/quantstack/runners/__init__.py`:

```python
class OperatingMode(str, Enum):
    MARKET = "market"           # 9:30-16:00 Mon-Fri
    EXTENDED = "extended"       # 16:00-20:00, 04:00-09:30 Mon-Fri
    OVERNIGHT = "overnight"     # 20:00-04:00 Mon-Fri
    WEEKEND = "weekend"         # Sat-Sun all day

def get_operating_mode() -> OperatingMode:
    """Determine current operating mode based on ET time and day of week."""
```

### 8.2 Graph Behavior Matrix

| Graph | Market | Extended | Overnight | Weekend |
|-------|--------|----------|-----------|---------|
| Trading | Full (300s) | Monitor-only (300s) | Off | Off |
| Research | Normal (120s) | Normal (180s) | Heavy (120s, full compute) | Heavy (300s, full compute) |
| Supervisor | Normal (300s) | Normal (300s) | Normal (300s) | Normal (300s) |

"Monitor-only" for trading graph means: run `data_refresh`, `safety_check`, `position_review`, `execute_exits` (stops still active). Skip: `plan_day`, `entry_scan`, `execute_entries`.

**Graph routing mechanism:** Add a conditional edge after `safety_check` that calls `get_operating_mode()`. If mode is `EXTENDED`, route to a truncated subgraph that goes directly from `safety_check` → `position_review` → `execute_exits` → `reflect` → `attribution` → END, skipping the parallel entry-scan branch entirely. This is a new router function in `src/quantstack/graphs/trading/graph.py` following the existing conditional edge pattern (e.g., `route_after_safety_check`).

### 8.3 Risk Gate Hard Block

Modify `src/quantstack/execution/risk_gate.py`:

Add a new check at the top of the risk gate's validation chain:

```python
def _check_trading_window(self) -> RiskDecision:
    """Reject new entries outside market hours.
    
    Extended hours: REJECT with reason 'Extended hours — no new entries'
    Overnight/Weekend: REJECT with reason 'Market closed'
    Market hours: PASS
    
    This is enforced at the risk gate level (hardest block) so no agent
    can bypass it regardless of graph routing.
    """
```

Exits (stop losses, trailing stops) are NOT blocked — only orders that *increase* absolute exposure. The check must look at current position state, not just order side — a sell can be a new short entry, and a buy can be covering a short. The logic: if the order would increase `abs(position_quantity)` (opening or adding), reject. If it would decrease `abs(position_quantity)` (closing or trimming), allow.

### 8.4 Runner Changes

Each runner in `src/quantstack/runners/` already calls `get_cycle_interval()`. Update the interval mapping to use `OperatingMode`:

```python
INTERVALS = {
    "trading": {
        OperatingMode.MARKET: 300,
        OperatingMode.EXTENDED: 300,   # was None (stopped)
        OperatingMode.OVERNIGHT: None,
        OperatingMode.WEEKEND: None,
    },
    "research": {
        OperatingMode.MARKET: 120,
        OperatingMode.EXTENDED: 180,
        OperatingMode.OVERNIGHT: 120,  # heavy research mode
        OperatingMode.WEEKEND: 300,
    },
    "supervisor": {
        OperatingMode.MARKET: 300,
        OperatingMode.EXTENDED: 300,
        OperatingMode.OVERNIGHT: 300,
        OperatingMode.WEEKEND: 300,
    },
}
```

Key change: trading graph now runs in extended hours (was `None`/stopped) but in monitor-only mode.

---

## Section 9: LLM Provider Unification (Item 9.8)

### 9.1 Audit & Remove Remaining Hardcoded Strings

Phase 5.6 removed most hardcoded model strings. This item finalizes the cleanup:

1. **Grep for remaining hardcoded patterns:** Search for string literals containing model names (`"claude"`, `"gpt-4"`, `"llama"`, `"qwen"`, `"gemini"`, `"mistral"`) outside of `llm_config.py` and `llm/provider.py`
2. **Replace each with `get_chat_model(tier)`** using the appropriate tier
3. **Remove direct Ollama/Groq client calls** — route through the provider layer

### 9.2 Single Configuration Table

Currently, LLM configuration is spread across:
- Environment variables (`LLM_MODEL_IC`, `LLM_MODEL_POD`, etc.)
- `llm_config.py` hardcoded defaults
- Agent YAML `llm_tier` fields

Consolidate into a single precedence chain:
1. Environment variable override (highest priority)
2. DB `llm_config` table (runtime-changeable)
3. `llm_config.py` defaults (code defaults)

New DB table: **`llm_config`** with `(tier, provider, model, fallback_order, updated_at)`.

```python
async def get_llm_config(tier: str) -> dict:
    """Resolve LLM config for a tier.
    
    Precedence: env var → DB row → code default.
    Returns: {provider, model, fallback_order, thinking_enabled, thinking_budget}.
    """
```

### 9.3 Provider Health Dashboard

Add an LLM provider status check to the supervisor's health_check node:
- Ping each configured provider with a minimal prompt
- Track: latency, availability, error rate over last hour
- Alert if primary provider for any tier is down and fallback is active

---

## Section 10: Research Fan-Out Default On (Item 9.9)

### 10.1 Flip Default

In `src/quantstack/graphs/research/graph.py`, change:

```python
# Before:
fan_out_enabled = os.environ.get("RESEARCH_FAN_OUT_ENABLED", "false").lower() == "true"

# After:
fan_out_enabled = os.environ.get("RESEARCH_FAN_OUT_ENABLED", "true").lower() == "true"
```

### 10.2 Rate Limiting Guard

Add a concurrency limiter to the fan-out path to prevent AV quota exhaustion:

```python
async def fan_out_hypotheses(state: ResearchState) -> dict:
    """Distribute hypotheses to parallel validation workers.
    
    NEW: Two-layer throttling:
    1. asyncio.Semaphore(10) to limit concurrent validation tasks (memory/connection pressure)
    2. Existing AV rate limiter (75/min) gates actual API calls (quota protection)
    
    The semaphore limits concurrency; the rate limiter limits rate. Both are needed —
    a semaphore alone doesn't prevent burst quota exhaustion if calls are fast.
    """
```

### 10.3 Quota Monitoring

Add AV call counter tracking (if not already present) that the fan-out node checks before launching workers. If calls in the current minute are above 60 (80% of 75 limit), throttle new launches by adding a 1-second delay between them.

---

## Dependency Order

Items can be parallelized in three batches:

**Batch 1 (foundation — no dependencies):**
- Section 1: Database schema (all items depend on this)
- Section 5: System-level alert lifecycle (Sections 2, 3, 4, 7 all emit system alerts — must be in place first)

**Batch 2 (depends on schema + alert infrastructure):**
- Section 2: Corporate actions monitor
- Section 3: Factor exposure monitor
- Section 4: Performance attribution node
- Section 7: EventBus ACK pattern
- Section 9: LLM provider unification
- Section 10: Research fan-out

**Batch 3 (depends on Batch 2 outputs):**
- Section 6: Dashboard alert integration (depends on alert data existing)
- Section 8: Multi-mode 24/7 (depends on Sections 3, 4 for mode-aware computation)

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Alert architecture | Separate system-level alerts | Equity alerts have different schema, lifecycle, and consumers than system alerts |
| Notifications | Dashboard only, no Discord | User doesn't have Discord active; dashboards (TUI + web) provide same visibility |
| Extended hours enforcement | Risk gate hard block | Agents can't bypass — defense at the deepest layer |
| Factor thresholds | Fully configurable (DB table) | Thresholds will need tuning as portfolio composition changes |
| Corporate actions scope | AV + EDGAR | M&A events can invalidate thesis — worth the complexity of EDGAR parsing |
| Attribution placement | Automatic graph node | Every cycle gets attribution — no agent decision needed, always-on observability |
| EventBus ACK | All risk events | Financial risk events must have delivery confirmation — partial ACK is worse than none |
| Fan-out default | On with rate limiter | 3-5x faster research; semaphore prevents quota exhaustion |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| EDGAR rate limiting / parsing failures | Graceful degradation: if EDGAR fails, AV data still covers splits/dividends. Log warning, don't block. |
| Factor exposure computation too slow for per-cycle | Precompute sector weights daily, only update beta and crowding per-cycle. Cache benchmark returns. |
| Attribution accounting identity violated (components don't sum to total) | Assert the identity in code. If violated, log the discrepancy and use "unattributed" bucket. |
| ACK monitor false positives during graph restart | Grace period: don't check ACKs for events published in the last 2 cycles after a graph restart. |
| Multi-mode transition edge cases (event published in market hours, ACK expected in extended hours) | Use absolute timestamps for ACK deadlines, not cycle counts. Mode transitions don't affect deadlines. |
| Alpaca auto-adjusts splits — risk of double adjustment | Reconciliation check: compare broker position qty vs DB qty before applying. If broker already adjusted, sync DB to broker and skip manual adjustment. |
| `edgartools` library may not parse 8-K items natively | Verify during implementation. Fallback: use EDGAR submissions API directly for filing metadata + flag for manual review instead of automated item extraction. |
