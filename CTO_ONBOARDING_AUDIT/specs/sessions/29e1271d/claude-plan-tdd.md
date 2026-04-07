# TDD Plan — Phase 9: Missing Roles & Scale

**Testing framework:** pytest with custom markers (slow, integration, requires_api, regression)
**Fixtures:** `trading_ctx` (full PostgreSQL-backed context), `paper_broker`, `kill_switch`, `portfolio`, `risk_state`, `sample_ohlcv_df`
**Conventions:** Tests in `tests/unit/` and `tests/integration/`. Autouse `reset_singletons_and_seeds` fixture prevents state pollution.

---

## Section 1: Database Schema

### Unit Tests
```python
# Test: ensure_tables() creates corporate_actions table with correct columns and constraints
# Test: ensure_tables() creates split_adjustments table with (symbol, effective_date, event_type) unique constraint
# Test: ensure_tables() creates system_alerts table with BIGSERIAL PK and all expected columns
# Test: ensure_tables() adds ACK columns to loop_events (requires_ack, expected_ack_by, acked_at, acked_by)
# Test: ensure_tables() creates dead_letter_events table
# Test: ensure_tables() creates factor_config table with default rows populated
# Test: ensure_tables() creates factor_exposure_history table
# Test: ensure_tables() creates cycle_attribution table
# Test: ensure_tables() is idempotent — running twice doesn't fail or duplicate default rows
```

### Integration Tests
```python
# Test: corporate_actions unique constraint rejects duplicate (symbol, event_type, effective_date, source)
# Test: corporate_actions unique constraint allows same symbol+date with different event_type
# Test: split_adjustments unique constraint rejects duplicate (symbol, effective_date, event_type)
# Test: system_alerts BIGSERIAL auto-increments on insert
# Test: loop_events existing rows have requires_ack=NULL (not FALSE) — verify NULL-safety
# Test: TradingState accepts cycle_attribution field without ValidationError (extra="forbid" compat)
```

---

## Section 2: Corporate Actions Monitor

### Unit Tests
```python
# Test: fetch_av_dividends parses AV response into list[CorporateAction] with correct fields
# Test: fetch_av_dividends handles empty response (no dividends) gracefully
# Test: fetch_av_dividends handles "None" string values in declaration/record dates
# Test: fetch_av_splits parses split response with correct split_ratio
# Test: fetch_edgar_8k_events parses 8-K items 1.01, 2.01, 3.03, 5.01 into CorporateAction
# Test: fetch_edgar_8k_events skips 8-K items we don't care about (e.g., 5.07)
# Test: fetch_edgar_8k_events handles missing CIK gracefully (logs warning, returns empty)
# Test: apply_split_adjustment computes correct new_qty and new_cost for 4:1 split
# Test: apply_split_adjustment computes correct values for 1:10 reverse split (ratio=0.1)
# Test: apply_split_adjustment asserts invariant: old_qty * old_cost == new_qty * new_cost
# Test: apply_split_adjustment is idempotent — second call for same (symbol, date) is no-op
# Test: apply_split_adjustment handles fractional shares on reverse split (rounds down)
# Test: apply_split_adjustment skips if broker already adjusted (qty reconciliation)
# Test: refresh_corporate_actions deduplicates on insert (unique constraint, no error)
# Test: CIK mapping loads from company_tickers.json format and resolves ticker → CIK
# Test: CIK mapping returns None for unknown ticker (doesn't crash)
```

### Integration Tests
```python
# Test: refresh_corporate_actions end-to-end with mocked AV + EDGAR responses
# Test: split auto-adjustment updates position in DB and writes audit row
# Test: M&A detection creates system alert with correct category and severity
```

---

## Section 3: Factor Exposure Monitor

### Unit Tests
```python
# Test: compute_factor_exposure calculates beta correctly for known return series
# Test: compute_factor_exposure returns sector_weights summing to 1.0 (within tolerance)
# Test: compute_factor_exposure handles single-position portfolio
# Test: compute_factor_exposure handles portfolio with no sector data (falls back gracefully)
# Test: check_factor_drift triggers alert when beta drift exceeds threshold
# Test: check_factor_drift triggers alert when top sector exceeds sector_max_pct
# Test: check_factor_drift triggers alert when momentum crowding exceeds threshold
# Test: check_factor_drift returns empty list when all metrics within thresholds
# Test: check_factor_drift reads thresholds from config dict (not hardcoded)
# Test: factor config defaults are correct (beta=0.3, sector=40, momentum=70, benchmark=SPY)
```

### Integration Tests
```python
# Test: factor_exposure_history row written on each computation
# Test: factor drift alert creates system_alert with category='factor_drift'
```

---

## Section 4: Performance Attribution Node

### Unit Tests
```python
# Test: compute_cycle_attribution components sum to total_pnl (accounting identity)
# Test: compute_cycle_attribution with zero fills returns all-zero components
# Test: compute_cycle_attribution factor_contribution reflects benchmark correlation
# Test: compute_cycle_attribution timing_contribution reflects entry vs VWAP difference
# Test: compute_cycle_attribution cost_contribution equals slippage + commissions
# Test: compute_cycle_attribution handles cycle with no active positions (empty portfolio)
# Test: attribution accounting identity assertion fires when components don't sum (logs + unattributed bucket)
```

### Integration Tests
```python
# Test: attribution_node reads state, computes, writes to cycle_attribution table
# Test: attribution_node returns dict with cycle_attribution key (TradingState compat)
# Test: attribution_node runs after reflect in trading graph node ordering
```

---

## Section 5: System-Level Alert Lifecycle

### Unit Tests
```python
# Test: create_system_alert returns alert ID and inserts row with status='open'
# Test: create_system_alert validates category against allowed values
# Test: create_system_alert validates severity against allowed values
# Test: acknowledge_alert sets status='acknowledged', acknowledged_by, acknowledged_at
# Test: acknowledge_alert on already-acknowledged alert is idempotent
# Test: escalate_alert bumps severity one level (warning→critical) and sets status='escalated'
# Test: escalate_alert on already-emergency severity doesn't go higher
# Test: resolve_alert sets status='resolved', resolution, resolved_at
# Test: resolve_alert on already-resolved alert is idempotent
# Test: query_system_alerts filters by severity correctly
# Test: query_system_alerts filters by status correctly
# Test: query_system_alerts filters by category correctly
# Test: query_system_alerts respects since_hours parameter
# Test: emit_system_alert (internal helper) writes same row as create_system_alert tool
```

### Integration Tests
```python
# Test: full lifecycle: create → acknowledge → resolve, verify all timestamps set
# Test: full lifecycle: create → escalate → resolve, verify severity bumped
# Test: all 5 tools registered in TOOL_REGISTRY
```

---

## Section 6: Dashboard Alert Integration

### Unit Tests
```python
# Test: alerts_widget query returns unresolved alerts sorted by severity then time
# Test: alerts_widget color mapping: emergency=red, critical=red, warning=yellow, info=dim
# Test: /api/alerts endpoint returns JSON list of alerts with correct schema
# Test: /api/alerts filters by status parameter
# Test: SSE stream includes system_alert event type
```

### Integration Tests
```python
# Test: TUI alerts widget renders with sample alert data (no crash)
# Test: web dashboard /api/alerts returns alerts from system_alerts table
# Test: emit_system_alert triggers dashboard event publication
```

---

## Section 7: EventBus ACK Pattern

### Unit Tests
```python
# Test: publish() sets requires_ack=True for events in ACK_REQUIRED_EVENTS
# Test: publish() sets requires_ack=False for events NOT in ACK_REQUIRED_EVENTS
# Test: publish() sets expected_ack_by = now + 600s for ACK-required events
# Test: ack() sets acked_at and acked_by on event row
# Test: ack() is idempotent — re-acking doesn't error or change timestamps
# Test: check_missed_acks returns empty list when all events ACKed on time
# Test: check_missed_acks detects event with expired expected_ack_by and NULL acked_at
# Test: check_missed_acks escalation: 1 cycle overdue → retry (re-publish)
# Test: check_missed_acks escalation: 5 cycles overdue → dead letter + CRITICAL alert
# Test: check_missed_acks ignores events with requires_ack=NULL (migration safety)
# Test: check_missed_acks grace period after graph restart (no false positives)
# Test: dead_letter_events row written with correct original_event_id and retry_count
```

### Integration Tests
```python
# Test: publish risk event → poll → ack → verify acked_at set in DB
# Test: publish risk event → don't ack → run check_missed_acks → verify alert created
# Test: publish non-risk event → verify requires_ack=False, no ACK monitoring
```

---

## Section 8: Multi-Mode 24/7 Operation

### Unit Tests
```python
# Test: get_operating_mode returns MARKET at 10:00 ET Monday
# Test: get_operating_mode returns EXTENDED at 17:00 ET Tuesday
# Test: get_operating_mode returns EXTENDED at 05:00 ET Wednesday
# Test: get_operating_mode returns OVERNIGHT at 22:00 ET Thursday
# Test: get_operating_mode returns WEEKEND at 14:00 ET Saturday
# Test: get_operating_mode handles NYSE holidays (market closed on holiday = weekend mode)
# Test: get_cycle_interval returns 300 for trading in MARKET mode
# Test: get_cycle_interval returns 300 for trading in EXTENDED mode (was None)
# Test: get_cycle_interval returns None for trading in OVERNIGHT mode
# Test: get_cycle_interval returns 120 for research in OVERNIGHT mode (heavy research)
# Test: risk gate _check_trading_window rejects new long entry in EXTENDED mode
# Test: risk gate _check_trading_window rejects new short entry in EXTENDED mode
# Test: risk gate _check_trading_window allows exit (sell to close long) in EXTENDED mode
# Test: risk gate _check_trading_window allows cover (buy to close short) in EXTENDED mode
# Test: risk gate _check_trading_window passes all orders in MARKET mode
# Test: risk gate checks absolute exposure change, not order side
```

### Integration Tests
```python
# Test: trading graph conditional edge routes to monitor-only subgraph in EXTENDED mode
# Test: trading graph runs full pipeline in MARKET mode
# Test: monitor-only subgraph visits: safety_check → position_review → execute_exits → reflect → attribution
# Test: monitor-only subgraph does NOT visit: plan_day, entry_scan, execute_entries
```

---

## Section 9: LLM Provider Unification

### Unit Tests
```python
# Test: no hardcoded model strings remain outside llm_config.py and llm/provider.py
# Test: get_llm_config precedence: env var overrides DB, DB overrides code default
# Test: get_llm_config returns code default when no env var and no DB row
# Test: get_llm_config returns DB row when present and no env var
# Test: get_llm_config returns env var when present (ignores DB and default)
# Test: llm_config table schema matches expected columns
```

### Integration Tests
```python
# Test: changing llm_config DB row changes subsequent get_chat_model() output
# Test: all agent YAML configs resolve to valid tiers via get_llm_for_agent()
```

---

## Section 10: Research Fan-Out Default On

### Unit Tests
```python
# Test: fan_out_enabled defaults to True (env var not set)
# Test: fan_out_enabled=False when RESEARCH_FAN_OUT_ENABLED=false
# Test: fan_out semaphore limits concurrent tasks to 10
# Test: AV rate limiter prevents >75 calls/min during fan-out
# Test: quota monitoring throttles new launches when calls > 60/min
```

### Integration Tests
```python
# Test: research graph uses fan-out path by default (fan_out_hypotheses node present)
# Test: research graph uses sequential path when RESEARCH_FAN_OUT_ENABLED=false
```
