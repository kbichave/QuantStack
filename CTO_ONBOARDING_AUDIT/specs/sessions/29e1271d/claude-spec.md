# Synthesized Spec — Phase 9: Missing Roles & Scale

## Overview

Phase 9 fills operational gaps in the QuantStack autonomous trading system: missing monitoring agents, incomplete alert lifecycle, and 24/7 operations capability. Nine items spanning corporate actions monitoring, factor exposure, performance attribution, system-level alerts, dashboard notifications, EventBus ACK patterns, multi-mode 24/7 operation, LLM provider unification, and research fan-out.

**Timeline:** Week 6-8 | **Effort:** 17-20 days | **Prerequisites:** Phases 1-3 and Phase 5 complete.

---

## Item 9.1: Corporate Actions Monitor

**Problem:** No monitoring for dividends, splits, or M&A on holdings. A 2:1 stock split could double apparent position size, triggering incorrect risk limits.

**Scope:** Full coverage — Alpha Vantage for dividends/splits, EDGAR 8-K for M&A detection.

**Design decisions from interview:**
- New `corporate_actions` table with `(symbol, event_type, effective_date, raw_payload)` grain
- Idempotent split adjustment via `split_adjustments` audit table
- Deterministic (no LLM needed) — daily scheduled job
- AV endpoints: `DIVIDENDS`, `SPLITS` (poll daily for all universe symbols)
- EDGAR: Parse 8-K items 1.01 (M&A signing), 2.01 (acquisition completion), 3.03 (splits/dividends) via `edgartools` library
- Cross-reference AV with EDGAR for validation
- Auto-adjust cost basis on splits: `new_qty = old_qty * split_ratio`, `new_cost = old_cost / split_ratio`, total cost invariant
- M&A announcements flag thesis review via system alert

**Existing codebase:**
- No existing corporate actions implementation
- Position management in `src/quantstack/execution/portfolio_state.py`
- Data acquisition pipeline in `src/quantstack/data/acquisition_pipeline.py`

---

## Item 9.2: Factor Exposure Monitor

**Problem:** No tracking of portfolio beta, sector tilts, or style exposure. Portfolio could unknowingly be 90% momentum.

**Design decisions:**
- Per-cycle factor exposure computation
- **Fully configurable:** thresholds AND benchmark stored in DB config table
- Default thresholds: beta drift >0.3, sector >40%, momentum crowding >70%
- Default benchmark: SPY (configurable to any symbol)
- Alerts via new system-level alert layer (item 9.4)

**Existing codebase:**
- Risk tools exist: `compute_risk_metrics()`, `compute_var()`, `stress_test_portfolio()`
- Portfolio risk module: `src/quantstack/risk/portfolio_risk.py`
- No dedicated factor exposure computation yet

---

## Item 9.3: Performance Attribution Per-Cycle

**Problem:** Attribution runs nightly only. Can't attribute intraday P&L in real-time.

**Design decisions:**
- **New trading graph node** (automatic, every cycle) — inserted after `reflect`, before END
- Decompose P&L into: factor (market, sector, style), timing (entry/exit quality), selection (stock-specific alpha), cost (execution quality)
- No LLM needed — deterministic computation
- Results stored in DB for dashboard display

**Existing codebase:**
- Attribution tools exist as stubs: `get_daily_equity()`, `get_strategy_pnl()`
- Benchmark module: `src/quantstack/performance/benchmark.py`
- Trading graph: `src/quantstack/graphs/trading/graph.py` (16-node StateGraph)

---

## Item 9.4: System-Level Alert Lifecycle

**Problem:** Alert system exists for equity alerts but no system-level alert lifecycle. Missing: create, acknowledge, escalate, resolve, query for operational alerts.

**Design decisions:**
- **Separate system-level alert layer** — new `system_alerts` table, distinct from `equity_alerts`
- 5 lifecycle tools: `create_system_alert`, `acknowledge_alert`, `escalate_alert`, `resolve_alert`, `query_system_alerts`
- Lifecycle: created → acknowledged → resolved/escalated
- Categories: risk_breach, service_failure, kill_switch, data_quality, performance_degradation
- Severity levels: info, warning, critical, emergency
- Alert history queryable for post-incident review

**Existing codebase:**
- Equity alert tools in `src/quantstack/tools/langchain/alert_tools.py` (stubs)
- Three equity tables: `equity_alerts`, `alert_exit_signals`, `alert_updates`
- Tool registry: `src/quantstack/tools/registry.py`

---

## Item 9.5: Dashboard Notifications (NOT Discord)

**Problem:** No real-time notification for critical events. Owner may not see kill switch activation for hours.

**Design decisions (changed from spec):**
- **No Discord** — user doesn't have Discord active. TODO note for future.
- **Integrate into both dashboards:**
  - TUI (Textual): new alerts widget on Overview tab
  - Web (FastAPI): alerts pane/banner with SSE streaming
- Trigger events: kill switch, daily loss >2%, position limit breach, system halt, 3+ consecutive graph failures
- Read from `system_alerts` table (item 9.4)

**Existing codebase:**
- TUI dashboard: `src/quantstack/tui/` (6 tabs, Textual framework)
- Web dashboard: `src/quantstack/dashboard/app.py` (FastAPI, SSE on port 8421)
- Dashboard events: `src/quantstack/dashboard/events.py`

---

## Item 9.6: EventBus ACK Pattern

**Problem:** Supervisor publishes events but never knows if trading graph received them. Fire-and-forget.

**Design decisions:**
- **All risk events require ACK:** RISK_WARNING, RISK_ENTRY_HALT, RISK_LIQUIDATION, RISK_EMERGENCY, IC_DECAY, REGIME_CHANGE, MODEL_DEGRADATION
- Extend existing PostgreSQL-based EventBus (not replace)
- New columns on `loop_events`: `requires_ack`, `expected_ack_by`, `acked_at`, `acked_by`
- New `dead_letter_events` table for missed ACKs
- Supervisor graph monitors for missing ACKs (background, no latency on happy path)
- Escalation: T1 (one cycle) → retry, T2 (3 cycles) → warn, T3 (5 cycles) → dead-letter + CRITICAL alert

**Existing codebase:**
- EventBus: `src/quantstack/coordination/event_bus.py` (299 lines, PostgreSQL append-only)
- 16 event types defined, poll-based, per-consumer cursors
- Supervisor graph: `src/quantstack/graphs/supervisor/graph.py`

---

## Item 9.7: Multi-Mode 24/7 Operation

**Problem:** System has no concept of extended hours. Trading and research don't run outside market hours.

**Design decisions:**
- Three modes: Market (9:30-16:00), Extended (16:00-20:00, 04:00-09:30), Overnight/Weekend
- **Risk gate enforcement (hard block)** for extended hours — rejects any order, can't be bypassed by agents
- Extended hours: trading graph runs in monitor-only mode (position monitoring, no new entries)
- Overnight/weekend: heavy research, ML training, data backfill
- Supervisor: always-on across all modes

**Existing codebase:**
- Runner intervals: `src/quantstack/runners/__init__.py` (market/after_hours/weekend)
- Risk gate: `src/quantstack/execution/risk_gate.py` (40KB)
- Current binary: market hours vs not (trading stops entirely after hours)

---

## Item 9.8: Unify LLM Provider Systems

**Problem:** Multiple LLM access paths coexist despite unified `get_chat_model()` system. Hardcoded strings in 6+ locations.

**Design decisions:**
- Route ALL model access through `get_chat_model()` with single configuration table
- Remove hardcoded Ollama, Groq, and model string references
- Provider fallback chain configurable per tier
- Prerequisites complete (Phase 5.6 removed hardcoded strings — now finalize)

**Existing codebase:**
- Primary routing: `src/quantstack/llm_config.py` (561 lines, 12 providers)
- Provider layer: `src/quantstack/llm/provider.py` (248 lines)
- 8 tiers, fallback order: bedrock → anthropic → openai → ollama

---

## Item 9.9: Research Fan-Out Default On

**Problem:** `RESEARCH_FAN_OUT_ENABLED=false` by default. Sequential is 3-5x slower.

**Design decisions:**
- Flip default to `true`
- Add rate limiting for AV quota during fan-out (max concurrent validations)
- Safety guard: if AV calls/min approach 75, throttle fan-out

**Existing codebase:**
- Fan-out logic: `src/quantstack/graphs/research/graph.py` line 74
- Conditional node registration based on env var
- Fan-out path: `fan_out_hypotheses → validate_symbol (parallel) → filter_results`
