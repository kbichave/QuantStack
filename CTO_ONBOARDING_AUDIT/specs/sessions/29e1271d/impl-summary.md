# Implementation Summary — Phase 9: Missing Roles & Scale

**Session ID:** 29e1271d
**Status:** COMPLETE (10/10 sections)
**Total tests added:** 90 unit tests across 8 test files

---

## Section-by-Section

### 01. Database Schema
- 6 migration functions in `db.py`: corporate_actions, system_alerts, eventbus_ack, factor_exposure, cycle_attribution, llm_config
- `cycle_attribution: dict = {}` added to `TradingState`
- Tests: `tests/unit/test_db_schema_phase9.py`

### 02. System Alerts
- 5 LangChain tools (emit/acknowledge/escalate/get/summary) + `emit_system_alert()` helper
- Registered in `TOOL_REGISTRY`, bound in supervisor `agents.yaml`
- Tests: `tests/unit/test_system_alerts.py` (30 tests)

### 03. Corporate Actions
- AlphaVantage dividend/split collectors, EDGAR 8-K M&A detector, CIK mapper
- Split auto-adjustment with broker reconciliation
- Scheduled refresh entry point in `scheduled_refresh.py`
- Tests: `tests/unit/test_corporate_actions.py` (22 tests)

### 04. Factor Exposure
- Factor computation: beta, sector, style, momentum crowding
- Configurable drift alerts via `factor_config` table
- Wired into supervisor `health_check` node
- Tests: `tests/unit/test_factor_exposure.py` (16 tests)

### 05. Performance Attribution
- Attribution engine: factor/timing/selection/cost decomposition
- New `attribution` node in trading graph (after reflect, before END)
- DB persistence to `cycle_attribution` table
- Tests: `tests/unit/test_performance_attribution.py` (11 tests)

### 06. Dashboard Alerts
- `AlertsCompact` TUI widget (T1 refresh, ALWAYS_ON)
- `GET /api/alerts` FastAPI endpoint with status/limit params
- `system_alert` CSS/JS type in dashboard
- `emit_system_alert` → `publish_event` SSE integration
- Tests: `tests/unit/test_dashboard_alerts.py` (20 tests)

### 07. EventBus ACK Protocol
- ACK fields on `Event` dataclass + `EventBus.ack()` method
- `check_missed_acks()` with 3-tier escalation: retry → warning → dead-letter
- Wired into supervisor `health_check` node
- Tests: `tests/unit/test_eventbus_ack.py` (12 tests)

### 08. Multi-Mode Operation
- `OperatingMode` enum: MARKET, EXTENDED, OVERNIGHT, WEEKEND
- `get_operating_mode()` with holiday awareness
- Mode-aware cycle intervals per graph
- Monitor-only routing via conditional edges in trading graph
- Risk gate `_check_market_hours()` hard block on exposure-increasing orders
- `operating_mode` field on `TradingState`
- Tests: `tests/unit/test_operating_mode.py` (16) + `tests/unit/test_risk_gate_trading_window.py` (9)

### 09. LLM Unification
- `get_llm_config(tier)`: 3-level precedence (env var > DB > code default) with 60s TTL cache
- `_read_llm_config_from_db()`: graceful fallback when table missing
- `check_provider_health()`: async provider liveness probe (10s timeout per provider)
- `mem0_client.py`: replaced hardcoded `"gpt-4o-mini"` with provider layer call
- `cost_queries.py`: added clarifying comment on intentional hardcoded price table
- Supervisor `health_check`: wired provider health + alert emission
- Tests: `tests/unit/test_llm_unification.py` (11 tests)

### 10. Research Fan-Out
- Default flipped to enabled (`RESEARCH_FAN_OUT_ENABLED=true`)
- `asyncio.Semaphore(10)` bounding concurrent fan-out workers
- AV rate limiter check (`get_calls_this_minute()`) before validation
- Tests: `tests/unit/test_research_fanout.py` (22 tests)

---

## Files Created
| File | Purpose |
|------|---------|
| `src/quantstack/risk/factor_exposure.py` | Factor exposure computation + drift alerts |
| `src/quantstack/risk/attribution_engine.py` | P&L decomposition engine |
| `src/quantstack/tools/functions/system_alerts.py` | Alert lifecycle helper |
| `src/quantstack/tools/langchain/alert_tools.py` | 5 LangChain alert tools |
| `src/quantstack/data/corporate_actions.py` | AV + EDGAR corporate action collectors |
| `src/quantstack/tui/widgets/alerts_widget.py` | TUI alert display widget |
| `tests/unit/test_llm_unification.py` | LLM unification tests |
| `tests/unit/test_research_fanout.py` | Fan-out throttling tests |
| `tests/unit/test_operating_mode.py` | Mode detection tests |
| `tests/unit/test_risk_gate_trading_window.py` | Market hours risk gate tests |
| `tests/unit/test_dashboard_alerts.py` | Dashboard alert tests |
| `tests/unit/test_eventbus_ack.py` | ACK protocol tests |

## Files Modified (key changes)
| File | What changed |
|------|-------------|
| `src/quantstack/db.py` | 6 new migration functions |
| `src/quantstack/graphs/state.py` | `cycle_attribution`, `operating_mode` fields |
| `src/quantstack/graphs/trading/graph.py` | Monitor-only conditional edges, attribution node |
| `src/quantstack/graphs/trading/nodes.py` | Mode-aware data_refresh |
| `src/quantstack/graphs/research/nodes.py` | Semaphore + AV rate guard in validate_symbol |
| `src/quantstack/graphs/supervisor/nodes.py` | Factor exposure, ACK monitoring, LLM health checks |
| `src/quantstack/execution/risk_gate.py` | `_check_market_hours()` method |
| `src/quantstack/coordination/event_bus.py` | ACK protocol on Event/EventBus |
| `src/quantstack/llm/provider.py` | `get_llm_config()`, `check_provider_health()` |
| `src/quantstack/memory/mem0_client.py` | Provider layer integration |
| `src/quantstack/dashboard/app.py` | `/api/alerts` endpoint, alert CSS/JS |
| `src/quantstack/data/fetcher.py` | `get_calls_this_minute()` accessor |
| `src/quantstack/data/scheduled_refresh.py` | Corporate actions phase |
| `src/quantstack/runners/__init__.py` | OperatingMode enum + intervals |

## Recurring Patterns
- **Deferred import patching:** Tests must patch at the source module (e.g., `quantstack.db.db_conn`) not where the deferred import occurs
- **TradingState `extra="forbid"`:** Every new field must be declared before any node returns it
- **Supervisor health_check:** New monitoring checks follow try/except pattern with `logger.warning` on failure
