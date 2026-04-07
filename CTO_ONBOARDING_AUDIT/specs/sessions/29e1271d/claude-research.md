# Research Findings — Phase 9: Missing Roles & Scale

---

## Part 1: Codebase Analysis

### 1. Alert System

**Location:** `src/quantstack/tools/langchain/alert_tools.py`

**Current state:** Stub tools with DB schema already created. Three tables exist:

| Table | Purpose |
|-------|---------|
| `equity_alerts` | Entry signals with thesis, risk params, debate verdict. Statuses: pending/watching/acted/expired/skipped |
| `alert_exit_signals` | Exit triggers with severity (info/warning/critical/auto_close), signal types (stop_loss_hit, thesis_invalidated, etc.) |
| `alert_updates` | Running commentary — thesis checks, price updates, regime changes |

**Tools defined (stubs):**
- `create_equity_alert()` — entry signals
- `get_equity_alerts()` — retrieve with filtering
- `update_alert_status()` — lifecycle transitions
- `create_exit_signal()` — exit triggers
- `add_alert_update()` — commentary tracking

**Gap for Phase 9:** Item 9.4 wants generic alert lifecycle tools (create/acknowledge/escalate/resolve/query). The existing alert system is equity-specific (trading alerts). Phase 9 needs a *system-level* alert layer (risk breaches, service failures, kill switch). These are complementary, not overlapping.

---

### 2. EventBus

**Location:** `src/quantstack/coordination/event_bus.py` (299 lines)

**Architecture:**
- PostgreSQL append-only event log (table: `loop_events`)
- Poll-based, not push — latency = one iteration cycle
- Per-consumer cursors via `loop_cursors` table
- 7-day TTL for events

**16 event types defined:**
- Strategy lifecycle: `STRATEGY_PROMOTED`, `STRATEGY_RETIRED`, `STRATEGY_DEMOTED`
- ML/Research: `MODEL_TRAINED`, `DEGRADATION_DETECTED`, `SCREENER_COMPLETED`, `UNIVERSE_REFRESHED`
- Health: `LOOP_HEARTBEAT`, `LOOP_ERROR`
- Agent activation: `MARKET_MOVE`, `IDEAS_DISCOVERED`
- Risk (6 types): `RISK_WARNING`, `RISK_SIZING_OVERRIDE`, `RISK_ENTRY_HALT`, `RISK_LIQUIDATION`, `RISK_EMERGENCY`, `RISK_ALERT`, `MODEL_DEGRADATION`, `IC_DECAY`, `REGIME_CHANGE`

**API:**
```python
bus = EventBus(conn)
bus.publish(event) → str  # Returns event_id
bus.poll(consumer_id, event_types) → list[Event]
bus.get_latest(event_type) → Event | None
bus.count_events(event_type, since) → int
```

**Gap for Phase 9 (item 9.6):** No ACK pattern. Fire-and-forget only. Need: `expected_ack_by` on publish, `ack(event_id)` on consumer, background monitor for missing ACKs.

---

### 3. Graph Runners & Multi-Mode

**Location:** `src/quantstack/runners/`, `src/quantstack/graphs/`

**Three independent async loops:**

| Graph | Market Hours | After Hours | Weekend |
|-------|-------------|-------------|---------|
| Trading | 300s (5min) | None (stopped) | None (stopped) |
| Research | 120s | 180s | 300s |
| Supervisor | 300s | 300s | 300s |

**Market hours detection:** `src/quantstack/runners/__init__.py`
- Constants: `MARKET_OPEN = 9:30 ET`, `MARKET_CLOSE = 16:00 ET`
- NYSE holidays through 2027
- `get_cycle_interval(graph_name)` returns interval or None (graph should sleep)

**Gap for Phase 9 (item 9.7):** Current modes are binary: market hours vs not. Need three modes (market, extended, overnight/weekend) with different behaviors per graph. Extended hours (4:00-9:30, 16:00-20:00) should run trading graph in monitor-only mode.

---

### 4. LLM Routing

**Location:** `src/quantstack/llm_config.py` (561 lines), `src/quantstack/llm/provider.py` (248 lines)

**Architecture:** Tier-based routing with multi-provider fallback chain.

**8 tiers:** ic, pod, assistant, decoder, workshop, autonomous_pm, research, bulk

**12 providers:** Bedrock, Anthropic, OpenAI, Vertex AI, Gemini, Azure OpenAI, Groq, Together AI, Fireworks AI, Mistral, Ollama, Custom OpenAI

**Fallback order:** `bedrock → anthropic → openai → ollama`

**Key functions:**
- `get_chat_model(tier, thinking=None)` — primary entry point
- `get_llm_for_agent(agent_name)` — resolves tier from agent name suffix
- `get_llm_for_role(role)` — role-based lookup

**Gap for Phase 9 (item 9.8):** Despite the unified system, hardcoded model strings exist in 6+ locations. Need to audit and route all through `get_chat_model()`.

---

### 5. Factor Exposure & Performance Attribution

**Attribution tools:** `src/quantstack/tools/langchain/attribution_tools.py`
- `get_daily_equity(start_date, end_date)` — equity curve, Sharpe, Sortino, max drawdown
- `get_strategy_pnl(strategy_id, ...)` — per-strategy P&L with credit breakdown (signal, regime, sizing, debate)

**Risk tools:** `src/quantstack/tools/langchain/risk_tools.py`
- `compute_risk_metrics()` — portfolio equity, daily P&L, concentration, exposure
- `compute_position_size()` — ATR or Kelly
- `compute_var()` — Value at Risk
- `stress_test_portfolio()` — market_crash, vol_spike, sector_rotation
- `compute_max_drawdown()`

**Related modules:**
- `src/quantstack/portfolio/optimizer.py` — portfolio optimization
- `src/quantstack/risk/portfolio_risk.py` — portfolio-level risk
- `src/quantstack/risk/monitoring.py` — risk monitoring
- `src/quantstack/performance/benchmark.py` — benchmark comparison

**Gap for Phase 9 (items 9.2, 9.3):** No per-cycle factor exposure computation (beta, sector tilt, style). Attribution runs nightly only. Need: per-cycle computation wired into trading graph, alerts on drift thresholds.

---

### 6. Corporate Actions

**Status:** No existing implementation. No tables, no tools, no data pipeline.

**Where to build:**
- New `corporate_actions` table
- New deterministic collector (no LLM needed)
- Position management needs `split_adjustment` handling
- Alert system integration for upcoming ex-dates

---

### 7. Notification / Discord

**Daily Digest:** `src/quantstack/coordination/daily_digest.py`
- `DailyDigest.generate()` → `DigestReport` dataclass
- Covers: positions, trades, strategy lifecycle, loop health, risk events, ML, screener
- `format_markdown()` for human-readable output

**Discord Client:** `src/quantstack/tools/discord/client.py` (80 lines)
- `DiscordClient(token, token_type="user")` — supports bot and user tokens
- `send_message(channel_id, content)` — basic message send
- Rate limit handling: respects 429 with `Retry-After`, 0.5s inter-request sleep
- Uses Discord REST API v10

**Gap for Phase 9 (item 9.5):** Discord client exists but uses token auth (DMs/channels), not webhooks. Need webhook-based sending for server channels. No integration with alert system or graph events. Need: webhook client, severity-based routing, batching for non-critical.

---

### 8. Tool Registry

**Location:** `src/quantstack/tools/registry.py`

**Architecture:** Central dictionary mapping tool names → tool objects. Agent configs in YAML reference tools by string name. Tool binding in `src/quantstack/graphs/tool_binding.py` supports three strategies:
1. **Anthropic native:** BM25 search with defer_loading
2. **Bigtool:** pgvector semantic search
3. **Full loading:** all tools bound upfront

**Pattern for new tools:** Import in registry.py, add to `TOOL_REGISTRY` dict, reference by name in agent YAML.

---

### 9. Testing Setup

**Framework:** pytest with custom markers (slow, integration, benchmark, requires_api, regression)

**Key fixtures in `tests/conftest.py`:**
- `trading_ctx` — fully-wired TradingContext backed by PostgreSQL
- Component fixtures: `signal_cache`, `risk_state`, `portfolio`, `paper_broker`, `kill_switch`, `tick_executor`
- `reset_singletons_and_seeds` (autouse) — prevents state pollution
- `sample_ohlcv_df` — 100-bar random walk
- OHLCV generators: `make_flat_market`, `make_monotonic_uptrend`, `make_impulse_wave_ohlcv`, etc.

**Test organization:** ~189 unit tests, 24 integration dirs, coordination tests, graph tests, property-based tests, benchmarks, regression suite.

---

### 10. Research Fan-Out

**Location:** `src/quantstack/graphs/research/graph.py`

**Control:** `RESEARCH_FAN_OUT_ENABLED` env var (default: `"false"`)

**Fan-out path (when enabled):**
`hypothesis_critique → fan_out_hypotheses → validate_symbol (parallel) → filter_results → strategy_registration → knowledge_update`

**Sequential path (default):**
`hypothesis_critique → signal_validation → backtest_validation → ml_experiment → strategy_registration → knowledge_update`

**Gap for Phase 9 (item 9.9):** Just need to flip default to `true` and add rate limiting guards for AV quota during parallel validation.

---

## Part 2: Web Research

### Corporate Actions Data Sources

**Alpha Vantage:**
- `DIVIDENDS` endpoint: ex_date, declaration_date, record_date, payment_date, amount. ~27yr history.
- `SPLITS` endpoint: effective_date, split_factor. Requires paid key.
- **No M&A endpoint.** Dividends and splits only.
- Data is EOD, appears after effective/ex date (not at announcement).

**EDGAR 8-K filings for corporate events:**

| 8-K Item | Event |
|----------|-------|
| 1.01 | M&A signing |
| 2.01 | Acquisition/disposition completion |
| 3.03 | Rights modification (splits, dividends) |
| 5.01 | Change in control (takeover) |
| 5.02 | C-suite changes |

**Python library:** `edgartools` (PyPI) — structured 8-K parsing. EDGAR API rate limit: 10 req/s with User-Agent header.

**Cost basis adjustment best practices:**
1. Adjust on effective date, not announcement
2. Formula: new_qty = old_qty * split_ratio; new_cost_per_share = old_cost / split_ratio; total cost unchanged (invariant)
3. Idempotency via `split_adjustments` audit table with `(symbol, effective_date)` unique constraint
4. Reverse splits: ratio < 1, watch for fractional shares (brokers cash out)

**Recommended layered approach:**
- Primary: AV for dividends/splits (structured, simple)
- Early warning: EDGAR 8-K for M&A
- Validation: Cross-reference with Alpaca position data (auto-adjusts on splits)

---

### Discord Webhook Patterns

**Rate limits:**
- Global: 50 req/s
- Per-webhook: 5 req/s (~30 messages/min)
- 10,000 invalid requests per 10 min → Cloudflare IP ban

**Batching strategy:**
- CRITICAL (immediate): kill switch, risk breach, execution failure, liquidation
- HIGH (15s batch): order confirmations
- MEDIUM (60s batch): research findings, lifecycle updates
- LOW (300s batch): data freshness, routine status

**Retry strategy:**
- 429: use server-provided `retry_after` value exactly
- 5xx: exponential backoff (1s, 2s, 4s, 8s, max 30s), max 3-5 retries
- 401/403: never retry (counts toward ban threshold)
- After max retries: dead-letter and local error log

**Message formatting:**
- Always use embeds for structured alerts (not plain text)
- Color by severity: red=CRITICAL, orange=HIGH, yellow=MEDIUM, green=LOW
- Embed limits: title 256 chars, description 4096, total 6000, 25 fields, 10 embeds/message
- Pattern: `[SEVERITY] Title` as embed title, context as inline fields, action required as non-inline field

---

### Event-Driven ACK/NACK Patterns

**Key insight:** Decouple ACK check from delivery path. Publisher never blocks waiting for ACK.

**Pattern for QuantStack's PostgreSQL-based EventBus:**
1. On publish: set `expected_ack_by = now + timeout` in the event row
2. Consumer calls `bus.ack(event_id)` after processing → sets `acked_at` timestamp
3. Background monitor (supervisor graph) queries for events where `acked_at IS NULL AND expected_ack_by < now()`
4. Escalation tiers: T1 (30s) retry, T2 (120s) warn, T3 (300s) dead-letter + CRITICAL alert

**Dead letter handling:**
- Persist to `dead_letter_events` table on every dead-letter transition
- Trading graph dead letters → CRITICAL alert (financial impact)
- Research graph dead letters → WARNING (can retry)
- Manual replay endpoint for re-publishing dead letters

**Implementation approach:** Extend existing `EventBus` rather than replacing it. Add columns to `loop_events`: `requires_ack BOOLEAN`, `expected_ack_by TIMESTAMPTZ`, `acked_at TIMESTAMPTZ`, `acked_by TEXT`. Add `dead_letter_events` table. Add `bus.ack(event_id, consumer_id)` method. Monitor query runs in supervisor graph.
