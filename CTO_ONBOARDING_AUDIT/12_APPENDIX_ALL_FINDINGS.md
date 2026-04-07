# 11 — Appendix: Complete Finding Registry

**Purpose:** Comprehensive cross-reference of all 164+ findings from all three audits. Every finding has a home section. MEDIUM findings from table rows (which had no IDs in the original) are catalogued here with section assignments.

---

## Tool Layer Findings (Not in Other Sections)

These findings relate to the tool registry and tool implementation quality. They don't fit cleanly into other sections but are important for the tool architecture cleanup.

### TC2: Entry Rule Validation Silently Drops Rules — CRITICAL

**Location:** `tools/_shared.py:113-143`

`validate_entry_rules()` checks rules against `_KNOWN_INDICATORS` and silently drops any rule using an unsupported indicator. A strategy with 5 entry rules might end up with 2 after silent pruning — fundamentally changing its behavior.

**Fix:** Return dropped rules as explicit warnings. Better: reject the strategy registration and tell the agent what rules aren't supported.

### TC3: No Tool Execution Timeout — CRITICAL

**Location:** All `@tool` functions

No timeout decorator or wrapper on any tool. A long-running backtest or hung DB query can block an agent indefinitely, consuming LLM tokens while waiting.

**Fix:** Add timeout wrapper to tool execution in `agent_executor.py` (e.g., 30s default, configurable per tool).

### TH1: Kill Switch & Risk Gate Have No Unit Tests — HIGH

**Location:** `execution_tools.py:38, 56` — marked "SACRED, NEVER BYPASSED"

The most critical safety invariants have zero test coverage verifying they actually block trades. If a refactor accidentally breaks the guard, rogue trades could execute.

**Fix:** Add pytest tests: (1) trigger kill switch → verify `execute_order` rejects, (2) set risk gate violation → verify order blocked.

### TH2: Execution Tools Missing 4 of 6 Critical Functions — HIGH

4 of 6 execution tools are stubbed: `get_fills`, `get_audit_trail`, `update_position_stops`, `check_broker_connection`. The `exit_evaluator` references `update_position_stops` for TIGHTEN verdicts — since it's stubbed, tightening stops never happens.

**Fix:** Implement the 4 stubbed execution tools. Priority: `update_position_stops` (blocks exit evaluator), `check_broker_connection` (blocks pre-trade health check).

### TH3: Risk Tools Missing VaR, Stress Testing, Drawdown — HIGH

4 of 6 risk tools are stubbed. `fund_manager` supposed to use `check_risk_limits` for portfolio-level gating — returns stub JSON. Risk assessment purely hardcoded in risk gate.

**Fix:** Implement `compute_var`, `stress_test_portfolio`, `check_risk_limits`, `compute_drawdown`.

### TH4: Storage Failures Silently Suppressed — HIGH

**Location:** `data_tools.py:47-48`

`fetch_market_data()` logs storage failures as warnings but returns success. Agent thinks data was persisted. Subsequent `load_market_data()` fails because data was never saved.

**Fix:** Either fail the tool or include `storage_warning` in response.

### TH5: Broad Exception Catching Hides Root Causes — HIGH

Every tool uses `except Exception` — API timeouts, code bugs, data issues all produce the same opaque error. Debugging requires log correlation.

**Fix:** Catch specific exceptions: `ConnectionError` (retry), `ValueError` (input validation), `TimeoutError` (retry with backoff).

---

## All MEDIUM Findings by Subsystem

### Graph Architecture — MEDIUM

| Finding | Location | Issue | Section |
|---------|----------|-------|---------|
| Cold-start data problem | Portfolio construction | Symbols with <5 days OHLCV rejected. New symbols can't trade for a week. | 08 |
| Fan-out disabled by default | `RESEARCH_FAN_OUT_ENABLED=false` | Sequential validation is 3-5x slower than fan-out mode | 09 |
| Tool category prompting is soft | `agent_executor.py:152-178` | System prompt guides tools but can't prevent wrong tool calls | 05 |
| No streaming between agents | All graphs | Each agent completes fully before next starts — no incremental handoff | 05 |
| Message pruning is heuristic | `agent_executor.py` | 150k char budget, drops oldest tool rounds — could lose critical context | 05 |
| Hypothesis loop can burn 360s | Research graph | 3 iterations x 120s each = 360s for hypothesis refinement | 09 |

### Tool Layer — MEDIUM

| Finding | Location | Issue | Section |
|---------|----------|-------|---------|
| Double-encoded JSON in strategy records | `_shared.py:99-100` | Guard against `json.loads(str)` suggests data layer inconsistency | 05 |
| No input validation on options Greeks | `options_tools.py:40-66` | Volatility and time_to_expiry not range-checked | 03 |
| Keyword-based tool search simplistic | `registry.py:287-317` | Embedding-based search available via pgvector but unused for tool discovery | 07 |
| No tool invocation telemetry | All tools | Can't measure tool usage frequency, latency, error rate per tool | 07 |
| ML training tools all stub | `ml_tools.py` | 5 tools all stub — ML pipeline non-functional through tool layer | 02 |
| Walk-forward validation stubbed | `backtest_tools.py:40-57` | Only single-pass backtest available (fixed in 2.6) | 02 |
| DB connections without context managers | `strategy_tools.py:23-34` | Missing `with pg_conn()` pattern in some tools | 04 |

### Execution & Risk — MEDIUM

| Finding | Location | Issue | Section |
|---------|----------|-------|---------|
| Trailing stop only per holding type | `holding_period.py:60-89` | Swing+ positions should mandate trailing stops | 03 |
| No single-day massive drawdown trigger | `kill_switch.py` | Has rolling 3-day check but no -10% single-day halt | 03 |
| Market circuit breaker only checks SPY | `kill_switch.py:364-373` | Should also check VIX >80, sector ETF halts | 03 |
| Fill polling uses wall-clock time | `alpaca_broker.py:145-146` | 30s timeout, 1s poll — may miss fills on clock skew | 03 |
| `require_ctx()` auto-initializes silently | `_state.py:75` | Creates context if missing — could mask config errors | 04 |

### Data Pipeline — MEDIUM

| Finding | Location | Issue | Section |
|---------|----------|-------|---------|
| No split/dividend adjustment verification | `data/validator.py` | Relies entirely on AV correctness — no cross-check | 08 |
| No forward-fill for trading day gaps | Data pipeline | Weekend/holiday gaps may affect indicator windows | 08 |
| HMM regime needs 120+ bars | `regime.py` | New symbols get rule-based fallback for ~4 months | 08 |
| ML model unavailability silent | `ml_signal.py` | Returns `{}` with no error log — could be broken for weeks | 08 |
| Cache warmer not integrated | `cache_warmer.py` | Exists but runs separately — data warm in cache but stale in engine | 08 |

### LLM & Prompts — MEDIUM

| Finding | Location | Issue | Section |
|---------|----------|-------|---------|
| No few-shot examples in any prompt | All agents | Quality loss ~5-15% vs. prompted with examples | 07 |
| No structured outputs (JSON schema mode) | All agents | Relies on fragile text parsing | 07 |
| No A/B testing framework | All agents | Can't iterate on prompt quality systematically | 09 |
| Backstories are 40-100 lines | `agents.yaml` | Token waste — most value in first 10 lines | 07 |
| No self-critique loop | All agents | Agents don't verify their own JSON before returning | 05 |
| Portfolio optimization two paths | `PortfolioOptimizerAgent` vs `fund_manager` | Potential conflicting sizing if both active | 05 |

### Ops & Infrastructure — MEDIUM

| Finding | Location | Issue | Section |
|---------|----------|-------|---------|
| No secrets manager | `.env` on disk | API keys, DB credentials unencrypted | 04 |
| No env var type validation | Startup | `RISK_MAX_POSITION_PCT=ten` silently fails | 04 |
| Ollama health check doesn't verify models | Docker health check | Service "healthy" but embedding model may not be pulled | 04 |
| No bind mount size limits | Docker volumes | Models/data/logs can fill host disk | 04 |
| No multi-host deployment path | Single host | Single point of failure for everything | 04 |
| Research graph may need >1GB | Docker memory limit | OOM risk during full model training | 04 |
| No deployment rollback procedure | Operations | Bad deploy requires manual git revert + rebuild | 04 |

---

## AutoResearchClaw — MEDIUM Findings

| Finding | Issue | Section |
|---------|-------|---------|
| Weekly schedule, should be nightly | Runs Sunday 20:00 only — 3 tasks/week vs. potential 96/night | 09 |
| Only reactive, never proactive | Only runs on failure (bug_fix) or drift. No gap detection feeds it. | 09 |
| Loop restarts via tmux send-keys | `tmux send-keys -t quantstack-loops:trading "C-c"` — fragile, fails silently if tmux session missing | 04 |
| No fix validation beyond syntax check | py_compile + import check only — no functional validation of fix | 09 |

---

## Retracted Findings (for the record)

| ID | Claimed | Actually Exists | Status |
|----|---------|----------------|--------|
| QS-E2 | Zero margin tracking | `core/risk/span_margin.py` (537 lines) | RETRACTED |
| QS-A1 (partial) | No reconciliation | `guardrails/agent_hardening.py:463-550` | RETRACTED |
| QS-I6 | No position reconciliation job | `execution_monitor._reconcile_loop()` | RETRACTED |
| QS-I7 | No job overlap detection | `strategy_lifecycle.py:422-441` heartbeat guard | RETRACTED |
| Loop-1 (partial) | Zero feedback loops | `hooks/trade_hooks.py:118-144` exists (incomplete) | DOWNGRADED |
| QS-I5 | No audit log | `audit/decision_log.py` with SHA256 hashes | DOWNGRADED to MEDIUM |

---

## Finding Count Summary (Final)

| Severity | Part I (CTO) | Part II (Quant Scientist) | Part III (Deep Ops) | Net (after retractions) |
|----------|-------------|--------------------------|--------------------|-----------------------|
| CRITICAL | 22 | 16 | 5 | **38** |
| HIGH | 34 | 26 | 5 | **60** |
| MEDIUM | 45 | 21 | 3 | **66** |
| RETRACTED | — | -5 | — | **-5** |
| **TOTAL** | **101** | **63** | **13** | **164** |

All 164 findings are now accounted for across sections 01-11.
