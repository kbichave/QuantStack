# QuantStack CTO Architecture Audit

**Auditor:** Principal Quant Scientist / Incoming CTO/ Staff Agnetic AI Scientist
**Date:** 2026-04-06
**Objective:** Identify inefficiencies in the agentic architecture, prompts, agents, tools, and infrastructure. Recommend path to 24/7 autonomous trading.

---

## Table of Contents

### Part I: CTO Architecture Audit
1. [Executive Summary](#executive-summary)
2. [Graph Architecture Findings](#graph-architecture)
3. [Tool Layer Findings](#tool-layer)
4. [Execution & Risk Management Findings](#execution-risk)
5. [Data Pipeline & Signal Engine Findings](#data-signals)
6. [LLM Routing & Prompt Engineering Findings](#llm-prompts)
7. [Ops & Infrastructure Findings](#ops-infra)
8. [Memory Architecture: RAG vs. File-Based Analysis](#memory-architecture)
9. [OpenClaw Feature Benchmark & Adoption Opportunities](#openclaw-benchmark)
10. [24/7 Readiness Gap Analysis](#247-readiness)
11. [Priority Recommendations](#recommendations)

### Part II: Principal Quant Scientist Deep Audit (68 new findings)
12. [Execution Layer Deep Audit: "The System Can't Actually Trade"](#12-execution-layer-deep-audit-the-system-cant-actually-trade)
13. [Signal Quality & Statistical Rigor: "We Don't Know If The Signals Work"](#13-signal-quality--statistical-rigor-we-dont-know-if-the-signals-work)
14. [Backtesting Validity: "The Backtests Can't Be Trusted"](#14-backtesting-validity-the-backtests-cant-be-trusted)
15. [ML Pipeline: "The Models Are Untested in Production Conditions"](#15-ml-pipeline-the-models-are-untested-in-production-conditions)
16. [Agent Architecture Deep Audit: "The Agents Don't Talk to Each Other"](#16-agent-architecture-deep-audit-the-agents-dont-talk-to-each-other)
17. [Feedback Loops & Learning: "The System Doesn't Learn From Its Mistakes"](#17-feedback-loops--learning-the-system-doesnt-learn-from-its-mistakes)
18. [Infrastructure & Security Deep Audit](#18-infrastructure--security-deep-audit)
19. [Revised Priority Recommendations (Combined)](#19-revised-priority-recommendations-cto--quant-scientist-combined)
20. [What The Previous CTO Got Right (And Missed)](#20-what-the-previous-cto-got-right-credit-where-due)

---

## Executive Summary

### Bottom Line

QuantStack is an **impressively architected autonomous trading system** — 21 specialized agents across 3 LangGraph pipelines, 16 concurrent signal collectors, a multi-layer risk gate, and deterministic exit enforcement. The engineering quality is high. The system is capable of generating alpha in paper trading.

**However, it is NOT ready for 24/7 unattended operation with real capital.**

The audit identified **22 CRITICAL findings, 34 HIGH findings, and 41 MEDIUM findings** across 8 subsystems (including Memory & Cost Optimization benchmarked against OpenClaw, Alert Coordination gap analysis, and AutoResearchClaw audit). The most dangerous issues are:

1. **Stop-losses are optional** — an LLM agent can place a trade with no downside protection
2. **Zero prompt caching configured** — paying 10x what we should on system prompt tokens (~$126/day vs $12.60/day)
3. **`search_knowledge_base` bypasses RAG entirely** — queries by recency, not relevance. Agents get random recent entries instead of semantically matching context
4. **Non-deterministic tool ordering destroys prompt cache** — even once caching is enabled, tool ordering variance will break it (1-line fix)
5. **Prompt injection vulnerabilities** — untrusted data injected into LLM prompts without sanitization
6. **No database backups** — all system state lives in a single PostgreSQL with no backup procedure
7. **Silent output validation failures** — LLM garbage silently continues with empty defaults
8. **Scheduler runs in tmux** — if it crashes, 13 critical jobs stop with no alerting

### Finding Summary by Subsystem

| Subsystem | CRITICAL | HIGH | MEDIUM | Overall Grade |
|-----------|----------|------|--------|---------------|
| Execution & Risk | 3 | 5 | 5 | B- |
| Graph Architecture | 3 | 4 | 6 | B |
| Data Pipeline & Signals | 3 | 5 | 5 | B |
| LLM Routing & Prompts | 3 | 5 | 8 | C+ |
| Ops & Infrastructure | 3 | 5 | 7 | C+ |
| Tools & Registry | 3 | 5 | 7 | C |
| Memory & Cost Optimization | 4 | 2 | 2 | C (massive cost leak) |
| Alert Coordination | 0 | 3 | 1 | D+ |
| AutoResearchClaw | 0 | 0 | 4 | B+ (under-utilized) |
| **TOTAL** | **22** | **34** | **45** | **B-** |

### What's Working Well

- **Signal Engine**: 16 concurrent collectors, 2-6s wall-clock, fault-tolerant, regime-adaptive weights
- **Risk Gate**: Multi-layer enforcement (daily loss, position caps, gross exposure, options DTE/premium)
- **Kill Switch**: Two-layer design (DB + sentinel file) survives process crashes
- **Agent Specialization**: 21 agents with clear, non-overlapping roles
- **Execution Monitor**: Deterministic exit rules evaluated on every price tick
- **Strategy Lifecycle**: Draft → backtested → forward_testing → live → retired with evidence-based gates
- **Self-Healing**: AutoResearchClaw patches tool failures, supervisor diagnoses and recovers
- **Observability**: Langfuse tracing on every node, LLM call, and tool invocation

### Key Decision: RAG vs. File-Based Memory

**Verdict: Stay with file-based memory. RAG would cost us ~8x more per session by destroying prompt cache hits.**

Our `.claude/memory/*.md` approach is architecturally correct for our scale (<50K tokens). The prompt caching economics are decisive: static memory prefixes get 90% token cost discount on subsequent calls, while RAG-retrieved chunks break the cache on every query. At 24/7 operation with 5-min cycles, cache hits compound to thousands in annual savings. OpenClaw (349K stars) validated this same conclusion — they use MD files as the primary layer with SQLite+vector only for overflow. See Section 8 for full analysis.

### What Must Change Before Real Capital

1. Mandatory stop-loss on every position (non-negotiable)
2. Deterministic tool ordering for prompt cache stability (1-line fix, 30-50% cost savings)
3. Prompt injection defense (sanitize all LLM inputs)
4. Output schema validation with retry (don't silently continue on parse failure)
5. Automated database backups (daily pg_dump + offsite storage)
6. Containerize the scheduler (move into Docker Compose)
7. Enable CI/CD (automated tests before deploy)

---

## Graph Architecture Findings

### Overall Assessment: SOLID MULTI-AGENT DESIGN WITH COORDINATION GAPS

Three LangGraph StateGraphs running as independent Docker services with PostgreSQL-backed inter-graph communication. Agent specialization is excellent. But serialization bottlenecks, checkpoint fragility, and slow inter-graph feedback loops would prevent reliable 24/7 operation.

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Docker Compose                      │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Trading     │  │  Research     │  │  Supervisor   │ │
│  │  Graph       │  │  Graph        │  │  Graph        │ │
│  │  5-min cycle │  │  10-min cycle │  │  5-min cycle  │ │
│  │  market hrs  │  │  market hrs   │  │  always-on    │ │
│  │  10 agents   │  │  7 agents     │  │  4 agents     │ │
│  └──────┬───────┘  └──────┬────────┘  └──────┬────────┘ │
│         │                 │                  │          │
│         └─────────┬───────┴──────────┬───────┘          │
│                   │                  │                   │
│          ┌────────▼────────┐  ┌──────▼──────┐           │
│          │   PostgreSQL    │  │   Langfuse   │           │
│          │   + pgvector    │  │   Tracing    │           │
│          └─────────────────┘  └──────────────┘           │
│          ┌─────────────────┐  ┌──────────────┐           │
│          │     Ollama      │  │  FinRL Worker │           │
│          │   Embeddings    │  │  (torch/RL)   │           │
│          └─────────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────┘
```

### Trading Graph — 16 Nodes, 2 Parallel Branches

**Pipeline Flow:**
```
START → data_refresh → safety_check
                      ↓ (halted? → END)
                      market_intel → daily_plan
                      ↓
        ┌─────────────┴──────────────┐  (PARALLEL)
        position_review              entry_scan
        ↓                            ↓ (earnings? → earnings_analysis)
        execute_exits                ↓
        └─────────────┬──────────────┘
                      merge_parallel
                      ↓
                      risk_sizing → (rejected? → END)
                      ↓
                      portfolio_construction
                      ↓
        ┌─────────────┴──────────────┐  (PARALLEL)
        portfolio_review             analyze_options
        └─────────────┬──────────────┘
                      merge_pre_execution
                      ↓
                      execute_entries → reflect → END
```

**Key Design Decisions:**
- Deterministic nodes (data_refresh, risk_sizing, portfolio_construction) have no LLM — fast and predictable
- Two parallel branch points maximize throughput
- Safety check is fail-fast (no retry) — correct for a kill switch gate
- Fund manager acts as final portfolio-level approval authority

### Research Graph — 8 Nodes, Self-Critique Loop

**Pipeline Flow:**
```
START → context_load → domain_selection → hypothesis_generation
                                          ↓
                                    hypothesis_critique
                                    ↓ (confidence < 0.7 AND attempts < 3?)
                                    ↓ YES: loop back to hypothesis_generation
                                    ↓ NO: forward
                                    signal_validation → (failed? → END)
                                    ↓
                                    backtest_validation → ml_experiment
                                    ↓
                                    strategy_registration → knowledge_update → END
```

**Self-Critique Loop (WI-8):** hypothesis_critic scores 0-1. If < 0.7 and attempts < 3, loops back to hypothesis_generation with critique feedback. Good design — prevents low-quality hypotheses from wasting backtest compute.

**Fan-Out Mode (WI-7, opt-in):** `RESEARCH_FAN_OUT_ENABLED=true` enables parallel per-symbol validation workers via LangGraph `Send()`. Disabled by default — enables 3-5x throughput when activated.

### Supervisor Graph — 7 Nodes, Linear Pipeline

```
START → health_check → diagnose_issues → execute_recovery
      → strategy_pipeline → strategy_lifecycle → scheduled_tasks → eod_data_sync → END
```

Always-on (runs weekends/holidays). Handles: heartbeat monitoring, self-healing playbooks, strategy promotion/retirement, community intel scans, EOD data sync.

### CRITICAL Findings

#### GC1: MemorySaver Checkpoints Lost on Container Restart

**Location:** All graph runners, `MemorySaver` (in-process only)

All three graphs use LangGraph's `MemorySaver` for mid-cycle checkpointing. This is in-process memory only. If a container crashes or is restarted mid-cycle, all intermediate state is lost. The `graph_checkpoints` Postgres table only captures final cycle outcomes, not intermediate node state.

**Impact:** A crash during `execute_entries` (after risk approval, before order submission) could leave the system in an inconsistent state — approved trades never executed, no record of the approval.

**Fix:** Switch to `PostgresSaver` or `SqliteSaver` for durable checkpointing. LangGraph supports this natively. Enables resumption from the exact node where the crash occurred.

#### GC2: No Inter-Graph Urgency Channel

**Location:** `coordination/event_bus.py`, all runners

Supervisor discovers critical issues (strategy IC decay, kill switch condition) but communicates via Postgres EventBus. The trading graph only polls at the start of each 5-min cycle. Worst case: 5 minutes of trading on a decayed strategy before the trading graph sees the IC_DECAY event.

**Impact:** In a fast-moving market, 5 minutes of stale-strategy trading could cause significant losses.

**Fix:** Add a "priority interrupt" mechanism — either a shared Redis/file sentinel that trading checks before each trade execution, or a direct process signal (SIGUSR1) from supervisor to trading container.

#### GC3: Serialization Bottleneck in Trading Pipeline

**Location:** `trading/graph.py`

The trade_debater (600s timeout) must complete before fund_manager (300s timeout) can evaluate. This means the critical path from entry_scan to execute_entries is 900s minimum — 15 minutes just for the LLM reasoning path.

In a 5-minute cycle, this means the trading graph regularly overruns its cycle interval. The watchdog timeout is 600s, but the full pipeline can take 900s+.

**Impact:** Missed entries due to stale signals by the time execution happens. Cycle overlap risk.

**Fix:** Stream debater outputs incrementally to fund_manager (process candidates as they're debated, not as a batch). Or: run debater with a tighter timeout (120s per candidate) and parallelize across candidates.

### HIGH Findings

#### GH1: Research Queue Contention — Missing Row Lock

**Location:** Research `context_load` node

When claiming tasks from `research_queue`, there's no `SELECT FOR UPDATE` lock. If two research cycles overlap (shouldn't happen in normal operation but possible on restart), both could claim the same task.

**Fix:** Add `FOR UPDATE SKIP LOCKED` to the claim query.

#### GH2: Covariance Matrix Falls Back to Identity Matrix

**Location:** `portfolio_construction` node

If OHLCV data is stale (>2 trading days), the portfolio optimizer uses `last_covariance` from state. If that's empty (first cycle), it falls back to an identity matrix scaled by 0.02. This means the optimizer assumes zero correlation between all assets — destroying diversification logic.

**Fix:** Never fall back to identity. Use EWMA covariance estimator that degrades gracefully, or block portfolio construction until fresh data is available.

#### GH3: 3+ Consecutive Failures Only Log — No Escalation

**Location:** `run_loop()` in all runners

When a graph fails 3+ consecutive cycles, it logs a CRITICAL message but takes no other action. It doesn't trigger the kill switch, notify the supervisor, or pause trading.

**Fix:** After 3 consecutive failures: (1) publish a GRAPH_FAILURE event to EventBus, (2) if trading graph, auto-trigger kill switch, (3) supervisor should have a dedicated handler for this event.

#### GH4: Supervisor Graph is Linear — No Conditional Routing

The supervisor runs all 7 nodes sequentially every cycle, even when everything is healthy. Strategy pipeline backtests every draft strategy every 5 minutes. EOD data sync runs at 2 PM.

**Fix:** Add conditional routing: skip strategy_pipeline if no new drafts. Skip eod_data_sync unless within EOD window. Skip diagnose_issues + execute_recovery if health_check returns healthy.

### MEDIUM Findings

| Finding | Location | Issue |
|---------|----------|-------|
| Cold-start data problem | Portfolio construction | Symbols with <5 days OHLCV rejected (vol floor). New symbols can't be traded for a week. |
| Fan-out disabled by default | `RESEARCH_FAN_OUT_ENABLED=false` | Sequential hypothesis validation is 3-5x slower than fan-out mode |
| Tool category prompting is soft guidance | `agent_executor.py:152-178` | System prompt injects tool guidance but can't prevent agents from calling wrong tools |
| No streaming between agents | All graphs | Each agent completes fully before next starts — no incremental handoff |
| Message pruning is heuristic | `agent_executor.py` | 150k char budget, drops oldest tool rounds — could lose critical context |
| Hypothesis loop can burn 360s | Research graph | 3 iterations × 120s each = 360s just for hypothesis refinement |

### Inter-Graph Communication Mechanisms

| Mechanism | Source | Consumer | Latency |
|-----------|--------|----------|---------|
| EventBus (Postgres) | Community Intel → IDEAS_DISCOVERED | Research context_load | 0-10 min |
| EventBus (Postgres) | Supervisor → IC_DECAY | Supervisor strategy_lifecycle | 0-5 min |
| research_queue table | External / drift detector | Research context_load | 0-10 min |
| strategies table | Research → register | Trading → daily_plan reads | 0-5 min |
| signal_ic table | Research ML experiments | Trading risk_sizing | 0-5 min |
| heartbeat table | All graphs write | Supervisor health_check reads | 0-5 min |
| graph_checkpoints table | All runners write | Dashboard reads | Real-time |

**Key Gap:** No mechanism for supervisor to urgently interrupt trading. All communication is poll-based with 5-10 minute latency.

### Resource Allocation (Docker)

| Service | Memory | CPU | Notes |
|---------|--------|-----|-------|
| postgres | 512m | — | May be tight with 60+ tables and pgvector |
| ollama | 4g | — | Embedding model is large |
| trading-graph | 1g | — | 10 agents, adequate |
| research-graph | 1g | — | 7 agents + ML training, may be tight |
| supervisor-graph | 512m | — | 4 agents, adequate |
| finrl-worker | 2g | — | torch + RL, may need more for training |
| langfuse | 512m | — | Adequate for self-hosted |
| dashboard | 256m | — | Lightweight |

---

## Tool Layer Findings

### Overall Assessment: SOLID INFRASTRUCTURE, 75% UNIMPLEMENTED

The tool architecture is well-designed — clean registry, three binding modes (Anthropic native, Bedrock/bigtool, legacy), proper state management, and sacred kill switch/risk gate guards. But **92 of 122 tools (75%) are stubbed** with placeholder code. Agents calling unimplemented tools get `{"error": "Tool pending implementation"}` and fail silently.

### Tool Implementation Status

| Category | Total | Implemented | Stubbed | Status |
|----------|-------|-------------|---------|--------|
| Signal & Data | 14 | 10 | 4 | Mostly ready |
| Execution | 6 | 2 | 4 | Critical gaps |
| Risk | 6 | 2 | 4 | Critical gaps |
| Options | 8 | 2 | 6 | Mostly stub |
| Strategy & Backtest | 7 | 4 | 3 | Partial |
| ML & Training | 5 | 0 | 5 | All stub |
| FinRL | 11 | 0 | 11 | All stub |
| Alert Lifecycle | 5 | 0 | 5 | All stub |
| Meta Orchestration | 6 | 0 | 6 | All stub |
| QC Research | 15 | 2 | 13 | Mostly stub |
| EWF | 2 | 2 | 0 | Complete |
| Portfolio/Calendar/System | 5 | 4 | 1 | Ready |
| Others (22 single-tool modules) | 22 | 1 | 21 | Mostly stub |
| **TOTAL** | **122** | **30** | **92** | **75% unimplemented** |

### CRITICAL Findings

#### TC1: 92 Stubbed Tools Registered in TOOL_REGISTRY

**Location:** `tools/registry.py:198-276`, all stub modules

All 92 unimplemented tools are registered in `TOOL_REGISTRY` and bound to agents via `agents.yaml`. When an agent invokes any of these, it gets:
```json
{"error": "Tool pending implementation", "status": "not_available"}
```

The agent then has to reason about this error, wasting an LLM round-trip and potentially making decisions without the intended data.

**Most impactful missing tools:**
- `get_fills`, `get_audit_trail`, `check_broker_connection` — execution monitoring blind spots
- `compute_var`, `stress_test_portfolio`, `check_risk_limits` — risk assessment gaps
- `train_model`, `predict_ml_signal`, `check_concept_drift` — ML pipeline broken
- All 11 FinRL tools — DRL model lifecycle completely unavailable
- All 5 alert tools — cannot create, track, or manage equity alerts
- `run_walkforward` — no walk-forward validation for strategies
- `validate_signal`, `diagnose_signal`, `detect_leakage` — no signal quality assurance

**Fix:** Either implement or remove from TOOL_REGISTRY. Agents should never be offered tools that don't work.

#### TC2: Entry Rule Validation Silently Drops Rules

**Location:** `tools/_shared.py:113-143`

`validate_entry_rules()` checks rules against `_KNOWN_INDICATORS` and silently drops any rule using an unsupported indicator. The caller (agent) never knows. A strategy with 5 entry rules might end up with 2 after silent pruning — fundamentally changing its behavior.

**Fix:** Return dropped rules as explicit warnings. Better: reject the strategy registration and tell the agent what rules aren't supported.

#### TC3: No Tool Execution Timeout

**Location:** All @tool functions

No timeout decorator or wrapper on any tool. A long-running backtest or hung DB query can block an agent indefinitely, consuming LLM tokens while waiting.

**Fix:** Add timeout wrapper to tool execution in `agent_executor.py` (e.g., 30s default, configurable per tool).

### HIGH Findings

#### TH1: Kill Switch & Risk Gate Have No Unit Tests

**Location:** `execution_tools.py:38, 56` — marked "SACRED, NEVER BYPASSED"

The most critical safety invariants in the entire system have zero test coverage verifying they actually block trades. If a refactor accidentally breaks the guard, rogue trades could execute.

**Fix:** Add pytest tests: (1) trigger kill switch → verify `execute_order` rejects, (2) set risk gate violation → verify order blocked.

#### TH2: Execution Tools Missing Critical Functions

4 of 6 execution tools are stubbed:
- `get_fills` — can't query fill history
- `get_audit_trail` — can't review execution decisions
- `update_position_stops` — can't adjust stop-losses after entry
- `check_broker_connection` — can't verify broker health before trading

The exit_evaluator agent references `update_position_stops` for TIGHTEN verdicts. Since it's stubbed, tightening stops never actually happens.

#### TH3: Risk Tools Missing VaR, Stress Testing, Drawdown

4 of 6 risk tools are stubbed. The fund_manager agent is supposed to use `check_risk_limits` for portfolio-level gating, but it returns stub JSON. The risk assessment is purely based on what's hardcoded in the risk gate — no dynamic scenario analysis.

#### TH4: Storage Failures Silently Suppressed

**Location:** `data_tools.py:47-48`

`fetch_market_data()` logs storage failures as warnings but returns success to the caller. The agent thinks data was persisted. A subsequent `load_market_data()` call will fail because the data was never actually saved.

**Fix:** Either fail the tool or include a `storage_warning` field in the response.

#### TH5: Broad Exception Catching Hides Root Causes

Every tool uses `except Exception` — API timeouts, code bugs, data issues, and configuration errors all produce the same opaque error JSON. Debugging requires log correlation rather than error classification.

**Fix:** Catch specific exception types: `ConnectionError` (retry), `ValueError` (input validation), `TimeoutError` (retry with backoff), `KeyError` (missing config).

### What's Working Well

| Component | Quality | Notes |
|-----------|---------|-------|
| `registry.py` | Excellent | Clean resolution, deferred loading, graceful degradation |
| `_shared.py` | Excellent | Shared backtest/strategy logic, async support, detailed logging |
| `_state.py` | Good | Global context management, IC cache, safe error handling |
| `_helpers.py` | Good | Common patterns encapsulated, no duplication |
| `models.py` | Good | 451 lines of Pydantic I/O models covering all major tools |
| `execute_order` | Excellent | Kill switch guard, risk gate check, auto-sizing, audit logging |
| `signal_brief` | Excellent | Lazy-loads SignalEngine, IC cache, multi-symbol parallel |
| `ewf_tools` | Excellent | Multi-timeframe, freshness windowing, graceful degradation |
| Tool binding | Good | 3 modes (Anthropic native, Bedrock/bigtool, legacy), fallback chain |

### Tool Architecture

```
agents.yaml (tool names per agent)
    ↓
registry.py (TOOL_REGISTRY: name → BaseTool)
    ↓
tool_binding.py (3 binding modes)
    ├→ Anthropic: defer_loading + BM25 search
    ├→ Bedrock: langgraph-bigtool + pgvector semantic search
    └→ Legacy: all tools bound upfront
    ↓
agent_executor.py (run_agent loop)
    ↓
@tool functions (langchain/)
    ↓
_shared.py / functions/ (implementation logic)
    ↓
DB / APIs / Broker
```

### MEDIUM Findings

| Finding | Location | Issue |
|---------|----------|-------|
| Double-encoded JSON in strategy records | `_shared.py:99-100` | Guard against `json.loads(str)` suggests data layer inconsistency |
| No input validation on options Greeks | `options_tools.py:40-66` | Volatility and time_to_expiry not range-checked |
| Keyword-based tool search is simplistic | `registry.py:287-317` | Embedding-based search (pgvector) already available but unused for tool discovery |
| No tool invocation telemetry | All tools | Can't measure tool usage frequency, latency, or error rate per tool |
| ML training tools completely missing | `ml_tools.py` | 5 tools all stub — ML pipeline effectively non-functional through tools |
| Walk-forward validation stubbed | `backtest_tools.py:40-57` | Can't validate strategy robustness — only single-pass backtest available |
| DB connections without context managers | `strategy_tools.py:23-34` | Missing `with pg_conn()` pattern in some tools |

---

## Execution & Risk Management Findings

### Overall Assessment: MEDIUM-HIGH RISK

The execution layer is well-architected with multi-layer enforcement (RiskGate -> OrderLifecycle -> Broker), but has critical gaps that would prevent safe 24/7 operation.

### What's Working Well

| Control | Implementation | Location |
|---------|---------------|----------|
| Daily loss limit (-2%) | Sentinel file survives restarts | `risk_gate.py:433-451` |
| Per-symbol position caps | 10% equity OR $20k (whichever lower) | `risk_gate.py:592-661` |
| Gross exposure limit | 150% max | `risk_gate.py:663-681` |
| Participation rate | Caps orders to 1% ADV | `risk_gate.py:466-477` |
| Options DTE bounds | Min 7, Max 60 | `risk_gate.py:519-563` |
| Kill switch | Auto-triggers on broker failures, SPY halt, rolling drawdown | `kill_switch.py:284-455` |
| OMS duplicate prevention | Same symbol+side within 60s rejected | `order_lifecycle.py:502-519` |
| Execution algo selection | Size-adaptive: IMMEDIATE/TWAP/VWAP/POV based on % ADV | `order_lifecycle.py:455-478` |
| Strategy breaker | Per-strategy circuit break at 5% drawdown or 3 consecutive losses | `strategy_breaker.py:54-200` |
| Paper broker realism | Half-spread slippage + sqrt volume impact + partial fills | `paper_broker.py:167-240` |

### CRITICAL Findings (Must Fix)

#### C1: Stop-Loss is OPTIONAL — Can Place Trades with `stop_price=None`

**Location:** `trade_service.py:99, 212-220`

The `ATRPositionSizer` requires a stop_loss parameter, but the entry point in `trade_service.py` allows `None`. This means an LLM agent can submit a trade with no stop-loss and it will pass the risk gate. For an autonomous system with no human oversight, this is an existential risk.

**Fix:** Enforce `stop_price is not None` in `OrderRequest` validation. Reject at OMS level.

#### C2: Bracket Order Fallback Silently Degrades to Plain Orders

**Location:** `trade_service.py:213-223`

If the broker doesn't support bracket orders (or conditions aren't met), the system silently falls back to a plain market/limit order — meaning the stop-loss and take-profit legs are never placed. The position enters with zero protection.

**Fix:** If bracket order fails, place SL/TP as separate contingent orders. Never allow a position without protective orders.

#### C3: Alpaca Broker Has No Bracket Order Support in `execute()`

**Location:** `alpaca_broker.py:106-148`

Only simple market/limit orders. No OCO, bracket, or trailing stop order types. This means even if the system intends bracket orders, Alpaca won't execute them.

**Fix:** Implement `execute_bracket()` using Alpaca's native bracket order API.

### HIGH Findings (Should Fix)

#### H1: Correlation Check is Post-Hoc Only

**Location:** `risk_gate.py:780-927`

Pairwise correlation (60-day window, threshold >0.80) is checked during the `monitor()` loop every 60s, NOT pre-trade. You can enter 5 highly correlated positions before the monitor catches it.

**Fix:** Add pre-trade correlation check in `risk_gate.check()` against existing portfolio.

#### H2: No Pre-Market / Post-Market Order Gating

**Location:** `alpaca_broker.py:145`

Alpaca warns but allows orders outside market hours. No enforcement. An agent could submit orders at 3 AM that execute at market open with gap risk.

**Fix:** Hard-reject orders outside configurable trading windows unless explicitly flagged as `extended_hours=True`.

#### H3: No Portfolio Heat Budget

**Location:** `risk_gate.py` — missing entirely

Daily loss limit (-2%) only triggers after losses are realized. There's no cap on total new notional deployed in a single day. An aggressive agent could deploy 100% of capital in one session.

**Fix:** Add `max_daily_notional_deployed` limit (e.g., 30% of equity per day for new positions).

#### H4: Sector Concentration Not Checked Pre-Entry

**Location:** `portfolio_risk.py:115`

Herfindahl sector concentration is computed in post-hoc monitoring only. Could end up 80% in tech before anyone notices.

**Fix:** Pre-trade sector check in risk gate.

#### H5: Regime Flip Only Alerts, Doesn't Force Exit

**Location:** `risk_gate.py:768-776`

If you entered a momentum trade in `trending_up` and regime flips to `ranging`, the system logs an alert but takes no action. The position stays open in a hostile regime.

**Fix:** Regime flip on active positions should trigger mandatory review with configurable auto-exit for high-severity regime mismatches.

### MEDIUM Findings

| Finding | Location | Issue |
|---------|----------|-------|
| Trailing stop only per holding type, not mandatory | `holding_period.py:60-89` | Swing+ positions should mandate trailing stops |
| No single-day massive drawdown trigger | `kill_switch.py` | Kill switch has rolling 3-day check but no -10% single-day emergency halt |
| Market circuit breaker only checks SPY | `kill_switch.py:364-373` | Should also check VIX >80, sector ETF halts |
| Fill polling uses wall-clock time | `alpaca_broker.py:145-146` | 30s timeout, 1s poll interval — may miss fills on clock skew |
| `require_ctx()` auto-initializes | `_state.py:75` | Silently creates context if missing — could mask configuration errors |

### Risk Limit Defaults

```
max_position_pct:       10% of equity per symbol
max_position_notional:  $20,000 hard cap per position
max_gross_exposure_pct: 150% (allows modest leverage)
max_net_exposure_pct:   100% net long
daily_loss_limit_pct:   -2% halts trading for day
min_daily_volume:       500k ADV minimum
max_participation_pct:  1% of ADV per order
max_premium_at_risk:    2% per options position, 8% total book
DTE bounds:             7-60 days
```

All overridable via environment variables — good for tuning, but means a misconfigured `.env` could silently weaken risk controls.

### Order Lifecycle State Machine

```
NEW → SUBMITTED → ACKNOWLEDGED → PARTIALLY_FILLED → FILLED
     ├→ REJECTED
     ├→ CANCELLED
     └→ EXPIRED
```

Thread-safe, terminal states enforced. Solid implementation.

### Exit Rule Priority (ExecutionMonitor)

```
1. Kill switch active       → immediate liquidation
2. Hard stop-loss hit       → market sell
3. Take profit hit          → limit sell
4. Trailing stop hit        → market sell
5. Time stop expired        → market sell
6. Intraday flatten (15:55) → market sell (intraday only)
```

Deterministic, priority-ordered. Good design. But relies on `stop_price` being set — see C1 above.

---

## Data Pipeline & Signal Engine Findings

### Overall Assessment: WELL-ENGINEERED, BUT 24/7 FRAGILE

The signal engine is impressively designed — 16 concurrent collectors, fault-tolerant, 2-6 second wall-clock, regime-conditional synthesis weights. The data pipeline is idempotent with rate limiting. But several single points of failure and cache invalidation gaps would cause silent degradation in 24/7 operation.

### What's Working Well

| Component | Strength | Details |
|-----------|----------|---------|
| SignalEngine | Fault-tolerant concurrency | 16 collectors via `asyncio.gather`, 10s per-collector timeout, failed collectors penalize confidence (-0.05 each) |
| Synthesis | Regime-adaptive weights | 4 regime profiles (trending_up/down, ranging, unknown), inactive voter weights redistributed |
| Data acquisition | Idempotent 14-phase pipeline | Delta-only fetches, `ON CONFLICT DO UPDATE`, rate-limited (75 req/min) |
| Rate limiting | Priority-based load shedding | Critical/normal/low tiers, daily quota tracked atomically in DB |
| Drift detection | PSI-based feature monitoring | 6 features tracked, CRITICAL severity triggers research_queue task |
| Data validation | OHLC consistency + freshness | Invalid bars discarded (never silently passed), staleness flagged |
| Streaming | WebSocket + REST fallback | `StreamManager` with `AlpacaStreamingAdapter`, passive mode if no credentials |
| Observability | Structured JSON logging | Trace-context-aware, 50MB rotation, 30-day retention |

### Signal Engine Architecture

```
16 Collectors (parallel, 10s timeout each):
  technical → regime → volume → risk → events → fundamentals →
  sentiment → macro → sector → flow → cross_asset → quality →
  ml_signal → statarb → options_flow → social

  ↓ asyncio.gather (return_exceptions=True)
  ↓ Failed collectors → {} with confidence penalty
  ↓ RuleBasedSynthesizer (regime-conditional weights)
  ↓ DriftDetector (PSI check, best-effort)
  ↓ SignalBrief (170 fields) → cached 1hr TTL
```

### CRITICAL Findings

#### DC1: Signal Cache Not Auto-Invalidated on Data Refresh

**Location:** `signal_engine/cache.py`

The cache has a 1-hour TTL, but intraday data refreshes every 5 minutes. A trading decision could be based on a 55-minute-old SignalBrief while fresh data sits in the DB unused. Manual `cache.invalidate(symbol)` exists but is never called automatically after data refresh.

**Impact:** Trading loop makes stale decisions in fast-moving markets.
**Fix:** Hook `cache.invalidate(symbol)` into `scheduled_refresh.py` after each intraday refresh cycle.

#### DC2: Alpha Vantage is a Single Point of Failure for Most Data

**Location:** `data/fetcher.py`, `data/acquisition_pipeline.py`

AV is the sole source for: options chains, earnings, macro indicators, news sentiment, fundamentals, insider/institutional data. Only OHLCV has a fallback (Alpaca). If AV goes down, 12 of 14 acquisition phases fail silently.

**Impact:** Extended AV outage → stale data across the entire system, with no alerting.
**Fix:** Add secondary providers for critical data (options: CBOE/Polygon, fundamentals: SEC EDGAR, macro: FRED API). Add alerting when >3 consecutive phase failures.

#### DC3: No Automatic Staleness Rejection in Signal Engine

**Location:** `signal_engine/engine.py`

The data `validator.py` flags stale data, but the signal engine doesn't check data freshness before running collectors. A collector can happily compute RSI on 3-week-old data and return a confident signal.

**Impact:** Stale signals presented as current → wrong trades.
**Fix:** Each collector should check `data_metadata.last_timestamp` and return `{}` if data is staler than a configurable threshold.

### HIGH Findings

#### DH1: Drift Detection Runs Post-Synthesis (Best-Effort)

**Location:** `signal_engine/engine.py:134-171`

PSI drift check happens AFTER the brief is already synthesized and cached. If drift is CRITICAL, the stale brief is still served for up to 1 hour (cache TTL).

**Fix:** Drift check should run pre-cache-store. If CRITICAL, either skip caching or tag brief with `low_confidence=True`.

#### DH2: Options Data Coverage Limited to Top 30 + Watched Symbols

**Location:** `data/scheduled_refresh.py`

Only ~30-50 symbols get options chains refreshed. The remaining universe gets `{}` from the options_flow collector, meaning GEX/gamma flip/IV skew/VRP signals are missing for most symbols.

**Fix:** Expand options refresh to full active universe, or at minimum, any symbol being evaluated for entry.

#### DH3: Sentiment Falls Back to Neutral (0.5) Silently

**Location:** `signal_engine/collectors/sentiment_alphavantage.py`

When no headlines are available or Groq times out, sentiment returns 0.5 (neutral). This is safe but misleading — a neutral sentiment signal carries the same weight as a real one in synthesis.

**Fix:** Return `{}` instead of fake neutral when data is unavailable. Let synthesis redistribute the weight.

#### DH4: Macro Indicators Lag by 30+ Days

**Location:** `data/acquisition_pipeline.py` — macro phase refreshes monthly

GDP, CPI, unemployment data is monthly at best. The macro collector may be using 6-week-old data. For regime detection this matters less, but for rate-sensitive strategies (bonds, REITs) it's a blind spot.

#### DH5: PostgreSQL Connection Pool May Exhaust Under Load

**Location:** `db.py` — default pool size 20

With 3 concurrent graph loops (research + trading + supervisor), each running multiple agents with DB queries, plus data acquisition and streaming persistence, 20 connections may not be enough.

**Fix:** Monitor pool utilization via `pg_stat_activity`. Consider raising to 50+ or using PgBouncer.

### MEDIUM Findings

| Finding | Location | Issue |
|---------|----------|-------|
| No split/dividend adjustment verification | `data/validator.py` | Relies entirely on AV correctness — no cross-check |
| No forward-fill for trading day gaps | Data pipeline | Weekend/holiday gaps may affect indicator windows |
| HMM regime needs 120+ bars for new symbols | `regime.py` | New universe additions get rule-based fallback for ~4 months |
| ML model unavailability silent | `ml_signal.py` | Returns `{}` with no error log — could be broken for weeks |
| Cache warmer not integrated into scheduled refresh | `cache_warmer.py` | Exists but runs separately — data may be warm in cache but stale in signal engine |

### Database Scale

60+ tables, well-organized by subsystem. Key concern: `ohlcv` table is the largest and uses composite PK `(symbol, timeframe, timestamp)`. With 50 symbols × 3 timeframes × years of data, this table will grow to hundreds of millions of rows. No partitioning strategy documented.

### Data Flow Architecture

```
Alpha Vantage ──┐
Alpaca IEX ─────┤
Groq LLM ───────┤──→ AcquisitionPipeline (14 phases)
Reddit/Stocktwits┘    ↓
                   PostgreSQL (60+ tables)
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
  SignalEngine   Trading Loop  Research Loop
  (16 collectors)  (positions)  (strategies)
        ↓
  SignalBrief → DriftDetector → Cache (1hr TTL)
```

---

## LLM Routing & Prompt Engineering Findings

### Overall Assessment: STRONG ARCHITECTURE, CRITICAL SECURITY GAPS

21 agents across 3 graphs with tier-stratified models. Role separation is clean. But prompt injection vulnerabilities, silent output validation failures, and zero token budget management make this unsafe for unsupervised 24/7 operation.

### Agent Inventory (21 Agents)

| Graph | Agent | Tier | Purpose |
|-------|-------|------|---------|
| **Research** (7) | quant_researcher | heavy (Sonnet) | Multi-week research programs, hypothesis generation |
| | ml_scientist | heavy | Feature engineering, model training, drift detection |
| | strategy_rd | heavy | Hypothesis gatekeeper, overfitting detection |
| | hypothesis_critic | medium (Haiku) | Pre-validation scoring (0-1 scale) |
| | community_intel | medium | 3-iteration web search loop |
| | domain_researcher | heavy | Investment/swing/options per-symbol research |
| | execution_researcher | heavy | TCA, factor exposure, portfolio construction |
| **Trading** (10) | daily_planner | medium | Watchlist ranking, entry candidates |
| | position_monitor | medium | Per-position thesis integrity assessment |
| | exit_evaluator | medium | HOLD/TRIM/TIGHTEN/CLOSE verdicts |
| | trade_debater | heavy | Structured bull/bear/risk debate |
| | fund_manager | heavy | Portfolio-level gates (correlation, concentration) |
| | options_analyst | heavy | Options structure selection |
| | earnings_analyst | medium | Earnings-specific strategies |
| | market_intel | medium | Pre-market macro news gathering |
| | trade_reflector | medium | Post-trade outcome classification |
| | executor | medium | Order submission & audit logging |
| **Supervisor** (4) | health_monitor | light | Service health, heartbeat freshness |
| | self_healer | medium | Root cause diagnosis, recovery playbooks |
| | portfolio_risk_monitor | medium | Factor/sector/correlation alerts |
| | strategy_promoter | medium | Evidence-based lifecycle management |

### CRITICAL Findings

#### LC1: Prompt Injection via Untrusted State (Multiple Nodes)

**Location:** `trading/nodes.py:80-92` (market_intel), `trading/nodes.py:245-260` (daily_plan), `research/nodes.py:219-224` (context_load)

Portfolio context, knowledge base entries, and market data API responses are injected directly into prompts via f-strings with no sanitization:

```python
# trading/nodes.py:80-92 — VULNERABLE
prompt = f"Portfolio: {json.dumps(portfolio_ctx, default=str)}\n"

# research/nodes.py:219-224 — VULNERABLE
prefetched_context = f"{str(knowledge_text)[:2000]}\n"  # From DB, unsanitized
```

An adversarial knowledge base entry, compromised API response, or malicious portfolio state could inject instructions that manipulate trading decisions. For an autonomous system managing real capital, this is an existential risk.

**Fix:** Use structured templates with field-level extraction instead of raw JSON dumps. Validate and escape all interpolated data. Consider using XML-tagged sections for clear boundary separation.

#### LC2: Silent JSON Parse Failures Mask Agent Errors

**Location:** `graphs/agent_executor.py:474-521`

All 21 agents are expected to return JSON. When parsing fails, `parse_json_response()` silently returns a fallback (`{}` or `[]`) with no retry:

| Node | Expected Output | On Parse Failure | Impact |
|------|----------------|-----------------|--------|
| daily_plan | `{"plan": "..."}` | Returns `{}` | No plan for the day — trades blind |
| entry_scan | `[{symbol, signal_strength}]` | Returns `[]` | Entries missed entirely |
| position_review | `[{symbol, thesis_intact}]` | Returns `[]` | Active positions unmonitored |
| fund_manager | `[{symbol, verdict}]` | Returns `[]` | Rejected entries treated as approved |
| risk_sizing | `[{symbol, action}]` | Returns `[]` | Risk assessment silently skipped |

**Fix:** Add schema validation (Pydantic models per agent output). On parse failure, retry once with "Please respond with valid JSON matching this schema: ...". Log all fallback events as warnings.

#### LC3: Temperature Hardcoded to 0.0 for All Agents

**Location:** `llm/provider.py:16`

```python
@dataclass(frozen=True)
class ModelConfig:
    temperature: float = 0.0  # ALL agents
    max_tokens: int = 4096    # ALL agents
```

Every agent — from creative hypothesis generation to deterministic order execution — uses temperature 0.0. This kills diversity in research (where you want novel hypotheses) and provides no benefit for tasks that are already deterministic.

**Fix:** Per-agent temperature in `agents.yaml`: hypothesis generation (0.7), debate (0.3-0.5), validation/parsing (0.0).

### HIGH Findings

#### LH1: Agent Tier Classification Has Dangerous Default

**Location:** `llm_config.py:120-135`

```python
def _classify_tier(agent_name: str) -> str:
    if agent_name in _SYNTHESIS_AGENTS: return "assistant"
    if "workshop" in agent_name: return "workshop"
    # ... pattern matching ...
    return "assistant"  # DEFAULT: expensive Sonnet tier!
```

Agents not matching naming conventions (daily_planner, position_monitor, exit_evaluator, trade_debater, fund_manager, health_monitor, self_healer — 10+ agents) default to the most expensive "assistant" tier (Sonnet 4.6). This is a cost leak.

**Fix:** Explicit tier assignment in `agents.yaml` per agent. Remove naming-convention-based classification. Log a warning if an agent falls to default tier.

#### LH2: No Runtime LLM Provider Fallback

**Location:** `llm_config.py:370-404`

Provider availability is checked at startup only. If Bedrock returns 429 (rate limit) or 500 mid-execution, there's no fallback to Anthropic API or Groq. The agent simply fails.

**Fix:** Wrap LLM calls with runtime fallback: on 429/500/timeout, try next provider in chain before returning error.

#### LH3: Hardcoded Model Names in 6+ Locations

| Location | Hardcoded Model | Should Be |
|----------|----------------|-----------|
| `tool_search_compat.py:21` | `anthropic/claude-sonnet-4-20250514` | `get_chat_model("evaluator")` |
| `trading/nodes.py:843` | `anthropic/claude-sonnet-4-20250514` | `get_chat_model("evaluator")` |
| `performance/trade_evaluator.py` | `anthropic/claude-sonnet-4-20250514` | `get_chat_model("evaluator")` |
| `mem0_client.py` | `gpt-4o-mini` | `get_chat_model("memory")` |
| `alpha_discovery/hypothesis_agent.py` | `groq/llama-3.3-70b-versatile` | `get_chat_model("bulk")` |
| `optimization/opro_loop.py` | `groq/llama-3.3-70b-versatile` | `get_chat_model("bulk")` |

**Fix:** All model references should go through `get_chat_model()` with appropriate tier.

#### LH4: No Token Budget Per Agent

No agent has a token limit. The only guard is a reactive 150k char message pruning threshold in `agent_executor.py`. A runaway agent (e.g., quant_researcher in a loop) could consume $50+ in a single cycle before being stopped.

**Fix:** Add `max_tokens_budget` to `AgentConfig`. Truncate backstory to 500 chars if over budget. Fallback to lighter model if budget exhausted.

#### LH5: EWF Analysis Called 3x Per Symbol Per Cycle

Three agents (daily_planner, position_monitor, trade_debater) each independently call `get_ewf_analysis` for the same symbol in the same trading cycle. 3x redundant API calls.

**Fix:** Fetch EWF once per symbol per cycle, store in graph state, share across agents.

### MEDIUM Findings

| Finding | Impact |
|---------|--------|
| No few-shot examples in any agent prompt | Quality loss ~5-15% vs. prompted with examples |
| No prompt caching (Claude API supports it) | ~20% wasted on repeated system prompt tokens |
| No structured outputs (JSON schema mode) | Relies on fragile text parsing |
| No cost tracking per agent per cycle | Can't optimize what you can't measure |
| No A/B testing framework for prompt variants | Can't iterate on prompt quality systematically |
| Backstories are 40-100 lines each | Token waste — most value in first 10 lines |
| No self-critique loop | Agents don't verify their own JSON before returning |
| Portfolio optimization has two paths | `PortfolioOptimizerAgent` (unused?) vs `fund_manager` node — potential for conflicting sizing |

### Prompt Quality Assessment

| Agent | Specificity | Examples | Structure | Validation Gates | Grade |
|-------|------------|----------|-----------|-----------------|-------|
| quant_researcher | Excellent (Fama-French, IC>0.02) | None | Good | IC, Sharpe, PF thresholds | B+ |
| trade_debater | Excellent (7-section structure) | None | Excellent | 3+3 evidence requirement | B+ |
| hypothesis_critic | Good (0-1 scale, 0.7 threshold) | None | Good | Checklist present | B |
| daily_planner | Good (watchlist criteria) | None | Moderate | Weak | B- |
| fund_manager | Good (concentration/correlation) | None | Moderate | Verdict enum implied | B- |
| All others | Moderate | None | Moderate | Weak or absent | C+ |

**Pattern:** Strong domain knowledge in prompts, but zero examples and weak output schema enforcement across the board.

---

## Ops & Infrastructure Findings

### Overall Assessment: PRODUCTION-GRADE SINGLE-HOST, NOT DURABLE

Comprehensive resilience patterns (kill switch, graceful shutdown, health checks, retry logic, drift detection). But critical gaps in data durability (no backups), scheduler availability (not containerized), and log persistence (not shipped) make this unsuitable for 24/7 unattended operation.

### Docker Compose Services (9 containers, ~10GB RAM)

| Service | Image | Memory | Health Check | Restart |
|---------|-------|--------|-------------|---------|
| postgres | pgvector:pg16 | 512m | pg_isready (10s/5 retries) | unless-stopped |
| langfuse-db | postgres:16-alpine | 256m | pg_isready (10s/5 retries) | unless-stopped |
| ollama | ollama/ollama | 4g | `ollama list` (15s/5 retries) | unless-stopped |
| langfuse | langfuse/langfuse:2 | 512m | HTTP health (15s/5 retries) | unless-stopped |
| trading-graph | Custom | 1g | Python heartbeat (60s/3 retries) | unless-stopped |
| research-graph | Custom | 1g | Python heartbeat (60s/3 retries) | unless-stopped |
| supervisor-graph | Custom | 512m | Python heartbeat (60s/3 retries) | unless-stopped |
| finrl-worker | Custom (torch) | 2g | HTTP SSE (60s/3 retries) | unless-stopped |
| dashboard | Custom | 256m | HTTP health (30s/3 retries) | unless-stopped |

### CRITICAL Findings

#### OC1: No Automated PostgreSQL Backup

**Impact:** ALL system state lives in PostgreSQL — 60+ tables including positions, strategies, fills, signal state, research queue, knowledge base. If Docker host disk fails, everything is lost.

No `pg_dump` scheduled job. No backup documentation. Volumes are `driver: local` (single host). This is the single highest-risk finding in the entire audit.

**Fix:** Add daily `pg_dump` → S3 upload. Add point-in-time recovery via WAL archiving. Test restore monthly.

#### OC2: APScheduler Not Containerized or Supervised

**Location:** `scripts/scheduler.py`

The scheduler runs 13 jobs (data refresh, strategy pipeline, community intel, etc.) but runs as a bare process in a tmux session. If it crashes, no jobs execute — no data refresh, no strategy promotion, no EOD sync. No supervisor (systemd, Docker) to restart it.

**Fix:** Add `scheduler` service to docker-compose.yml with health check and restart policy.

#### OC3: CI/CD Completely Disabled

**Location:** `.github/workflows/ci.yml.disabled`, `release.yml.disabled`

No automated tests on push. No image registry. No staging environment. No rollback procedure. Manual `docker compose build` on the host.

For a system trading real capital, deploying untested code is an existential risk.

**Fix:** Enable CI pipeline. At minimum: run test suite + type check + build image on every push to main.

### HIGH Findings

#### OH1: Langfuse Traces Grow Unbounded

Langfuse keeps all traces indefinitely. The `langfuse-db` container has only 256MB memory. With 3 graphs producing traces every 5-10 minutes, the DB will fill disk within weeks.

**Fix:** Add retention cleanup job to scheduler: delete traces older than 30 days.

#### OH2: No Log Aggregation or Alerting

Logs go to Docker json-file driver (50MB rotation, 5 files max per container) and local JSONL files (50MB rotation, 30-day retention). No shipping to ELK/Splunk/CloudWatch. No alerting on ERROR/CRITICAL volume spikes.

If the host goes down, all logs are lost. No way to detect a pattern of failures across containers.

**Fix:** Add Filebeat/Fluentd sidecar → Elasticsearch or CloudWatch. Add alerts for error rate anomalies.

#### OH3: Kill Switch is Manual Reset Only

Once triggered, the kill switch requires manual `reset()` call. No auto-recovery even for transient conditions. No escalation (alert after 1 hour, auto-reset after investigation). No stale kill switch detection (could sit triggered for days if owner is AFK).

**Fix:** Add tiered recovery: (1) Discord alert immediately, (2) if transient condition resolved, auto-reset after configurable cooldown, (3) escalate to email/SMS after 4 hours.

#### OH4: Health Checks Are Coarse

Health checks only verify heartbeat file presence and age. They don't validate:
- Whether the graph cycle succeeded (vs. errored but wrote heartbeat)
- Strategy generation rate (research producing nothing for days)
- Trading performance (IC/Sharpe degrading)
- Error counts in final state

**Fix:** Add granular health metrics: cycle success rate (last 10), error rate, IC trend, strategy pipeline throughput.

#### OH5: No OOM Monitoring

Docker memory limits are hard caps. If a container exceeds its limit, Docker kills it silently (OOMKilled). No alerting, no logging of the OOM event beyond Docker daemon logs.

**Fix:** Add container stats monitoring. Alert on memory usage >80% of limit.

### MEDIUM Findings

| Finding | Impact |
|---------|--------|
| No secrets manager (plain .env on disk) | API keys, DB credentials unencrypted |
| No env var type validation | Misconfigured `RISK_MAX_POSITION_PCT=ten` silently fails |
| Ollama health check doesn't verify models loaded | Service "healthy" but `mxbai-embed-large` may not be pulled |
| No bind mount size limits | Models/data/logs can fill host disk |
| No multi-host deployment path | Single host = single point of failure for everything |
| Research graph may need >1GB for ML training | OOM risk during full model training |
| No deployment rollback procedure | Bad deploy requires manual git revert + rebuild |

### Startup Flow (`start.sh`)

```
1. Validate .env + Docker presence
2. Start infrastructure (postgres, langfuse-db, ollama, langfuse)
3. Wait for health checks (120s deadline)
4. Pull Ollama models (mxbai-embed-large, llama3.2)
5. Run DB migrations (advisory lock, idempotent)
6. Bootstrap universe if empty
7. Preflight checks (wallet balance, stale positions)
8. Data freshness check (trigger background sync if >1 day old)
9. RAG embedding check (ingest if empty)
10. Start graph services + dashboard + finrl-worker
11. Poll graph health checks (60s, warning-only)
```

Idempotent and well-sequenced. Main gap: no confirmation prompt before data refresh (can delay startup 5-15 min).

### Shutdown Flow (`stop.sh`)

```
1. Activate kill switch (DB + sentinel file, two-layer)
2. Docker compose down (SIGTERM → 90s grace → SIGKILL)
3. Graph runners catch SIGTERM → set should_stop → exit loop
4. DB connections committed/rolled back
5. Langfuse flushed (trading + supervisor only — research is missing cleanup callback)
```

Clean two-layer kill switch is excellent design. Gap: research graph doesn't register Langfuse cleanup callback.

### Resilience Patterns (What's Working)

| Pattern | Implementation | Quality |
|---------|---------------|---------|
| Kill switch | Two-layer (DB + file sentinel) | Excellent |
| Graceful shutdown | SIGTERM handlers + cleanup callbacks | Good (research gap) |
| Exponential backoff | `resilient_call()` with configurable retries | Good |
| DB reconnection | Pool discards broken connections, retries | Good |
| Crash-only recovery | Execution monitor rebuilds from DB + broker | Excellent |
| Watchdog timeout | Per-graph configurable (600s/1800s/600s) | Good |
| Heartbeat tracking | DB-persisted, Docker health-check integrated | Good |
| Drift detection | PSI-based, auto-triggers kill switch >50% drift | Good |

### Scheduler Jobs (13 Total)

| Job | Frequency | Critical? |
|-----|-----------|-----------|
| `strategy_pipeline_10m` | Every 10 min | Yes — backtests drafts |
| `av_intraday_5min_5m` | Every 5 min (market hours) | Yes — data freshness |
| `intraday_quote_refresh` | Every 15 min (market hours) | Yes — position P&L |
| `data_refresh_08:00` | Daily 8 AM | Yes — full universe refresh |
| `eod_data_refresh_16:30` | Daily 4:30 PM | Yes — EOD close data |
| `daily_attribution_16:10` | Daily 4:10 PM | Medium — P&L attribution |
| `credit_regime_intraday_2h` | 3x daily | Medium |
| `credit_regime_eod_16:45` | Daily | Medium |
| `memory_compaction` | 2x weekly | Low |
| `strategy_lifecycle_weekly` | Weekly Sunday | Medium |
| `community_intel_weekly` | Weekly Sunday | Low |
| `autoresclaw_weekly` | Weekly Sunday | Low |
| `strategy_lifecycle_monthly` | Monthly 1st | Medium |

All of these stop running if the scheduler process dies (see OC2).

---

## Memory Architecture: RAG vs. File-Based Analysis

### Overall Assessment: CURRENT FILE-BASED APPROACH IS CORRECT — BUT LEAVING MONEY ON THE TABLE

QuantStack uses markdown files in `.claude/memory/*.md` with `MEMORY.md` as an index. This is the right architecture for our scale. **RAG would actively hurt us** due to prompt cache destruction. However, we're missing several optimizations that OpenClaw and other systems have adopted.

### The Prompt Cache Economics (The Decisive Argument)

Claude's prompt caching provides **90% cost reduction** on cached input tokens. This is the single most important factor in the RAG vs. file-based decision.

| | Base Input | Cache Write (5-min TTL) | Cache Read |
|---|---|---|---|
| **Sonnet 4** | $3/MTok | $3.75/MTok (1.25x) | $0.30/MTok (0.1x) |
| **Haiku 3.5** | $0.80/MTok | $1.00/MTok | $0.08/MTok (0.1x) |
| **Opus 4** | $15/MTok | $18.75/MTok | $1.50/MTok (0.1x) |

**Cache invalidation rule**: The cache is a cumulative hash of everything up to the breakpoint. Any change in the prefix (system prompt, tools, memory) breaks the cache for everything after it.

**Why RAG destroys caching**: Each RAG query retrieves different chunks based on the current question. The retrieved context changes between requests → prefix hash changes → cache breaks → full price every call.

**Concrete cost comparison for QuantStack** (Sonnet 4, 20K token static context, 50 requests/session):

| Approach | First Request | Subsequent 49 Requests | Session Total |
|---|---|---|---|
| **Static MD (cache hits)** | 20K × $3.75/MTok = $0.075 | 49 × 20K × $0.30/MTok = $0.294 | **$0.369** |
| **RAG (no cache hits)** | 20K × $3.00/MTok = $0.060 | 49 × 20K × $3.00/MTok = $2.940 | **$3.000** |

**Static memory is ~8x cheaper per session.** With 3 graph services running cycles every 5-10 minutes, 24/7, this compounds to thousands of dollars per month.

Additionally: our Trading Graph runs 5-min cycles. The 5-minute cache TTL means each cycle's first call refreshes the cache and all subsequent agent calls within that cycle hit cache. **Our cycle interval perfectly matches the cache TTL.**

### Memory System Comparison Matrix

| System | Memory Type | Persistence | Auto-learning | Search | Prompt Cache Compatible |
|--------|-----------|-------------|--------------|--------|------------------------|
| **QuantStack (current)** | File-based MD | Disk | Yes (agent writes) | Linear scan | ✅ Yes (static prefix) |
| **OpenClaw** | MD + SQLite + vector hybrid | Disk + SQLite | Yes (dreaming system) | Hybrid BM25 + vector | ⚠️ Partial (static core cached, retrieval not) |
| **Claude Code** | File-based MD | Disk | Yes (auto memory) | File read on demand | ✅ Yes |
| **mem0** | Vector store (Qdrant) | Qdrant + LLM extraction | Yes | Semantic similarity | ❌ No (chunks vary per query) |
| **Letta/MemGPT** | Tiered (main + archival) | In-context + vector | Yes (self-managed) | LLM-driven paging | ❌ No (each memory op = LLM call) |
| **Zep/Graphiti** | Temporal knowledge graph | Graph DB | Yes | Graph traversal + temporal | ❌ No (assembled context varies) |
| **Devin** | Curated knowledge base | Cloud | Semi (AI suggests, human approves) | Selective retrieval | ⚠️ Partial |
| **Cursor** | .cursor/rules/ + index | Disk | No | Semantic codebase search | ✅ Yes (rules are static) |
| **LangGraph Memory** | Thread + namespace stores | Checkpointer + store | Yes | Embedding search (namespace) | ⚠️ Depends on usage |

### RAG: When It Makes Sense (And When It Doesn't)

#### When RAG Wins

- Memory corpus exceeds ~50K tokens (too large for context window)
- You need to search through hundreds of historical trades for pattern matching
- Multiple agents need different subsets of a very large shared knowledge base
- Sessions are infrequent (>5 min gaps → cache expires anyway, so no cache advantage lost)

#### When RAG Loses (Our Situation)

- Memory corpus is <50K tokens (ours is ~15-20K active)
- High-frequency calls (5-min cycles) where cache hits compound savings
- Domain-specific semantic traps: "stop loss at 2%" and "take profit at 2%" are semantically similar but operationally opposite — embedding search gets this wrong
- Every RAG component is a new failure mode: embedding model, vector DB, chunking strategy, reranker
- Retrieval variance: sometimes it finds the right context, sometimes it doesn't. For an autonomous trading system, non-deterministic memory is a risk

#### RAG Cost Overhead (Annual)

| Component | Cost |
|---|---|
| Embedding API calls | ~$50-200/yr |
| Vector DB hosting (Qdrant Cloud) | $300-1,200/yr |
| **Lost prompt cache savings** | **$2,000-10,000/yr** (the real cost) |
| Additional latency per request | 50-200ms |
| Maintenance/debugging time | Significant |

#### File-Based Cost Overhead (Annual)

| Component | Cost |
|---|---|
| Token waste from irrelevant context | ~$100-500/yr |
| Manual curation time | 1-2 hrs/month |
| Infrastructure cost | $0 |

### CRITICAL Finding: `search_knowledge_base` Doesn't Actually Use RAG

**Finding ID: MC0 — Knowledge Base Tool Bypasses pgvector Entirely**

**Location:** `tools/langchain/learning_tools.py:25-31`

The `search_knowledge_base` tool — used by 15+ agents for institutional memory retrieval — does NOT use the pgvector RAG pipeline. It runs a simple SQL query:

```sql
SELECT id, category, content, metadata, created_at
FROM knowledge_base
ORDER BY created_at DESC
LIMIT 5
```

This is **recency-only retrieval** — no semantic search, no query relevance, no embedding comparison. The `query` parameter is accepted but never used in the SQL. An agent asking "What momentum strategies failed on AAPL?" gets the 5 most recent entries regardless of topic.

Meanwhile, a proper semantic search function exists in `rag/query.py:156-203` (`search_knowledge_base()`) that does pgvector cosine similarity search. But the LangChain tool never calls it.

**Impact:** Agents make decisions based on irrelevant context. The knowledge base could have 500 entries about options strategies, but if the 5 most recent are about macro indicators, that's what the momentum researcher sees.

**Fix:** Replace the SQL recency query with a call to `rag.query.search_knowledge_base(query=query, n_results=top_k)`. One import, one function call change.

### CRITICAL Finding: No pgvector Index — Sequential Scan on Every RAG Query

**Finding ID: MC0b — Missing HNSW/IVFFlat Index on Embeddings Table**

**Location:** `rag/query.py:17-34`

The embeddings table schema creates a B-tree index on `collection` and a GIN index on `metadata->ticker`, but **no vector index** (HNSW or IVFFlat) on the `embedding` column:

```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_collection ON embeddings (collection);
CREATE INDEX IF NOT EXISTS idx_embeddings_metadata_ticker ON embeddings USING GIN ((metadata -> 'ticker'));
-- NO: CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)
```

Every `search_similar()` call does a full sequential scan with `embedding <=> query::vector` across all rows. At small scale this is fine, but as the knowledge base grows (500+ entries), query latency will degrade from <10ms to 100ms+.

**Fix:** Add HNSW index:
```sql
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON embeddings
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

### CRITICAL Finding: Zero Prompt Caching Configured

**Finding ID: MC0c — Anthropic Prompt Caching Not Enabled**

**Location:** Searched entire codebase — no `cache_control`, `CacheControl`, or `ephemeral` references found.

Claude's prompt caching (90% cost reduction on cached tokens) is the single largest cost optimization available to QuantStack. With 21 agents running every 5-10 minutes with static system prompts (backstories, tool definitions), the cache should hit on nearly every call after the first.

**But it's not enabled.** The `ChatBedrock` and `ChatAnthropic` instantiation in `llm/provider.py` doesn't set any cache control breakpoints. Every call pays full input token price.

**Impact:** At Sonnet $3/MTok input, with ~20K tokens of system prompt per agent, 21 agents, ~100 cycles/day: ~$126/day in system prompt tokens. With caching: ~$12.60/day. **We're paying 10x what we should.**

**Fix:** Add `cache_control` breakpoints to system message construction in `agent_executor.py:152-178`:
```python
SystemMessage(content=base, additional_kwargs={"cache_control": {"type": "ephemeral"}})
```
For Bedrock: use `anthropic_beta: ["prompt-caching-2024-07-31"]` header.

### CRITICAL Finding: Tool Ordering Breaks Prompt Cache

**Finding ID: MC1 — Non-Deterministic Tool Ordering Destroys Cache Hits**

**Location:** `tools/registry.py` → tool resolution order

When agents bind tools from `TOOL_REGISTRY`, the tool definitions are injected into the system prompt. If the ordering varies between invocations (dict ordering, import order, or any non-deterministic resolution), the entire system prompt hash changes → prompt cache breaks → full-price input tokens on every call.

**Impact:** With 21 agents making calls every 5-10 minutes, non-deterministic tool ordering could be costing us 30-50% more on API spend than necessary.

**Fix:** Sort tool definitions alphabetically by name before injection in `tool_binding.py`. This is a 1-line change with massive cost impact. OpenClaw implemented this exact fix in v2026.4.5 ("deterministic MCP tool ordering").

### HIGH Finding: No Compaction Layer for Mid-Cycle Context Bloat

**Finding ID: MC2 — Context Accumulates Unbounded Within Graph Cycles**

**Location:** All graph runners, `agent_executor.py`

Each agent in a cycle accumulates context (market data, signal briefs, position states, tool results). By the time the 10th agent in the Trading Graph runs, the context window contains all prior agents' outputs. The only guard is the reactive 150K char message pruning in `agent_executor.py`, which drops oldest tool rounds — potentially losing critical context.

**Fix:** Add a compaction step between parallel branches that summarizes intermediate state using a haiku-tier model. This maps to our existing model tier system (haiku during market hours is already cheaper). The compacted summary replaces raw outputs, keeping context lean for downstream agents.

### HIGH Finding: Memory Entries Have No Temporal Decay

**Finding ID: MC3 — Stale Memory Entries Persist Indefinitely**

**Location:** `.claude/memory/*.md`

Memory entries about market reads, strategy states, and session handoffs have no expiry mechanism. The 2026-04-04 EWF market reads (already flagged as "low-trust") will persist in memory until manually pruned. Over time, the memory index grows, pushing useful context further down and wasting tokens on outdated information.

**Fix:** Add date metadata to all temporal memory entries. Implement automated pruning: market reads >7 days old → archive. Strategy states → validate against current DB. Session handoffs >30 days → archive. This is what OpenClaw's "Dreaming" system does with `recencyHalfLifeDays` and `maxAgeDays`.

### MEDIUM Finding: No Lazy Loading of Topic Files

**Finding ID: MC4 — All Memory Loaded Regardless of Relevance**

**Location:** `.claude/memory/MEMORY.md` (index loaded at session start)

MEMORY.md is loaded in full at session start (first 200 lines). All entries — community intel scans, strategy registries, architecture findings, EWF reads — load regardless of whether the current session needs them. A trading session doesn't need community intel scan details; a research session doesn't need trade journal entries.

**Fix:** Structure MEMORY.md as a lean index with category headers. Implement session-type-aware loading: Trading sessions load strategy + execution memories; Research sessions load hypothesis + intel memories; Supervisor loads health + lifecycle memories. This is Claude Code's native pattern with topic files read on-demand.

### Recommendation: Enhanced File-Based Memory (Not RAG)

**Keep what we have. Optimize the economics.**

| Priority | Action | Effort | Annual Savings |
|----------|--------|--------|---------------|
| 1 | **Deterministic tool ordering** in `registry.py` | 1 line | $2,000-5,000 (cache hits) |
| 2 | **Compaction with haiku** between graph branches | Medium | $1,000-3,000 (smaller contexts) |
| 3 | **Temporal decay** on memory entries | Low | $200-500 (leaner memory) |
| 4 | **Session-type-aware loading** | Medium | $500-1,000 (relevant context only) |
| 5 | **Enable Claude prompt caching** explicitly | Low | 20-50% on system prompt tokens |

**When to revisit**: If memory corpus exceeds 50K tokens, or if we need semantic search over 500+ historical trade outcomes (archival RAG for history, not for active memory).

---

## OpenClaw Feature Benchmark & Adoption Opportunities

### Why Benchmark Against OpenClaw?

**OpenClaw** (https://github.com/openclaw/openclaw, 349K+ stars) is a self-hosted autonomous AI assistant that runs 24/7 — the same operational profile as QuantStack. While its domain is personal productivity (not trading), it has solved hard infrastructure problems that directly map to ours: prompt cache economics at scale, memory decay for always-on agents, cross-provider failover, context window management across multi-agent sessions, and task cancellation propagation. These are domain-agnostic operational patterns.

**What we're borrowing:** Not the product features (messaging, browser, skills) — the **runtime engineering** for cost, reliability, and memory lifecycle in a system that never sleeps.

**Relevant architecture facts:**
- Multi-LLM: OpenAI, Claude, Gemini, Groq, Mistral, Ollama (same providers we use)
- Isolated agent workspaces with inter-agent messaging (analogous to our 3 graph services)
- Three memory backends: Builtin (SQLite + FTS5 + vector), QMD (local sidecar), Honcho (AI-native)
- Cron-driven automation with per-job tool restrictions (analogous to our scheduler + agent tool bindings)

### Feature Benchmark: OpenClaw vs. QuantStack

| Capability | OpenClaw | QuantStack | Gap | Priority |
|------------|----------|------------|-----|----------|
| **Memory persistence** | MD files + SQLite hybrid | MD files only | Minor — our scale doesn't need SQLite | Low |
| **Memory promotion/decay** | 3-phase Dreaming (Light/Deep/REM) with `recencyHalfLifeDays` | Manual pruning | Significant — stale entries persist | HIGH |
| **Prompt cache stability** | Deterministic tool ordering, fingerprint normalization, duplicate removal | Non-deterministic tool binding | Critical — we're likely breaking cache | **CRITICAL** |
| **Context compaction** | Dedicated compaction model, configurable per agent | 150K char reactive pruning | Significant — we lose context | HIGH |
| **Model failover** | Capped retries per provider → cross-provider fallback, cooldown config | Startup-only provider check | Significant — outages crash cycles | HIGH |
| **Inter-agent messaging** | `sessions_send` with reply-back coordination | PostgreSQL polling (5-10 min latency) | Moderate — our event bus works but is slow | MEDIUM |
| **Per-job tool restrictions** | `cron --tools` allowlist per scheduled job | All agents see all registered tools | Moderate — defense-in-depth gap | MEDIUM |
| **Task lifecycle** | TaskFlow with managed/mirrored modes, sticky cancel | `require_ctx` check at session start | Moderate — kill switch propagation delay | MEDIUM |
| **Browser automation** | CDP-based Chrome management | WebFetch + web search | Low — API data sources sufficient | Low |
| **Cost tracking** | Indirect (cache optimization) | None | Moderate — can't optimize what we can't measure | MEDIUM |

### Adoptable Patterns (Ranked by Impact)

#### OC-1: Deterministic Tool Ordering for Cache Stability [CRITICAL]

**OpenClaw v2026.4.5:** Sorts MCP tool definitions deterministically, removes duplicates from agent prompts, and normalizes system-prompt fingerprints to prevent cache-breaking whitespace changes.

**QuantStack adaptation:**
1. Sort `TOOL_REGISTRY` entries alphabetically by name before binding in `tool_binding.py`
2. Remove duplicate tool descriptions from agent system prompts (each agent gets tools listed in YAML + the full backstory — ensure no duplication)
3. Normalize whitespace in `agents.yaml` backstories so formatting changes don't break cache

**Effort:** Low (1-3 line changes in `tool_binding.py` and `registry.py`)
**Impact:** 30-50% reduction in prompt token costs from improved cache hit rates

#### OC-2: Tiered Memory Promotion (Dreaming Pattern) [HIGH]

**OpenClaw v2026.4.5:** Three-phase consolidation:
- **Light sleep**: Quick recall promotion (signal detected → short-term memory)
- **Deep sleep**: Substantial context integration (signal validated → medium-term)
- **REM**: Lasting-truth extraction (strategy principle confirmed → permanent memory)

Memories age via `recencyHalfLifeDays` and `maxAgeDays`. Weighted short-term recall decides what gets promoted to `dreams.md`.

**QuantStack adaptation:**
- **Light**: Research Graph discovers a signal → add to `research_findings.md` with 7-day TTL
- **Deep**: Signal passes backtest validation → promote to `validated_signals.md` with 30-day TTL
- **REM**: Strategy produces live profit → extract the lasting principle to `strategy_registry.md` (permanent)
- **Aging**: Market reads (EWF, sentiment) expire after 7 days. Strategy states validate against DB monthly. Session handoffs archive after 30 days.
- **Tooling**: Add `compact-memory` skill (already exists in skill list) as a weekly scheduled task that applies TTL rules

**Effort:** Medium (memory template changes + scheduler job)
**Impact:** Keeps memory lean and relevant, prevents stale context from misleading agents

#### OC-3: Dedicated Compaction Model [HIGH]

**OpenClaw v2026.4.1+:** Configurable `compaction.model` that runs a cheaper LLM to summarize conversation history before it exceeds context limits.

**QuantStack adaptation:**
- After the parallel merge points in the Trading Graph (`merge_parallel`, `merge_pre_execution`), run a haiku-tier compaction step that summarizes the parallel branch outputs into a structured brief
- This prevents context bloat when 4-5 agents' full outputs are concatenated for downstream agents
- Cost: ~$0.08/MTok (haiku cache read) vs. carrying 50K+ tokens of raw output at $3/MTok (sonnet input)

**Effort:** Medium (add compaction nodes to graph definitions)
**Impact:** 40-60% context size reduction at merge points, faster downstream agent responses

#### OC-4: Provider Failover with Rotation Cooldowns [HIGH]

**OpenClaw v2026.4.5:** Caps same-provider retries before cross-provider fallback. Configurable `rateLimitedProfileRotations` cooldown.

**QuantStack adaptation:**
- Wrap `get_chat_model()` with a retry-failover chain: Anthropic API → Bedrock Claude → Groq (for haiku-tier tasks)
- On 429/500/timeout: retry same provider 2x → switch to next provider → cooldown original for 5 minutes
- Already have AWS credentials for Bedrock and Groq API key configured
- Addresses the 2026-04-05 Anthropic API credit exhaustion incident that blocked the EWF analyzer

**Effort:** Medium (modify `llm_config.py` provider resolution)
**Impact:** Eliminates single-provider outage as a system-wide failure mode

#### OC-5: Per-Job Tool Allowlists [MEDIUM]

**OpenClaw v2026.4.1+:** `cron --tools` restricts which tools each scheduled job can invoke.

**QuantStack adaptation:**
- Research Graph agents should NOT have access to: `execute_order`, `cancel_order`, `activate_kill_switch`
- Trading Graph agents should NOT have access to: `register_strategy`, `train_model`
- Supervisor Graph agents should have read-only access to execution tools
- Implement in `agents.yaml` by adding a `blocked_tools` field per agent, enforced in `agent_executor.py`

**Effort:** Low (YAML config + 5-line guard in executor)
**Impact:** Defense-in-depth — prevents agent confusion from calling wrong-domain tools

#### OC-6: TaskFlow Cancel Propagation [MEDIUM]

**OpenClaw v2026.4.2:** "Sticky cancel intent" propagates cancellation from parent to all child tasks immediately.

**QuantStack adaptation:**
- When kill switch fires, propagate cancellation to all in-flight graph nodes, not just check at `require_ctx`
- Implement via LangGraph's interrupt mechanism or a shared threading.Event that all nodes check before execution
- Currently: kill switch sets DB flag + sentinel file → each node checks on next `require_ctx()` call → delay of up to one full node execution time
- With cancel propagation: interrupt signal → all nodes abort within seconds

**Effort:** Medium (modify graph runners + node wrappers)
**Impact:** Reduces kill-switch-to-halt time from minutes to seconds

### OpenClaw Features NOT Worth Adopting

| Feature | Why Not |
|---------|---------|
| **SQLite memory backend** | Our corpus is <50K tokens. SQLite adds complexity without benefit at this scale. |
| **Browser automation (CDP)** | All our data sources are API-based. Browser scraping adds attack surface and fragility. |
| **WebSocket Gateway** | We don't need messaging platform integration. LangGraph handles our orchestration. |
| **Honcho (AI-native memory)** | Cloud-dependent external service. Against our local-first architecture. |
| **ClawHub skill registry** | We have our own tool registry. The skill ecosystems don't overlap. |
| **OpenClaw-RL (weight fine-tuning)** | Requires training infrastructure we don't have. Prompt engineering is more practical at our scale. |

### OpenClaw's Memory Architecture Deep-Dive

For reference, OpenClaw's three-backend memory system:

```
┌─────────────────────────────────────────────┐
│              Context Engine                  │
│  ingest → assemble (token budget) → compact │
│         → post-turn persistence             │
└──────────┬────────────────────┬─────────────┘
           ↓                    ↓
┌──────────────────┐  ┌─────────────────────┐
│  MEMORY.md       │  │  memory/YYYY-MM-DD  │
│  (long-term)     │  │  (daily notes)      │
│  loaded at start │  │  today + yesterday  │
└────────┬─────────┘  └─────────┬───────────┘
         ↓                      ↓
┌──────────────────────────────────────────────┐
│           Builtin Backend (SQLite)           │
│  ~400-token chunks, 80-token overlap         │
│  FTS5 (BM25) + vector similarity (hybrid)    │
│  Auto-detects embedding provider from keys   │
│  Debounced reindex on file change (1.5s)     │
└──────────────────────────────────────────────┘

┌──────────────────┐
│   DREAMS.md      │
│  3-phase cycle:  │
│  Light → Deep    │
│  → REM           │
│  Aging params:   │
│  recencyHalfLife │
│  maxAgeDays      │
└──────────────────┘
```

**Key insight**: OpenClaw's hybrid approach keeps MEMORY.md as a static, always-loaded prefix (cache-friendly) while using SQLite+vector for overflow search. The Dreaming system handles temporal decay. This is the architecture we should aspire to IF our memory corpus grows past 50K tokens — but we're not there yet.

### Operational Gap Summary: What OpenClaw Solved That We Haven't

These are the runtime engineering gaps — problems any 24/7 multi-agent system faces regardless of domain:

| Problem | OpenClaw Solution | QuantStack Current State | Trading Impact |
|---------|------------------|------------------------|----------------|
| **API cost at scale** | Deterministic tool ordering, fingerprint normalization, duplicate removal → stable prompt cache | Non-deterministic binding → cache breaks likely | 30-50% wasted spend across 21 agents |
| **Stale context poisoning** | Dreaming system with `recencyHalfLifeDays`, `maxAgeDays` → automatic decay | Manual pruning → stale EWF reads, expired theses persist | Agents trade on outdated market reads |
| **Context window bloat** | Dedicated compaction model per agent → summarize before overflow | Reactive 150K char pruning → drops oldest rounds (may lose critical data) | Fund manager loses position_review context |
| **Provider outages** | Capped retries → cross-provider fallback → cooldown rotation | Startup-only check → mid-session failures crash the cycle | 2026-04-05 Anthropic credit exhaustion blocked EWF analyzer for hours |
| **Kill switch latency** | Sticky cancel intent → immediate propagation to all child tasks | Poll-based `require_ctx` → up to one full node execution delay | Minutes of rogue trading after kill switch fires |
| **Agent tool confusion** | Per-job tool allowlists via `cron --tools` | All 122 tools visible to all agents (92 of which are stubbed) | Research agent could theoretically call `execute_order` |
| **Alert coordination** | `sessions_send` with reply-back → agent-to-agent messaging with confirmation | PostgreSQL EventBus polling → 5-10 min latency | Supervisor detects IC decay but Trading Graph trades on stale strategy for 5 more minutes |

**Net assessment**: OpenClaw has solved 7 operational problems that directly affect QuantStack's 24/7 reliability and cost structure. The top 4 (cache stability, memory decay, compaction, failover) would reduce API costs by ~30-50% and eliminate several failure modes identified in earlier sections of this audit. The alert coordination pattern (`sessions_send` with reply-back) directly addresses our GC2 finding (no inter-graph urgency channel).

### Alert Coordination: The Broken Feedback Loop

OpenClaw's `sessions_send` with reply-back coordination exposes a critical gap in QuantStack: **our alert and inter-graph coordination system is half-built and half-connected.**

#### What Exists (Foundation Is Solid)

| Component | Status | Location |
|-----------|--------|----------|
| Alert DB schema (3 tables) | ✅ Implemented | `db.py:1059-1136` — `equity_alerts`, `alert_exit_signals`, `alert_updates` |
| EventBus (poll-based) | ✅ Implemented | `coordination/event_bus.py` — append-only log, per-consumer cursors, 7-day TTL |
| Kill switch (two-layer) | ✅ Implemented | `execution/kill_switch.py` — DB flag + sentinel file, 4 auto-triggers |
| Daily digest → Discord | ✅ Implemented | `coordination/daily_digest.py` — EOD webhook, one-way broadcast |
| Strategy lifecycle events | ✅ Working | Supervisor publishes `IC_DECAY`, `DEGRADATION_DETECTED` to EventBus |
| Degradation enforcer | ✅ Working | `coordination/degradation_enforcer.py` — strategy breaker trips → event |

#### What's Broken or Missing

**CRITICAL: 5 Alert Lifecycle Tools Are All Stubbed**

| Tool | Status | Impact |
|------|--------|--------|
| `create_equity_alert()` | Returns `{"error": "Tool pending implementation"}` | Research can't persist trade ideas as actionable alerts |
| `get_equity_alerts()` | Stubbed | Trading can't load pending alerts from research |
| `update_alert_status()` | Stubbed | No lifecycle tracking (pending → watching → acted/expired) |
| `create_exit_signal()` | Stubbed | Can't record stop hits, target reaches, thesis breaks |
| `add_alert_update()` | Stubbed | No commentary timeline, no thesis evolution tracking |

**HIGH: EventBus Is Only Half-Connected**

The Supervisor publishes events (`IC_DECAY`, `DEGRADATION_DETECTED`, `REGIME_CHANGE`) but the **Trading Graph never polls them**. No code in `trading/nodes.py` calls `bus.poll()` for supervisor findings. The supervisor is shouting into the void.

```
CURRENT STATE:
  Supervisor → publishes IC_DECAY → EventBus → ??? (nobody reads)
  Research   → polls IDEAS_DISCOVERED → EventBus ← (one consumer)
  Trading    → never polls anything from EventBus

REQUIRED STATE:
  Supervisor → publishes IC_DECAY → EventBus → Trading reads at cycle start
  Supervisor → publishes RISK_EMERGENCY → EventBus → Trading checks pre-trade
  Research   → publishes STRATEGY_READY → EventBus → Trading picks up next cycle
  Kill Switch→ publishes KILL_TRIGGERED → EventBus → All loops abort immediately
```

**HIGH: Kill Switch Doesn't Publish to EventBus**

When the kill switch fires, it sets a DB flag and writes a sentinel file. But it **never publishes a `KILL_SWITCH_TRIGGERED` event**. The supervisor can't detect kill switch activation via its normal polling loop — it would have to directly check the sentinel file (which it doesn't do). This means:
- Kill switch fires → trading loop stops (checks `require_ctx()`) → supervisor doesn't know for up to 5 minutes
- No event trail of when/why the kill switch was triggered

**HIGH: No Real-Time Alert Routing**

Discord integration is daily digest only (17:00 ET broadcast). When critical events happen mid-day — kill switch trigger, strategy degradation, risk limit breach — there's no notification. The system operates blind between daily digests.

**MEDIUM: Research → Trading Alert Flow Is Completely Broken**

The intended flow:
```
Research discovers validated strategy
  → creates equity_alert (STUBBED)
  → Trading reads pending alerts (STUBBED)
  → Trading acts on alert → updates status (STUBBED)
  → Exit signal recorded (STUBBED)
```

Every step in this chain is stubbed. Research findings reach Trading only through the `strategies` table (slow, indirect) or EventBus `IDEAS_DISCOVERED` events (which Trading doesn't poll).

#### OpenClaw Pattern That Fixes This

OpenClaw's `sessions_send` with reply-back solves the coordination problem:

```
# OpenClaw pattern (pseudocode):
supervisor.sessions_send(
    to="trading_graph",
    message={"type": "IC_DECAY", "strategy": "momentum_aapl", "action": "halt"},
    reply_back=True  # blocks until trading confirms receipt
)
# Trading graph receives, processes, replies with confirmation
# Supervisor knows the message was received and acted on
```

**QuantStack adaptation** (using what we have):

1. **Immediate**: Make Trading Graph poll EventBus at `safety_check` node for `IC_DECAY`, `RISK_EMERGENCY`, `KILL_SWITCH_TRIGGERED` events. This is 5-10 lines in `trading/nodes.py`.

2. **Short-term**: Add `KILL_SWITCH_TRIGGERED` event type to EventBus. Publish from `kill_switch.trigger()`. All loops poll this.

3. **Short-term**: Implement the 5 alert tools. The DB schema already exists (`equity_alerts`, `alert_exit_signals`, `alert_updates`). The tools are just CRUD wrappers.

4. **Medium-term**: Add real-time Discord/Slack routing for CRITICAL events. Extend `daily_digest.py` with an `urgent_alert()` function that sends immediately (not batched).

5. **Medium-term**: Add reply-back confirmation. When supervisor publishes `IC_DECAY`, trading graph should publish `IC_DECAY_ACK` confirming it received and halted the strategy. Supervisor monitors for ACK within 2 cycles — if missing, escalate.

| Fix | Effort | Impact | Priority |
|-----|--------|--------|----------|
| Trading polls EventBus for supervisor events | Low (5-10 lines) | Closes the feedback loop entirely | **P0** |
| Kill switch publishes to EventBus | Low (3 lines) | Cross-loop kill switch awareness | **P0** |
| Implement 5 alert lifecycle tools | Medium (CRUD wrappers, schema exists) | Enables research→trading alert flow | **P1** |
| Real-time Discord for CRITICAL events | Low (extend daily_digest) | Human-in-the-loop for emergencies | **P1** |
| Reply-back ACK pattern | Medium (new event types + supervisor monitor) | Guaranteed delivery confirmation | **P2** |

---

## 24/7 Readiness Gap Analysis

### Current State: Market-Hours-Only, Attended Operation

The system currently runs trading + research graphs during market hours only (9:30 AM - 4:00 PM ET, Mon-Fri), with supervisor always-on. A human must start the system (`./start.sh`), monitor tmux sessions, and manually reset the kill switch after incidents.

### Gap Matrix: What Must Change for 24/7

| Capability | Current State | 24/7 Requirement | Gap Severity |
|------------|--------------|-------------------|-------------|
| **Data durability** | No backups | Automated backup + restore tested | CRITICAL |
| **Process supervision** | tmux + manual | Containerized + auto-restart | CRITICAL |
| **Stop-loss enforcement** | Optional (can be None) | Mandatory on every position | CRITICAL |
| **Prompt injection defense** | None | Sanitized input boundaries | CRITICAL |
| **Output validation** | Silent fallback to `{}` | Schema validation + retry | CRITICAL |
| **Inter-graph urgency** | 5-10 min poll latency | Sub-second interrupt channel | HIGH |
| **Checkpoint durability** | In-process MemorySaver | PostgresSaver (survives crashes) | HIGH |
| **Kill switch recovery** | Manual reset only | Tiered auto-recovery | HIGH |
| **Market hours gating** | Alpaca warns but allows | Hard reject outside hours | HIGH |
| **Pre-trade correlation** | Post-hoc monitoring only | Pre-entry portfolio check | HIGH |
| **Portfolio heat budget** | Missing | Daily notional deployment cap | HIGH |
| **Signal cache freshness** | 1hr TTL, no auto-invalidation | Invalidate on data refresh | HIGH |
| **Data provider redundancy** | AV primary, Alpaca OHLCV fallback only | Multi-provider for all data types | HIGH |
| **LLM provider runtime fallback** | Startup-only check | Per-call fallback on 429/500 | HIGH |
| **Log aggregation** | Local files only | Centralized + alerting | HIGH |
| **CI/CD** | Disabled | Automated test + deploy | MEDIUM |
| **Multi-host deployment** | Single Docker host | At minimum hot standby | MEDIUM |
| **Cost tracking** | None | Per-agent per-cycle LLM spend | MEDIUM |
| **Structured LLM outputs** | Text-parsed JSON | JSON schema mode | MEDIUM |

### The 24/7 Operating Model

For truly autonomous 24/7 operation, the system needs three modes:

**1. Market Hours Mode (9:30-16:00 ET Mon-Fri)**
- Trading graph: 5-min cycles, full pipeline
- Research graph: 10-min cycles
- Execution monitor: real-time tick-level exit enforcement
- Data refresh: 5-min intraday bars, 15-min quote refresh

**2. Extended Hours Mode (16:00-20:00, 04:00-09:30 ET)**
- Position monitoring only (no new entries)
- Earnings/macro event processing
- EOD data sync + strategy pipeline
- News sentiment monitoring for pre-market alerts

**3. Overnight/Weekend Mode (20:00-04:00 ET, weekends)**
- Research graph: full compute budget (heavier ML training, community intel)
- Supervisor: health monitoring + strategy lifecycle
- Data acquisition: historical backfill, macro indicator refresh
- No trading, no position changes

**Current gap:** The system has no concept of extended hours or overnight modes. Trading and research graphs simply don't run outside market hours. The supervisor runs always-on but doesn't shift behavior based on time of day.

### Failure Recovery Requirements for 24/7

| Failure Scenario | Current Recovery | Required Recovery |
|-----------------|-----------------|-------------------|
| LLM provider down | Next cycle retries (5-10 min) | Immediate fallback to next provider |
| DB connection lost | Exponential backoff reconnect | Same + alert after 3 failures |
| Container OOM killed | Docker restart (unless-stopped) | Same + alert + memory profiling |
| Broker API down | Kill switch after 3 failures | Same + position monitoring via alternative feed |
| Bad deploy (broken code) | Manual rollback | Automated canary + rollback on error rate spike |
| Data provider outage | Stale data used silently | Alert + block trading on stale signals |
| Kill switch triggered | Manual reset required | Auto-investigate → tiered recovery → manual escalation |
| Disk full | Silent failure | Proactive monitoring + alerting at 80% |
| Network partition | Graphs operate on stale state | Detect staleness → reduce position sizes → alert |

### Estimated Effort to Reach 24/7

| Phase | Work | Timeline |
|-------|------|----------|
| **Phase 1: Safety hardening** | Mandatory stop-loss, prompt sanitization, output validation, DB backups | 2 weeks |
| **Phase 2: Operational resilience** | Containerize scheduler, durable checkpoints, log aggregation, CI/CD | 2 weeks |
| **Phase 3: Autonomy** | Kill switch auto-recovery, inter-graph urgency channel, LLM runtime fallback, multi-mode operation | 3 weeks |
| **Phase 4: Scale** | Multi-host deployment, hot standby, advanced monitoring, cost optimization | 4 weeks |

---

## Priority Recommendations

### P0 — Must Fix Before Real Capital (Week 1-2)

| # | Finding | Subsystem | Fix |
|---|---------|-----------|-----|
| 1 | **Mandatory stop-loss** | Execution (C1) | Reject `OrderRequest` if `stop_price is None`. Enforce at OMS level. |
| 2 | **Deterministic tool ordering** | Memory/Cost (MC1) | Sort tool defs alphabetically in `tool_binding.py`. 1-line fix, 30-50% prompt cost savings. |
| 3 | **Prompt injection defense** | LLM (LC1) | Replace f-string interpolation with structured templates. Sanitize all DB/API-sourced data before prompt injection. |
| 4 | **Output schema validation** | LLM (LC2) | Add Pydantic models per agent output. On parse failure, retry once with schema hint. Log all fallback events. |
| 5 | **Automated DB backups** | Ops (OC1) | Daily `pg_dump` → S3. WAL archiving for PITR. Test restore monthly. |
| 6 | **Containerize scheduler** | Ops (OC2) | Add `scheduler` service to docker-compose.yml with health check + restart. |
| 7 | **Bracket order fallback** | Execution (C2) | If bracket fails, place SL/TP as separate contingent orders. Never allow unprotected positions. |
| 8 | **Durable checkpoints** | Graph (GC1) | Switch from `MemorySaver` to `PostgresSaver`. Enables crash recovery from exact node. |
| 9 | **Trading Graph polls EventBus** | Alerts (AC1) | Add `bus.poll()` at `safety_check` node for `IC_DECAY`, `RISK_EMERGENCY`. 5-10 lines. Closes supervisor→trading feedback loop. |
| 10 | **Kill switch publishes to EventBus** | Alerts (AC2) | Add `KILL_SWITCH_TRIGGERED` event in `kill_switch.trigger()`. 3 lines. Cross-loop awareness. |

### P1 — Should Fix for Reliable Operation (Week 3-4)

| # | Finding | Subsystem | Fix |
|---|---------|-----------|-----|
| 8 | **Inter-graph urgency channel** | Graph (GC2) | Add Redis pub/sub or file sentinel for sub-second supervisor → trading interrupts. |
| 9 | **Signal cache auto-invalidation** | Data (DC1) | Hook `cache.invalidate(symbol)` into scheduled_refresh after each intraday cycle. |
| 10 | **Pre-trade correlation check** | Execution (H1) | Add pairwise correlation check in `risk_gate.check()` against existing portfolio. |
| 11 | **Market hours hard gating** | Execution (H2) | Reject orders outside configurable windows unless `extended_hours=True`. |
| 12 | **Portfolio heat budget** | Execution (H3) | Add `max_daily_notional_deployed` limit (e.g., 30% of equity/day). |
| 13 | **LLM runtime failover** | LLM (LH2) + Memory (OC-4) | On 429/500, retry 2x → cross-provider fallback (Anthropic → Bedrock → Groq). Cooldown failed provider 5 min. |
| 14 | **Per-agent temperature** | LLM (LC3) | Hypothesis generation=0.7, debate=0.3, validation=0.0. Configure in agents.yaml. |
| 18.5 | **Haiku compaction at merge points** | Memory (MC2) | Add compaction nodes after parallel merges in Trading Graph. Haiku summarizes branch outputs. 40-60% context reduction. |
| 18.6 | **Implement 5 alert lifecycle tools** | Alerts (AC3) | CRUD wrappers over existing `equity_alerts` schema. Enables research→trading alert flow. |
| 18.7 | **Real-time Discord for CRITICAL events** | Alerts (AC4) | Extend `daily_digest.py` with `urgent_alert()` for kill switch, risk breach, IC decay. |
| 15 | **Kill switch auto-recovery** | Ops (OH3) | Tiered: immediate alert → investigate → auto-reset if transient → escalate if persistent. |
| 16 | **Log aggregation + alerting** | Ops (OH2) | Ship logs to centralized system. Alert on ERROR rate spikes. |
| 17 | **Enable CI/CD** | Ops (OC3) | Re-enable ci.yml. Run tests + type check on every push. |
| 18 | **Data staleness rejection** | Data (DC3) | Each collector checks `data_metadata.last_timestamp`. Return `{}` if stale. |

### P2 — Improve for 24/7 Autonomy (Week 5-8)

| # | Finding | Subsystem | Fix |
|---|---------|-----------|-----|
| 19 | **Multi-mode operation** | Graph | Add extended_hours and overnight modes to graph runners. |
| 20 | **Reduce serialization bottleneck** | Graph (GC3) | Stream debater outputs to fund_manager incrementally. |
| 21 | **Data provider redundancy** | Data (DC2) | Add FRED (macro), SEC EDGAR (fundamentals), CBOE (options) as fallback providers. |
| 22 | **Drift check pre-cache** | Data (DH1) | Run PSI before caching brief. Tag CRITICAL briefs as low-confidence. |
| 23 | **Remove hardcoded models** | LLM (LH3) | All 6+ hardcoded model strings → `get_chat_model()` with appropriate tier. |
| 24 | **Token budget per agent** | LLM (LH4) | Add `max_tokens_budget` to AgentConfig. Truncate backstory if over. |
| 25 | **EWF deduplication** | LLM (LH5) | Fetch EWF once per symbol per cycle, cache in graph state. |
| 26 | **Regime flip auto-exit** | Execution (H5) | Mandatory review on regime flip. Auto-exit for severe mismatches. |
| 27 | **Structured LLM outputs** | LLM | Use Claude's JSON schema mode. Eliminate text parsing. |
| 28 | **Cost tracking** | LLM | Log tokens_in/out per call. Aggregate by agent/day/graph. Alert on budget. |
| 29 | **OHLCV table partitioning** | Data | Partition by symbol or timeframe for query performance at scale. |
| 30 | **Langfuse retention cleanup** | Ops (OH1) | Schedule trace deletion >30 days. |
| 31 | **Memory temporal decay** | Memory (MC3) | Add TTL metadata to memory entries. Market reads expire 7d, strategies validate monthly, handoffs archive 30d. |
| 32 | **Session-type memory loading** | Memory (MC4) | Lean MEMORY.md index with category-aware lazy loading. Trading sessions skip research memories and vice versa. |
| 33 | **Per-agent tool allowlists** | Memory (OC-5) | Block execution tools from Research agents, block strategy tools from Trading agents. Defense-in-depth. |
| 34 | **Kill switch cancel propagation** | Memory (OC-6) | Replace poll-based kill switch checking with sticky cancel intent. Interrupt all in-flight nodes within seconds. |

### P3 — Scale and Optimize (Week 9+)

| # | Finding | Subsystem | Fix |
|---|---------|-----------|-----|
| 31 | Multi-host deployment | Ops | Hot standby host. Shared Postgres (RDS). Container registry. |
| 32 | Prompt caching | LLM | Enable Claude's prompt caching for static system prompts (~20% token savings). |
| 33 | Few-shot examples | LLM | Add 1-2 good/bad examples per agent prompt (+5-15% quality). |
| 34 | A/B testing framework | LLM | Compare prompt variants systematically. |
| 35 | VIX-based circuit breaker | Execution | Auto-scale positions at VIX >40, halt at VIX >80. |
| 36 | Options chain expansion | Data (DH2) | Refresh options for full active universe, not just top 30. |
| 37 | PgBouncer | Ops | Connection pooler for multi-process DB access. |
| 38 | Research fan-out default | Graph | Enable `RESEARCH_FAN_OUT_ENABLED=true` for 3-5x research throughput. |

---

## Appendix: All Critical Findings Cross-Reference

| ID | Finding | Subsystem | Location |
|----|---------|-----------|----------|
| C1 | Stop-loss is optional | Execution | `trade_service.py:99, 212-220` |
| C2 | Bracket order silent degradation | Execution | `trade_service.py:213-223` |
| C3 | Alpaca no bracket order support | Execution | `alpaca_broker.py:106-148` |
| DC1 | Signal cache not auto-invalidated | Data | `signal_engine/cache.py` |
| DC2 | Alpha Vantage SPOF | Data | `data/fetcher.py` |
| DC3 | No staleness rejection in collectors | Data | `signal_engine/engine.py` |
| LC1 | Prompt injection via untrusted state | LLM | `trading/nodes.py:80-92`, `research/nodes.py:219-224` |
| LC2 | Silent JSON parse failures | LLM | `agent_executor.py:474-521` |
| LC3 | Temperature hardcoded to 0.0 | LLM | `llm/provider.py:16` |
| GC1 | MemorySaver lost on crash | Graph | All graph runners |
| GC2 | No inter-graph urgency channel | Graph | `coordination/event_bus.py` |
| GC3 | Serialization bottleneck (900s) | Graph | `trading/graph.py` |
| OC1 | No automated DB backup | Ops | Docker volumes |
| OC2 | Scheduler not containerized | Ops | `scripts/scheduler.py` |
| OC3 | CI/CD disabled | Ops | `.github/workflows/` |
| TC1 | 92/122 tools stubbed (75%) | Tools | `tools/registry.py`, all stub modules |
| TC2 | Entry rules silently dropped | Tools | `tools/_shared.py:113-143` |
| TC3 | No tool execution timeout | Tools | All @tool functions |
| MC1 | Non-deterministic tool ordering breaks prompt cache | Memory/Cost | `tools/registry.py`, `tool_binding.py` |
| MC0 | `search_knowledge_base` tool bypasses pgvector entirely | RAG/Memory | `tools/langchain/learning_tools.py:25-31` |
| MC0b | No HNSW index on embeddings table (sequential scan) | RAG/Memory | `rag/query.py:17-34` |
| MC0c | Zero prompt caching configured (Anthropic cache_control) | Cost | `llm/provider.py`, `agent_executor.py` |

---

## AutoResearchClaw Audit

### Overall Assessment: EXCELLENT FOUNDATION, UNDER-UTILIZED

AutoResearchClaw (`scripts/autoresclaw_runner.py`, 683 lines) is a sophisticated self-healing system that translates failures into automated fixes. The architecture is production-grade with proper safety rails. But it runs **weekly** when it should run **nightly**, and its scope is limited to bug fixes when it could be doing much more.

### How It Works

```
Tool fails 3x consecutively
  → record_tool_error() inserts into `bugs` table
  → Supervisor's _bug_fix_watcher thread (60s poll) detects open bug
  → Links bug to research_queue task
  → Dispatches autoresclaw_runner.py (blocking, 90-min timeout)
  → Runner builds task-specific prompt from context
  → Invokes: `researchclaw run --topic "<prompt>" --auto-approve --agent claude-code`
  → ARC edits source files directly in src/quantstack/
  → _apply_bug_fix() validates:
    1. Reads fix_summary.md for confidence + human-review flags
    2. Checks git diff for changed files
    3. REJECTS if: protected files touched, low confidence, needs human review, syntax error
    4. If valid: git add + git commit + update bugs table + write to session_handoffs.md
    5. Restarts affected loops via tmux send-keys
```

### Safety Rails (Excellent)

| Guard | Implementation | Location |
|-------|---------------|----------|
| Protected files | `risk_gate.py`, `kill_switch.py`, `db.py` can NEVER be modified | `autoresclaw_runner.py:341` |
| Low confidence revert | If ARC reports `## Confidence: low` → auto-revert all changes | `autoresclaw_runner.py:373-376` |
| Human review flag | If ARC writes `## Requires Human Review` → revert + note | `autoresclaw_runner.py:365-371` |
| Syntax validation | `py_compile` on every changed .py file | `autoresclaw_runner.py:408-424` |
| No dependency changes | Cannot modify `pyproject.toml` | Prompt constraint |
| Audit trail | Every fix/revert written to `session_handoffs.md` | `autoresclaw_runner.py:449-457` |
| Timeout | 90-min max per task, 1-hour default | `autoresclaw_runner.py:57` |

### Task Types Supported

| Task Type | Trigger | Prompt Builder | Example |
|-----------|---------|---------------|---------|
| `bug_fix` | Tool fails 3x consecutively OR trade loss >1% | `_build_prompt_bug_fix()` | Fix broken `fetch_market_data` tool |
| `ml_arch_search` | DriftDetector finds PSI ≥ 0.25 | `_build_prompt_ml_arch_search()` | Retrain model for drifted symbol |
| `rl_env_design` | Manual or research queue | `_build_prompt_rl_env_design()` | Design new RL gymnasium environment |
| `strategy_hypothesis` | Coverage gap identified | `_build_prompt_strategy_hypothesis()` | Research and backtest new strategy |

### What's Under-Utilized

**Gap 1: Weekly schedule, should be nightly.**
The scheduler runs ARC at Sunday 20:00 — once per week. The bug_fix_watcher polls every 60s but only for bugs linked to research_queue. Nightly autoresearch (AR-1 pattern) would use ARC's infrastructure for 96 experiments/night instead of 3 tasks/week.

**Gap 2: Only reactive, never proactive.**
ARC only runs when something breaks (bug_fix) or drifts (ml_arch_search). It never proactively searches for improvements. The `strategy_hypothesis` task type exists but is only queued manually or by community_intel — no automated gap detection feeds it.

**Gap 3: Loop restarts via tmux send-keys.**
`_restart_loops_after_fix()` uses `tmux send-keys -t quantstack-loops:trading "C-c" ""` — fragile. If tmux session doesn't exist, restart fails silently. Should use Docker Compose restart or SIGHUP to graph containers.

**Gap 4: No fix validation beyond syntax check.**
ARC validates: py_compile + import check. But it doesn't run the affected tool to verify the fix actually works. A syntactically valid but logically wrong fix passes all guards.

**Fix:** Add a functional validation step: after py_compile, invoke the fixed tool with a test input and verify non-error response.

---

## Advanced Research Integration: Frontier AI Techniques for QuantStack

### CTO Perspective

The audit above documents what QuantStack *is*. This section documents what it *should become*. I've reviewed eight frontier AI research systems — Karpathy's autoresearch, DGM-Hyperagents, Mimosa, Kosmos, ResearchGym, MAGNET, AI-Supervisor, and OrgAgent — and mapped their architectures against QuantStack's subsystems. The goal: identify concrete, implementable upgrades that push QuantStack from ~75% autonomous to >95% autonomous with compounding self-improvement.

The core insight: **QuantStack already has the bones of a frontier system.** Three decoupled graphs, event-driven coordination, self-healing via AutoResearchClaw, regime-conditional synthesis, drift-triggered research. What's missing is the *meta layer* — the system doesn't improve *how it improves*. It discovers strategies, but it doesn't discover better ways to discover strategies. It fixes bugs, but it doesn't fix the patterns that cause bugs.

---

### AR-1. Autoresearch Loop (Karpathy Pattern)

**What it is:** Karpathy's autoresearch runs ~100 experiments overnight on a single GPU. Humans write instructions in markdown. The agent modifies `train.py`, measures bits-per-byte, keeps improvements, discards regressions. Three files. One metric. Fixed time budgets.

**What QuantStack has today:**
- Research graph runs hypothesis → critique → backtest → ML experiment (10-min cycles)
- AutoResearchClaw runs weekly bug fixes and ML architecture searches
- Strategy pipeline backtests drafts every 10 minutes

**Gap:** QuantStack's research loop is *LLM-heavy and slow*. One hypothesis per cycle. 360s just for the critique loop. No fixed time budget — a bad hypothesis can burn 15 minutes of compute before failing validation. AutoResearchClaw runs weekly, not nightly.

**Concrete integration:**

```
┌─────────────────────────────────────────────────────────────┐
│                OVERNIGHT AUTORESEARCH MODE                    │
│                (20:00 ET → 04:00 ET, 8 hours)                │
│                                                               │
│  program.md (human-written strategy themes, constraints)      │
│       ↓                                                       │
│  ┌─────────────────────────────────────────────────┐         │
│  │  FIXED 5-MIN EXPERIMENT BUDGET (per iteration)   │         │
│  │                                                   │         │
│  │  1. Generate hypothesis (30s, Haiku)              │         │
│  │  2. Compute features (60s, deterministic)         │         │
│  │  3. Train LightGBM draft (120s, CPU)              │         │
│  │  4. Measure OOS IC + Sharpe (30s, deterministic)  │         │
│  │  5. Keep if IC > prev_best, else discard (0s)     │         │
│  │                                                   │         │
│  │  ~96 experiments per night                        │         │
│  └─────────────────────────────────────────────────┘         │
│       ↓                                                       │
│  Winners registered as 'draft' → morning strategy pipeline    │
└─────────────────────────────────────────────────────────────┘
```

**Implementation path:**
1. Add `overnight_mode` to research graph runner (check time-of-day, switch pipeline)
2. Create `autoresearch_node.py` with fixed 5-min budget per experiment
3. Use Haiku (not Sonnet) for hypothesis generation — speed matters more than depth here
4. Single metric: OOS IC on purged holdout. No Sharpe, no drawdown, no PF — just IC. Filter later.
5. Store experiment log in `autoresearch_experiments` table (hypothesis, IC, runtime, kept/discarded)
6. Morning strategy pipeline evaluates overnight winners with full backtest + Sharpe + drawdown

**Why this matters:** 96 experiments/night vs. ~6 hypotheses/day today = **16x research throughput**. The fixed budget prevents the "bad hypothesis burns 15 minutes" problem. IC-only filtering is intentionally crude — the strategy pipeline does proper validation. This is the funnel top.

**Estimated effort:** 1 week. No new infrastructure. Reuse existing `ModelTrainer`, `feature_importance`, and `labeling` modules.

---

### AR-2. Metacognitive Self-Modification (DGM-Hyperagents Pattern)

**What it is:** DGM-Hyperagents separates task agents (solve the objective) from meta agents (improve the task agents). The meta agent can edit prompts, modify tool selections, adjust thresholds, and — critically — improve its own modification procedure. This creates a recursive self-improvement loop.

**What QuantStack has today:**
- AutoResearchClaw patches tool bugs (task-level self-repair)
- `opro_loop.py` exists for prompt optimization but uses hardcoded Groq model
- `prompt_tuner.py` does few-shot example selection
- No system currently optimizes *how agents work*, only *what they produce*

**Gap:** QuantStack's self-healing is *reactive* (fix what broke) not *proactive* (improve what's working). No agent reviews whether the research graph's hypothesis quality is improving. No agent tunes the trading graph's entry criteria based on realized alpha. The system runs the same prompts with the same thresholds forever.

**Concrete integration:**

```
┌──────────────────────────────────────────────────────────┐
│                  META-IMPROVEMENT LAYER                    │
│           (Weekly cycle, Sunday overnight)                 │
│                                                            │
│  ┌────────────────┐    ┌─────────────────────────┐        │
│  │  TASK AGENTS   │    │  META AGENTS             │        │
│  │  (existing 21) │    │                           │        │
│  │                │    │  meta_prompt_optimizer     │        │
│  │  Research      │◄───│  meta_threshold_tuner     │        │
│  │  Trading       │◄───│  meta_tool_selector       │        │
│  │  Supervisor    │◄───│  meta_architecture_critic  │        │
│  └────────────────┘    └─────────────────────────┘        │
│                                                            │
│  Feedback signals:                                         │
│  - Hypothesis accept/reject ratio (research quality)       │
│  - Entry win rate by agent recommendation (trading quality) │
│  - Tool invocation success rate (tool quality)             │
│  - Time-to-decision per node (efficiency)                  │
│  - LLM token spend per profitable trade (cost efficiency)  │
│  - Alpha decay rate per strategy (strategy durability)     │
└──────────────────────────────────────────────────────────┘
```

**Implementation path:**
1. **meta_prompt_optimizer**: Weekly, analyze `trade_reflector` outcomes. For each agent, compute: what fraction of its recommendations led to profitable outcomes? If below threshold, generate prompt variant with more specificity on failure mode. A/B test next week.
2. **meta_threshold_tuner**: Monthly, analyze hypothesis_critic's 0.7 confidence threshold. Is it too lenient (passing bad hypotheses) or too strict (rejecting winners)? Adjust by 0.05 increments. Same for risk gate thresholds — are we rejecting trades that would have been profitable?
3. **meta_tool_selector**: Track per-agent tool invocation patterns. If `quant_researcher` never uses `compute_alpha_decay` but strategies keep decaying, inject that tool into `always_loaded_tools`. If an agent calls a tool that always returns stub JSON, remove it from the binding.
4. **meta_architecture_critic**: Quarterly, compare QuantStack's realized Sharpe to the S&P benchmark. Identify which subsystem is the bottleneck (signal quality? sizing? execution? research throughput?) and generate an architecture improvement proposal.

**The recursive piece:** `meta_prompt_optimizer` itself has a prompt. If its suggestions consistently don't improve agent performance, `meta_architecture_critic` can modify the optimizer's prompt. This is the DGM-Hyperagents "improving how you improve" pattern.

**Why this matters:** Without a meta layer, QuantStack's agents are frozen in capability. The prompts written on day 1 are the prompts running on day 365. The 0.7 confidence threshold is never validated. The tool bindings never evolve. A meta layer turns static agents into *learning* agents.

**Estimated effort:** 3 weeks. Requires outcome attribution data (partially exists in `outcome_tracker.py` and `ic_attribution.py`) and a new `meta/` module.

---

### AR-3. Explicit Knowledge Graph (AI-Supervisor Pattern)

**What it is:** AI-Supervisor maintains a structured knowledge graph of research findings, gaps, contradictions, and consensus. Agents query the graph to avoid redundant work and identify unexplored territory.

**What QuantStack has today:**
- `knowledge_base` table with text entries (search via pgvector embeddings)
- `workshop_lessons.md` with research findings
- `strategy_registry` tracks strategies but not the *reasoning* behind them
- `research_queue` tracks tasks but not their *relationships*

**Gap:** QuantStack's knowledge is *flat text*. The system can't answer: "Which alpha factors have we tested for AAPL? Which failed and why? What's the unexplored space?" It can't detect that two strategies are based on the same underlying factor (and would correlate in drawdown). It can't identify that a new paper contradicts an assumption in an active strategy.

**Concrete integration:**

```
┌──────────────────────────────────────────────────────┐
│              ALPHA KNOWLEDGE GRAPH                     │
│                                                        │
│  Nodes:                                                │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐           │
│  │ Factor   │  │ Strategy │  │ Hypothesis │           │
│  │ momentum │──│ AAPL_sw1 │──│ H_234      │           │
│  │ mean_rev │  │ QQQ_opt3 │  │ H_567      │           │
│  │ carry    │  └──────────┘  └────────────┘           │
│  └─────────┘       │              │                    │
│       │        ┌───▼────┐    ┌────▼─────┐             │
│  ┌────▼─────┐  │ Result │    │ Evidence │             │
│  │ Regime   │  │ IC=0.04│    │ Paper X  │             │
│  │ trending │  │ DD=12% │    │ Backtest │             │
│  └──────────┘  └────────┘    └──────────┘             │
│                                                        │
│  Edges:                                                │
│  - FACTOR --uses--> STRATEGY                           │
│  - HYPOTHESIS --tested_by--> BACKTEST                  │
│  - STRATEGY --correlates_with--> STRATEGY (>0.6 corr)  │
│  - FACTOR --contradicted_by--> PAPER                   │
│  - REGIME --favors--> FACTOR                           │
│  - STRATEGY --decayed_because--> FACTOR_SHIFT          │
└──────────────────────────────────────────────────────┘
```

**Implementation path:**
1. Add `alpha_knowledge_graph` table with node/edge schema (use PostgreSQL JSON columns, not a separate graph DB — keep infrastructure simple)
2. Hook into `strategy_registration` node: on every new strategy, extract factors and add to graph
3. Hook into `trade_reflector` node: on every trade outcome, update factor-regime performance edges
4. Add `query_knowledge_graph` tool to `quant_researcher` and `hypothesis_critic` agents
5. Add gap discovery: "What factor-regime combinations have <3 backtests?" → feed to research queue
6. Add contradiction detection: when community_intel finds a paper that challenges an active factor, flag for review

**Killer feature — Correlated Drawdown Prevention:** When `fund_manager` evaluates a new entry, query the graph: "Does this strategy share >2 factors with any existing position?" If yes, apply a correlation haircut to position sizing. This is smarter than the current pairwise price correlation check (H1) because it catches *factor* correlation, not just *price* correlation.

**Why this matters:** The knowledge graph is the system's *institutional memory*. Without it, QuantStack rediscovers the same dead-end hypotheses, misses factor crowding, and can't systematically fill research gaps. With it, every research cycle builds on every prior cycle.

**Estimated effort:** 2 weeks for schema + basic tooling. 2 more weeks for gap discovery and contradiction detection.

---

### AR-4. Hierarchical Governance Model (OrgAgent Pattern)

**What it is:** OrgAgent separates agents into governance (decides what to do), execution (does it), and compliance (audits it). Improved performance 102% while *reducing* token consumption 74% on benchmarks.

**What QuantStack has today:**
- Research graph: flat pipeline (context_load → hypothesis → critique → backtest → register)
- Trading graph: parallel branches but single authority chain (daily_plan → debate → fund_manager → execute)
- Supervisor graph: linear pipeline (health → diagnose → recover)
- No explicit governance/compliance separation

**Gap:** The `fund_manager` agent is simultaneously governance (decides allocation), execution (sizes positions), and compliance (checks concentration). This conflation means a single LLM call makes all three decisions, with no adversarial check. The OrgAgent insight is that *separation of concerns* at the agent level reduces errors AND tokens.

**Concrete integration:**

```
┌──────────────────────────────────────────────────────────┐
│            HIERARCHICAL TRADING DESK (OrgAgent)           │
│                                                            │
│  ┌──────────────────────────────────────────────┐         │
│  │  GOVERNANCE LAYER (CIO Agent)                 │         │
│  │  - Sets daily capital allocation per strategy  │         │
│  │  - Declares regime and conviction level         │         │
│  │  - Approves/rejects strategy activations        │         │
│  │  - Weekly: rebalance strategy weights           │         │
│  │  Tier: heavy (Sonnet), runs once per day        │         │
│  └──────────────────────┬───────────────────────┘         │
│                         │ mandates                         │
│  ┌──────────────────────▼───────────────────────┐         │
│  │  EXECUTION LAYER (Strategy Agents, parallel)  │         │
│  │  - momentum_agent: scans trending setups       │         │
│  │  - mean_rev_agent: scans mean-reversion        │         │
│  │  - options_agent: scans vol structures          │         │
│  │  - earnings_agent: scans earnings plays         │         │
│  │  Each: medium (Haiku), runs every 5 min         │         │
│  │  Constrained by governance capital allocation   │         │
│  └──────────────────────┬───────────────────────┘         │
│                         │ proposals                        │
│  ┌──────────────────────▼───────────────────────┐         │
│  │  COMPLIANCE LAYER (Risk Officer Agent)        │         │
│  │  - Validates proposals against risk_gate       │         │
│  │  - Checks factor correlation (knowledge graph) │         │
│  │  - Enforces sector/regime/heat constraints     │         │
│  │  - Cannot be overridden by execution layer     │         │
│  │  Tier: medium (Haiku), deterministic checks    │         │
│  │  IMMUTABLE — same protection as risk_gate.py   │         │
│  └──────────────────────────────────────────────┘         │
│                                                            │
│  Token savings: governance runs 1x/day (not every cycle)   │
│  Execution agents are narrower (fewer tools, shorter       │
│  prompts) → cheaper per call                               │
│  Compliance is mostly deterministic (minimal LLM)          │
└──────────────────────────────────────────────────────────┘
```

**Implementation path:**
1. Split `fund_manager` into CIO (governance, daily) + risk_officer (compliance, per-trade)
2. Split `trade_debater` into strategy-specific execution agents (momentum, mean_rev, options, earnings)
3. CIO agent sets `daily_mandate`: capital allocation per strategy type, max new positions, regime stance
4. Execution agents propose trades *within their mandate* — can't exceed allocation
5. Risk officer validates against risk_gate + knowledge graph factor correlation + mandate compliance
6. **Key insight:** CIO runs once per morning (heavy model, one expensive call). Strategy agents run every 5 min (medium model, cheap calls). This *inverts* the cost structure — expensive reasoning happens less often.

**Why this matters:** OrgAgent's token reduction of 74% is directly applicable. Today, QuantStack runs `trade_debater` (heavy, Sonnet) every 5-minute cycle at ~$0.05-0.15 per call. That's $150-450/day. With OrgAgent separation: CIO at $0.15/day + 4 strategy agents at Haiku prices (~$0.01/call x 80 calls/day = $3.20/day) + deterministic compliance = **~$10/day vs. $150-450/day.** Same or better quality because execution agents have narrower scope.

**Estimated effort:** 3 weeks. Requires splitting `trading/nodes.py` into governance/execution/compliance modules and updating `agents.yaml`.

---

### AR-5. Long-Horizon Parallel Research (Kosmos Pattern)

**What it is:** Kosmos runs 12-hour research cycles, executing ~42,000 lines of code and reading 1,500 papers per run. Independent scientists validated 79.4% of its findings. The key is *parallel* long-horizon exploration — multiple research threads running simultaneously.

**What QuantStack has today:**
- Research graph runs 10-min cycles (short horizon)
- Fan-out mode (`RESEARCH_FAN_OUT_ENABLED`) does parallel per-symbol validation
- Community intel does 3-pass web search (short, keyword-based)
- No multi-stream parallel research

**Gap:** QuantStack researches one hypothesis at a time, in serial. Even with fan-out, it's parallel *validation* of a single hypothesis across symbols. Kosmos's pattern is parallel *exploration* — multiple independent research streams, each pursuing different alpha themes.

**Concrete integration:**

```
┌──────────────────────────────────────────────────────────┐
│           PARALLEL RESEARCH STREAMS (Weekend Mode)        │
│           (Friday 20:00 → Monday 04:00, 56 hours)        │
│                                                            │
│  Stream 1: Factor Mining                                   │
│  ┌──────────────────────────────────────────┐             │
│  │ Academic paper scan → extract testable    │             │
│  │ factors → compute IC across universe →     │             │
│  │ rank by IC stability → register winners    │             │
│  └──────────────────────────────────────────┘             │
│                                                            │
│  Stream 2: Regime Research                                 │
│  ┌──────────────────────────────────────────┐             │
│  │ Historical regime labeling → strategy     │             │
│  │ performance by regime → build regime-     │             │
│  │ conditional allocation model → test OOS   │             │
│  └──────────────────────────────────────────┘             │
│                                                            │
│  Stream 3: Cross-Asset Signals                             │
│  ┌──────────────────────────────────────────┐             │
│  │ Scan bond-equity-FX-commodity rels →      │             │
│  │ test lead-lag → build cross-asset signals │             │
│  │ → validate with current universe          │             │
│  └──────────────────────────────────────────┘             │
│                                                            │
│  Stream 4: Portfolio Construction Research                  │
│  ┌──────────────────────────────────────────┐             │
│  │ Test alternative optimizers (risk parity, │             │
│  │ Black-Litterman, hierarchical risk) →     │             │
│  │ compare OOS Sharpe vs. current HRP        │             │
│  └──────────────────────────────────────────┘             │
│                                                            │
│  Coordinator: merge findings, deduplicate,                 │
│  update knowledge graph, prioritize Monday pipeline        │
└──────────────────────────────────────────────────────────┘
```

**Implementation path:**
1. Add `weekend_mode` to research graph runner
2. Create 4 research stream configs in `agents.yaml` (each a sub-graph with its own agents)
3. Use `LangGraph Send()` to launch parallel streams (already supported by fan-out infrastructure)
4. Each stream writes to `research_discoveries` table with stream ID
5. Monday morning: coordinator node merges, deduplicates, ranks by expected IC, feeds top-N to strategy pipeline
6. **Literature integration**: Use web search tools (already in community_intel) but with targeted queries: "machine learning alpha factor {year}" "market microstructure anomaly" "cross-asset momentum"

**Why this matters:** 56 weekend hours x 4 parallel streams = 224 stream-hours of research. At autoresearch speed (~96 experiments/8 hours), that's **~2,688 experiments per weekend**. Current system: ~6 hypotheses on a weekend day = ~12/weekend. That's a **224x throughput increase**.

**Estimated effort:** 2 weeks for weekend mode + stream orchestration. Reuses existing research graph infrastructure.

---

### AR-6. Consensus-Based Signal Validation (AI-Supervisor + ResearchGym Insights)

**What it is:** AI-Supervisor uses consensus mechanisms — multiple agents must agree before a finding is integrated. ResearchGym found that successful research agents need "hypothesis humility" and explicit experiment prioritization. Combined: don't trust any single agent's recommendation.

**What QuantStack has today:**
- `hypothesis_critic` scores 0-1, single agent gatekeeps
- `trade_debater` provides structured bull/bear debate, but single agent
- `fund_manager` makes final allocation, single agent
- Signal synthesis is rule-based (not agent-based), which is actually more robust

**Gap:** Every decision point relies on a *single LLM call*. If that call hallucinates, misinterprets data, or has a bad prompt day, the error propagates. The debate agent is one agent playing both sides — it's not adversarial in the true sense.

**Concrete integration:**

```
┌──────────────────────────────────────────────────────┐
│          ADVERSARIAL CONSENSUS PROTOCOL               │
│                                                        │
│  For HIGH-STAKES decisions (>$5k position size):       │
│                                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│  │ Bull     │  │ Bear     │  │ Neutral  │              │
│  │ Advocate │  │ Advocate │  │ Arbiter  │              │
│  │ (Sonnet) │  │ (Sonnet) │  │ (Sonnet) │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │              │              │                    │
│       ▼              ▼              ▼                    │
│  ┌──────────────────────────────────────┐              │
│  │  CONSENSUS RULES:                     │              │
│  │  - 3/3 agree: execute at full size    │              │
│  │  - 2/3 agree: execute at 50% size     │              │
│  │  - 1/3 or 0/3: reject                 │              │
│  │  - Any agent cites kill-switch        │              │
│  │    condition: immediate reject         │              │
│  └──────────────────────────────────────┘              │
│                                                        │
│  For LOW-STAKES decisions (<$5k):                      │
│  Single trade_debater (current behavior, faster)       │
└──────────────────────────────────────────────────────┘
```

**Implementation path:**
1. Add `consensus_mode` flag to trading graph (triggered when proposed position size > threshold)
2. Create `bull_advocate` and `bear_advocate` agent configs — same data, opposing mandates
3. Use `LangGraph Send()` to run all three in parallel (no serialization bottleneck)
4. Deterministic consensus node applies the voting rules — no LLM needed for the merge
5. Log all three perspectives in `trade_decisions` table for post-hoc analysis

**Why this matters:** ResearchGym showed that single-agent research has a 6.7% success rate. Multi-agent consensus doesn't guarantee better outcomes, but it catches more *obvious errors* — the bear advocate sees the risk the bull advocate ignores. The key insight: only apply this to high-stakes decisions. Low-stakes decisions should be fast (single agent).

**Estimated effort:** 1 week. Minimal infrastructure — three agent configs, one `Send()` node, one deterministic merge.

---

### AR-7. Error-Driven Iteration (MAGNET + ResearchGym Pattern)

**What it is:** MAGNET focuses improvement effort on *failure modes*. ResearchGym found agents need "patience and experimentation discipline" — resist premature optimization, track what's been tried, and prioritize high-signal experiments.

**What QuantStack has today:**
- `trade_reflector` classifies outcomes (win/loss/breakeven) with P&L attribution
- `outcome_tracker.py` tracks trade outcomes
- `skill_tracker.py` tracks agent calibration
- No systematic feedback from *losses* to *research priorities*

**Gap:** When a strategy loses money, the reflector logs "stopped out, market reversed." That log sits in the database. No system reads it and says: "Three momentum strategies stopped out this week due to overnight gaps → research overnight risk hedging → add to research queue with HIGH priority." The *pain signal* from losses doesn't drive the *research agenda*.

**Concrete integration:**

```
Loss Analysis Pipeline (runs daily after market close):
────────────────────────────────────────────────────────

1. COLLECT: All losing trades from today
        ↓
2. CLASSIFY by failure mode:
   - Stop hit (market moved against thesis)
   - Time stop (thesis didn't play out in window)
   - Regime mismatch (entered trending, market ranged)
   - Data stale (signal was based on old data)
   - Factor crowding (multiple positions hit same factor)
   - Black swan (>3 sigma move, unforeseeable)
        ↓
3. AGGREGATE: Failure mode frequency over trailing 30 days
        ↓
4. PRIORITIZE: Top failure modes by cumulative P&L impact
        ↓
5. GENERATE: Research tasks targeting top failure modes
   - "Regime mismatch" (3 occurrences, -$1,200) →
     queue: "Improve intraday regime detection, test 15-min HMM"
   - "Factor crowding" (2 occurrences, -$800) →
     queue: "Implement factor exposure pre-trade check"
   - "Overnight gap" (4 occurrences, -$2,100) →
     queue: "Research overnight hedging with VIX calls"
        ↓
6. FEED: research_queue with priority = f(cumulative_loss)
```

**Implementation path:**
1. Add `loss_analyzer` node to supervisor graph (daily, 16:30 ET)
2. Create failure mode taxonomy as enum (start with 6 modes above, evolve)
3. Deterministic aggregation — no LLM needed for counting and ranking
4. LLM-assisted research task generation (Haiku, one call: "Given these failure modes, what research would address them?")
5. Priority scoring: `priority = cumulative_loss_30d * recency_weight`
6. Research graph's `context_load` already polls `research_queue` — this slots in naturally

**Why this matters:** This closes the learning loop. Today: loss → log → forgotten. Tomorrow: loss → classify → prioritize → research → new strategy/hedge → prevented future loss. This is the *compounding* in "compound capital." Every loss makes the system smarter.

**Estimated effort:** 1 week. The infrastructure already exists (outcome_tracker, research_queue, event_bus). Just needs the orchestration node.

---

### AR-8. Dynamic Tool Discovery (Mimosa Pattern)

**What it is:** Mimosa uses Model Context Protocol (MCP) to dynamically discover available tools and synthesize task-specific agent workflows. No hardcoded tool lists — the orchestrator finds what's available and composes agents on the fly.

**What QuantStack has today:**
- `TOOL_REGISTRY` with 122 tools (92 stubbed)
- Static `agents.yaml` binds tools by name at startup
- `tool_binding.py` supports deferred loading + BM25 search
- No dynamic tool discovery or composition

**Gap:** When a new data source becomes available (say, FRED API for macro, or a new broker's options chain), someone must manually add a tool, register it, and update `agents.yaml`. The system can't discover new capabilities. Worse: 92 stubbed tools pollute the registry — agents waste LLM calls on tools that don't work.

**Concrete integration:**

```
┌──────────────────────────────────────────────────────┐
│              DYNAMIC TOOL LIFECYCLE                    │
│                                                        │
│  1. REGISTRY CLEANUP (immediate):                      │
│     - Remove all 92 stubbed tools from TOOL_REGISTRY   │
│     - Add them to PLANNED_TOOLS registry (metadata only)│
│     - Agents never see planned tools in their bindings  │
│                                                        │
│  2. TOOL HEALTH MONITORING (per cycle):                │
│     - Track invocation count, success rate, latency    │
│     - If success_rate < 50% over 24h: disable tool     │
│     - If latency > 10s p95: flag for optimization      │
│     - If invocation_count = 0 over 7d: candidate for   │
│       removal from agent bindings                      │
│                                                        │
│  3. TOOL SYNTHESIS (weekly, via AutoResearchClaw):     │
│     - Check PLANNED_TOOLS for implementation priority  │
│     - Implement top-priority tools based on:           │
│       a. Agent request frequency (how often agents     │
│          tried to use the stub and got error)           │
│       b. Knowledge graph gap (what data is missing)    │
│       c. Loss analysis need (what would prevent losses)│
│     - Auto-test, validate, register                    │
│                                                        │
│  4. CAPABILITY ANNOUNCEMENT (via event bus):           │
│     - When new tool implemented: publish TOOL_ADDED    │
│     - Agents receive capability updates at cycle start │
│     - Meta-layer updates tool bindings in agents.yaml  │
└──────────────────────────────────────────────────────┘
```

**Implementation path:**
1. **Immediate:** Split `TOOL_REGISTRY` into `ACTIVE_TOOLS` (30 working) and `PLANNED_TOOLS` (92 stubs). Agents only bind to `ACTIVE_TOOLS`.
2. Add `tool_health` table: `(tool_name, invocations_24h, success_rate, p95_latency, last_invoked)`
3. Wrapper in `agent_executor.py`: after each tool call, update `tool_health`
4. Weekly AutoResearchClaw task: implement highest-priority planned tool based on demand signals
5. Event bus: `TOOL_ADDED` event → all graphs reload tool bindings at next cycle start

**Why this matters:** Removes the "92 stubbed tools" finding (TC1) systematically. Creates a demand-driven implementation pipeline — tools get built because agents *need* them, not because a human guessed they might be useful. The health monitoring also catches degradation (API changes, credential expiry) before agents fail silently.

**Estimated effort:** 1 week for registry split + health monitoring. Tool synthesis is ongoing (AutoResearchClaw already runs weekly).

---

### AR-9. Fixed-Budget Experiment Discipline (ResearchGym Insight)

**What it is:** ResearchGym found that AI agents succeed only 6.7% of the time on research tasks, with key failure modes: impatience, poor resource management, overconfidence in weak hypotheses, context collapse on long tasks. The fix: explicit experiment budgets, patience protocols, and memory management.

**What QuantStack has today:**
- No per-experiment compute budget
- No per-cycle token budget per agent (finding LH4)
- 150k char message pruning is the only guard
- Hypothesis loop can burn 360s on a weak hypothesis
- No concept of "experiment prioritization" — FIFO from research_queue

**Gap:** The system has no resource awareness. A research cycle might spend $5 on a hypothesis that had a 5% chance of passing validation. No prioritization means low-value research tasks consume the same resources as high-value ones.

**Concrete integration:**

```
EXPERIMENT BUDGET PROTOCOL:
──────────────────────────

Per-Cycle Budget:
  - Token budget: 50k tokens/cycle (research), 30k tokens/cycle (trading)
  - Wall-clock budget: 600s (research), 300s (trading)
  - LLM cost budget: $0.50/cycle (research), $0.20/cycle (trading)

Per-Experiment Budget (within research cycle):
  - Hypothesis generation: 5k tokens, 30s
  - Hypothesis critique: 3k tokens, 20s
  - Signal validation: 2k tokens + 60s compute
  - Backtest: 0 tokens, 120s compute
  - ML experiment: 0 tokens, 180s compute

Prioritization Formula:
  priority = (expected_IC x regime_fit x novelty_score) / estimated_compute_cost

  where:
  - expected_IC: from knowledge graph (similar factors' historical IC)
  - regime_fit: 1.0 if factor matches current regime, 0.5 otherwise
  - novelty_score: 1.0 if untested, 0.7 if variant of tested factor
  - estimated_compute_cost: from historical experiment runtimes

Patience Protocol:
  - Never discard a hypothesis after 1 backtest. Run 3 time windows.
  - If IC > 0.01 but < 0.02 (marginal): test with different feature engineering
  - If IC < 0.01 across 3 windows: discard and log reason in knowledge graph
  - Track "experiments per discovery" ratio — if declining, meta-agent
    investigates (are we searching in explored space?)
```

**Implementation path:**
1. Add `BudgetTracker` class to `agent_executor.py`: tracks tokens, wall-clock, cost per cycle
2. Add budget fields to `AgentConfig` in `agents.yaml`
3. When budget exhausted: graceful exit at next node boundary (don't kill mid-tool-call)
4. Add `prioritize_experiments` deterministic function: scores research_queue tasks by priority formula
5. Add patience protocol to `hypothesis_critique`: require 3 backtest windows before reject

**Why this matters:** Without budgets, a single bad research cycle can cost $5-10 in LLM tokens with zero output. With budgets: predictable cost per day ($15-20 for research, $10-15 for trading), no runaway agents, and the highest-value experiments get compute first.

**Estimated effort:** 1 week. BudgetTracker is straightforward. Prioritization formula needs tuning (start with equal weights, let meta_threshold_tuner optimize).

---

### AR-10. Autonomous Feature Factory (MAGNET Pattern)

**What it is:** MAGNET automatically generates training datasets from raw sources, removing the manual data engineering bottleneck. Applied to crypto prediction (41-54.9% hit rate) and video safety (0.93+ accuracy).

**What QuantStack has today:**
- 14-phase `AcquisitionPipeline` fetches raw data from Alpha Vantage + Alpaca
- `labeling.py` does triple-barrier labeling for ML
- `feature_importance.py` does SHAP analysis
- Feature engineering is agent-guided (ml_scientist tells tools what to compute)
- No automated feature discovery or dataset synthesis

**Gap:** Features are manually specified. The ml_scientist agent asks for specific features (RSI_14, MACD_signal, etc.) based on its prompt. There's no systematic exploration of feature space — no auto-generation of interaction features, no lag optimization, no automated alternative data integration.

**Concrete integration:**

```
AUTONOMOUS FEATURE FACTORY:
──────────────────────────

Phase 1: Feature Enumeration (overnight, weekly)
  - Base features: OHLCV, volume, returns (already computed)
  - Auto-generated features:
    ┌─────────────────────────────────────────────┐
    │ For each base feature X:                     │
    │   - Lags: X_lag_1, X_lag_5, X_lag_21        │
    │   - Rolling: X_roll_5_mean, X_roll_21_std   │
    │   - Cross: X * Y for top-10 correlated pairs │
    │   - Regime-conditional: X_when_trending       │
    │   - Time features: X_day_of_week, X_month    │
    │   - Residualized: X - beta*SPY               │
    └─────────────────────────────────────────────┘
  
Phase 2: Feature Screening (overnight, weekly)
  - Compute IC for all features across universe
  - Filter: IC > 0.01 AND IC_stability > 0.5
  - Drop: features with >0.95 correlation to kept features
  - Output: curated feature matrix (~50-100 features)

Phase 3: Feature Monitoring (daily)
  - PSI drift check per feature (already exists)
  - IC decay tracking per feature
  - Auto-replace: if feature IC decays >50%, regenerate variant
```

**Implementation path:**
1. Add `feature_factory.py` to `src/quantstack/ml/`: enumerate, screen, monitor
2. Enumeration is pure computation — no LLM needed, runs on CPU overnight
3. Screening uses existing `trainer.py` + `feature_importance.py`
4. Hook into overnight autoresearch mode: feature factory runs first, then strategy experiments use the curated feature set
5. Store feature metadata in `feature_registry` table: name, IC, stability, last_computed, status

**Why this matters:** Manual feature engineering is QuantStack's biggest research bottleneck. ml_scientist asks for 15-20 predefined features. A feature factory generates 500+ candidates, screens to 50-100, and *auto-replaces decaying features*. This is the MAGNET pattern applied to alpha generation — remove the human bottleneck in data preparation.

**Estimated effort:** 2 weeks. Feature enumeration is straightforward. Screening reuses existing ML infrastructure.

---

### Advanced Research Integration Roadmap

| Phase | Techniques | Timeline | Prerequisites | Expected Impact |
|-------|-----------|----------|---------------|-----------------|
| **Phase 1: Foundation** | Registry cleanup (AR-8), Error-driven iteration (AR-7), Experiment budgets (AR-9) | Week 1-2 | P0 safety fixes from main audit | Cost control, learning loop, clean tool layer |
| **Phase 2: Research Multiplier** | Autoresearch loop (AR-1), Feature factory (AR-10), Long-horizon parallel research (AR-5) | Week 3-5 | Phase 1, overnight mode in graph runner | 16x-224x research throughput |
| **Phase 3: Intelligence Layer** | Knowledge graph (AR-3), Consensus validation (AR-6) | Week 5-7 | Phase 2 (need research output to populate graph) | Institutional memory, fewer bad trades |
| **Phase 4: Self-Improvement** | Meta layer (AR-2), Hierarchical governance (AR-4), Dynamic tool discovery (AR-8 phase 2) | Week 7-10 | Phase 3 (need performance data for meta-optimization) | Compounding improvement, 74% token reduction |

### Cost-Benefit Summary

| Technique | Implementation Cost | Annual Token Savings | Alpha Impact |
|-----------|-------------------|---------------------|--------------|
| Autoresearch loop | 1 week | -$2k (more research) | +16x hypothesis throughput |
| Meta layer | 3 weeks | +$5k (meta agents) | Compounding quality improvement |
| Knowledge graph | 4 weeks | $0 (deterministic) | Prevent redundant research, factor crowding |
| OrgAgent hierarchy | 3 weeks | **-$50k** (74% token reduction) | Same or better trade quality |
| Consensus validation | 1 week | +$3k (3x LLM calls on big trades) | Fewer catastrophic trades |
| Error-driven iteration | 1 week | $0 | Systematic loss reduction |
| Feature factory | 2 weeks | $0 (CPU compute) | 10x feature coverage |
| Experiment budgets | 1 week | **-$15k** (bounded spend) | Predictable costs |
| Dynamic tool discovery | 1 week | $0 | Self-improving tool layer |
| Parallel research streams | 2 weeks | +$5k (weekend compute) | 224x weekend research throughput |
| **NET** | **20 weeks** | **~$54k/year savings** | **Compounding alpha + self-improvement** |

### The Endgame: Self-Improving Autonomous Trading Company

```
TODAY (v1.0):
  Human writes strategy → System backtests → System trades → Human reviews losses

PHASE 2 (v2.0, +5 weeks):
  System generates 96 hypotheses/night → System filters → System trades → System logs losses

PHASE 3 (v3.0, +7 weeks):
  System generates hypotheses → Knowledge graph prevents redundancy →
  Consensus validates → System trades → Loss analyzer feeds research queue

PHASE 4 (v4.0, +10 weeks):
  Meta layer improves hypothesis generation quality →
  Meta layer tunes risk thresholds from realized outcomes →
  Meta layer optimizes its own optimization process →
  System compounds intelligence as it compounds capital
```

**The v4.0 system doesn't just make money — it gets better at making money, and it gets better at getting better.** That's the Karpathy/Hyperagents synthesis: autoresearch for breadth, metacognition for depth, knowledge graphs for memory, hierarchical governance for efficiency, and error-driven iteration for resilience.

This is what separates a trading system from a trading *company*.

---
---

# PART II: PRINCIPAL QUANT SCIENTIST DEEP AUDIT

**Auditor:** Principal Quant Scientist (ex-Citadel/Two Sigma caliber review)
**Date:** 2026-04-06
**Scope:** Everything the previous CTO audit missed — execution reality, statistical rigor, agent failure modes, infrastructure security, missing roles, and the feedback loops that separate a demo from a fund.

**Bottom line:** The CTO audit was competent surface-level architecture review. It found real issues (stop-loss optional, prompt injection, no backups). But it missed the **quant substance** — the statistical validity of signals, the execution realism gap, the absence of feedback loops that make a system learn, and the missing roles that a 2,000-person fund staffs for good reason. This system doesn't just have engineering gaps. It has **alpha validation gaps** that mean we don't even know if the strategies work.

---

## Revised Scorecard (CTO + Quant Scientist Combined)

| Subsystem | CTO Grade | Quant Scientist Grade | Combined | New Findings |
|-----------|-----------|----------------------|----------|-------------|
| Execution & Risk | B- | **D** | **D+** | No real algo execution, no margin tracking, no Greeks risk, no TCA feedback |
| Signal Quality & Statistical Rigor | (not audited) | **D-** | **D-** | No IC computation, no confidence intervals, no decay modeling, no look-ahead bias protection |
| Graph Architecture | B | **C** | **C+** | Race conditions, no state validation, errors don't block execution |
| Data Pipeline & Signals | B | **C-** | **C** | No point-in-time semantics, survivorship bias, loose validation |
| ML Pipeline | (not audited) | **D** | **D** | No hyperparameter optimization, weak drift detection, no multicollinearity check |
| LLM Routing & Prompts | C+ | C+ | **C+** | (CTO audit was thorough here) |
| Ops & Infrastructure | C+ | **D+** | **D+** | Root containers, exposed ports, default passwords, no migration versioning |
| Tools & Registry | C | C | **C** | (CTO audit was thorough here) |
| Missing Agent Roles | (not audited) | **F** | **F** | No compliance, no reconciliation, no corporate actions, no factor exposure |
| Feedback Loops & Learning | (not audited) | **F** | **F** | Zero closed loops — losses don't drive research, outcomes don't tune parameters |
| Backtesting Validity | (not audited) | **D-** | **D-** | Transaction costs underapplied, no Monte Carlo, no survivorship adjustment |
| Security & Compliance | (not audited) | **D** | **D** | No SEC audit trail, no immutable logs, no best execution compliance |
| **OVERALL** | **B-** | **D+** | **C-** | **47 new CRITICAL/HIGH findings** |

### Updated Finding Count (Post-Verification)

| Severity | CTO Audit | Quant Scientist (New) | Retracted | Net New | Total |
|----------|-----------|----------------------|-----------|---------|-------|
| CRITICAL | 22 | 19 | 3 | **16** | **38** |
| HIGH | 34 | 28 | 2 | **26** | **60** |
| MEDIUM | 45 | 21 | 0 | **21** | **66** |
| **TOTAL** | **101** | **68** | **5** | **63** | **164** |

### Verification & Corrections (Intellectual Honesty)

**I initially reported 68 findings. Upon code-level verification, 5 were wrong:**

| Retracted Finding | What I Claimed | What Actually Exists | My Error |
|-------------------|---------------|---------------------|----------|
| QS-E2: "Zero margin tracking" | No margin/leverage awareness | `core/risk/span_margin.py` (537 lines): full SPAN with 16-scenario stress test. `core/risk/options_risk.py` (444 lines): portfolio Greeks tracking with limits | **Didn't find the module. Searched too narrowly.** |
| QS-A1: "No reconciliation" | No broker-vs-system check | `guardrails/agent_hardening.py:463-550`: detects phantom/unknown positions + quantity mismatches. `execution_monitor.py`: `_reconcile_loop()` | **Exists as guardrail function, not as agent — I was looking for an agent when a function does it.** |
| QS-I7: "No job overlap detection" | Scheduler starts duplicate jobs | `strategy_lifecycle.py:422-441`: heartbeat guard checks for in-progress runs within 9 min, skips gracefully | **Real guard exists. My agent searched scheduler.py but the guard is in lifecycle.py.** |
| Loop-1: "Zero feedback loops" | Trade outcomes don't drive research | `hooks/trade_hooks.py:118-144`: loss > 1% → `research_queue` bug_fix task + ReflexionMemory + CreditAssigner | **One-directional feedback exists. Not "zero." Loop is incomplete (no auto-weight update), but exists.** |
| QS-I5: "No audit log" | No immutable audit trail | `audit/decision_log.py`: append-only design with SHA256 context hashes | **Exists but immutability is code convention, not DB constraint. Downgraded to MEDIUM.** |

**Also refined:**
| Finding | Original Claim | Corrected Assessment |
|---------|---------------|---------------------|
| QS-E3: "Greeks risk absent" | Zero delta/gamma/vega limits | `options_risk.py` tracks and limits Greeks. But `risk_gate.py` options path only checks DTE + premium. **Gap is in the risk gate integration, not the capability.** |
| Loop-1: "No loss → research loop" | Zero closed loops | One-way loop exists (trade_hooks → research_queue). Missing: auto-weight adjustment, failure mode taxonomy, systematic loss classification. **Downgraded from "broken" to "incomplete."** |

**Am I over-engineering like the previous CTO?** Let me be explicit about what I'm NOT recommending:
- NOT recommending a rewrite of the execution layer — the architecture is sound, the gaps are incremental
- NOT recommending new infrastructure (no Kafka, no Redis, no separate vector DB)
- NOT recommending new agents for the sake of agents — the "missing roles" can mostly be deterministic functions, not LLM agents
- Every fix I recommend uses existing infrastructure (PostgreSQL, existing modules, existing tool framework)
- The previous CTO almost pushed you into RAG migration that would've cost 8x more per session and added 3 new failure modes. I'm recommending fixes within what you already have.

---

## 12. Execution Layer Deep Audit: "The System Can't Actually Trade"

### The Core Problem

The CTO audit found stop-losses are optional (C1), bracket orders fail silently (C2), and Alpaca has no bracket support (C3). Those are real issues. But they're surface-level. The deeper problem: **the execution layer is a prototype that simulates trading, not a system that can trade.**

A real execution desk has: algorithm scheduling (TWAP/VWAP/POV), market impact modeling, TCA feedback loops, margin management, Greeks monitoring, intraday circuit breakers, partial fill tracking, and best execution compliance. QuantStack has none of these.

### CRITICAL Findings (New)

#### QS-E1: No Real Execution Algorithm Implementation — TWAP/VWAP Selected But Never Executed

**Location:** `order_lifecycle.py:455-478`, `paper_broker.py:243-276`

The order lifecycle selects an execution algorithm (IMMEDIATE/TWAP/VWAP/POV) based on order size vs. ADV. This looks sophisticated. But the paper broker executes **everything as a single fill**. There is no time-slicing, no child order generation, no participation rate enforcement.

```
What the system claims:    "Order 5% ADV → VWAP algorithm selected"
What actually happens:      Single market order, instant fill, one price
What should happen:         50 child orders over 2 hours, each ≤0.1% ADV
```

**Impact:** Paper trading results show no market impact. Live trading with the same sizes would incur 10-50 bps of impact cost per trade. A strategy that backtests at Sharpe 1.2 with instant fills may have Sharpe 0.4 with realistic execution.

**Fix:** Implement TWAP/VWAP as child order generators. Paper broker simulates fills against historical tick/bar data with realistic participation constraints.

#### ~~QS-E2: Zero Margin/Leverage Tracking~~ — RETRACTED

**Verification found:** `core/risk/span_margin.py` (537 lines) implements full SPAN-style margin with 16-scenario stress testing, scanning risk, delivery margin, and inter-spread credits. `core/risk/options_risk.py` (444 lines) tracks portfolio delta/gamma/theta/vega with enforced limits. **This finding was wrong — margin tracking is extensively implemented.** The gap is narrower than claimed: the `risk_gate.py` options path doesn't call the Greeks manager, but the capability exists and is real.

#### QS-E3: Options Greeks Risk Completely Absent from Risk Gate

**Location:** `risk_gate.py:519-589`

The options risk checks are:
- DTE bounds (7-60 days) ✓
- Premium at risk (2% per position, 8% total) ✓
- Delta/gamma/vega/theta limits: **NONE**
- Portfolio Greeks aggregation: **NONE**

A 50-contract short straddle at 21 DTE (high gamma, high vega) gets the same risk treatment as a 50-delta call spread (low gamma, low vega). The straddle can lose $100K on a 5% move because gamma explodes near expiration. The system has no model for this.

**Missing checks that any options desk enforces:**
- Portfolio delta exposure limit (e.g., ±$50K per 1% move)
- Portfolio gamma limit (e.g., max $10K P&L per 1% squared)
- Portfolio vega limit (e.g., max $5K per 1 vol point)
- Theta budget (acceptable daily time decay)
- Pin risk near expiration (DTE < 3 AND near strike)

**Fix:** Add Greeks aggregation to `portfolio_state.py`. Add Greeks limits to risk gate. Block any trade that would push portfolio Greeks outside limits.

#### QS-E4: Liquidity Risk Severely Underestimated

**Location:** `risk_gate.py:453-465`

Current liquidity check: `if daily_volume < min_daily_volume: reject`. That's it. Missing:

- **Bid-ask spread check**: Can trade a 500K ADV stock, but if spread is 50 bps, execution cost is 25 bps per side
- **Market depth**: 500K ADV doesn't mean you can trade 5K shares at tight spread — depth may only support 500 shares at NBBO
- **Time-of-day liquidity**: A symbol liquid at 10:00 AM may be illiquid at 3:55 PM
- **Stressed liquidity**: If entire portfolio tries to exit simultaneously (drawdown event), spreads widen 10x
- **Exit liquidity**: System checks entry liquidity but not "can I get out in a crisis?"

**Impact:** In a flash crash or vol spike, the system will attempt to exit positions through illiquid markets, incurring catastrophic slippage. The paper broker doesn't model this — it always fills at half-spread.

**Fix:** Add `LiquidityModel` that estimates depth from historical intraday volume profiles. Pre-trade check: `if order_size > estimated_depth_at_current_time * 0.1: scale down or reject`. Stressed liquidity test: `if all_positions_exit_simultaneously, what is total slippage cost?`

#### QS-E5: No Intraday Drawdown Circuit Breaker

**Location:** `risk_gate.py:433-451`

Daily loss limit (-2%) triggers a halt. But this only fires **after losses are realized**. There is no:
- Real-time unrealized P&L monitoring against threshold (e.g., -3% unrealized → reduce exposure)
- Velocity check (lost -1% in 5 minutes → halt regardless of daily limit)
- Single-position circuit breaker (one position down -20% → force review)

**Impact:** Can lose 5% in 5 minutes before daily halt triggers (which requires trades to close, not just mark-to-market).

**Fix:** Add `IntraDayCircuitBreaker` to execution monitor. Check unrealized P&L every tick cycle. Thresholds: -1.5% unrealized → halt new entries, -2.5% unrealized → begin systematic exit, -5% → emergency liquidation.

#### QS-E6: TCA Feedback Loop Completely Missing

**Location:** `tca_engine.py`, `tca_storage.py`, `tca_recalibration.py`

Pre-trade TCA exists — estimates cost using Almgren-Chriss model with fixed coefficients. Post-trade TCA exists — stores realized costs. But **there is no feedback loop between them**:

- If realized slippage is consistently 2x forecast, coefficients don't adjust
- If slippage is worse at 3:00 PM, the algorithm doesn't learn
- If a symbol's spread widened 3x overnight, estimates are stale
- `tca_recalibration.py` requires 50 trades per segment to fit — will be non-estimated for months

**Impact:** The system makes position sizing decisions based on cost estimates that may be 2-5x wrong. Alpha that looks positive pre-trade is negative post-cost.

**Fix:** Implement daily recalibration: compare realized vs. forecast, update Almgren-Chriss parameters per symbol/time-of-day bucket. Until 50 trades accumulated: use conservative multiplier (2x forecast cost).

#### QS-E7: Best Execution Compliance Infrastructure Absent

**Location:** Missing entirely

SEC Rule 606 and FINRA Rule 5310 require broker-dealers to demonstrate best execution. Even for a proprietary system, audit trail of execution quality is essential for:
- Demonstrating that fills were at or better than NBBO at time of order
- Tracking execution venue quality (if using multiple brokers)
- Documenting execution algorithm selection rationale
- Maintaining fill-level audit trail with timestamps, quantities, prices, venues

QuantStack has none of this. The `fills` table stores basic fill data but no NBBO reference, no venue data, no algorithm selection rationale.

**Fix:** Add `execution_audit` table: `(order_id, nbbo_bid, nbbo_ask, fill_price, fill_venue, algo_selected, algo_rationale, timestamp_ns)`. Populate on every fill.

### HIGH Findings (New)

#### QS-E8: Slippage Model Is Regime-Agnostic

**Location:** `paper_broker.py:243-276`

Paper broker uses square-root impact model with fixed spread. Same slippage in calm markets and during a vol spike. Same slippage for a $10 stock and a $500 stock. No adjustment for:
- Current VIX level (high vol = wider spreads)
- Time of day (open/close = wider spreads)
- Earnings proximity (IV spike = wider options spreads)
- Market cap tier (small cap = less depth)

**Fix:** Parameterize slippage by: `spread_multiplier = base_spread * (1 + vix_adjustment + time_of_day_adjustment + event_premium)`.

#### QS-E9: Partial Fills Overwrite Price History

**Location:** `order_lifecycle.py:266-292`

When a partial fill arrives, the fill price is updated but previous partial fill prices are overwritten, not accumulated. Can't reconstruct:
- Average fill price from sequence of partial fills
- Execution VWAP for post-trade TCA
- Fill trajectory (did we get worse prices over time?)

**Fix:** Add `fill_legs` table: `(order_id, leg_sequence, quantity, price, timestamp)`. Compute VWAP from legs.

#### QS-E10: Correlation Check Is Stub — Alert Only, No Veto

**Location:** `risk_gate.py:781-825`

The CTO audit noted correlation is post-hoc (H1). The deeper problem: the correlation spike detection method is a **stub**. It checks 60-day rolling correlation, but:
- If correlation data is unavailable, it returns "alert" not "reject"
- No pre-trade veto power — only monitoring
- No stressed correlation modeling (correlations spike to 0.95 in crashes)

**Fix:** Pre-trade correlation gate: `if adding_position_corr_with_existing > 0.7: apply_concentration_haircut(50%)`. Stressed correlation: use DCC-GARCH or simply `min(historical_corr, 0.9)` as stress case.

#### QS-E11: No Borrowing Costs, Funding Rates, or Settlement Fees

**Location:** `costs.py:26-91`

Transaction cost model includes commissions and estimated spread. Missing:
- Short borrowing costs (can be 5-30% annualized for hard-to-borrow)
- Funding rate for leveraged positions
- Settlement fees (T+1 for equities)
- Assignment/exercise fees for options
- Exchange fees per contract

**Impact:** P&L estimates off by 10-30 bps for leveraged/short portfolios. A strategy that looks profitable before funding costs may be underwater after.

#### QS-E12: Smart Order Router Is Single-Venue

**Location:** `smart_order_router.py:96-118`

Routes to Alpaca or IBKR. No:
- Venue splitting (split large order across venues for better fill)
- Dark pool access
- Price improvement analytics
- Venue fee optimization (maker/taker)

For current scale (paper trading, small positions), this is acceptable. Before scaling to real capital >$100K, multi-venue routing becomes important.

---

## 13. Signal Quality & Statistical Rigor: "We Don't Know If The Signals Work"

### The Core Problem

The CTO audit assessed signal engine *architecture* (16 collectors, fault-tolerant, regime-adaptive). Architecture is fine. But **no one validated that these signals actually predict returns**. The system has no running IC computation, no confidence intervals, no decay modeling, and a stubbed look-ahead bias detector. This means: we're trading on signals whose predictive power is unknown.

At a real quant fund, this section alone would halt the trading operation.

### CRITICAL Findings (New)

#### QS-S1: No Signal IC Computation or Tracking

**Location:** `qc_research_tools.py:39` (stubbed), `db.py:2191-2206` (`signal_ic` table exists but empty)

The `signal_ic` database table exists. The `compute_information_coefficient()` tool is defined. But it returns `{"error": "Tool pending implementation"}`. **No signal has ever been validated against forward returns.**

This is the quant equivalent of a pharmaceutical company selling drugs without clinical trials.

**What must exist:**
- Daily IC computation: rank correlation of signal value vs. 1/5/21-day forward returns
- IC stability: standard deviation of daily IC over rolling 63-day window
- IC decay curve: IC at lag 0, 1, 5, 21 days — how fast does alpha decay?
- IC by regime: does the signal work in all regimes or only trending?
- Statistical significance: t-stat of IC, reject if t < 2.0

**Impact:** Every trade the system makes is based on unvalidated signals. Sharpe ratio in backtest could be entirely due to overfitting, look-ahead bias, or survivorship bias. We literally don't know.

**Fix:** Implement `ICTracker` module. Compute IC daily for all 22 collectors. Store in `signal_ic` table. Add gate: `if rolling_63d_IC < 0.02: disable collector from synthesis`. Alert if IC negative for >5 consecutive days.

#### QS-S2: No Signal Confidence Intervals or Uncertainty Quantification

**Location:** `synthesis.py:137-247`

Signals output point estimates: `consensus_conviction = 0.75`. No confidence bounds. This means:
- Position sizing can't use Kelly criterion (needs probability, not point estimate)
- Portfolio optimization can't properly weight uncertain signals
- No distinction between "0.75 conviction based on 5 confirming signals" vs. "0.75 conviction based on 1 noisy signal"

**What should exist:**
- Bootstrap confidence intervals on conviction: `0.75 [0.65, 0.85]`
- Bayesian posterior from collector agreement: high agreement → narrow interval
- Propagation to position sizing: `size = kelly_fraction * conviction * (1 - confidence_width)`

**Fix:** Add `uncertainty_estimate` field to `SignalBrief`. Compute from collector agreement distribution. Propagate to position sizing as a scaling factor.

#### QS-S3: Signal Decay Not Modeled — 59-Minute-Old Signal = 1-Minute-Old Signal

**Location:** `cache.py`, `engine.py:109-117`

Signal cache TTL is 60 minutes. A signal generated 59 minutes ago has identical weight to one generated 1 minute ago. In fast-moving markets, a 59-minute-old RSI signal is essentially noise.

**What should exist:**
- Exponential decay: `effective_conviction = conviction * exp(-age_minutes / half_life)`
- Per-collector half-life calibrated from IC decay curves
- Technical signals: half-life ~15 minutes (fast-moving)
- Fundamental signals: half-life ~24 hours (slow-moving)
- Macro signals: half-life ~7 days

**Impact:** The system makes trades based on stale signals with full confidence. In a reversal scenario, this means entering positions after the edge has already disappeared.

#### QS-S4: No Look-Ahead Bias Detection — Tool Stubbed

**Location:** `qc_research_tools.py:104-113`

`check_lookahead_bias()` returns `{"error": "Tool pending implementation"}`. No automated check that features at signal time don't include future data:

- Earnings data from Alpha Vantage: when is it "known"? If pulled at market close, it may include data announced at 16:05 ET but used for a signal computed at 16:00 ET
- Options flow collector: uses live delta/gamma, but IC may be computed against 5-day forward returns — the features overlap with the prediction window
- Fundamentals refreshed nightly but signals fire intraday — fundamentals include stale quarterly data that may already be priced in

**Impact:** If look-ahead bias exists, backtest IC is inflated 0.10+. Strategies that appear profitable are actually unprofitable live.

**Fix:** Add `FeatureTimestamp` metadata to every feature: `(feature_value, as_of_timestamp, known_since_timestamp)`. Validate: `known_since_timestamp < signal_time < forward_return_start`. Flag any violation.

#### QS-S5: Signal Correlation/Redundancy Not Tracked

**Location:** `engine.py:213-269`, `synthesis.py:52-93`

22 collectors run independently with fixed weights. But:
- Technical RSI and ML direction signals are often >0.7 correlated (both measure momentum)
- No correlation matrix between signals computed
- No VIF (variance inflation factor) analysis
- Weights in `_WEIGHT_PROFILES` are static, not adjusted for actual independence

**Impact:** System claims 22 independent signal sources. Effective independent signal count may be 10-12. Diversification benefit is overstated. Conviction scores are overconfident because correlated signals "agree" but carry the same information.

**Fix:** Weekly: compute pairwise signal correlation matrix. If `corr(signal_A, signal_B) > 0.7`: halve the weight of the weaker signal. Report effective signal count = eigenvalues > 0.1 of correlation matrix.

#### QS-S6: No Position Sizing From Signal Conviction

**Location:** Strategy execution layer — **missing link**

Signals produce conviction [0.05, 0.95]. But there is no evidence that position sizing scales with conviction. The ATR-based sizer in `risk_gate.py` uses stop distance and equity fraction — not signal strength.

A 0.95 conviction signal and a 0.10 conviction signal could get the same position size if they have the same stop distance. This is alpha-destroying: you want to bet big on high-conviction and small on low-conviction.

**What should exist:** `position_size = base_size * min(conviction / conviction_threshold, 2.0)` or Kelly: `size = (2 * conviction - 1) / odds_ratio`.

**Fix:** Add conviction-scaling to `ATRPositionSizer`. When conviction < 0.3: reduce size by 50%. When conviction > 0.8: allow up to 1.5x base size (still capped by risk gate).

### HIGH Findings (New)

#### QS-S7: Regime Detection Is Coarse and Unstable

**Location:** `regime.py`, `synthesis.py:52-93`

HMM regime model with 3 states (trending_up, trending_down, ranging). Issues:
- Minimum 120 bars for fitting — too few for stable HMM estimation
- No secondary regimes (trending_up_low_vol vs. trending_up_high_vol)
- No regime transition zone detection — **most losses occur during regime transitions**
- During 2-3 day regime flip, signals can be wrong for 48 hours
- Weight profiles are flat within regime (no transition weighting)

**Fix:** Add transition probability output from HMM. When P(transition) > 0.3: reduce all signal weights by 50% and halve position sizes. Add vol-conditioned sub-regimes.

#### QS-S8: Conviction Scaling Is Ad-Hoc, Not Empirically Calibrated

**Location:** `synthesis.py:382-420`

Adjustments are additive and fixed:
- ADX > 25: +0.10 conviction bonus
- HMM stability > 0.8: +0.05 bonus
- Weekly-daily conflict: -0.15 penalty
- Collector failures: -0.20 penalty

These magnitudes are not empirically calibrated. Why +0.10 for ADX and not +0.05 or +0.20? The adjustments don't scale with baseline conviction (a +0.10 bonus on 0.15 base conviction is a 67% increase; on 0.85 base it's an 11% increase).

**Fix:** Replace additive adjustments with multiplicative factors calibrated from historical IC: `adjusted_conviction = base_conviction * adx_factor * stability_factor * conflict_factor`. Calibrate factors quarterly from realized signal-to-return performance.

#### QS-S9: Conflicting Signals Resolved by Weighted Average, Not by Reducing Exposure

**Location:** `synthesis.py:252-510`

When technical says "bullish" but ML says "bearish" and sentiment says "neutral," the system computes a weighted average — which lands at "slightly bullish" or "slightly bearish." It then trades on this weak signal.

**What should happen:** Signal conflict = low confidence = reduced position size or no trade. Professional systematic funds use conflict detection as a **filter**, not a blender. When signals disagree, the correct action is usually to wait for agreement.

**Fix:** Add conflict detection: `if max_signal - min_signal > 0.5: flag as "conflicting"`. When conflicting: cap conviction at 0.3 regardless of weighted average. Or: skip trade entirely and log.

---

## 14. Backtesting Validity: "The Backtests Can't Be Trusted"

### CRITICAL Findings (New)

#### QS-B1: Transaction Costs Underapplied in Backtests

**Location:** `tca_recalibration.py`, `finrl/config.py:default_transaction_cost=0.001`

The default backtest transaction cost is 10 bps (commissions only). Missing:
- Bid-ask spread: 1-5 bps for liquid large-cap, 10-50 bps for small-cap/illiquid
- Market impact: 2-20 bps depending on order size vs. ADV
- Opportunity cost: unfilled limit orders represent missed alpha

A strategy with 50 bps gross alpha and 10 bps modeled cost looks like 40 bps net. Realistic cost of 30-40 bps means actual net alpha is 10-20 bps — possibly not statistically different from zero.

**Impact:** Strategy selection is biased toward high-turnover strategies that appear profitable with unrealistic costs but are unprofitable in practice.

**Fix:** Use 30 bps as default all-in cost until TCA feedback loop provides realized estimates. For options: use 5% of bid-ask spread as cost floor.

#### QS-B2: No Survivorship Bias Adjustment in Backtests

**Location:** `universe.py`, data acquisition pipeline

Backtests use current universe constituents. If backtesting a 2020-2024 period, only symbols that survived to 2026 are included. Companies that went bankrupt (e.g., Bed Bath & Beyond 2023) are excluded, creating a positive bias of 2-5% annual returns.

`delisted_at` column exists in the database but there's no evidence it's used to filter the backtest universe point-in-time.

**Fix:** Add `universe_as_of(date)` function that returns only symbols that were active (not delisted, not pre-IPO) at the given date. Use in all backtests.

#### QS-B3: Monte Carlo Simulation Stubbed — Can't Quantify Overfitting Risk

**Location:** `qc_research_tools.py:52-63`

`run_monte_carlo()` returns `{"error": "pending implementation"}`. Without Monte Carlo:
- Can't compute confidence intervals on backtest Sharpe ratio
- Can't test parameter sensitivity (does strategy break with slightly different parameters?)
- Can't estimate probability of ruin under path-dependent drawdown
- A backtest Sharpe of 0.8 could have 95% CI of [0.2, 1.4] — essentially noise

**Fix:** Implement bootstrap Monte Carlo: resample daily returns with replacement, compute Sharpe distribution. Reject strategies where lower 5th percentile Sharpe < 0.3.

#### QS-B4: Walk-Forward Validation Exists But Not Enforced

**Location:** `walkforward.py`, `qc_backtesting_tools.py` (stubbed)

Walk-forward framework exists in `core/research/walkforward.py` but:
- The tool wrapper `run_walkforward()` is stubbed
- No mandatory WFV gate in the strategy validation pipeline
- Strategies can proceed from `draft` → `backtested` without OOS testing
- No OOS Sharpe ratio gate (if OOS Sharpe < 0.5 × IS Sharpe, reject as overfit)

**Fix:** Make WFV mandatory before any strategy advances past `backtested`. Gate: OOS Sharpe must be ≥ 50% of IS Sharpe. Log ratio for ongoing monitoring.

### HIGH Findings (New)

#### QS-B5: No Point-in-Time Data Semantics

**Location:** `data/manager.py`, `data/storage.py`

Features used in backtests don't have explicit "as-of-date" and "known-since-date" fields. Example: Q3 2024 earnings released 2025-01-25 but labeled as 2024-09-30. If a signal fires on 2025-01-24, it shouldn't use Q3 data — but it does because there's no PIT enforcement.

For fundamentals data, this inflates backtest IC by 0.05-0.20 (the "perfect foresight" artifact).

**Fix:** Add `(value, as_of_date, available_date)` triple to all fundamental features. Filter: `available_date < signal_timestamp`.

#### QS-B6: Feature Multicollinearity Unchecked

**Location:** `core/features/` (80+ feature files)

150+ features with no VIF analysis, no correlation matrix, no dimensionality reduction. Technical indicators (RSI, MACD, Stochastic, Williams %R) are 0.7+ correlated — they all measure momentum. Including all of them makes the model think it has 150 independent features when effective rank is ~30.

**Impact:** Massive overfitting. Model trains on 150 features, 100+ of which are redundant. Regularization partially compensates but doesn't solve the root problem.

**Fix:** Weekly feature correlation audit. Remove features with VIF > 10. Use PCA or autoencoders for dimensionality reduction before model training.

---

## 15. ML Pipeline: "The Models Are Untested in Production Conditions"

### CRITICAL Findings (New)

#### QS-M1: No Hyperparameter Optimization

**Location:** `ml/trainer.py:46-63`

Hyperparameters are hardcoded: `learning_rate=0.05, max_depth=6, n_estimators=500`. No grid search, random search, or Bayesian optimization. Default parameters are a starting point, not an optimal configuration.

**Impact:** Model leaves 10-20% of achievable performance on the table. A Bayesian hyperparameter search over 100 configurations typically improves IC by 15-30% over defaults.

**Fix:** Add `optuna` or `hyperopt` integration. Run 100-trial Bayesian optimization with purged cross-validation on each model retraining. Cache optimal parameters per symbol/horizon.

#### QS-M2: Concept Drift Detection Is Weak — Only PSI on 6 Features

**Location:** `learning/drift_detector.py:45-52, 87-110`

Drift monitoring: PSI (Population Stability Index) on 6 hardcoded features (RSI, ATR, ADX, Bollinger Band, volume ratio, regime). Missing:

- Feature-to-target correlation shift (feature used to predict returns, now doesn't)
- Label distribution change (fewer winners, more losers)
- Feature interaction drift (correlation between features shifted)
- Cross-validation performance degradation on recent data
- PSI thresholds not calibrated per feature (generic 0.10/0.25)

**Impact:** Model trained in trending market deployed in ranging market. Features still in valid range (PSI fine) but coefficients are wrong. System keeps trading on a broken model.

**Fix:** Add: (a) rolling IC per feature — alert when IC drops >50% from training period, (b) label distribution monitoring — alert when win rate drops below 40%, (c) per-feature PSI threshold calibration from historical distributions.

#### QS-M3: No Model Versioning or A/B Testing

**Location:** `ml/ml_signal.py:54-169`

Models saved as `{symbol}_latest.joblib`. "Latest" overwrites previous. No:
- Model version history
- Mapping from model version to training parameters, data window, validation metrics
- A/B testing: run old and new model in parallel, compare live performance
- Rollback capability: if new model worse, revert to previous

**Impact:** Can't answer "which model is running for AAPL right now?" or "was performance better before or after last retrain?" or "should we roll back?"

**Fix:** Save as `{symbol}_{timestamp}_{ic_value}.joblib`. Add `model_registry` table: `(symbol, version, trained_at, train_ic, oos_ic, is_active, config_hash)`. Keep last 5 versions.

### HIGH Findings (New)

#### QS-M4: Feature Importance Is Unreliable — Only MDI Implemented

**Location:** `ml/feature_importance.py:24-135`

Three importance methods mentioned (MDI, MDA, SFI) but only MDI (built-in `feature_importances_`) actually runs. MDI is known to be biased toward high-cardinality features. SHAP explainer exists in `explainer.py:82-124` but is "best-effort" — falls back to MDI on failure.

**Fix:** Implement all three methods. Use consensus: feature is "important" only if ranked top-20 by ≥2/3 methods. Discard features that are important by only one method (likely noise).

#### QS-M5: Cross-Validation Doesn't Account for Regime Changes

**Location:** `ml/trainer.py:225-244`, `core/validation/purged_cv.py`

Purged K-fold CV exists (good for preventing leakage). But:
- Fixed `test_size=0.2` regardless of holding period
- No stratification on regime label — model tested on trending data but not on ranging data in same fold
- No expanding-window CV to simulate production learning

**Impact:** Model achieves 0.04 IC in backtest CV (mix of regimes) but 0.01 IC in live (currently in ranging regime where model was weakly tested).

**Fix:** Add regime-stratified CV: ensure each fold contains proportional representation of all regime types.

---

## 16. Agent Architecture Deep Audit: "The Agents Don't Talk to Each Other"

### CRITICAL Findings (New)

#### QS-A1: Missing Agent Roles — A Real Trading Desk Has These

A 2,000-person fund has specialized roles that QuantStack lacks entirely:

| Missing Role | What It Does | Why It's Critical | Priority |
|-------------|-------------|-------------------|----------|
| **Compliance Officer** | Pre-trade regulatory validation, position limits, insider trading windows | No SEC/FINRA compliance checks. System could violate Reg SHO, position limits | CRITICAL |
| ~~**Trade Reconciliation**~~ | ~~Post-execution verification~~ | **RETRACTED**: `guardrails/agent_hardening.py:463-550` already implements `reconcile_blackboard_with_portfolio()` with phantom/unknown position detection. `execution_monitor.py` runs `_reconcile_loop()` | ~~CRITICAL~~ N/A |
| **Corporate Actions** | Monitor dividends, splits, mergers, spinoffs on holdings | Ex-date causes basis error. Merger announcement = thesis change. Not monitored | HIGH |
| **Factor Exposure Monitor** | Track portfolio beta, sector tilts, style exposure continuously | Tool exists but no agent manages it. Portfolio could drift to 100% tech without awareness | HIGH |
| **Performance Attribution** | Decompose P&L by factor, timing, selection, cost in real-time | Only runs nightly in supervisor. Trading agents make decisions without attribution context | HIGH |
| **Counterparty Risk** | Monitor broker health, clearing risk, concentration limits | Single broker dependency with no health monitoring | MEDIUM |
| **Market Microstructure** | Analyze bid-ask dynamics, trade flow patterns, volume clocks | No understanding of how orders interact with market. Fills modeled as instant | MEDIUM |

**Fix (Priority):** Implement Compliance Officer and Trade Reconciliation agents first. Compliance = pre-trade regulatory check node in trading graph. Reconciliation = daily job comparing `positions` table vs. broker API response.

#### QS-A2: Race Condition — Parallel Branches Can Conflict on Same Symbol

**Location:** `trading/graph.py:177-182`

Position review and entry scan run in parallel:
```
plan_day → position_review → execute_exits → merge_parallel
        → entry_scan → earnings_router → merge_parallel
```

If `position_review` decides to exit XYZ and `entry_scan` simultaneously decides to enter XYZ, the merged state has conflicting orders. No transactional guarantee prevents this.

**Impact:** Buy and sell orders for same symbol submitted in same cycle. At best: wasted commission. At worst: unintended position change.

**Fix:** Add conflict resolution at `merge_parallel`: if same symbol appears in both exits and entries, exits take priority (risk-off bias). Or: sequential execution with exits completing before entries.

#### QS-A3: State Schema Has No Validation — Typos Silently Break Pipeline

**Location:** `graphs/state.py:54-91`

Node returns are merged into `TradingState` via dict update. If a node returns `{"daly_plan": "..."}` (typo) instead of `{"daily_plan": "..."}`, the typo key is silently added and `daily_plan` remains stale from the prior cycle.

No schema validation, no required-field checking, no type checking at state boundaries.

**Impact:** One LLM hallucination or parsing error in node output silently corrupts the entire pipeline for the rest of the cycle.

**Fix:** Add Pydantic validation at each state merge point. Each node declares its output schema. `merge_parallel` validates against schema before accepting.

#### QS-A4: Errors Accumulate But Never Block Execution

**Location:** `trading/nodes.py:99-103, 144-148, 185-189, 277-281`

Every node has the same pattern: catch exception → append to `errors` list → continue execution. Even critical nodes like `data_refresh` (line 185-189) and `daily_planner` (line 277-281) — if they crash, the graph continues with stale/empty data.

The `errors` list accumulates but no downstream node checks it. There is no circuit breaker pattern. A graph cycle can have 5 errors from 5 different nodes and still reach `execute_entries`.

**Impact:** Trades executed based on corrupted or missing data because upstream failures don't propagate as blocking conditions.

**Fix:** Add error-count check before `execute_entries`: `if len(errors) > 2: skip execution this cycle, log, alert`. Critical nodes (data_refresh, safety_check, risk_sizing) should be blocking: if they fail, halt the pipeline.

### HIGH Findings (New)

#### QS-A5: No Circuit Breaker for Repeatedly Failing Nodes

If `daily_planner` fails 5 consecutive cycles, the graph still calls it on cycle 6. No exponential backoff, no circuit breaker, no fallback to a simpler plan.

**Fix:** Track per-node failure count. After 3 consecutive failures: skip node and use safe defaults. After 5: alert and halt graph. Reset counter on success.

#### QS-A6: Message Pruning Loses Critical Context

**Location:** `agent_executor.py:111-143`

When conversation exceeds 150K chars, oldest tool rounds are dropped. By the time `fund_manager` (the 10th agent) runs, it may have lost `position_review` results from the beginning of the cycle — the very data it needs to make allocation decisions.

No summary of pruned content. No priority-based retention (keep risk data, drop verbose market intel). Just FIFO deletion.

**Fix:** Implement priority-based pruning: tag messages with `priority=critical|normal|verbose`. Prune verbose first. Or: add compaction step (haiku-tier summarization) at merge points instead of raw message accumulation.

#### QS-A7: Event Bus Cursor Updates Are Not Atomic

**Location:** `coordination/event_bus.py:228-251`

After polling events, the cursor is updated via DELETE + INSERT. If the process crashes between DELETE and INSERT, the cursor is lost. Next poll re-reads all events since the beginning, causing duplicate processing.

For `STRATEGY_PROMOTED` events, this means a strategy could be promoted twice. For `IC_DECAY` events, a harmless duplicate. For `KILL_SWITCH_TRIGGERED`, a harmless duplicate. But the non-atomicity is a bug class.

**Fix:** Use single `INSERT ... ON CONFLICT DO UPDATE` (upsert) for cursor management.

#### QS-A8: No Dead Letter Queue for Invalid Agent Outputs

When `parse_json_response()` fails on agent output, the output is silently replaced with `{}` or `[]`. No record of what the agent actually said, why it failed to parse, or how often this happens per agent.

**Fix:** Add `agent_dead_letters` table: `(agent_name, cycle_id, raw_output, parse_error, timestamp)`. Monitor frequency per agent — high DLQ rate = prompt quality issue.

#### QS-A9: Tool Access Control Is Absent — Any Agent Can Call Any Tool

**Location:** `tool_binding.py:49-79`

All tools in an agent's YAML config are bound. But there's no **negative** access control. If `hypothesis_critic` config accidentally includes `execute_order`, it can submit trades. Research agents have no explicit block on execution tools. Trading agents have no block on `register_strategy`.

**Fix:** Add `blocked_tools` per graph (not per agent): Research graph → blocks all execution tools. Trading graph → blocks strategy registration tools. Supervisor → read-only execution tools.

---

## 17. Feedback Loops & Learning: "The System Doesn't Learn From Its Mistakes"

### The Core Problem

This is the single biggest gap the CTO audit missed entirely. QuantStack has **zero closed feedback loops**. Losses don't drive research priorities. Realized execution costs don't calibrate the cost model. Signal IC degradation doesn't adjust signal weights. Strategy underperformance doesn't trigger strategy demotion.

A real fund has hundreds of people whose job is closing these loops. QuantStack has zero loops closed.

### CRITICAL: The Five Missing Loops

#### Loop 1: Trade Outcome → Research Priority (INCOMPLETE — not broken)

**Correction:** Verification found `hooks/trade_hooks.py:118-144` implements a real feedback path:
```
on_trade_close() fires →
  IF loss > 1%: INSERT research_queue (task_type='bug_fix', priority=5-7)
    with context: symbol, strategy_id, regime_at_entry/exit, conviction, debate_verdict
  ALSO: feeds PromptTuner (outcome records), ReflexionMemory (episodic), CreditAssigner (step-level)
```

**What exists:** One-directional feedback from losses → research queue. The research graph picks up `bug_fix` tasks and investigates. This is a real closed loop for catastrophic losses (>1%).

**What's still missing:**
- No failure mode taxonomy (all losses are "bug_fix", not classified as regime_mismatch vs. factor_crowding vs. data_stale)
- No aggregation of failure modes over time (can't say "3 regime mismatches this week")
- No auto-adjustment of signal weights based on outcomes
- Threshold is binary (>1% loss) — no graduated response for smaller but systematic losses

**Fix:** Extend `trade_hooks.py` with failure mode classification. Add aggregation query in supervisor. The hard part (research_queue insertion) is already done.

#### Loop 2: Realized Execution Cost → Cost Model Calibration (BROKEN)

**Current state:**
```
tca_engine forecasts 15 bps slippage
  → trade executes with 35 bps actual slippage
  → tca_storage records the 35 bps
  → tca_recalibration needs 50 trades to fit (months away)
  → next trade still forecasts 15 bps
```

**What should exist:**
```
tca_engine forecasts 15 bps
  → trade executes with 35 bps
  → ewma update: forecast = 0.9 * 15 + 0.1 * 35 = 17 bps
  → next trade uses 17 bps estimate
  → positions sized more conservatively when cost is high
```

**Fix:** Implement EWMA recalibration: after every fill, update cost model parameters. Don't wait for 50 trades — use Bayesian updating with informative prior.

#### Loop 3: Signal IC Degradation → Weight Adjustment (BROKEN)

**Current state:**
```
technical collector IC drops from 0.05 to 0.01 over 2 months
  → nobody notices (IC not tracked)
  → technical still gets 25% weight in synthesis
  → conviction inflated by stale signal
```

**What should exist:**
```
IC_tracker detects technical IC dropped below 0.02 for 21 days
  → synthesis reduces technical weight by 50%
  → conviction drops, position sizes shrink
  → OR: drift detector queues "investigate technical signal degradation"
  → research identifies: regime shifted to ranging, technical signals weak
  → synthesis switches to ranging weight profile automatically
```

**Fix:** Daily IC per collector. Rolling 21-day average. If IC < 0.02 for any collector: halve its weight in synthesis. Publish `SIGNAL_DEGRADATION` event to EventBus. Research graph picks up investigation task.

#### Loop 4: Strategy Performance → Strategy Demotion (PARTIALLY BROKEN)

**Current state:**
```
strategy_promoter checks: forward_testing → live promotion gate
strategy_breaker checks: 5% drawdown OR 3 consecutive losses → circuit break
```

**What's missing:**
```
Strategy live for 30 days, Sharpe = 0.2 (backtest was 1.5)
  → no automatic demotion. Strategy keeps trading until drawdown hits 5%
  → by then, already lost 3-4% of allocation
```

**What should exist:**
```
Live Sharpe < 50% of backtest Sharpe for 21 trading days
  → automatic demotion to forward_testing
  → reduce position size by 75%
  → queue research task: "investigate strategy degradation for {strategy_id}"
```

**Fix:** Add live-vs-backtest Sharpe comparison to `strategy_promoter`. Threshold: live Sharpe < 50% of backtest for 21+ days → auto-demote. This catches slow bleed strategies that don't trigger the hard circuit breaker.

#### Loop 5: Agent Decision Quality → Prompt Improvement (BROKEN)

**Current state:**
```
trade_debater recommends ENTER on AAPL
  → fund_manager approves
  → position loses 3%
  → trade_reflector logs "thesis was wrong"
  → debater's next call uses the exact same prompt
  → no learning
```

**What should exist:**
```
trade_debater recommendations tracked: 60% win rate
  → meta_prompt_optimizer analyzes losses
  → identifies: debater overweights momentum, underweights vol regime
  → generates prompt variant with stronger vol awareness
  → A/B test: variant vs. original for 2 weeks
  → winner becomes new default prompt
```

**Fix:** See AR-2 (Meta Layer). Near-term: track per-agent decision quality (recommendation → outcome). Alert when win rate drops below baseline. Manual prompt improvement until meta layer is built.

---

## 18. Infrastructure & Security Deep Audit

### CRITICAL Findings (New)

#### QS-I1: Containers Run as Root

**Location:** `Dockerfile` — no `USER` directive

All containers (trading, research, supervisor, finrl-worker) run as root inside the container. If any container is compromised (prompt injection → code execution → shell access), the attacker has root privileges.

**Fix:** Add `RUN useradd -r quantstack && chown -R quantstack:quantstack /app` and `USER quantstack` to Dockerfile.

#### QS-I2: PostgreSQL Exposed with Default Passwords

**Location:** `docker-compose.yml:36, 40`

```yaml
ports:
  - "5434:5432"    # Exposed to localhost (and possibly network)
environment:
  POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-quantstack}  # Default: "quantstack"
```

If `.env` is not set, the database uses password "quantstack" and is accessible on port 5434. All system state — positions, strategies, fills, API keys in some tables — is accessible to anyone on the local network.

**Fix:** Bind to localhost only: `127.0.0.1:5434:5432`. Remove default passwords. Require `.env` to exist with non-default values.

#### QS-I3: No Database Transaction Isolation for Position Updates

**Location:** `db.py` — default `READ COMMITTED` isolation

When two agents simultaneously read and update a position (e.g., execution monitor tightening stop while trading graph sizing a new entry on same symbol), the default READ COMMITTED isolation allows both to read the same stale state. One update overwrites the other.

**Fix:** Use `SELECT FOR UPDATE` on position rows during modification. Or: set isolation to `SERIALIZABLE` for the position update connection pool.

#### QS-I4: No Migration Versioning — Can't Track What Ran

**Location:** `db.py:517-553`

30+ migration functions called sequentially. No `schema_version` table. Can't answer: "Which migrations have run? Did migration #17 fail last time? Is the schema consistent?"

If a migration fails mid-way, the next startup re-runs all migrations. Idempotent `CREATE TABLE IF NOT EXISTS` helps, but `ALTER TABLE` migrations may fail or duplicate.

**Fix:** Add `schema_migrations` table: `(version, name, applied_at)`. Check before running each migration. Skip already-applied migrations.

### HIGH Findings (New)

#### QS-I5: Audit Log Exists But Not DB-Enforced (Downgraded from CRITICAL to MEDIUM)

**Verification found:** `audit/decision_log.py` implements an append-only log with SHA256 context hashes for data integrity. Every IC analysis, pod synthesis, and execution decision is logged. This is better than "no audit log."

**Remaining gap:** Immutability is enforced by code convention ("No deletes" comment), not by database constraints. A direct `DELETE FROM decision_events` SQL would succeed. No triggers, no `REVOKE DELETE`, no write-once table.

**Fix:** Add `REVOKE DELETE, UPDATE ON decision_events FROM quantstack;` at the DB level. Or use a PostgreSQL trigger that blocks DELETE/UPDATE operations.

#### ~~QS-I6: No Position Reconciliation Job~~ — RETRACTED

**Verification found:** `guardrails/agent_hardening.py:463-550` implements `reconcile_blackboard_with_portfolio()` that detects phantom positions, unknown positions, and quantity mismatches (>5% divergence). `execution_monitor.py` runs a `_reconcile_loop()` as a background task. **Reconciliation exists as a guardrail function, not a scheduled job, but the capability is real.**

#### ~~QS-I7: Scheduler Job Overlap Not Detected~~ — RETRACTED

**Verification found:** `strategy_lifecycle.py:422-441` implements a heartbeat-based guard: checks for in-progress runs within 9 minutes and gracefully skips with `report.skipped = True`. **Overlap prevention exists in the lifecycle module, not in scheduler.py directly, which is why the initial search missed it.**

#### QS-I8: No SBOM or Dependency Vulnerability Scanning

No Software Bill of Materials generated. No automated scanning for CVEs in Python dependencies. A vulnerability in `numpy`, `pandas`, or `httpx` could be exploited without awareness.

**Fix:** Add `pip audit` to CI pipeline. Generate SBOM with `cyclonedx-py`. Run weekly vulnerability scan.

---

## 19. Revised Priority Recommendations (CTO + Quant Scientist Combined)

### P0 — EXISTENTIAL RISK: Must Fix Before Any Real Capital (Week 1-2)

| # | Finding | Source | Fix |
|---|---------|--------|-----|
| 1 | **Mandatory stop-loss** | CTO C1 | Reject `OrderRequest` if `stop_price is None` |
| 2 | **Implement IC computation & tracking** | QS-S1 | Daily IC per collector. Gate: IC < 0.02 → disable collector |
| 3 | **Prompt injection defense** | CTO LC1 | Structured templates, sanitize all interpolated data |
| 4 | **Automated DB backups** | CTO OC1 | Daily pg_dump → S3. WAL archiving. Monthly restore test |
| 5 | **Output schema validation** | CTO LC2 | Pydantic models per agent output. Retry on parse failure |
| 6 | **Fix PostgreSQL security** | QS-I2 | Bind localhost only, remove default passwords, enforce .env |
| 7 | **Run containers as non-root** | QS-I1 | Add USER directive to Dockerfile |
| 8 | **Add margin tracking** | QS-E2 | Compute margin utilization. Block trades if >80% utilized |
| 9 | **Options Greeks risk limits** | QS-E3 | Portfolio delta/gamma/vega limits in risk gate |
| 10 | **Transaction cost reality** | QS-B1 | Default 30 bps all-in until TCA calibrated |

### P1 — STATISTICAL VALIDITY: Must Fix Before Trusting Any Backtest (Week 2-4)

| # | Finding | Source | Fix |
|---|---------|--------|-----|
| 11 | **Look-ahead bias detection** | QS-S4 | Feature timestamp validation. known_since < signal_time |
| 12 | **Survivorship bias adjustment** | QS-B2 | `universe_as_of(date)` function. Use in all backtests |
| 13 | **Walk-forward mandatory gate** | QS-B4 | OOS Sharpe ≥ 50% of IS Sharpe to advance strategy |
| 14 | **Signal confidence intervals** | QS-S2 | Bootstrap intervals. Propagate to position sizing |
| 15 | **Signal decay modeling** | QS-S3 | Per-collector half-life. Exponential decay on cached signals |
| 16 | **Feature multicollinearity audit** | QS-B6 | VIF analysis. Remove features with VIF > 10 |
| 17 | **Hyperparameter optimization** | QS-M1 | Optuna/hyperopt with purged CV |
| 18 | **Conviction → position sizing** | QS-S6 | Scale position size with signal conviction |

### P2 — OPERATIONAL SAFETY: Must Fix for 24/7 Unattended (Week 4-6)

| # | Finding | Source | Fix |
|---|---------|--------|-----|
| 19 | **Compliance Officer agent** | QS-A1 | Pre-trade regulatory validation node |
| 20 | **Trade Reconciliation agent** | QS-A1 | Daily broker-vs-system position check |
| 21 | **Intraday circuit breaker** | QS-E5 | Unrealized P&L triggers at -1.5%, -2.5%, -5% |
| 22 | **Errors block execution** | QS-A4 | Error count > 2 → skip execute_entries |
| 23 | **Race condition fix** | QS-A2 | Exits complete before entries (sequential, not parallel) |
| 24 | **State schema validation** | QS-A3 | Pydantic at state merge points |
| 25 | **Position reconciliation job** | QS-I6 | Daily broker check, alert on mismatch |
| 26 | **Migration versioning** | QS-I4 | schema_migrations table |
| 27 | **Audit log immutability** | QS-I5 | Hash-chained append-only table + S3 export |
| 28 | **TCA feedback loop** | QS-E6 | EWMA cost model update after every fill |

### P3 — LEARNING LOOPS: What Makes a System into a Company (Week 6-10)

| # | Finding | Source | Fix |
|---|---------|--------|-----|
| 29 | **Loss → research priority loop** | Loop 1 | loss_analyzer in supervisor, failure mode taxonomy |
| 30 | **IC degradation → weight adjustment** | Loop 3 | Daily IC per collector, auto-weight reduction |
| 31 | **Live vs. backtest Sharpe demotion** | Loop 4 | Auto-demote if live < 50% backtest for 21d |
| 32 | **Agent decision quality tracking** | Loop 5 | Per-agent recommendation → outcome tracking |
| 33 | **Real execution algo implementation** | QS-E1 | TWAP/VWAP child order generation |
| 34 | **Signal correlation tracking** | QS-S5 | Weekly correlation matrix, weight adjustment |
| 35 | **Concept drift: IC + label + interaction** | QS-M2 | Beyond PSI: rolling IC, win rate, feature correlation |
| 36 | **Model versioning + A/B** | QS-M3 | Model registry table, keep last 5 versions |
| 37 | **Regime transition detection** | QS-S7 | HMM transition probability, reduce exposure during flips |
| 38 | **Monte Carlo validation** | QS-B3 | Bootstrap Sharpe CI, reject if 5th percentile < 0.3 |

---

## 20. What The Previous CTO Got Right (Credit Where Due)

The CTO audit was not bad. It correctly identified:

1. **Stop-loss optional** — the single most dangerous bug
2. **Prompt caching economics** — the RAG vs. file-based analysis was excellent and saved the company from a costly mistake
3. **92 stubbed tools** — massive issue correctly flagged
4. **Prompt injection** — serious security vulnerability
5. **No backups** — existential infrastructure risk
6. **EventBus half-connected** — supervisor shouting into void
7. **MemorySaver crashes** — durable checkpoints needed
8. **OpenClaw benchmarking** — smart to learn from similar systems
9. **Advanced research integration (AR-1 through AR-10)** — genuinely visionary roadmap

### What The CTO Missed

1. **Statistical validity of signals** — the most important question ("do our signals predict returns?") was never asked
2. **Execution realism** — assessed architecture, not whether the execution layer actually works
3. **Feedback loops** — identified individual components but missed that none are connected into learning cycles
4. **Missing roles** — counted agents (21) but didn't compare to what a real desk needs
5. **Backtesting validity** — assessed the code quality of backtests, not whether the backtests can be trusted
6. **ML pipeline substance** — noted tools are stubbed, didn't assess whether the implemented ML is statistically sound
7. **Security fundamentals** — caught prompt injection but missed root containers, exposed ports, default passwords
8. **Race conditions** — described parallel execution as a feature, didn't identify the state conflicts it creates
9. **Error propagation** — noted individual error handling issues, missed that errors never block execution

### The Fundamental Difference

The CTO audit asked: **"Is this well-engineered?"** Answer: mostly yes.

This audit asks: **"Can this make money?"** Answer: we don't know, and we can't know until the statistical validation gaps are closed.

Engineering quality without statistical validity is a beautiful machine that might be pointing in the wrong direction. Fix the statistics first. Then fix the engineering. Then close the loops.

---

## Appendix: Complete Finding Cross-Reference (All 169 Findings)

### New Findings from Quant Scientist Audit

| ID | Finding | Subsystem | Severity | Location |
|----|---------|-----------|----------|----------|
| QS-E1 | No real algo execution (TWAP/VWAP phantom) | Execution | CRITICAL | `order_lifecycle.py:455-478` |
| ~~QS-E2~~ | ~~Zero margin/leverage tracking~~ | ~~Execution~~ | **RETRACTED** | SPAN margin exists in `core/risk/span_margin.py` (537 lines) |
| QS-E3 | Options Greeks risk absent from gate | Execution | CRITICAL | `risk_gate.py:519-589` |
| QS-E4 | Liquidity risk severely underestimated | Execution | CRITICAL | `risk_gate.py:453-465` |
| QS-E5 | No intraday drawdown circuit breaker | Execution | CRITICAL | `risk_gate.py:433-451` |
| QS-E6 | TCA feedback loop missing | Execution | CRITICAL | `tca_engine.py` |
| QS-E7 | Best execution compliance absent | Execution | CRITICAL | Missing entirely |
| QS-E8 | Slippage model regime-agnostic | Execution | HIGH | `paper_broker.py:243-276` |
| QS-E9 | Partial fills overwrite price history | Execution | HIGH | `order_lifecycle.py:266-292` |
| QS-E10 | Correlation check is stub, alert-only | Execution | HIGH | `risk_gate.py:781-825` |
| QS-E11 | No borrowing/funding costs | Execution | HIGH | `costs.py:26-91` |
| QS-E12 | Smart order router single-venue | Execution | HIGH | `smart_order_router.py:96-118` |
| QS-S1 | No signal IC computation | Signal | CRITICAL | `qc_research_tools.py:39` |
| QS-S2 | No signal confidence intervals | Signal | CRITICAL | `synthesis.py:137-247` |
| QS-S3 | Signal decay not modeled | Signal | CRITICAL | `cache.py`, `engine.py` |
| QS-S4 | No look-ahead bias detection | Signal | CRITICAL | `qc_research_tools.py:104-113` |
| QS-S5 | Signal correlation not tracked | Signal | CRITICAL | `engine.py:213-269` |
| QS-S6 | No conviction → position sizing | Signal | CRITICAL | Strategy execution (missing) |
| QS-S7 | Regime detection coarse/unstable | Signal | HIGH | `regime.py`, `synthesis.py` |
| QS-S8 | Conviction scaling ad-hoc | Signal | HIGH | `synthesis.py:382-420` |
| QS-S9 | Conflicting signals blended, not filtered | Signal | HIGH | `synthesis.py:252-510` |
| QS-B1 | Transaction costs underapplied | Backtest | CRITICAL | `tca_recalibration.py` |
| QS-B2 | No survivorship bias adjustment | Backtest | CRITICAL | `universe.py` |
| QS-B3 | Monte Carlo simulation stubbed | Backtest | CRITICAL | `qc_research_tools.py:52-63` |
| QS-B4 | Walk-forward not enforced | Backtest | CRITICAL | `walkforward.py` |
| QS-B5 | No point-in-time data semantics | Backtest | HIGH | `data/manager.py` |
| QS-B6 | Feature multicollinearity unchecked | Backtest | HIGH | `core/features/` |
| QS-M1 | No hyperparameter optimization | ML | CRITICAL | `ml/trainer.py:46-63` |
| QS-M2 | Concept drift detection weak | ML | CRITICAL | `drift_detector.py:45-52` |
| QS-M3 | No model versioning or A/B | ML | CRITICAL | `ml/ml_signal.py:54-169` |
| QS-M4 | Feature importance unreliable | ML | HIGH | `feature_importance.py:24-135` |
| QS-M5 | CV doesn't account for regimes | ML | HIGH | `ml/trainer.py:225-244` |
| QS-A1 | 5 missing agent roles (was 7, reconciliation retracted) | Agents | CRITICAL | Architecture (absent) |
| QS-A2 | Race condition: parallel branches | Agents | CRITICAL | `trading/graph.py:177-182` |
| QS-A3 | State schema no validation | Agents | CRITICAL | `graphs/state.py:54-91` |
| QS-A4 | Errors don't block execution | Agents | CRITICAL | `trading/nodes.py:99-281` |
| QS-A5 | No circuit breaker for failing nodes | Agents | HIGH | All graph runners |
| QS-A6 | Message pruning loses context | Agents | HIGH | `agent_executor.py:111-143` |
| QS-A7 | Event bus cursor not atomic | Agents | HIGH | `event_bus.py:228-251` |
| QS-A8 | No dead letter queue | Agents | HIGH | `agent_executor.py:474-521` |
| QS-A9 | No tool access control | Agents | HIGH | `tool_binding.py:49-79` |
| QS-I1 | Containers run as root | Infra | CRITICAL | `Dockerfile` |
| QS-I2 | PostgreSQL exposed, default passwords | Infra | CRITICAL | `docker-compose.yml:36-40` |
| QS-I3 | No transaction isolation for positions | Infra | CRITICAL | `db.py` |
| QS-I4 | No migration versioning | Infra | HIGH | `db.py:517-553` |
| QS-I5 | Audit log exists but no DB enforcement | Infra | **MEDIUM** (downgraded) | `audit/decision_log.py` — SHA256 hashes, but no REVOKE DELETE |
| ~~QS-I6~~ | ~~No position reconciliation job~~ | ~~Infra~~ | **RETRACTED** | `guardrails/agent_hardening.py:463-550` + `execution_monitor._reconcile_loop()` |
| ~~QS-I7~~ | ~~Scheduler job overlap not detected~~ | ~~Infra~~ | **RETRACTED** | `strategy_lifecycle.py:422-441` heartbeat guard |
| QS-I8 | No SBOM or vuln scanning | Infra | MEDIUM | CI pipeline (absent) |
| Loop-1 | Trade outcome → research priority (incomplete, not broken) | Learning | **HIGH** (downgraded) | `hooks/trade_hooks.py:118-144` — exists but no failure taxonomy |
| Loop-2 | Realized cost → cost model (broken) | Learning | CRITICAL | No closed loop |
| Loop-3 | IC degradation → weight adjustment (broken) | Learning | CRITICAL | No closed loop |
| Loop-4 | Live performance → strategy demotion (partial) | Learning | HIGH | `strategy_promoter` |
| Loop-5 | Agent quality → prompt improvement (broken) | Learning | HIGH | No closed loop |

---

## Final Verdict

**The previous CTO audit graded this system B-. That was generous.**

When you include statistical validity, execution realism, feedback loops, missing roles, security fundamentals, and backtesting integrity, **this system is a C- at best**.

The architecture is genuinely impressive — 21 agents, 3 graphs, event-driven coordination, self-healing, multi-layer risk gate. But architecture without substance is a demo, not a fund.

**The path from C- to A:**
1. **Week 1-2:** Safety hardening (P0) — stop the bleeding
2. **Week 2-4:** Statistical validation (P1) — know if signals work
3. **Week 4-6:** Operational safety (P2) — survive 24/7
4. **Week 6-10:** Learning loops (P3) — compound intelligence
5. **Week 10+:** Advanced research integration (AR-1 through AR-10) — become a company

The CTO's AR-1 through AR-10 roadmap is the right destination. This audit provides the foundation that must exist before that roadmap is meaningful. You can't build a self-improving research machine on top of unvalidated signals and phantom execution.

**Fix the statistics. Fix the execution. Close the loops. Then scale.**

---

# PART III: DEEP OPERATIONAL AUDIT — THE HARD QUESTIONS

**Auditor:** CTO / Principal Quant Scientist / Staff Agentic AI Scientist
**Date:** 2026-04-06
**Scope:** Code-level investigation of the questions a CTO asks before signing off on real capital. Not architecture. Not recommendations. What the Python actually executes when things go wrong.

---

## DO-1. The Learning Modules Are Fully Built But Have Zero Callers

Part II identified that feedback loops are broken. This section proves it at the code level: **5 fully-implemented learning modules exist with zero consumers.**

### The Ghost Component Registry

| Component | File | Lines | Implemented | `write()` Called | `read()` Called | Verdict |
|-----------|------|-------|-------------|-----------------|----------------|---------|
| `OutcomeTracker` | `learning/outcome_tracker.py` | ~200 | Yes — Bayesian regime_affinity | Yes — on fill hook | **NO** — `get_regime_strategies()` returns stub | **SINK** |
| `SkillTracker` | `learning/skill_tracker.py` | ~250 | Yes — per-agent IC, confidence adj (0.5-1.5x) | **NEVER** — 0 callers for `update_agent_skill()` | **NEVER** — 0 callers for `get_confidence_adjustment()` | **GHOST** |
| `ICAttribution` | `learning/ic_attribution.py` | ~200 | Yes — rolling Spearman IC per collector | **NEVER** — 0 callers for `record()` | **NEVER** — 0 callers for `get_weights()` | **GHOST** |
| `ExpectancyEngine` | `learning/expectancy_engine.py` | ~200 | Yes — Kelly, win rate, quality scores | **NEVER** — 0 callers in entire repo | **NEVER** — sizing uses `core/kelly_sizing.py` instead | **ORPHAN** |
| `StrategyBreaker` | `execution/strategy_breaker.py` | ~200 | Yes — ACTIVE(1.0x)→SCALED(0.5x)→TRIPPED(0.0x) | **NEVER** — `record_trade()` never invoked | **NEVER** — `get_scale_factor()` never checked | **GHOST** |
| `TradeEvaluator` | `performance/trade_evaluator.py` | ~150 | Yes — LLM-as-judge, 6-dimension scoring | Yes — from reflection node | **NEVER** — scores written to DB, nobody reads | **SINK** |

### Step-by-Step Trace: What Happens When a Trade Loses -2%

```
1. Trade closes (execution_monitor detects price crossed stop_price)
   → on_trade_close() hook fires
   → Writes to reflection_journal .................. LOGGING ONLY
   → Inserts into research_queue if loss > 1% ....... QUEUED, but not prioritized by failure mode
   → Credit assignment computes "worst step" ........ NEVER READ by any policy

2. Fill hook fires → OutcomeTracker.apply_learning()
   → regime_affinity updated: 0.50 → 0.49 ......... WRITTEN to DB
   → get_regime_strategies() tool would read it ..... STUBBED: returns "Tool pending implementation"
   → SkillTracker.update_agent_skill() .............. NOT CALLED
   → ICAttribution.record() ......................... NOT CALLED

3. Reflection node runs (30 min later, end of cycle)
   → TradeEvaluator scores on 6 dimensions ......... WRITTEN to trade_quality_scores table
   → Nobody queries trade_quality_scores ............ DEAD DATA

4. Next morning, daily_planner runs
   → Calls search_knowledge_base() .................. CAN find past loss IF written to KB
   → Does NOT call get_regime_strategies() .......... Would get stub anyway
   → Does NOT check StrategyBreaker.get_scale_factor() .. Breaker state never updated
   → Position sized via Kelly ........................ AS IF LOSS NEVER HAPPENED
```

**Time from loss to system adaptation: NEVER. The loss is recorded but the system behavior is unchanged.**

### The 6 Readpoints That Must Be Wired (2-3 days total)

| # | Missing Wire | From → To | Fix |
|---|-------------|-----------|-----|
| 1 | Implement `get_regime_strategies()` | `meta_tools.py` → reads `strategies.regime_affinity` | Replace stub with DB query, return affinity-weighted allocations |
| 2 | `risk_sizing` checks regime_affinity | `trading/nodes.py` → `outcome_tracker.get_affinity()` | Multiply Kelly fraction by affinity before sizing |
| 3 | `execute_entries` checks strategy_breaker | `trading/nodes.py` → `strategy_breaker.get_scale_factor()` | Reduce/block orders for SCALED/TRIPPED strategies |
| 4 | Trade hooks populate SkillTracker | `trade_hooks.py` → `skill_tracker.update_agent_skill()` | Call on every trade close with agent_name + outcome |
| 5 | Signal engine records IC attribution | `signal_engine/engine.py` → `ic_attribution.record()` | After synthesis, record each collector's contribution vs. forward return |
| 6 | `daily_planner` queries quality scores | `trading/nodes.py` → `SELECT * FROM trade_quality_scores` | Surface patterns to planner: "exit_evaluator gives HOLD on positions that then lose >3%" |

---

## DO-2. Position Protection: Stop-Loss Optional, Brackets Silently Degrade

### The Entry Code Path (`trade_service.py:212-223`)

```python
# ALL FOUR conditions must be true for bracket order:
if (stop_price is not None                              # ← default is None
    and target_price is not None                         # ← default is None
    and getattr(broker, "supports_bracket_orders", ...)()
    and hasattr(broker, "execute_bracket")):
    fill = broker.execute_bracket(order, stop_price, target_price)
else:
    fill = broker.execute(order)  # ← plain market order, ZERO protection
```

If ANY condition fails → silent fallback to plain order.

### The Bracket Fallback (`alpaca_broker.py:execute_bracket()`)

```python
except Exception as e:
    logger.warning(f"Bracket submission failed ({e}), falling back to plain order")
    return self.execute(req)  # ← ANY Alpaca hiccup drops protection silently
```

Rate limit, timeout, malformed request → plain order. Agent thinks stops are set. They're not.

### The "Safety Net" (`execution_monitor.py`)

The ExecutionMonitor CAN enforce stops on next price tick, but:
- Polls every 5-60s (not real-time OCO)
- If bracket failed silently, `stop_price` may be in DB but NO broker-side order exists
- Gap through stop → filled at gap price, not stop price
- **This is software-emulated stop with 5-60s latency, not a broker-held OCO**

### What Must Change

1. Make `stop_price` non-optional in `OrderRequest` — reject at validation
2. Verify bracket legs after submission — query broker for active child orders
3. Persist bracket leg IDs to DB (currently in-memory `Fill` only, lost on crash)
4. On bracket failure: place SL as separate contingent order, NEVER fall back to plain
5. On startup: reconcile all open positions have active SL orders at broker

---

## DO-3. Options Theta, Greeks, and Assignment: Unmonitored

### ExecutionMonitor Exit Rules (ALL 6 are equity-centric)

```
1. Kill switch active?         → EXIT
2. Hard stop-loss hit?         → EXIT (price vs stop_price — works for options premium)
3. Take profit hit?            → EXIT
4. Trailing stop hit?          → EXIT
5. Time stop expired?          → EXIT
6. Intraday flatten (15:55)?   → EXIT
```

**Zero options-specific rules.** The monitor stores `option_contract` and `underlying_symbol` but evaluates them identically to equities.

### What's Missing

| Rule | What It Prevents | Status |
|------|-----------------|--------|
| Theta acceleration: DTE < 7 AND theta/premium > 5%/day → TIGHTEN | Final-week decay eating premium | **NOT IMPLEMENTED** |
| Pin risk: DTE < 3 AND abs(underlying - strike)/strike < 1% → EXIT | Assigned at expiration on worthless-looking position | **NOT IMPLEMENTED** |
| Assignment risk: short call ITM + ex-div within 2 days → EXIT/ROLL | Early assignment, short stock position | **NOT IMPLEMENTED** |
| IV crush: post-earnings + IV dropped >30% → reassess | Debit positions lose edge after event | **NOT IMPLEMENTED** |
| Max theta loss: cumulative decay > 40% of premium paid → EXIT | Slow bleed on OTM options | **NOT IMPLEMENTED** |

The system computes Greeks on demand (`core/options/engine.py:293-364`) but doesn't *monitor* them on open positions.

---

## DO-4. Fed Events: Calendar Exists, Enforcement Doesn't

### What's Implemented (Good)

The `events` collector outputs `has_fomc_24h`, `has_macro_event`, `next_event_desc`. The `macro` collector computes yield curve, TED spread, rate momentum. Both feed into `SignalBrief`. Agents see the data.

### What's NOT Implemented (Critical)

Agent prompts say "reduce sizing 50% within 24h of FOMC" (agents.yaml:13, 241). But this is **prompt guidance, not code**. No hard rule in `risk_gate.py` checks the macro calendar.

**What's needed in risk_gate.py (mandatory, not agent-discretion):**

| Condition | Sizing Multiplier | Restriction |
|-----------|-------------------|-------------|
| FOMC within 4 hours | 0.5x | No naked options |
| CPI/NFP within 2 hours | 0.75x | — |
| VIX > 30 | 0.7x | — |
| VIX > 50 | 0.0x (paper only) | No live orders |
| Multiple events same day | Most restrictive | — |

---

## DO-5. Intel Beyond Tools: Web Search Broken, SEC Data Empty

| Intel Source | Status | Notes |
|-------------|--------|-------|
| SignalBrief (16 collectors) | **Working** | 170 fields per symbol |
| Knowledge base (pgvector) | **Working but sparse** | Depends on what was written |
| Web search (breaking news) | **BROKEN** | Returns "not configured — set SEARCH_API_KEY". Cost: $20/mo to fix (Tavily/Brave). |
| SEC filings (10-K, 10-Q, 8-K) | **EMPTY** | `sec_filings` table exists, never populated. EDGAR API is free. |
| Insider trading data | **EMPTY** | `insider_trades` table exists, never populated. SEC Form 4 via EDGAR. |
| Earnings transcripts | **NO TOOL** | No tool exists. Seeking Alpha or company IR pages. |
| Analyst ratings | **NO TOOL** | No tool exists. |
| Unusual options activity | **PARTIAL** | GEX/gamma/DEX only, no unusual volume detection |

**The market_intel agent** tries to call `web_search` for breaking news. It gets an error. It falls back to training data — which is months stale.

---

## DO-6. Real-Time Trading: Not Equipped

### Latency Budget

| Stage | Current | Real-Time Requirement |
|-------|---------|----------------------|
| Price feed | 1-min bars + 5-min quote polls + Alpaca IEX (15-min delayed) | Tick-level L1/L2, real-time |
| Signal generation | 5-min graph cycle + 30-300s LLM reasoning | <5 seconds, mostly deterministic |
| Entry latency (signal → order) | **5-60 minutes** | <5 seconds |
| Exit latency (price cross → order) | 5-60 seconds (ExecutionMonitor polling) | <1 second |
| Data freshness | 5 minutes (scheduled_refresh interval) | Sub-second |

**The Alpaca IEX problem:** Free tier quotes are 15 minutes delayed. The system trades on stale prices. By the time an order hits NBBO, the market has moved.

### What Must Change for Real-Time

1. **Event-driven architecture** — WebSocket-triggered signals, not 5-min polling
2. **Split LLM from execution** — LLM sets daily parameters (watchlist, sizing, risk). Deterministic engine executes within parameters on tick events.
3. **Real-time data feed** — Polygon ($199/mo) or Alpaca SIP for tick data
4. **Low-latency hosting** — Docker on Mac = 50-200ms to Alpaca. AWS us-east-1 = 1-5ms.

---

## DO-7. Provider Switching: Two Uncoordinated LLM Systems

### The Problem

| System | File | Providers | Used By |
|--------|------|-----------|---------|
| **A** | `llm/provider.py` + `llm/config.py` | 5 | LangGraph agents |
| **B** | `llm_config.py` | 12 | CrewAI agents, bulk tasks, opro |

Different env vars. Different tier names. Different fallback chains. No shared state.

### What Breaks on `LLM_PROVIDER=openai`

| Issue | Impact |
|-------|--------|
| 6 hardcoded model strings (tool_search_compat, nodes.py, trade_evaluator, mem0_client, hypothesis_agent, opro_loop) | Those calls still hit Anthropic/Groq regardless of setting |
| `thinking` (extended thinking) silently disabled | No error, agents just produce shallower reasoning |
| Deferred tool loading → falls back to full loading | More tokens per call, less efficient |
| Prompt caching → not available | ~20% more token spend on repeated system prompts |
| Tool binding → Path 2 (BigTool) instead of Path 1 (Anthropic native) | Different tool selection behavior, potential quality difference |

### What's Needed

1. **Unify Systems A and B** — one `get_model(tier)` for all code paths
2. **Replace all 6 hardcoded strings** with `get_chat_model()` calls
3. **Provider capability matrix** — detect: supports_thinking, supports_deferred_tools, supports_structured_output. Route based on capabilities, not class name checks.

---

## DO-8. SEC Compliance: Zero Guardrails

### What's Not Enforced

| Requirement | Statute | Risk | Status |
|------------|---------|------|--------|
| **Wash Sale** | 26 USC §1091 | Sell at loss + rebuy within 30 days → loss disallowed for tax | **NOT TRACKED** |
| **Pattern Day Trader** | FINRA 4210 | >3 day trades in 5 days on <$25k → account restricted | **NOT ENFORCED** |
| **Reg T Margin** | 12 CFR 220 | 50% initial margin equity, higher for options | **NOT CALCULATED** |
| **Tax Lot Tracking** | IRS Form 8949 | Must identify which lots sold, FIFO/LIFO/specific ID | **NOT IMPLEMENTED** |
| **Short-Term vs Long-Term** | 26 USC §1222 | <1yr hold = ordinary income rate | **NOT DISTINGUISHED** |

The DB has `insider_trades` and `sec_filings` tables — both exist in schema but are **never populated and never queried**.

### Minimum Compliance for Live Trading

```
Add to risk_gate.py:

CHECK 1: Wash Sale
  → Query trades: sold {symbol} at loss within 30 cal days?
  → If YES: flag wash_sale_risk, adjust cost basis in tax_lots table
  → Don't block trade — just track for tax reporting

CHECK 2: PDT
  → Count round-trips (buy+sell same day) in rolling 5 business days
  → If count >= 3 AND account_value < $25k → REJECT

CHECK 3: Margin
  → Long equity: require 50% cash
  → Short equity: require 150%
  → Options: max_loss as margin requirement
  → If insufficient → REJECT

CHECK 4: Tax Lots
  → On every BUY: create tax_lot(symbol, qty, price, date, lot_id)
  → On every SELL: match via FIFO, compute realized gain/loss
  → Track short-term vs long-term automatically
```

| Check | Effort | Priority |
|-------|--------|----------|
| Wash sale tracking | 2 days | **P0 for live** |
| PDT enforcement | 1 day | **P0 if account < $25k** |
| Tax lot tracking | 3 days | **P0 for live** |
| Margin calculation | 3 days | P1 |
| Form 8949 export | 2 days | P2 |

---

## DO-9. Rate Limiting: Per-Process, Not Shared

`data/fetcher.py` has a per-process rate limiter:
```python
def _wait_for_rate_limit(self):
    if self._call_count >= self.rate_limit:
        time.sleep(wait_time)  # ← BLOCKING, per-process only
```

Three Docker containers × 75 req/min each = **225 req/min to Alpha Vantage**. AV limit is 75/min. The daily quota guard helps (`system_state` table) but has race conditions across processes.

**Fix:** Shared rate limiter via PostgreSQL advisory lock or Redis token bucket. Replace `time.sleep()` with `asyncio.sleep()` to avoid blocking the event loop.

---

## DO-10. Broker Abstraction: Clean but Alpaca-Coupled

`BrokerInterface` ABC is well-designed — `UnifiedOrder`, `UnifiedPosition`, `UnifiedBalance`. AlpacaBroker, PaperBroker, ETradeBroker all implement it.

**What's tightly coupled to Alpaca:**
- Fill polling (30s, 1s interval) — IBKR uses async callbacks
- Time-in-Force enums — IBKR has different constants
- Options order format — different contract specs
- Kill switch `cancel_orders` — different API
- Data feed (Alpaca IEX) — separate from broker abstraction

**Effort to add IBKR:** 2-4 days equity, 4-6 days including options.

---

## Deep Operational Audit: Readiness Verdict

| Question | Answer | Severity |
|----------|--------|----------|
| Do we have a feedback agent? | 5 learning modules exist. Zero close the loop. Losses don't change behavior. | **CRITICAL** |
| Does the system put OCO for each trade? | No. Stop-loss optional. Brackets silently degrade. Positions enter unprotected. | **CRITICAL** |
| How does it handle theta decay? | Computes on demand. No monitoring. No auto-exit. No OPEX/pin/assignment handling. | **HIGH** |
| How does it handle Fed events? | Calendar exists. Agents told to reduce sizing. Nothing enforced in risk gate. | **HIGH** |
| How does it use intel beyond tools? | Knowledge base (sparse), web search (broken), SEC filings (empty). | **HIGH** |
| Are we equipped for real-time? | No. 5-min cycles, 15-min delayed quotes, 5-60 min entry latency. | **CRITICAL** |
| Can we switch LLM providers? | Two uncoordinated systems. 6 hardcoded models. Silent feature loss. | **MEDIUM** |
| Can we switch brokers? | Yes for equity (2-4 days). Options/data tightly coupled to Alpaca. | **LOW** |
| Do we have SEC compliance? | Zero. No wash sale, PDT, margin, tax lots, or restricted list. | **CRITICAL for live** |
| Can the system handle concurrent load? | Rate limiter per-process. 3 containers = 3x API calls. Race conditions. | **HIGH** |

### The Three Things That Must Be True Before Real Capital

1. **Every position has a broker-held stop order** — not optional, not software-emulated, not silently degraded.
2. **Losses change system behavior** — wire the 6 readpoints (2-3 days). A system that doesn't learn from losses will repeat them.
3. **SEC compliance exists** — wash sale tracking, PDT enforcement, tax lot tracking. Without these, the legal risk exceeds the market risk.
