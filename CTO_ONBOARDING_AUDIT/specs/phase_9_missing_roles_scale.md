# Phase 9: Missing Roles & Scale — Deep Plan Spec

**Timeline:** Week 6-8
**Effort:** 17-20 days
**Gate:** Corporate actions monitored. Factor exposure tracked. Alert lifecycle operational. 24/7 mode available.

---

## Context

This spec is part of the QuantStack CTO Onboarding Audit implementation plan (164 findings, overall grade C-). Phase 9 fills operational gaps: missing monitoring agents, incomplete alert lifecycle, and the 24/7 operations capability. These items are important but not existential — they improve operational completeness and scale readiness.

**Full audit reference:** [`CTO_ONBOARDING_AUDIT/`](../README.md)
**Primary audit sections:** [`06_AGENT_ARCHITECTURE.md` §5.6](../06_AGENT_ARCHITECTURE.md) (missing roles), [`01_SAFETY_HARDENING.md`](../01_SAFETY_HARDENING.md) (alerts), [`04_OPERATIONAL_RESILIENCE.md` §4.12](../04_OPERATIONAL_RESILIENCE.md) (multi-mode), [`08_COST_OPTIMIZATION.md`](../08_COST_OPTIMIZATION.md) (LLM unification)

---

## Objective

Add the missing operational roles (corporate actions, factor exposure, performance attribution), build the alert lifecycle system, enable real-time notifications for critical events, achieve multi-mode 24/7 operation, and unify the LLM provider systems.

---

## Items

### 9.1 Corporate Actions Monitor

- **Finding:** QS-A1 | **Severity:** HIGH | **Effort:** 2 days
- **Audit section:** [`06_AGENT_ARCHITECTURE.md` §5.6](../06_AGENT_ARCHITECTURE.md)
- **Problem:** No monitoring for dividends, splits, or mergers on holdings. A 2:1 stock split could double apparent position size, triggering incorrect risk limits. M&A announcements may invalidate investment thesis.
- **Fix:** Daily check for corporate actions on all holdings via AV/EDGAR. Auto-adjust cost basis on splits. Flag thesis changes on M&A. Alert on upcoming ex-dividend dates for covered call positions.
- **Key files:** New corporate actions monitor (deterministic, no LLM needed), position management
- **Acceptance criteria:**
  - [ ] Corporate actions checked daily for all holdings
  - [ ] Stock splits auto-adjust cost basis and position quantity
  - [ ] M&A announcements flag thesis review

### 9.2 Factor Exposure Monitor

- **Finding:** QS-A1 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`06_AGENT_ARCHITECTURE.md` §5.6](../06_AGENT_ARCHITECTURE.md)
- **Problem:** No tracking of portfolio beta, sector tilts, or style exposure. Tool exists but no agent calls it. Portfolio could unknowingly be 90% momentum without detection.
- **Fix:** Per-cycle factor exposure computation. Alert on drift: beta deviates >0.3 from target, single sector >40%, momentum crowding >70%.
- **Key files:** Factor exposure tools, supervisor monitoring
- **Acceptance criteria:**
  - [ ] Factor exposure computed per cycle
  - [ ] Alerts on beta drift, sector concentration, style crowding

### 9.3 Performance Attribution Per-Cycle

- **Finding:** QS-A1 | **Severity:** HIGH | **Effort:** 2 days
- **Audit section:** [`06_AGENT_ARCHITECTURE.md` §5.6](../06_AGENT_ARCHITECTURE.md)
- **Problem:** Performance attribution currently runs only in nightly supervisor batch. Can't attribute intraday P&L to factors, timing, selection, or cost in real-time.
- **Fix:** Move attribution to per-cycle computation. Decompose P&L into: factor contribution (market, sector, style), timing (entry/exit quality), selection (stock-specific alpha), and cost (execution quality vs. benchmark).
- **Key files:** Performance attribution module, trading graph state
- **Acceptance criteria:**
  - [ ] Per-cycle P&L attribution available
  - [ ] Factor vs. timing vs. selection vs. cost decomposition

### 9.4 Implement 5 Alert Lifecycle Tools

- **Finding:** CTO AC3 | **Severity:** HIGH | **Effort:** 3 days
- **Audit section:** [`01_SAFETY_HARDENING.md`](../01_SAFETY_HARDENING.md)
- **Problem:** Alert system exists but lifecycle tools are incomplete. Missing: create, acknowledge, escalate, resolve, and query alert tools for the full lifecycle.
- **Fix:** Implement full alert lifecycle:
  1. `create_alert(severity, category, message, context)` — generates alert with unique ID
  2. `acknowledge_alert(alert_id, agent_name)` — marks as being investigated
  3. `escalate_alert(alert_id, reason)` — bumps severity, notifies owner
  4. `resolve_alert(alert_id, resolution)` — closes with resolution notes
  5. `query_alerts(filters)` — search by severity, status, category, time range
- **Key files:** Alert system, tool registry
- **Acceptance criteria:**
  - [ ] All 5 alert lifecycle tools implemented and registered
  - [ ] Alerts have full lifecycle: created → acknowledged → resolved/escalated
  - [ ] Alert history queryable for post-incident review

### 9.5 Real-Time Discord for CRITICAL Events

- **Finding:** CTO AC4 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`01_SAFETY_HARDENING.md`](../01_SAFETY_HARDENING.md)
- **Problem:** No real-time notification for critical events. Owner may not see kill switch activation, position limit breach, or system halt for hours.
- **Fix:** Discord webhook for: kill switch triggered, daily loss >2%, position limit breach, system halt, 3+ consecutive graph failures.
- **Key files:** Notification system, `scripts/daily_digest.py`
- **Acceptance criteria:**
  - [ ] CRITICAL events trigger immediate Discord notification
  - [ ] Kill switch, loss limits, system halt all notify in real-time
  - [ ] Notification includes event context (positions affected, reason)

### 9.6 Reply-Back ACK Pattern (EventBus)

- **Finding:** CTO AC5 | **Severity:** HIGH | **Effort:** 2 days
- **Depends on:** Phase 1 item 1.7 (EventBus wiring)
- **Audit section:** [`01_SAFETY_HARDENING.md`](../01_SAFETY_HARDENING.md)
- **Problem:** Supervisor publishes events but never knows if trading graph received them. Fire-and-forget pattern. Critical events (IC_DECAY, KILL_SWITCH) may be missed without detection.
- **Fix:** Add ACK pattern to EventBus: publisher sets expected_ack_by timestamp; consumer sends ACK event on receipt; supervisor monitors for missing ACKs and escalates.
- **Key files:** EventBus implementation, graph runners
- **Acceptance criteria:**
  - [ ] Critical events require ACK within 1 cycle
  - [ ] Missing ACK triggers escalation
  - [ ] ACK pattern doesn't add latency to normal event flow

### 9.7 Multi-Mode Operation (24/7)

- **Finding:** CTO 24/7 Readiness | **Severity:** MEDIUM | **Effort:** 3-5 days
- **Depends on:** Phases 1-3 substantially complete
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md` §4.12](../04_OPERATIONAL_RESILIENCE.md)
- **Problem:** System has no concept of extended hours or overnight modes. Trading and research simply don't run outside market hours.
- **Fix:** Three modes:
  - **Market Hours** (9:30-16:00 Mon-Fri): Full trading + research
  - **Extended Hours** (16:00-20:00, 04:00-09:30): Position monitoring only, no new entries, earnings processing
  - **Overnight/Weekend** (20:00-04:00, weekends): Heavy research, ML training, data backfill, community intel
- **Key files:** Graph runners, mode detection, agent configuration loading
- **Acceptance criteria:**
  - [ ] Graph runners detect current mode and adjust behavior
  - [ ] Extended hours: no new entries, monitoring only
  - [ ] Overnight: research graph gets full compute budget

### 9.8 Unify LLM Provider Systems

- **Finding:** DO-7 | **Severity:** HIGH | **Effort:** 2 days
- **Depends on:** Phase 5 item 5.6 (remove hardcoded model strings)
- **Audit section:** [`08_COST_OPTIMIZATION.md`](../08_COST_OPTIMIZATION.md)
- **Problem:** Multiple LLM provider systems coexist: `get_chat_model()` (primary), hardcoded strings (6+ locations), Ollama direct calls, Groq direct calls. Configuration fragmented.
- **Fix:** Unify all model access through `get_chat_model()` with a single configuration table. Provider fallback chain configurable per tier. Remove all alternative access paths.
- **Key files:** LLM routing layer, all files touching LLM providers
- **Acceptance criteria:**
  - [ ] Single LLM access path through `get_chat_model()`
  - [ ] Provider configuration in one place
  - [ ] Changing provider config switches all model calls

### 9.9 Research Fan-Out Default On

- **Finding:** CTO (MEDIUM) | **Severity:** MEDIUM | **Effort:** 0.5 day
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md`](../04_OPERATIONAL_RESILIENCE.md)
- **Problem:** `RESEARCH_FAN_OUT_ENABLED=false` by default. Sequential validation is 3-5x slower than fan-out mode.
- **Fix:** Change default to `true`. Add safety guards for fan-out (rate limiting, max concurrent).
- **Key files:** Research graph configuration, `.env` defaults
- **Acceptance criteria:**
  - [ ] Research fan-out enabled by default
  - [ ] Rate limiting prevents AV quota exhaustion during fan-out

---

## Dependencies

- **Depends on:** Phase 1 (safety), Phases 1-3 for 9.7 (24/7 mode)
- **9.6 depends on Phase 1 item 1.7** (EventBus wiring)
- **9.8 depends on Phase 5 item 5.6** (hardcoded strings removed first)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| 9.1: Corporate actions data may be incomplete or delayed | Cross-reference AV with EDGAR for critical events. Alert on any detected discrepancy. |
| 9.7: 24/7 mode increases compute costs | Overnight mode uses Haiku-only for research. Weekend compute budgeted at $50/weekend. |
| 9.5: Discord webhook rate limits | Batch non-critical alerts. CRITICAL events always immediate. |

---

## Validation Plan

1. **Corporate actions (9.1):** Simulate stock split → verify position quantity and cost basis adjusted.
2. **Factor exposure (9.2):** Build portfolio with 80% tech → verify sector concentration alert.
3. **Alerts (9.4):** Create → acknowledge → resolve alert → verify full lifecycle in DB.
4. **Discord (9.5):** Trigger kill switch → verify Discord message within 30 seconds.
5. **24/7 mode (9.7):** Run system at 22:00 ET → verify research graph active, trading graph dormant.
