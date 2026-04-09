# P15 Implementation Plan: Autonomous Fund Integration

## 1. Background

P15 is the capstone phase. All prior phases (P00-P14) provide the subsystems — P15 wires them into a self-running, self-improving autonomous trading company that operates 24/7 with zero human intervention. This phase is primarily integration, configuration, and operational hardening rather than new feature development.

## 2. Anti-Goals

- **Do NOT build new subsystems** — P15 integrates what exists, it doesn't add features
- **Do NOT remove human override capability** — kill switch, manual demote, manual research override always available
- **Do NOT deploy without 7-day burn-in** — system must run 7 days unattended before "autonomous" label
- **Do NOT allow unbounded autonomy** — every autonomous decision has a ceiling and a circuit breaker
- **Do NOT skip reconciliation** — broker state vs system state must match within $100 tolerance

## 3. 24/7 Multi-Mode Scheduler

### 3.1 Mode Configuration

New `src/quantstack/config/operating_modes.py`:
- `MARKET_HOURS` (9:30-16:00 ET Mon-Fri): full trading cycle, lightweight research, health monitoring
- `EXTENDED_HOURS` (16:00-20:00, 04:00-09:30 ET): position monitoring only, earnings processing, EOD reconciliation
- `OVERNIGHT_WEEKEND`: no equity trading, full research compute (hypothesis generation, ML training, RL training), community intel, strategy lifecycle
- `CRYPTO_FUTURES` (24/7, conditional on P12): crypto + futures trading in all modes

### 3.2 Scheduler Integration

Extend `scripts/scheduler.py`:
- Mode auto-detection from system clock + market calendar
- Graph activation/deactivation per mode
- Resource allocation: research gets full compute overnight, trading gets priority during market hours
- Mode transition hooks: EOD reconciliation on market→extended, data sync on extended→overnight

## 4. Five Closed Feedback Loops

### 4.1 Loop Verification Framework

New `src/quantstack/autonomous/loop_verifier.py`:
- For each of the 5 feedback loops, define: trigger condition, expected behavior change, measurement
- Run verification daily: did the loop actually close in the last 24h?
- Alert if any loop hasn't closed in 48h (indicates broken feedback)

### 4.2 Loop Definitions

1. **Trade outcome → research priority** (P00 + P10): trade loss → research queue entry with priority boost
2. **Realized cost → cost model** (P02): TCA feedback updates cost estimates for future execution
3. **IC degradation → signal weight** (P05): IC drops below 0.02 → weight floors to minimum
4. **Live performance → strategy demotion** (P00): 3 consecutive losing weeks → strategy demoted
5. **Agent quality → prompt improvement** (P10): agent win rate < 40% → few-shot injection from library

### 4.3 Loop Health Dashboard

`get_loop_health()` returns per-loop:
- `last_triggered`: timestamp
- `last_behavior_change`: what changed and when
- `status`: `healthy` (closed in last 24h), `stale` (24-48h), `broken` (>48h)

## 5. Decision Authority Matrix

### 5.1 Authority Configuration

New `src/quantstack/autonomous/authority_matrix.py`:
- Define: decision_type, authorized_agent, approval_required, ceiling, override_mechanism
- Every autonomous decision logged with: who decided, what inputs, what ceiling applied

### 5.2 Ceiling Enforcement

Extend risk gate with authority ceilings:
- Max single position: 5% of portfolio
- Max daily new positions: 3
- Max strategy promotion per week: 1
- Max signal weight change per cycle: 10% relative

### 5.3 Escalation

When an agent wants to exceed ceiling:
- Log the request with full context
- Do NOT execute — flag for human review
- Continue operating within ceiling in the meantime

## 6. Position Reconciliation

### 6.1 Reconciler

New `src/quantstack/execution/reconciler.py`:
- Compare: system positions table vs broker API positions
- Run: every mode transition + every 4 hours
- Tolerance: $100 notional difference per position
- On mismatch: alert, log, adjust system state to match broker (broker is source of truth)

### 6.2 P&L Reconciliation

- Compare: system P&L calculation vs broker P&L
- Alert threshold: >1% daily P&L discrepancy
- Log discrepancies for investigation

## 7. Health & Safety Dashboard

### 7.1 Unified Status

Extend `status.sh` and supervisor graph:
- Portfolio: positions, unrealized P&L, Greeks exposure, margin utilization
- Signals: per-collector IC health, stale data alerts, synthesis weights
- Agents: per-agent quality score, win rate, last cycle time
- Research: pipeline depth, hypothesis velocity, strategy lifecycle counts
- System: DB health, API rate limits, container status

### 7.2 Alerting

Discord webhook alerts for:
- Kill switch triggered
- Drawdown > 5% from peak
- Agent win rate < 30% for 5 cycles
- Data staleness > configured threshold
- Reconciliation mismatch
- Feedback loop broken (>48h without closing)

### 7.3 Weekly Report

Automated report every Sunday 20:00 ET:
- Weekly Sharpe, max drawdown, Calmar ratio, Sortino ratio
- Alpha decomposition: signal vs execution vs timing
- Top 3 winners/losers with causal attribution (P13)
- Research velocity: hypotheses generated, validated, rejected
- System health: uptime, error rate, loop health

## 8. Performance Benchmarking

### 8.1 Benchmark Tracking

New `src/quantstack/autonomous/benchmarks.py`:
- Track against: SPY (market), 60/40 portfolio, equal-weight universe
- Compute: information ratio, tracking error, alpha, beta
- Rolling windows: 1w, 1m, 3m, YTD

### 8.2 Attribution

Decompose returns into:
- Signal alpha: return from signal-driven entries
- Execution alpha: return from execution optimization (P02)
- Timing alpha: return from entry/exit timing vs benchmark
- Risk management: return preserved by risk gate rejections

## 9. Disaster Recovery

### 9.1 Backup

- Automated daily `pg_dump` → local backup directory
- Retention: 30 days of daily backups
- Verify: weekly restore test to separate DB

### 9.2 Container Restart Policies

- All services: `restart: unless-stopped` in docker-compose
- Health checks: `/health` endpoint with 30s interval
- Graceful degradation: if research graph down, trading continues; if trading down, kill switch engages

### 9.3 Kill Switch Layering

- Layer 1: Per-position stop loss (automatic)
- Layer 2: Portfolio drawdown circuit breaker (automatic)
- Layer 3: Agent-level kill switch (supervisor graph)
- Layer 4: System-level kill switch (manual + auto on critical failure)

## 10. Burn-In Protocol

### 10.1 7-Day Validation

Before declaring autonomous:
1. All 5 feedback loops closed at least once
2. No kill switch triggers from bugs (real risk triggers OK)
3. Reconciliation matches within tolerance for 7 consecutive days
4. Weekly report generates successfully
5. At least 1 strategy promoted and 1 strategy managed through lifecycle

### 10.2 Go-Live Checklist

- [ ] All feature flags enabled for production features
- [ ] Kill switch tested (trigger + recovery)
- [ ] Backup restore tested
- [ ] Discord alerts verified
- [ ] Mode transitions tested (market → extended → overnight → market)

## 11. Testing

- Loop verifier: mock loop triggers → verify detection and alerting
- Authority matrix: attempt ceiling breach → verify rejection
- Reconciler: inject position mismatch → verify detection and correction
- Mode scheduler: mock clock → verify correct mode activation
- Dashboard: verify all metrics compute without error
- Burn-in: simulated 7-day run with injected failures
