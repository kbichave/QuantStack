# P15: Autonomous Fund Integration

**Objective:** Final integration phase — wire everything together into a self-running, self-improving autonomous trading company that operates 24/7 with zero human intervention.

**Scope:** All graphs, all subsystems

**Depends on:** ALL previous phases (this is the capstone)

**Effort estimate:** 2 weeks

---

## What Changes

### 15.1 24/7 Multi-Mode Operation
```
Market Hours (9:30-16:00 ET Mon-Fri):
  - Trading graph: full cycle (entry/exit/monitor/execute)
  - Research graph: lightweight (quick hypothesis checks)
  - Supervisor: health monitoring + risk monitoring

Extended Hours (16:00-20:00, 04:00-09:30 ET):
  - Trading graph: position monitoring only (no new entries)
  - Research graph: earnings processing, overnight signals
  - Supervisor: EOD reconciliation, data sync

Overnight/Weekend:
  - Trading graph: dormant (or crypto/futures trading if P12 complete)
  - Research graph: FULL COMPUTE — hypothesis generation, ML training, RL training
  - Supervisor: community intel, strategy lifecycle, data acquisition
```

### 15.2 The Five Closed Loops (All Must Be Active)
1. **Trade outcome → research priority** (P00 wiring + P10 prioritization)
2. **Realized cost → cost model** (P02 TCA feedback)
3. **IC degradation → signal weight** (P05 adaptive synthesis)
4. **Live performance → strategy demotion** (P00 strategy breaker)
5. **Agent quality → prompt improvement** (P10 meta-learning)

### 15.3 Autonomous Decision Authority Matrix
| Decision | Authority | Override |
|----------|-----------|---------|
| Enter new position | fund_manager agent (with risk gate) | Kill switch |
| Exit position | exit_evaluator + execution_monitor | Always allowed |
| Promote strategy | strategy_promoter (data-driven gates) | Manual demote |
| Retire strategy | strategy_promoter | Manual reinstate |
| Increase exposure | fund_manager (within risk limits) | Risk gate caps |
| Halt trading | kill_switch (auto) | Manual reset |
| Deploy new model | A/B testing framework (P03) | Manual rollback |
| Modify signal weights | IC attribution (P05) | Floor at 50% of static |
| Research priority | autonomous prioritizer (P10) | Manual override queue |

### 15.4 Health & Safety Dashboard
- Unified dashboard: positions, P&L, Greeks, signal health, agent quality, research pipeline
- Alerting: Discord/email for kill switch, drawdown, agent failures, data staleness
- Weekly automated report: Sharpe, alpha decomposition, top winners/losers, research velocity

### 15.5 Performance Benchmarking
- Benchmark against: SPY (market), 60/40 (conservative), equal-weight universe
- Track: Sharpe ratio, max drawdown, Calmar ratio, Sortino ratio, information ratio
- Attribution: how much alpha from signals vs execution vs portfolio optimization vs timing

### 15.6 Disaster Recovery
- Automated DB backup (daily pg_dump → S3)
- Container restart policies with health checks
- Kill switch + circuit breaker layered defense
- Position reconciliation (broker vs system state)

## Acceptance Criteria

1. System runs 24/7 for 7 days without human intervention
2. All 5 feedback loops are actively closing (measurable behavior change)
3. Research generates and validates hypotheses autonomously
4. Trading decisions informed by learning loop outcomes
5. Kill switch triggers correctly on simulated failure scenarios
6. Weekly performance report generated automatically

## The Harvard-IB Fund Benchmark

When P15 is complete, the system should:
- **Research like a quant analyst:** Generate hypotheses, validate statistically, reject failures
- **Trade like a portfolio manager:** Size positions by conviction, manage risk holistically
- **Execute like a trader:** Minimize market impact, adapt to liquidity conditions
- **Monitor like a risk manager:** Real-time Greeks, circuit breakers, regime awareness
- **Learn like a principal:** Every trade improves the next decision
- **Operate like a fund:** 24/7, multi-asset, autonomous, self-healing
