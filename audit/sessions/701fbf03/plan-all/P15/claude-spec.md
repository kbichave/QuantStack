# P15 Spec: Autonomous Fund Integration

## Deliverables

### D1: 24/7 Operating Modes
- Market hours, extended hours, overnight/weekend mode definitions
- Scheduler integration with mode auto-detection
- Mode transition hooks (EOD reconciliation, data sync)
- Market calendar awareness (holidays, early close, halts)

### D2: Feedback Loop Verification
- 5 loops defined with trigger conditions and expected behavior changes
- Daily verification: did each loop close in last 24h?
- Alert if any loop stale (>48h without closing)

### D3: Decision Authority Matrix
- Per-decision-type: authorized agent, ceiling, override mechanism
- Ceiling enforcement in risk gate
- Escalation: log + flag for human review when ceiling exceeded

### D4: Position Reconciliation
- Broker vs system state comparison every 4h + mode transitions
- $100 tolerance per position
- Corporate action awareness
- Broker is source of truth on mismatch

### D5: Health Dashboard + Alerting
- Unified status: portfolio, signals, agents, research, system
- Discord alerts for kill switch, drawdown, broken loops
- Weekly automated performance report

### D6: Benchmarking
- Track vs SPY, 60/40, equal-weight
- Information ratio, tracking error, alpha/beta
- Return decomposition: signal, execution, timing, risk management

### D7: Disaster Recovery
- Daily pg_dump, 30-day retention
- Container restart policies with health checks
- Kill switch layering (4 layers)
- Weekly restore test

### D8: Burn-In Protocol
- 7-day unattended operation validation
- All 5 loops must close at least once
- No bug-triggered kill switches (market triggers OK)
- Go-live checklist

## Dependencies
- ALL prior phases (P00-P14) — this is the capstone integration phase
