# P15 Research: Autonomous Fund Integration

## Codebase Research

### What Exists
- **Scheduler**: `scripts/scheduler.py` — existing scheduling infrastructure, market hour awareness
- **Strategy lifecycle**: `src/quantstack/autonomous/strategy_lifecycle.py` — promote/demote/retire pipeline
- **Risk gate**: `src/quantstack/execution/risk_gate.py` — position limits, kill switch integration
- **Execution monitor**: `src/quantstack/execution/execution_monitor.py` — trade monitoring
- **Trade hooks**: `src/quantstack/hooks/trade_hooks.py` — pre/post trade hooks
- **Status scripts**: `status.sh` — existing health dashboard
- **Docker Compose**: multi-service orchestration (postgres, langfuse, ollama, 3 graphs)
- **Supervisor graph**: health monitoring, self-healing existing

### What's Needed (Gaps)
1. **Operating modes**: No formal mode system (market/extended/overnight/weekend)
2. **Feedback loop verification**: No automated loop closure detection
3. **Decision authority matrix**: No formal ceiling/escalation framework
4. **Position reconciliation**: No broker vs system state comparison
5. **Automated reporting**: No weekly report generation
6. **Benchmark tracking**: No systematic benchmark comparison
7. **Burn-in protocol**: No formal validation for autonomous operation

## Domain Research

### Autonomous Trading Operations
- Key principle: defense in depth — multiple independent safety layers
- Reconciliation is non-negotiable: broker state is source of truth, system state must match
- Mode-based scheduling: different activities during market vs after-hours vs weekends
- All autonomous systems need an escalation path — ceiling + alert, never unbounded authority

### Feedback Loop Verification
- Each loop needs: trigger condition, expected behavior change, measurement metric
- Healthy loop: closes within 24-48h of trigger
- Broken loop: indicates disconnected subsystem — alert immediately
- Loop health is the primary measure of system autonomy quality
