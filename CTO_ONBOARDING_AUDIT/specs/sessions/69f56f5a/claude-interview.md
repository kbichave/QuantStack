# Interview Transcript — Phase 3: Operational Resilience

## Round 1

### Q1: Priority order if time runs short?

**Answer:** Follow spec order as-is. Items are already prioritized by severity.

### Q2: Log aggregation stack choice?

**Answer:** Fluent Bit + Loki + Grafana. Full observability stack, free, ~200MB, pairs with existing Prometheus metrics.

### Q3: Kill switch auto-recovery — what should NEVER auto-reset?

**Answer:** Auto-reset broker failures ONLY. Everything else requires manual reset with reason logged.

Detailed reasoning:
- **Broker failures** — pure infrastructure. Connectivity restored = known state, no signal about market or model health. Safe to auto-reset.
- **Drawdown (6% over 3 days)** — NEVER auto-reset. Systematic signal that something is wrong (regime change, model failure, bug compounding). Auto-recovery at 50% sizing doesn't fix root cause, just loses money slower. Requires human review.
- **Model drift (>50%)** — NEVER auto-reset. Trading with a broken model at half size is a slower bleed, not a recovery. Drift this severe needs investigation.
- **SPY halt** — NEVER auto-reset. Looks transient but post-halt open is often the most volatile 15 minutes. Auto-resuming into that with momentum or mean-reversion models is exactly when you get blown up.

**Principle:** Auto-reset only when the condition is purely external and infrastructure-level with no information content about model or market state.

### Q4: Rate limiter implementation approach?

**Answer:** PL/pgSQL function. Single atomic DB call, cleanest approach.

## Round 2

### Q5: CI/CD scope — re-enable as-is or modernize?

**Answer:** Re-enable + fix breakage only. Get CI green first, iterate later.

### Q6: Migration versioning approach?

**Answer:** Full Alembic migration. Proper up/down migrations, version chain, auto-generate from models.

### Q7: Notification escalation channels?

**Answer:** Discord + email. No AWS available — will use a Google mail bot (user to create). Email escalation is a TODO for the user to set up the Google mail bot credentials.

### Q8: Docker monitoring stack?

**Answer:** Add everything to docker-compose.yml — cAdvisor + Prometheus + Grafana, fully self-contained.

## Round 3

### Q9: Secrets management approach?

**Answer:** Harden .env only. chmod 600, startup validation, rotation docs. Sufficient for single-host.

### Q10: Langfuse retention cleanup?

**Answer:** Don't do cleanup now. Add it as a feature with config flag, default to OFF. Defer actual cleanup to later.

### Q11: Env var validation — fail-fast or warnings?

**Answer:** Hard crash on invalid. Better to not start than to trade with `RISK_MAX_POSITION_PCT=ten`.
