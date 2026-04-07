<!-- PROJECT_CONFIG
runtime: python-pip
test_command: pytest tests/
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-env-validation
section-02-alembic-migrations
section-03-cicd-pipeline
section-04-rate-limiter
section-05-log-aggregation
section-06-monitoring-stack
section-07-health-check-granularity
section-08-kill-switch-recovery
section-09-langfuse-retention
section-10-sbom-scanning
END_MANIFEST -->

# Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-env-validation | - | 02, 08 | Yes |
| section-02-alembic-migrations | - | 04 | Yes |
| section-03-cicd-pipeline | - | 10 | Yes |
| section-04-rate-limiter | 02 | - | No |
| section-05-log-aggregation | - | 06, 07 | Yes |
| section-06-monitoring-stack | 05 | 07 | No |
| section-07-health-check-granularity | 06 | - | No |
| section-08-kill-switch-recovery | 01 | - | No |
| section-09-langfuse-retention | - | - | Yes |
| section-10-sbom-scanning | 03 | - | No |

## Execution Order

1. **Batch 1** (no dependencies, parallel): section-01-env-validation, section-02-alembic-migrations, section-03-cicd-pipeline, section-05-log-aggregation, section-09-langfuse-retention
2. **Batch 2** (after batch 1): section-04-rate-limiter (needs 02), section-06-monitoring-stack (needs 05), section-08-kill-switch-recovery (needs 01)
3. **Batch 3** (after batch 2): section-07-health-check-granularity (needs 06), section-10-sbom-scanning (needs 03)

## Section Summaries

### section-01-env-validation
Env var type validation module (`src/quantstack/config/validation.py`), .env permissions check in `start.sh`, .env.example documentation, credential rotation docs. Hard crash on invalid values.

### section-02-alembic-migrations
Full Alembic setup with baseline migration for 37+ existing tables. Advisory lock integration. `USE_ALEMBIC` fallback flag. Hand-written migrations (no autogenerate).

### section-03-cicd-pipeline
Re-enable `ci.yml` and `release.yml` from `.disabled`. Triage and fix failures (ruff, mypy, pytest, bandit, trivy). Branch protection. Post-deployment smoke test in `start.sh`.

### section-04-rate-limiter
PL/pgSQL `consume_token()` function with `clock_timestamp()`. `rate_limit_buckets` table via Alembic migration. Non-blocking Python integration with circuit breaker fallback to per-process limiter.

### section-05-log-aggregation
Fluent Bit + Loki + Grafana in docker-compose.yml. LogQL alert rules. Discord + email contact points. Loki volume protection. Supervisor Discord alerts remain as parallel alerting path.

### section-06-monitoring-stack
cAdvisor + Prometheus added to docker-compose.yml. Prometheus scrape config for cAdvisor + all graph `/metrics` endpoints. PromQL alert rules for memory, OOM, container restarts. Grafana dashboard with alert panels at top.

### section-07-health-check-granularity
Extend supervisor health check node with cycle success rate, error count, strategy generation velocity. Prometheus gauges for health metrics. Grafana alert rules for thresholds. IC monitoring deferred.

### section-08-kill-switch-recovery
AutoRecoveryManager (broker failures only, MAX_AUTO_RESETS_PER_DAY=2). KillSwitchEscalationManager (Discord 0min, email 4h, emergency 24h). Sizing ramp-back (0.5→0.75→1.0). Optional `reason` parameter on reset().

### section-09-langfuse-retention
Config stub only: `LANGFUSE_RETENTION_ENABLED=false`, `LANGFUSE_RETENTION_DAYS=30` env vars. Scheduler job that checks flag and logs disabled message. No actual cleanup logic.

### section-10-sbom-scanning
Add `pip-audit` and `cyclonedx-py` steps to CI pipeline. Depends on CI being green from section-03.
