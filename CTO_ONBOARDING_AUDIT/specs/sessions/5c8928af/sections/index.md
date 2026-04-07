<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-psycopg3-migration
section-02-stop-loss-enforcement
section-03-prompt-injection-defense
section-04-output-schema-validation
section-05-non-root-containers
section-06-durable-checkpoints
section-07-eventbus-integration
section-08-database-backups
section-09-containerize-scheduler
section-10-transaction-isolation
section-11-testing-strategy
END_MANIFEST -->

# Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-psycopg3-migration | - | 06, 10 | Yes |
| section-02-stop-loss-enforcement | 01 | - | Yes (after 01) |
| section-03-prompt-injection-defense | - | - | Yes |
| section-04-output-schema-validation | - | - | Yes |
| section-05-non-root-containers | - | - | Yes |
| section-06-durable-checkpoints | 01 | - | No (needs 01) |
| section-07-eventbus-integration | - | - | Yes |
| section-08-database-backups | - | - | Yes |
| section-09-containerize-scheduler | - | - | Yes |
| section-10-transaction-isolation | 01 | - | No (needs 01) |
| section-11-testing-strategy | 01-10 | - | No (final) |

## Execution Order

1. **Batch 1** (no dependencies): section-01-psycopg3-migration, section-03-prompt-injection-defense, section-04-output-schema-validation, section-05-non-root-containers, section-07-eventbus-integration, section-08-database-backups, section-09-containerize-scheduler
2. **Batch 2** (after section-01): section-02-stop-loss-enforcement, section-06-durable-checkpoints, section-10-transaction-isolation
3. **Batch 3** (after all): section-11-testing-strategy

## Section Summaries

### section-01-psycopg3-migration
Migrate db.py and 15+ files from psycopg2 to psycopg3. Replace ThreadedConnectionPool with ConnectionPool. Update PgConnection wrapper, JSON handling, all placeholder patterns. Audit for integer-indexed row access.

### section-02-stop-loss-enforcement
Six-layer stop-loss defense: validation at trade_service and OMS, bracket-or-contingent pattern for all brokers (Alpaca, PaperBroker, E*Trade), post-submission verification, startup reconciliation with auto-fix, bracket leg persistence, stop-loss priority circuit breaker.

### section-03-prompt-injection-defense
Allowlist-first defense: field-level extraction via Pydantic models (primary), XML-tagged templates, injection pattern monitoring/alerting (secondary), Dual LLM separation enforcing research agents can't access execution tools. Migrate research graph first.

### section-04-output-schema-validation
21 Pydantic output models (one per agent), enhanced parse_and_validate() with retry, fail-CLOSED fallback audit (safety_check must halt on failure), dead letter queue table, DLQ monitoring in supervisor.

### section-05-non-root-containers
Add USER directive to Dockerfile, create quantstack user, fix volume mount permissions, update kill switch sentinel path, add init: true to all docker-compose services.

### section-06-durable-checkpoints
Switch all three runners from MemorySaver to PostgresSaver (langgraph-checkpoint-postgres ~=3.0). Shared checkpointer factory with dedicated connection pool. setup() as deployment step. 48-hour checkpoint data retention with pruning.

### section-07-eventbus-integration
Kill switch publishes KILL_SWITCH_TRIGGERED to EventBus (best-effort, never blocks activation). Trading graph polls at safety_check + before execute_entries + before execute_exits. All runners poll at cycle start.

### section-08-database-backups
Daily pg_dump with 30-day retention, WAL archiving with 7-day retention, backup verification via pg_restore --list, flock for concurrent protection, supervisor health check for stale backups, restore runbook.

### section-09-containerize-scheduler
Fix ibkr_mcp import chain, add scheduler Docker service with health endpoint, unless-stopped restart policy, SIGTERM graceful shutdown.

### section-10-transaction-isolation
SELECT FOR UPDATE on position rows, 5s lock timeout with one retry, single-row constraint (no multi-row locks), cover all 5 write paths (broker fill, trade_service metadata, execution monitor, reconciliation, kill switch closer).

### section-11-testing-strategy
End-to-end kill switch propagation test, chaos/failure injection for bracket SL fallback, concurrent writer stress tests, comprehensive integration test suite covering all 10 items.
