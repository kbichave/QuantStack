# Section 09: Disaster Recovery

## Objective

Establish automated backup, container restart policies, and layered kill switch hardening to ensure the system recovers from failures without human intervention.

## Files to Create

### `scripts/backup.sh`

Automated `pg_dump` backup script with retention management.

### `scripts/restore_test.sh`

Weekly restore verification script.

### `src/quantstack/autonomous/kill_switch_layers.py`

Layered kill switch coordination.

## Files to Modify

### `docker-compose.yml`

Add restart policies and health checks to all services.

### `start.sh`

Integrate backup scheduling.

## Implementation Details

### Database Backup (`scripts/backup.sh`)

```bash
#!/usr/bin/env bash
# Daily pg_dump to local backup directory
BACKUP_DIR="${QUANTSTACK_BACKUP_DIR:-$HOME/.quantstack/backups}"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

pg_dump "$TRADER_PG_URL" | gzip > "$BACKUP_DIR/quantstack_$TIMESTAMP.sql.gz"

# Prune backups older than RETENTION_DAYS
find "$BACKUP_DIR" -name "quantstack_*.sql.gz" -mtime +$RETENTION_DAYS -delete
```

- Run daily at 02:00 ET via scheduler or cron
- Backup directory defaults to `~/.quantstack/backups/`
- 30-day retention (configurable via `BACKUP_RETENTION_DAYS`)

### Restore Test (`scripts/restore_test.sh`)

```bash
#!/usr/bin/env bash
# Weekly: restore latest backup to a test DB, verify table counts
TEST_DB="quantstack_restore_test"
LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/quantstack_*.sql.gz | head -1)

createdb "$TEST_DB" 2>/dev/null || dropdb "$TEST_DB" && createdb "$TEST_DB"
gunzip -c "$LATEST_BACKUP" | psql "$TEST_DB"

# Verify key tables exist and have rows
for table in positions strategies trade_outcomes signal_weights; do
    count=$(psql -tA "$TEST_DB" -c "SELECT COUNT(*) FROM $table;")
    echo "$table: $count rows"
done

dropdb "$TEST_DB"
```

- Run weekly (Sunday 03:00 ET)
- Log results to `reconciliation_log` table with type `backup_verify`

### Docker Compose Changes

For every service in `docker-compose.yml`:

```yaml
services:
  trading-graph:
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

Add health check endpoints to each graph service if not already present.

### Graceful Degradation Rules

| Service Down | System Behavior |
|-------------|-----------------|
| Research graph | Trading continues, research paused |
| Trading graph | Kill switch engages, positions monitored via supervisor |
| Supervisor graph | Trading continues, no self-healing — alert for human review |
| PostgreSQL | Full halt (kill switch) — no state = no trading |
| Langfuse | Continue without tracing (log locally) |

### Kill Switch Layers (`kill_switch_layers.py`)

Four layers, each independent:

```python
class KillSwitchLayers:
    def check_all_layers(self, portfolio_state: PortfolioState) -> KillSwitchResult: ...

    def _layer1_position_stops(self, portfolio_state) -> bool: ...    # per-position stop loss
    def _layer2_portfolio_drawdown(self, portfolio_state) -> bool: ... # portfolio-level circuit breaker
    def _layer3_agent_kill(self, agent_name: str) -> bool: ...        # per-agent disable
    def _layer4_system_kill(self) -> bool: ...                        # full system halt
```

Layer behavior:
- **Layer 1** (per-position stop): Automatic. Fires close order for individual position. Other trading continues.
- **Layer 2** (portfolio drawdown): Automatic. Fires when drawdown > configured threshold (default 10% from peak). Closes all new order flow, existing positions monitored.
- **Layer 3** (agent kill): Triggered by supervisor when agent quality degrades. Disables specific agent from trading. Other agents continue.
- **Layer 4** (system kill): Manual trigger OR auto on critical failure (DB down, 3+ agents killed). All trading halted. Requires manual restart.

### Integration with Existing Kill Switch

`src/quantstack/execution/kill_switch.py` already exists. The new `KillSwitchLayers` wraps and extends it:
- Layer 4 delegates to the existing `kill_switch.py` mechanism
- Layers 1-3 are new, finer-grained controls
- Existing kill switch behavior is preserved — this is purely additive

## Test Requirements

- `tests/unit/autonomous/test_kill_switch_layers.py`:
  - Test each layer fires independently
  - Test layer precedence (layer 4 overrides all)
  - Test that layer 1 (position stop) does not affect other positions
  - Test that layer 2 threshold is configurable
  - Test layer 3 agent-specific isolation

- Integration test for backup/restore is manual (requires DB) — document in runbook

## Acceptance Criteria

1. Daily backup runs without manual intervention and produces valid gzip files
2. Restore test verifies data integrity (table existence + row counts)
3. All Docker services have restart policies and health checks
4. Kill switch layers are independent — failure of one layer does not disable others
5. Existing kill switch behavior is fully preserved (no regression)
6. Graceful degradation rules are documented and tested for each service-down scenario
7. Backup retention prunes old files correctly (no disk space leak)
