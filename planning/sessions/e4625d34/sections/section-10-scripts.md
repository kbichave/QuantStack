# Section 10: Start/Stop/Status Scripts

## Overview

This section replaces the existing tmux-based `start.sh`, `stop.sh`, and `status.sh` scripts with Docker Compose equivalents. The new scripts manage a containerized stack of 8 services (postgres, langfuse-db, langfuse, ollama, chromadb, trading-crew, research-crew, supervisor-crew) instead of spawning tmux windows with `claude` CLI invocations.

**Dependencies:** Section 09 (runners must exist as `python -m quantstack.runners.*` entrypoints). Section 14 depends on this section for Docker resource configuration.

**Files to modify:**
- `/Users/kshitijbichave/Personal/Trader/start.sh` (rewrite)
- `/Users/kshitijbichave/Personal/Trader/stop.sh` (rewrite)
- `/Users/kshitijbichave/Personal/Trader/status.sh` (rewrite)

**Existing behavior preserved:** The current scripts handle `.env` loading, DB migrations, universe bootstrap, preflight checks, data freshness, credit regime display, kill switch, and memory compaction. All of these must survive in the new version, but the execution target changes from tmux+claude to Docker Compose.

---

## Tests

Write these tests before implementing the scripts. They validate script behavior by parsing script content and, where feasible, running subcommands in isolation.

```python
# File: tests/unit/test_scripts.py

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_start_sh_checks_for_docker():
    """start.sh must verify Docker and docker compose are installed."""
    content = (PROJECT_ROOT / "start.sh").read_text()
    assert "docker" in content.lower()
    # Must check for both docker daemon and compose plugin
    assert "docker compose" in content or "docker-compose" in content


def test_start_sh_checks_for_env_file():
    """start.sh must abort if .env is missing."""
    content = (PROJECT_ROOT / "start.sh").read_text()
    assert ".env" in content
    # Should exit non-zero when .env is absent
    assert "exit 1" in content


def test_start_sh_starts_infrastructure_before_crews():
    """Infrastructure services (postgres, ollama, chromadb, langfuse) must
    start and pass health checks before crew services start."""
    content = (PROJECT_ROOT / "start.sh").read_text()
    infra_pos = content.find("docker compose up")
    # There should be at least two `docker compose up` invocations:
    # one for infra, one for crews
    second_up = content.find("docker compose up", infra_pos + 1)
    assert second_up > infra_pos, (
        "Expected two docker compose up calls: infra first, crews second"
    )


def test_stop_sh_sends_graceful_shutdown():
    """stop.sh must use 'docker compose down', not kill -9."""
    content = (PROJECT_ROOT / "stop.sh").read_text()
    assert "docker compose down" in content
    assert "kill -9" not in content


def test_status_sh_displays_container_status_heartbeats_positions():
    """status.sh must show container health, heartbeats, and position count."""
    content = (PROJECT_ROOT / "status.sh").read_text()
    assert "docker compose ps" in content or "docker compose" in content
    assert "heartbeat" in content.lower()
    assert "position" in content.lower()


def test_startup_waits_for_health_checks():
    """start.sh must wait for infrastructure health checks before proceeding."""
    content = (PROJECT_ROOT / "start.sh").read_text()
    # Should contain a wait/health-check loop or --wait flag
    assert "--wait" in content or "healthy" in content.lower()
```

---

## Implementation: start.sh

The new `start.sh` is a 13-step Docker Compose launcher. It replaces all tmux logic with container orchestration. The script is a sequential bash script with `set -euo pipefail`.

### Step-by-step specification

1. **Preamble and .env check.** Source `.env`. If `.env` does not exist, print an error referencing `.env.example` and exit 1.

2. **Check prerequisites.** Verify `docker` is on PATH. Verify `docker compose version` succeeds (Compose V2 plugin). If either is missing, print an actionable error message and exit 1. Note: tmux and claude CLI checks are removed since they are no longer needed.

3. **Validate required env vars.** Check that `TRADER_PG_URL`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and `ALPHA_VANTAGE_API_KEY` are set (non-empty). Exit 1 with a list of missing vars if any are absent.

4. **Start infrastructure services.** Run:
   ```
   docker compose up -d postgres langfuse-db ollama chromadb langfuse
   ```

5. **Wait for infrastructure health checks.** Poll each infrastructure service until healthy or until a 120-second timeout. Use `docker compose ps --format json` to check health status. The services to wait for: `postgres`, `ollama`, `chromadb`, `langfuse`. Print progress dots. If timeout is reached, print which services are still unhealthy and exit 1.

6. **Pull Ollama models.** Run:
   ```
   docker compose exec ollama ollama pull mxbai-embed-large
   docker compose exec ollama ollama pull llama3.2
   ```
   These are idempotent (no-op if model already exists). Print model status.

7. **Run DB migrations.** Run migrations via a one-shot container:
   ```
   docker compose run --rm trading-crew python -m quantstack.db --migrate
   ```
   This reuses the same image as the trading-crew service. The `--rm` flag removes the container after completion.

8. **Bootstrap universe if empty.** Run a one-shot container that checks `SELECT COUNT(*) FROM universe WHERE is_active = TRUE`. If zero, run `quantstack-bootstrap`. Same one-shot container pattern as step 7.

9. **Run preflight checks.** One-shot container runs `PreflightCheck`. If preflight fails (returns non-zero), print blockers and exit 1. Same logic as current start.sh step 8.

10. **Check data freshness and trigger sync if stale.** One-shot container checks max OHLCV timestamp for SPY. If more than 1 trading day stale, run the data sync in a detached one-shot container (background). Print sync status.

11. **Trigger one-time RAG ingestion.** One-shot container checks if ChromaDB collections are empty (via `chromadb` HTTP API or Python client). If empty, run `python -m quantstack.rag.ingest` to ingest `.claude/memory/` files. This only runs once on first startup.

12. **Display credit regime and system status.** One-shot container runs credit regime check (same logic as current step 8b). Print regime status. This is informational and does not block startup.

13. **Start crew services.** Run:
    ```
    docker compose up -d trading-crew research-crew supervisor-crew
    ```

14. **Wait for crew health checks.** Poll crew service health (heartbeat-based) with a 60-second timeout. Unlike infrastructure, crew health checks may take longer since the first cycle needs to complete. If a crew is not healthy after 60s, print a warning but do not exit (the crew may just be in its first cycle).

15. **Print status summary.** Print a box with:
    - Langfuse URL: `http://localhost:3000`
    - How to view logs: `docker compose logs -f trading-crew`
    - How to stop: `./stop.sh`
    - How to check status: `./status.sh`

### Removed from current start.sh

- tmux session creation and window management (replaced by Docker Compose)
- claude CLI check (no longer used)
- SIGTERM trap writing kill switch to DB (runners handle their own shutdown)
- Memory compaction step (moved to supervisor crew scheduled tasks)
- Community intelligence background scan (moved to supervisor crew scheduled tasks)

### Preserved from current start.sh

- `.env` loading and validation
- DB migration (idempotent)
- Universe bootstrap on first run
- Preflight checks (Alpaca connection, AV key, data freshness)
- Data freshness check with background sync
- Credit regime display

---

## Implementation: stop.sh

The new `stop.sh` replaces tmux session killing with Docker Compose graceful shutdown.

### Step-by-step specification

1. **Source .env** (if it exists).

2. **Print banner** ("QuantStack -- Graceful Shutdown").

3. **Activate kill switch.** Both layers, identical to current stop.sh:
   - Layer A: Write `kill_switch = 'active'` to DB `system_state` table via `psycopg2`.
   - Layer B: Write sentinel file at `~/.quantstack/KILL_SWITCH_ACTIVE` with `triggered_at` and `reason`.

   This uses a host-side Python invocation (not containerized) since the DB is accessible from the host via the exposed postgres port. If DB write fails, log a warning but continue (sentinel file is the fallback).

4. **Run `docker compose down`.** This sends SIGTERM to all containers. Each crew runner's graceful shutdown handler (from section 07) will:
   - Set `should_stop = True`
   - Wait for current cycle to complete (up to 60s)
   - Flush Langfuse traces
   - Persist state to PostgreSQL
   - Exit cleanly

   Docker Compose has a `stop_grace_period` of 90 seconds (configured in `docker-compose.yml` per section 01). After 90s, containers receive SIGKILL.

5. **Print completion banner.** Include:
   - Kill switch status (active)
   - How to restart: `./start.sh`
   - How to reset kill switch (same command as current stop.sh)
   - How to check status: `./status.sh`

### Removed from current stop.sh

- tmux session detection and killing
- Polling `loop_heartbeats` table for running iterations (Docker Compose handles this via stop_grace_period + SIGTERM handlers)

### Preserved from current stop.sh

- Kill switch activation (both DB and sentinel layers)
- Graceful wait for in-flight work (now handled by Docker's stop_grace_period instead of manual polling)

---

## Implementation: status.sh

The new `status.sh` replaces the delegation to `scripts/dashboard.py` with a self-contained bash script that queries Docker Compose and the database.

### Step-by-step specification

1. **Parse arguments.** Support `--watch` flag and optional `--interval N` (default 10s).

2. **Print header** ("QuantStack Status Dashboard").

3. **Container health.** Run `docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Health}}"` and display the formatted output. Color-code: green for healthy, red for unhealthy, yellow for starting.

4. **Crew heartbeats.** For each crew (trading, research, supervisor), read the heartbeat timestamp. Two approaches (try in order):
   - Query `loop_heartbeats` table via `psycopg2` for last heartbeat per loop.
   - If DB is unreachable, check Docker container health status as a fallback.

   Display: crew name, last heartbeat timestamp, age in seconds, and status (OK if within threshold, STALE if exceeded). Thresholds: trading 120s, research 600s, supervisor 360s.

5. **Active positions.** Query the database:
   ```sql
   SELECT COUNT(*), COALESCE(SUM(unrealized_pnl), 0)
   FROM positions WHERE status = 'open'
   ```
   Display: position count and total unrealized P&L.

6. **Current regime.** Query:
   ```sql
   SELECT value FROM loop_iteration_context
   WHERE key = 'current_regime' ORDER BY updated_at DESC LIMIT 1
   ```
   Display the regime string.

7. **Langfuse dashboard URL.** Print `http://localhost:3000`.

8. **Ollama model status.** Run `docker compose exec ollama ollama list` and display loaded models.

9. **ChromaDB collection sizes.** Query ChromaDB HTTP API at `http://localhost:8000/api/v1/collections` and display collection names with document counts.

10. **Watch mode.** If `--watch` was passed, clear screen and repeat steps 2-9 every `--interval` seconds until Ctrl+C.

---

## Key Design Decisions

**One-shot containers for startup tasks (steps 7-11).** DB migrations, bootstrap, and preflight run as ephemeral containers (`docker compose run --rm`) rather than host-side Python. This ensures the same Python environment and dependencies are used regardless of what's installed on the host. The tradeoff is ~2-3 seconds of container startup overhead per step, but this only happens once at launch.

**Kill switch stays host-side in stop.sh.** The kill switch write in stop.sh uses host-side Python (not a container) because `docker compose down` is about to tear down all containers. Writing via a container that's about to be stopped introduces a race condition. The host has direct access to postgres via the exposed port.

**status.sh is bash, not Python.** The current status.sh delegates to `scripts/dashboard.py`. The new version is self-contained bash to minimize dependencies and work even when the Python environment is broken. Database queries use one-liner `python3 -c` invocations for portability.

**No tmux references.** All tmux logic is removed. The only process manager is Docker Compose with `restart: unless-stopped` on crew services.
