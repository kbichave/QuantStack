# Section 9: Containerize Scheduler

## Problem

The scheduler (`scripts/scheduler.py`) runs ~30+ APScheduler jobs in a tmux session on the host. This has three problems:

1. **No restart supervision.** If the process crashes (OOM, unhandled exception, host reboot), all scheduled jobs stop silently. No health check detects this.
2. **Broken import chain.** Importing from `quantstack` transitively pulls in `ibkr_mcp` (via `src/quantstack/data/adapters/ibkr.py`), which is not installed in most environments. The scheduler itself never uses IBKR functionality.
3. **Inconsistent runtime.** The scheduler runs on the bare host while all other services run in Docker. Different Python versions, different installed packages, different environment variables.

The fix: isolate the `ibkr_mcp` import so it only triggers when IBKR is actually requested, then add the scheduler as a Docker Compose service with health checking and auto-restart.

## Dependencies

- None. This section is independently implementable. The scheduler already exists and works when the import chain is clean.
- If section-01 (psycopg3 migration) is completed first, ensure the scheduler's `open_db()` and `pg_conn()` calls work with the new driver.

## Tests First

Write these tests before implementing. Place in `tests/integration/test_scheduler_container.py` (or split across unit/integration as appropriate).

### Import chain tests

```python
# Test: "from quantstack.runners import scheduler" succeeds without ibkr_mcp
# Verify that the quantstack package can be imported without ibkr_mcp installed.
# This validates the lazy import fix is working.

# Test: "python scripts/scheduler.py --dry-run" runs without import errors
# Run the scheduler CLI in dry-run mode and assert exit code 0.
# This is the end-to-end validation that the full import chain is clean.
```

### Docker service tests

```python
# Test: scheduler container starts and health check passes
# Build the image, start the scheduler service, wait for the health check
# to report healthy. Assert container status is "running" and health is "healthy".

# Test: scheduler container auto-restarts after kill (within 60s)
# Kill the scheduler container process (docker kill), then poll for restart.
# The "unless-stopped" restart policy should bring it back within 60 seconds.

# Test: APScheduler has all expected jobs registered
# Hit the health endpoint and verify the job list contains all expected job labels.
# Compare against the JOBS list in scheduler.py.

# Test: health endpoint returns job list and next_run times
# GET /health on port 8422 and verify the response is valid JSON with
# "status", "jobs" (list), and each job entry has "id" and "next_run".

# Test: SIGTERM triggers clean APScheduler shutdown
# Send SIGTERM to the scheduler process inside the container. Verify the
# process exits cleanly (exit code 0) and logs "Scheduler stopped".
```

## Implementation

### Step 1: Fix the ibkr_mcp import chain

The root cause is in `src/quantstack/data/adapters/ibkr.py` (line 36):

```python
from ibkr_mcp.connection import IBKRConnectionManager
```

This is a top-level import in a module that gets pulled in when other parts of `quantstack.data` are imported. The registry (`src/quantstack/data/registry.py`) already handles this correctly -- it comments out the IBKR import and defers it to `from_settings()`. But other import paths may still trigger the `ibkr.py` module.

**Fix approach: guard the import in `ibkr.py` itself.** Move the `ibkr_mcp` and `ib_insync` imports inside a `try/except ImportError` at module level, and raise a clear error at adapter instantiation time if the dependency is missing.

In `src/quantstack/data/adapters/ibkr.py`, replace the bare top-level imports:

```python
# Current (breaks when ibkr_mcp not installed):
import ib_insync as ib
from ibkr_mcp.connection import IBKRConnectionManager

# Fixed (deferred with clear error at usage time):
try:
    import ib_insync as ib
    from ibkr_mcp.connection import IBKRConnectionManager
    _IBKR_AVAILABLE = True
except ImportError:
    _IBKR_AVAILABLE = False
```

Then in the `IBKRDataAdapter.__init__()` method, add a guard:

```python
if not _IBKR_AVAILABLE:
    raise ImportError(
        "IBKRDataAdapter requires 'ib_insync' and 'ibkr_mcp'. "
        "Install with: uv pip install -e '.[ibkr]'"
    )
```

Also check `src/quantstack/data/streaming/ibkr_tick.py` and `src/quantstack/data/streaming/ibkr_stream.py` for the same pattern -- they reference `IBKRConnectionManager` and should get the same `try/except ImportError` guard.

**Validation:** After this change, the following must succeed in an environment without `ibkr_mcp` installed:

```bash
python -c "from quantstack.data.registry import DataProviderRegistry; print('OK')"
python scripts/scheduler.py --dry-run
```

### Step 2: Add a health endpoint to the scheduler

The scheduler needs an HTTP health endpoint so Docker can health-check it. Add a lightweight `threading.Thread`-based HTTP server (no Flask dependency needed) to `scripts/scheduler.py`.

**Health server specification:**

- Bind to `0.0.0.0:8422` (port chosen to avoid conflicts with dashboard on 8421)
- `GET /health` returns HTTP 200 with JSON body:

```json
{
  "status": "running",
  "uptime_seconds": 3847,
  "jobs": [
    {"id": "data_refresh_08:00", "next_run": "2026-04-07T08:00:00-04:00"},
    {"id": "daily_attribution_16:10", "next_run": "2026-04-07T16:10:00-04:00"}
  ],
  "job_count": 30
}
```

- The handler reads the APScheduler instance's `get_jobs()` to populate the job list and next run times
- If APScheduler is not running (scheduler crashed internally but process is alive), return HTTP 503 with `{"status": "degraded"}`

**Implementation sketch** (use `http.server.HTTPServer` + `BaseHTTPRequestHandler` on a daemon thread):

```python
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

_scheduler_ref = None  # Set by start_scheduler()
_start_time = None

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        # Build response from _scheduler_ref.get_jobs()
        # Return 200 if scheduler is running, 503 if degraded
        ...

    def log_message(self, format, *args):
        pass  # Suppress access logs

def _start_health_server(port: int = 8422):
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
```

Call `_start_health_server()` from `start_scheduler()` before `scheduler.start()`. Store the APScheduler instance in `_scheduler_ref` so the handler can inspect it.

### Step 3: Handle SIGTERM for graceful shutdown

The scheduler currently catches `KeyboardInterrupt` and `SystemExit` but does not explicitly handle SIGTERM. Docker sends SIGTERM on `docker stop`. APScheduler's `BlockingScheduler` will not shut down cleanly unless SIGTERM is translated.

Add signal handling in `start_scheduler()`:

```python
import signal

def start_scheduler(dry_run: bool = False) -> None:
    # ... existing setup ...
    scheduler = BlockingScheduler(timezone=TIMEZONE)

    def _handle_sigterm(signum, frame):
        logger.info("Received SIGTERM — shutting down scheduler")
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # ... add jobs ...

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")
```

`scheduler.shutdown(wait=False)` stops the scheduler without waiting for running jobs to complete. This is acceptable because:
- Most jobs are subprocess calls that will be killed by Docker's `stop_grace_period` anyway
- The `data_refresh` job can take up to 2 hours; waiting would exceed any reasonable grace period
- All jobs are idempotent and will run again on next schedule

### Step 4: Add Docker Compose service

Add to `docker-compose.yml`, after the `dashboard` service and before the `finrl-worker` service:

```yaml
scheduler:
  build:
    context: .
    dockerfile: Dockerfile
  container_name: quantstack-scheduler
  command: ["python", "scripts/scheduler.py"]
  depends_on:
    postgres:
      condition: service_healthy
  extra_hosts:
    - "host.docker.internal:host-gateway"
  env_file:
    - path: .env
      required: false
  environment:
    TRADER_PG_URL: ${TRADER_PG_URL:-postgresql://${USER}@host.docker.internal:5432/quantstack}
  ports:
    - "127.0.0.1:8422:8422"
  volumes: *graph-volumes
  healthcheck:
    test: ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8422/health')\" 2>/dev/null || exit 1"]
    interval: 60s
    timeout: 15s
    retries: 3
    start_period: 30s
  mem_limit: 512m
  memswap_limit: 512m
  stop_grace_period: 120s
  restart: unless-stopped
  networks:
    - quantstack-net
  logging:
    driver: json-file
    options:
      max-size: "50m"
      max-file: "5"
```

**Key configuration decisions:**

- **`depends_on: postgres`**: The scheduler's startup freshness check queries the database. It must wait for Postgres to be healthy.
- **`stop_grace_period: 120s`**: The longest regularly-running job is `data_refresh` at up to 2 hours, but we cannot hold Docker stop for 2 hours. 120 seconds is a reasonable compromise -- gives most jobs time to finish, and `data_refresh` is fully idempotent so killing it mid-run is safe.
- **`mem_limit: 512m`**: Matches the plan. The scheduler itself is lightweight but spawns subprocesses (`acquire_historical_data.py`) that can use significant memory. Those subprocesses run inside the same container cgroup, so the limit applies to the total.
- **`restart: unless-stopped`**: Automatic restart on crash. Docker will back off exponentially on repeated failures.
- **Port 8422**: Health endpoint. Exposed on localhost only (127.0.0.1 binding) for security.
- **No dependency on ollama or langfuse**: The scheduler runs deterministic Python jobs, not LLM agents. It only needs the database.

**Volume mounts**: Uses the same `*graph-volumes` anchor as graph services. The scheduler needs:
- `./scripts:/app/scripts` -- the scheduler script itself plus `acquire_historical_data.py` and others it invokes
- `./src:/app/src` -- quantstack package source
- `./data:/app/data` -- market data cache written by data refresh jobs
- `./.claude/memory:/app/.claude/memory` -- memory compaction job reads/writes these files
- `~/.quantstack:/root/.quantstack` -- kill switch sentinel path

### Step 5: Update start.sh and status.sh

If `start.sh` currently launches the scheduler in tmux, remove that logic. The scheduler now starts automatically via Docker Compose.

If `status.sh` checks the scheduler's tmux session, replace that check with a Docker container health query:

```bash
# Old: check tmux session
# New: check Docker container
docker inspect --format='{{.State.Health.Status}}' quantstack-scheduler 2>/dev/null || echo "not running"
```

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Subprocess calls (e.g., `acquire_historical_data.py`) may behave differently inside Docker | Data refresh jobs fail silently | Run `--dry-run` first to verify all subprocess paths resolve. Check `WORKDIR` is correct inside container. |
| 512MB memory limit may be insufficient for data refresh subprocesses | OOM kill during data refresh | Monitor with `docker stats`. Increase to 1GB if needed. The subprocess runs in the same cgroup. |
| The `community_intel_weekly` job shells out to `claude` CLI | Claude CLI not available in container | This job will fail gracefully (subprocess returns non-zero, logged as warning). Either install Claude CLI in the image or restructure this job to use the API directly. Not a blocker for containerization. |
| `ibkr_mcp` guard may miss an import path | ImportError at runtime for unrelated code | Grep for all `ibkr_mcp` and `ib_insync` imports across the codebase. The known sites are: `data/adapters/ibkr.py`, `data/streaming/ibkr_tick.py`, `data/streaming/ibkr_stream.py`. All three need the guard. |

## Files to Create or Modify

- **Modify**: `src/quantstack/data/adapters/ibkr.py` -- guard `ibkr_mcp` and `ib_insync` imports
- **Modify**: `src/quantstack/data/streaming/ibkr_tick.py` -- guard `ibkr_mcp` import
- **Modify**: `src/quantstack/data/streaming/ibkr_stream.py` -- guard `ibkr_mcp` import (if applicable)
- **Modify**: `scripts/scheduler.py` -- add health endpoint, SIGTERM handler
- **Modify**: `docker-compose.yml` -- add `scheduler` service
- **Modify**: `scripts/start.sh` -- remove tmux-based scheduler launch (if present)
- **Modify**: `scripts/status.sh` -- update scheduler health check to use Docker
- **Create**: `tests/integration/test_scheduler_container.py` -- container and import chain tests
