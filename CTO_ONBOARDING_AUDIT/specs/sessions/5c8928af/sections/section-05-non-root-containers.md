# Section 05: Non-Root Containers

## Problem

Every container in the QuantStack Docker stack runs as root. The Dockerfile has no `USER` directive. If any container is compromised (prompt injection leading to code execution, dependency supply chain attack, etc.), the attacker has root access to the container filesystem, mounted volumes, and the Docker socket if exposed. Running as root also means accidental writes to read-only paths go undetected until production.

## Scope

This section covers four changes:

1. Create a non-root `quantstack` user in the Dockerfile
2. Fix volume mount ownership so the non-root user can write to `/app/logs` and the kill switch sentinel path
3. Update the kill switch sentinel path for the non-root user context
4. Add `init: true` to all services in `docker-compose.yml` to prevent zombie process accumulation

## Dependencies

None. This section is fully parallelizable with all other sections in Batch 1.

## Tests (Write These First)

These tests validate the security hardening before and after implementation. They are integration-level tests that require a built Docker image.

```python
# File: tests/integration/test_non_root_containers.py

import subprocess
import pytest

pytestmark = pytest.mark.integration


def test_container_runs_as_non_root():
    """Container process runs as 'quantstack' user, not root.

    Verifies the USER directive in the Dockerfile is effective.
    Run: docker run --rm <image> whoami
    Expected output: 'quantstack'
    """


def test_application_can_write_logs():
    """The quantstack user can write to /app/logs inside the container.

    Run: docker run --rm <image> touch /app/logs/test.log
    Expected: exit code 0
    """


def test_kill_switch_sentinel_writable():
    """The quantstack user can create and write the kill switch sentinel file.

    The sentinel path is /data/quantstack/KILL_SWITCH_ACTIVE (set via
    KILL_SWITCH_SENTINEL env var in the Dockerfile). The /data/quantstack
    directory must be owned by the quantstack user.

    Run: docker run --rm <image> touch /data/quantstack/KILL_SWITCH_ACTIVE
    Expected: exit code 0
    """


def test_all_services_pass_health_checks():
    """After rebuilding images, all services reach healthy state.

    Run: docker compose up -d --build && docker compose ps
    Expected: all services show 'healthy' within their start_period + retries.
    This is a manual/CI verification step rather than a pytest assertion.
    """


def test_init_prevents_zombie_processes():
    """With init: true, PID 1 is tini (or docker-init), not the application process.

    Run: docker run --rm --init <image> cat /proc/1/cmdline
    Expected: contains 'docker-init' or 'tini'

    Without init: true, orphaned child processes from APScheduler or subprocess
    calls become zombies because PID 1 (the app) does not reap them.
    """
```

## Implementation

### 1. Dockerfile Changes

**File:** `Dockerfile`

The current Dockerfile installs system deps, builds TA-Lib, installs Python packages, copies source code, and sets the entrypoint -- all as root. The `USER` directive must be added after all root-requiring operations (package installs, directory creation, chown) but before the `ENTRYPOINT`.

**What to add, and where:**

After the line `RUN mkdir -p /data/quantstack` (line 60 in current Dockerfile), add user creation and ownership:

```dockerfile
# Create non-root user for runtime.
# -r: system user (no home dir clutter, no login shell)
# -m: create /home/quantstack for any user-space state
# -s /bin/false: no interactive shell
RUN useradd -r -m -s /bin/false quantstack

# Transfer ownership of application and data directories.
# /app: application code, logs, scripts
# /data: kill switch sentinel, backups, runtime state
RUN chown -R quantstack:quantstack /app /data
```

Then, immediately before the `ENTRYPOINT` line, add:

```dockerfile
USER quantstack
```

The final Dockerfile structure (relevant tail end) becomes:

```dockerfile
# ... (system deps, TA-Lib build, uv install, COPY, pip install -- all as root) ...

COPY scripts/ ./scripts/
RUN mkdir -p /data/quantstack

# Create non-root runtime user
RUN useradd -r -m -s /bin/false quantstack
RUN chown -R quantstack:quantstack /app /data

ENV KILL_SWITCH_SENTINEL=/data/quantstack/KILL_SWITCH_ACTIVE

COPY scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Drop to non-root for all runtime operations
USER quantstack

ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
```

**Why `-r -m` flags:** `-r` creates a system user (UID below 1000, excluded from login screens). `-m` creates `/home/quantstack` so the user has a home directory for any tools that write to `~` (e.g., Python's `site.USER_BASE`). `-s /bin/false` prevents interactive shell access.

**Why chown both /app and /data:** The application writes logs to `/app/logs` and the kill switch sentinel to `/data/quantstack/KILL_SWITCH_ACTIVE`. Both must be writable by the `quantstack` user. The TA-Lib shared library in `/usr/local/lib` is world-readable by default and needs no ownership change.

**uv cache mount:** The `--mount=type=cache,target=/root/.cache/uv` build cache still works because it runs during `RUN` (build time, as root). The `USER quantstack` directive only affects runtime.

### 2. Volume Mount Permissions in docker-compose.yml

**File:** `docker-compose.yml`

The current `x-graph-volumes` anchor includes:

```yaml
- ~/.quantstack:/root/.quantstack
```

This path is root-specific (`/root/`). With the non-root user, `~` inside the container resolves to `/home/quantstack`, not `/root`. However, the Dockerfile already sets `KILL_SWITCH_SENTINEL=/data/quantstack/KILL_SWITCH_ACTIVE`, which means the kill switch does not depend on the home directory path. The `~/.quantstack` mount is used for drift baselines, FinRL state, Qdrant, and structured logs.

**Change the mount target** from `/root/.quantstack` to `/home/quantstack/.quantstack`:

```yaml
x-graph-volumes: &graph-volumes
  - ./src:/app/src
  - ./scripts:/app/scripts
  - ./models:/app/models
  - ./data:/app/data
  - ./logs:/app/logs
  - ./.claude/memory:/app/.claude/memory
  - ~/.quantstack:/home/quantstack/.quantstack   # was /root/.quantstack
```

**No other volume changes needed.** The bind-mounted directories (`./src`, `./scripts`, `./logs`, etc.) are owned by the host user. Inside the container, the `quantstack` user's UID may differ from the host UID. On Docker Desktop for Mac, this is handled transparently by the file-sharing layer. On Linux hosts, you may need to ensure the host directories are writable by the container UID, or use `user:` directive in compose to map UIDs. For the current single-user local deployment, Docker Desktop handles this.

### 3. Kill Switch Sentinel Path

**No code changes required.** The Dockerfile already sets:

```dockerfile
ENV KILL_SWITCH_SENTINEL=/data/quantstack/KILL_SWITCH_ACTIVE
```

And `kill_switch.py` (line 77) reads this env var:

```python
os.getenv("KILL_SWITCH_SENTINEL", "~/.quantstack/KILL_SWITCH_ACTIVE")
```

The env var takes precedence, so the default `~/.quantstack/` path is never used inside Docker. The `/data/quantstack` directory is owned by `quantstack` (from the `chown` in step 1), so the sentinel file is writable.

**Outside Docker** (local dev), the default `~/.quantstack/KILL_SWITCH_ACTIVE` path still works because local dev runs as the host user who owns `~/.quantstack/`.

Also note: `preflight.py` (line 185) hardcodes `~/.quantstack/KILL_SWITCH_ACTIVE` without checking the env var. This is a latent bug -- it should use the same env var lookup. Flag this for a follow-up fix (outside this section's scope, but worth noting).

### 4. Add `init: true` to All Services in docker-compose.yml

**File:** `docker-compose.yml`

Add `init: true` to every service that runs application code (the three graph services, dashboard, finrl-worker). This causes Docker to inject `tini` as PID 1, which properly reaps zombie child processes.

**Why this matters:** APScheduler (used by the scheduler service, future Section 9) and subprocess calls in tools can spawn child processes. If the parent exits without waiting, the child becomes a zombie. Without an init process, zombies accumulate until the container is restarted. With `init: true`, tini reaps them automatically.

The infrastructure services (postgres, langfuse-db, langfuse, ollama) already handle process management correctly via their own entrypoints, but adding `init: true` to them is harmless and consistent.

Add `init: true` to each service definition. For example, for `trading-graph`:

```yaml
trading-graph:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: quantstack-trading-graph
    init: true                    # <-- ADD THIS
    command: ["python", "-m", "quantstack.runners.trading_runner"]
    # ... rest unchanged ...
```

Repeat for: `research-graph`, `supervisor-graph`, `dashboard`, `finrl-worker`. Optionally add to `postgres`, `langfuse-db`, `langfuse`, `ollama` for consistency.

## Verification Checklist

After implementation, verify manually (or in CI):

1. `docker build -t quantstack .` succeeds
2. `docker run --rm quantstack whoami` prints `quantstack`
3. `docker run --rm quantstack touch /app/logs/test.log` exits 0
4. `docker run --rm quantstack touch /data/quantstack/KILL_SWITCH_ACTIVE` exits 0
5. `docker run --rm quantstack cat /proc/1/cmdline` shows `tini` or `docker-init` (when run with `--init`)
6. `docker compose up -d --build` starts all services, `docker compose ps` shows all healthy
7. Run a full paper trading cycle -- all graph services complete without permission errors

## Failure Modes

- **Host UID mismatch on Linux:** If deploying on a Linux host (not Docker Desktop), the container UID for `quantstack` may not match the host UID that owns the bind-mounted directories. Symptom: `Permission denied` on volume writes. Fix: either use `user: "1000:1000"` in compose to force the container UID to match the host, or use named volumes instead of bind mounts.
- **Build cache invalidation:** The `chown -R /app` step runs after `COPY src/`, so any source change triggers a re-chown. This adds a few seconds to rebuilds but is unavoidable without splitting the chown into multiple layers.
- **Third-party images:** The `ollama`, `postgres`, and `langfuse` images have their own user models. Adding `init: true` is safe, but do not add a `user:` directive to these -- they manage their own user switching internally.
