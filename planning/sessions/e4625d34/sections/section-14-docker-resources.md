# Section 14: Docker Resource Limits and Cost Estimation

## Overview

This section adds resource limits, log rotation, Langfuse retention cleanup, and cost estimation validation to the Docker Compose stack. It is the final section in the implementation sequence and depends on section 10 (scripts) being complete -- all 8 services must already be defined in `docker-compose.yml` before resource constraints are applied.

The goal: ensure the full stack runs reliably on a 16GB machine without OOM kills, logs do not consume unbounded disk, Langfuse traces are pruned after 30 days, and the monthly LLM cost stays within the $60-240 budget.

---

## Dependencies

- **Section 01 (Scaffolding):** `docker-compose.yml` with all 8 services defined (postgres, langfuse-db, langfuse, ollama, chromadb, trading-crew, research-crew, supervisor-crew).
- **Section 10 (Scripts):** `start.sh`, `stop.sh`, `status.sh` must exist -- this section does not modify their logic, only the Docker Compose resource constraints they rely on.

---

## Tests (Write First)

File: `tests/unit/test_docker_resources.py`

```python
"""Tests for Docker resource limits, log rotation, and Langfuse retention."""
import yaml
import pytest


@pytest.fixture
def compose_config():
    """Load and parse docker-compose.yml."""
    # Load the docker-compose.yml from the project root.
    # Implementation should use yaml.safe_load on the file contents.


class TestDockerResourceLimits:
    """Verify every service has explicit memory limits."""

    def test_all_services_have_memory_limits(self, compose_config):
        """Each service in docker-compose.yml must declare a mem_limit or
        deploy.resources.limits.memory so Docker enforces a ceiling."""

    def test_total_memory_under_10gb(self, compose_config):
        """Sum of all service memory limits must be < 10 GB so the stack
        fits on a 16 GB machine with room for the OS and dev tools."""

    def test_expected_per_service_limits(self, compose_config):
        """Spot-check that high-memory services (ollama, crews) have the
        correct limits from the design spec:
          postgres      512m
          langfuse-db   256m
          langfuse      512m
          ollama        4g
          chromadb      1g
          trading-crew  1g
          research-crew 1g
          supervisor-crew 512m
        """


class TestLogRotation:
    """Verify log rotation is configured for every service."""

    def test_all_services_have_logging_config(self, compose_config):
        """Each service must specify a logging driver with max-size and
        max-file to prevent unbounded log growth on disk."""

    def test_log_max_size_is_50m(self, compose_config):
        """Log file max-size should be 50m per the design spec."""

    def test_log_max_file_is_5(self, compose_config):
        """Log file rotation should keep at most 5 files per service."""
```

File: `tests/unit/test_langfuse_retention.py`

```python
"""Tests for Langfuse trace retention cleanup."""
import pytest
from datetime import datetime, timedelta


class TestLangfuseRetention:
    """Verify the retention cleanup function works correctly."""

    def test_cleanup_removes_traces_older_than_30_days(self):
        """Traces older than 30 days should be deleted from the
        langfuse-db. The cleanup function queries the Langfuse
        Postgres database directly (not the Langfuse API) and
        deletes rows from the traces table where timestamp < now - 30d."""

    def test_cleanup_preserves_recent_traces(self):
        """Traces younger than 30 days must not be deleted."""

    def test_cleanup_is_idempotent(self):
        """Running cleanup twice in a row does not error or delete
        anything extra."""
```

---

## Implementation Details

### 14.1 Docker Compose Resource Limits

Add `mem_limit` to every service in `docker-compose.yml`. Use the short-form `mem_limit` key (Compose v2 syntax) rather than the nested `deploy.resources.limits.memory` which requires `docker stack deploy` or swarm mode.

**Target file:** `/Users/kshitijbichave/Personal/Trader/docker-compose.yml`

Per-service limits:

| Service | `mem_limit` | Rationale |
|---------|------------|-----------|
| `postgres` | `512m` | Sufficient for quantstack workload; PostgreSQL shared_buffers default is 128MB |
| `langfuse-db` | `256m` | Lightweight metadata store for Langfuse traces |
| `langfuse` | `512m` | Next.js app + trace ingestion pipeline |
| `ollama` | `4g` | Must hold mxbai-embed-large (~2GB) and llama3.2 (~2GB) in memory simultaneously |
| `chromadb` | `1g` | HNSW vector index + HTTP server; grows with collection size |
| `trading-crew` | `1g` | Python process + CrewAI agent overhead + in-cycle memory |
| `research-crew` | `1g` | Same profile as trading-crew |
| `supervisor-crew` | `512m` | Lighter workload -- monitoring and scheduling only, uses haiku-tier LLM |

**Total: ~9.5 GB**, leaving ~6.5 GB on a 16 GB machine for macOS, Docker Desktop overhead, and developer tools.

Each crew service should also set `memswap_limit` equal to `mem_limit` (disables swap) so OOM behavior is deterministic rather than degrading silently into swap thrashing.

### 14.2 Log Rotation Configuration

Add a top-level `x-logging` YAML anchor and reference it from every service. This avoids repeating the logging block 8 times.

```yaml
x-logging: &default-logging
  driver: json-file
  options:
    max-size: "50m"
    max-file: "5"
```

Then in each service:

```yaml
services:
  postgres:
    logging: *default-logging
    # ... rest of service config
```

**Disk budget:** 8 services x 5 files x 50 MB = 2 GB maximum log footprint. In practice, infrastructure services (postgres, chromadb) produce far less than 50 MB per file, so actual usage will be well under 1 GB.

### 14.3 Langfuse Trace Retention

Langfuse's self-hosted edition does not have a built-in retention policy setting (as of the version pinned in section 01). Implement a cleanup function that the supervisor crew calls on a daily schedule.

**Target file:** `src/quantstack/health/langfuse_retention.py`

The function should:

1. Connect to the `langfuse-db` PostgreSQL instance (connection string from `LANGFUSE_DATABASE_URL` env var).
2. Delete rows from the `traces` table where `created_at < now() - interval '30 days'`.
3. Also clean up dependent tables (`observations`, `scores`, `events`) that reference deleted traces, using cascading deletes or explicit multi-table cleanup in the correct FK order.
4. Log the number of rows deleted.
5. Be idempotent -- safe to call multiple times.

Function signature:

```python
def cleanup_langfuse_traces(retention_days: int = 30) -> int:
    """Delete Langfuse traces older than retention_days.

    Connects to the Langfuse Postgres instance and removes old trace data
    to prevent unbounded storage growth. Returns the number of traces deleted.

    Depends on LANGFUSE_DATABASE_URL env var pointing to the langfuse-db service.
    """
```

The supervisor crew's `scheduled_tasks` step (defined in section 05) should call this function once per day during the post-market window.

### 14.4 Cost Estimation Validation

No runtime code is needed for cost estimation -- this is a design-time validation. The numbers below should be referenced during the section 13 verification phase (48-hour dry run) to confirm actual costs match projections.

**LLM call volume model (market hours, 5-min trading cycles):**

| Crew | Cycles/Day | LLM Calls/Cycle | Daily Calls |
|------|-----------|-----------------|-------------|
| Trading | 78 | ~11 | ~858 |
| Research | 39 | ~8 | ~312 |
| Supervisor | 78 | ~3 | ~234 |
| **Total** | | | **~1,404** |

**Per-call token budget:** ~2,000 input tokens + ~500 output tokens (average across tiers).

**Daily cost by provider:**

| Provider | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) | Estimated Daily Cost |
|----------|---------------------------|----------------------------|---------------------|
| Bedrock Sonnet | $3.00 | $15.00 | $5-8 |
| Anthropic Direct | $3.00 | $15.00 | $5-8 |
| OpenAI GPT-4o | $2.50 | $10.00 | $4-6 |
| Gemini 2.5 Pro | $1.25 | $5.00 | $2-4 |

**Monthly range: $60-240** depending on provider, after-hours cycle reduction, and weekend pauses.

During the 48-hour verification phase (section 13), check Langfuse's cost dashboard against these projections. If daily cost exceeds $12 (1.5x the high estimate), investigate which agents are consuming the most tokens and whether backstory length or tool output verbosity can be reduced.

### 14.5 Applying Changes to Existing docker-compose.yml

The current `docker-compose.yml` at `/Users/kshitijbichave/Personal/Trader/docker-compose.yml` defines two services (`api` and `alpaca-mcp`) with no resource limits and no log rotation. When sections 01 and 10 are complete, this file will have all 8 services. At that point, this section's changes are:

1. Add the `x-logging` anchor at the top level.
2. Add `logging: *default-logging` to every service.
3. Add `mem_limit` and `memswap_limit` to every service with the values from the table in 14.1.
4. Add `LANGFUSE_DATABASE_URL` to the crew services' environment blocks (needed by the retention cleanup function).

---

## Verification Checklist

After implementation, confirm:

- [ ] `docker compose config` parses without errors (valid YAML after anchors expand).
- [ ] `docker compose up -d` starts all services; none are OOM-killed within 10 minutes of idle operation.
- [ ] `docker compose logs --tail=100 <service>` shows log output (driver is working).
- [ ] On the host, `ls -la /var/lib/docker/containers/*/` shows rotated log files (after generating enough output).
- [ ] `cleanup_langfuse_traces()` runs without error against an empty Langfuse DB (idempotent on zero rows).
- [ ] After 48-hour verification phase, Langfuse cost dashboard shows daily cost within the $2-12 range.
