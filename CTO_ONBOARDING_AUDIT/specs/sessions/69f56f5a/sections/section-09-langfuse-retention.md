# Section 9: Langfuse Retention (Config Stub)

## Background

QuantStack uses Langfuse for observability tracing across all three LangGraph graphs (trading, research, supervisor). Every node, LLM call, and tool invocation is traced via callback handlers. The `langfuse-db` service runs with a 256MB memory limit. With 3 graphs producing traces every 5-10 minutes, the Langfuse database will fill disk within weeks if old traces are never purged.

However, the stakeholder decision is to defer actual cleanup logic. This section wires the configuration and scheduler plumbing only -- no trace deletion is implemented. When the owner is ready to enable retention cleanup, they flip a flag and the scheduler job is already in place, waiting for the deletion logic to be filled in.

This section has no dependencies on other sections and can be implemented in parallel with anything else.

## Tests First

All tests go in `tests/unit/test_langfuse_retention.py`. These validate the configuration defaults, the scheduler job existence, and the stub behavior.

```python
# tests/unit/test_langfuse_retention.py

import os
import pytest


class TestLangfuseRetentionConfig:
    """Verify env var defaults and validation for Langfuse retention settings."""

    def test_retention_enabled_defaults_to_false(self, monkeypatch):
        """LANGFUSE_RETENTION_ENABLED must default to 'false' when unset.

        The stub is intentionally disabled by default -- enabling cleanup
        requires an explicit opt-in to prevent accidental trace deletion.
        """
        monkeypatch.delenv("LANGFUSE_RETENTION_ENABLED", raising=False)
        # Import or call the config reader, assert it returns False

    def test_retention_days_defaults_to_30(self, monkeypatch):
        """LANGFUSE_RETENTION_DAYS must default to 30 when unset.

        30 days matches the Loki log retention period configured in
        section-05 (log aggregation), keeping observability data aligned.
        """
        monkeypatch.delenv("LANGFUSE_RETENTION_DAYS", raising=False)
        # Import or call the config reader, assert it returns 30


class TestLangfuseRetentionSchedulerJob:
    """Verify the scheduler job stub exists and behaves correctly."""

    def test_scheduler_job_exists(self):
        """A 'langfuse_retention_cleanup' entry must exist in the JOBS list
        in scripts/scheduler.py, or a callable function must be importable.
        """
        # Verify the job function is defined and callable

    def test_job_logs_disabled_when_flag_false(self, monkeypatch, caplog):
        """When LANGFUSE_RETENTION_ENABLED=false, the job must log a message
        indicating cleanup is disabled and return without doing anything.

        Expected log message (or substring):
        'Langfuse retention cleanup is disabled'
        """
        monkeypatch.setenv("LANGFUSE_RETENTION_ENABLED", "false")
        # Call the job function, assert log contains 'disabled'

    def test_job_logs_would_delete_when_flag_true(self, monkeypatch, caplog):
        """When LANGFUSE_RETENTION_ENABLED=true, the job must log a message
        indicating what it *would* delete (placeholder -- no actual deletion).

        Expected log message (or substring):
        'would delete traces older than 30 days'
        """
        monkeypatch.setenv("LANGFUSE_RETENTION_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_RETENTION_DAYS", "30")
        # Call the job function, assert log contains 'would delete'

    def test_job_respects_custom_retention_days(self, monkeypatch, caplog):
        """When LANGFUSE_RETENTION_DAYS=14, the 'would delete' message
        must reflect 14 days, not the default 30.
        """
        monkeypatch.setenv("LANGFUSE_RETENTION_ENABLED", "true")
        monkeypatch.setenv("LANGFUSE_RETENTION_DAYS", "14")
        # Call the job function, assert log contains '14 days'
```

## Implementation

### 9.1 Add env vars to `.env.example`

**File:** `.env.example`

Add the following block (near the bottom, with the other optional operational settings):

```
# Langfuse trace retention (stub -- actual cleanup not yet implemented)
# Set LANGFUSE_RETENTION_ENABLED=true when ready to enable scheduled cleanup.
# LANGFUSE_RETENTION_DAYS controls how many days of traces to keep.
LANGFUSE_RETENTION_ENABLED=false    # boolean: true/false (default: false)
LANGFUSE_RETENTION_DAYS=30          # positive integer (default: 30)
```

These two variables are optional. The system must function correctly when they are absent (falling back to the defaults: `false` and `30`).

### 9.2 Add scheduler job function to `scripts/scheduler.py`

**File:** `scripts/scheduler.py`

Add a new job function `run_langfuse_retention_cleanup` following the same pattern as every other job in the file (accepts `dry_run: bool = False`, uses the `label`/`timestamp`/`logger` convention).

The function logic is:

1. Read `LANGFUSE_RETENTION_ENABLED` from `os.environ`, default to `"false"`.
2. If not `"true"` (case-sensitive match on `"true"`): log an INFO message -- `"Langfuse retention cleanup is disabled. Set LANGFUSE_RETENTION_ENABLED=true to enable."` -- and return.
3. Read `LANGFUSE_RETENTION_DAYS` from `os.environ`, default to `"30"`. Parse as int.
4. Log an INFO message: `"Langfuse retention cleanup: would delete traces older than {days} days (implementation pending)"`.
5. Return. No database calls. No Langfuse API calls. No deletion.

Function signature:

```python
def run_langfuse_retention_cleanup(dry_run: bool = False) -> None:
    """Langfuse trace retention stub -- config wiring only.

    When LANGFUSE_RETENTION_ENABLED=true, logs what it would delete.
    Actual deletion logic is deferred until the owner opts in and
    the Langfuse cleanup API or direct DB pruning is implemented.

    Schedule: Sunday 02:00 ET (weekly).
    """
```

### 9.3 Register the job in the JOBS list

**File:** `scripts/scheduler.py`

Add an entry to the `JOBS` list (in the weekly section, between existing Sunday jobs):

```python
# Langfuse trace retention stub — wiring only, cleanup logic deferred.
{"trigger": {"hour": 2, "minute": 0, "day_of_week": "sun"}, "func": run_langfuse_retention_cleanup, "label": "langfuse_retention_cleanup_sun02:00"},
```

Schedule rationale: Sunday 02:00 ET is a quiet period -- no other jobs compete, and it runs well before the Sunday 17:00-20:00 research/lifecycle block.

### 9.4 Register in the `--run-now` CLI map

**File:** `scripts/scheduler.py`

Add to the `func_map` dictionary inside the `main()` function:

```python
"langfuse_retention_cleanup": run_langfuse_retention_cleanup,
```

This allows manual invocation via `python scripts/scheduler.py --run-now langfuse_retention_cleanup`.

### 9.5 Update the startup banner

**File:** `scripts/scheduler.py`

Add a line to the `start_scheduler()` print block in the weekly section:

```
  02:00 Sun          — Langfuse retention cleanup (stub, disabled by default)
```

## Env Var Validation Integration

If section-01 (env validation) is implemented, `LANGFUSE_RETENTION_ENABLED` should be validated as a boolean (`true`/`false` only) and `LANGFUSE_RETENTION_DAYS` as a positive integer. Both are optional with defaults. This is a cross-section dependency -- the validation module in `src/quantstack/config/validation.py` handles it. If section-01 is not yet complete, the scheduler job reads the raw env var strings directly, which is safe since the job is a no-op stub.

## What This Section Does NOT Do

- No Langfuse API calls. No database queries against `langfuse-db`.
- No trace deletion logic. The `would delete` log message is the entire implementation.
- No changes to `docker-compose.yml`, Langfuse configuration, or any graph code.
- No new Python dependencies.

The purpose is purely to wire the plumbing so that when someone is ready to implement actual cleanup, the config, schedule, and CLI entry point are already in place. They only need to replace the log statement with real deletion logic.

## Verification

After implementation, verify with:

```bash
# Check that the job appears in the schedule
python scripts/scheduler.py --dry-run | grep langfuse

# Run the job manually (should log "disabled" message)
python scripts/scheduler.py --run-now langfuse_retention_cleanup

# Run with flag enabled (should log "would delete" message)
LANGFUSE_RETENTION_ENABLED=true python scripts/scheduler.py --run-now langfuse_retention_cleanup

# Run tests
pytest tests/unit/test_langfuse_retention.py -v
```
