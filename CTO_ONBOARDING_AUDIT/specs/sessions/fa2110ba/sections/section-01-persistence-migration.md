# Section 01: Persistence Migration (StrategyBreaker + ICAttributionTracker)

## Problem

Two critical stateful modules persist to JSON files on disk instead of PostgreSQL:

- **StrategyBreaker** (`src/quantstack/execution/strategy_breaker.py`) saves to `~/.quantstack/strategy_breakers.json`
- **ICAttributionTracker** (`src/quantstack/learning/ic_attribution.py`) saves to `~/.quantstack/ic_attribution.json`

This violates the project's hard rule ("DB writes use `db_conn()` context managers") and creates a concrete failure mode: when a Docker container restarts, the JSON files are lost. A TRIPPED strategy silently resumes trading at full size. IC attribution history vanishes, resetting all per-collector quality signals to cold-start defaults.

Both modules are otherwise well-structured. The migration is purely a persistence layer swap — no behavioral changes to the public APIs.

## Dependencies

None. This section has no upstream dependencies and can be implemented immediately.

**Blocks:** section-02 (ghost module audit) and section-03 (readpoint wiring) depend on this migration being complete, since those sections wire StrategyBreaker and ICAttributionTracker into live trading paths where persistence reliability is critical.

## New Database Tables

### `strategy_breaker_states`

Add to `src/quantstack/db.py` inside a new `_migrate_strategy_breaker_pg(conn)` function, called from `run_migrations_pg()`.

```sql
CREATE TABLE IF NOT EXISTS strategy_breaker_states (
    strategy_id   TEXT PRIMARY KEY,
    status        TEXT NOT NULL DEFAULT 'ACTIVE',
    scale_factor  FLOAT NOT NULL DEFAULT 1.0,
    consecutive_losses INTEGER NOT NULL DEFAULT 0,
    peak_equity   FLOAT NOT NULL DEFAULT 0.0,
    current_equity FLOAT NOT NULL DEFAULT 0.0,
    drawdown_pct  FLOAT NOT NULL DEFAULT 0.0,
    tripped_at    TIMESTAMPTZ,
    reason        TEXT NOT NULL DEFAULT '',
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Key design choices:
- `strategy_id` as primary key (one row per strategy, upserted on every state change)
- `updated_at` column for debugging and audit trail
- `TIMESTAMPTZ` for `tripped_at` to handle timezone-aware comparisons in cooldown logic
- No foreign key to `strategies` table — breaker states may exist for strategies not yet in the registry (defensive)

### `ic_attribution_data`

Add to `src/quantstack/db.py` inside a new `_migrate_ic_attribution_pg(conn)` function, called from `run_migrations_pg()`.

```sql
CREATE TABLE IF NOT EXISTS ic_attribution_data (
    id             SERIAL PRIMARY KEY,
    collector      TEXT NOT NULL,
    signal_value   FLOAT NOT NULL,
    forward_return FLOAT NOT NULL,
    recorded_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ic_attribution_collector
    ON ic_attribution_data (collector, recorded_at DESC);
```

Key design choices:
- Append-only table (one row per observation), matching the current list-of-observations model
- Index on `(collector, recorded_at DESC)` for efficient windowed queries ("last N observations for collector X")
- No `symbol` column in the table — the current `ICAttributionTracker.record()` accepts `symbol` but aggregates observations across symbols per collector. The symbol is used only for logging context. If per-symbol IC is needed later, add the column then. Keeping it out now avoids a premature schema commitment.
- `SERIAL` primary key for ordered retrieval matching the current list-append behavior

## Tests (Write First)

Create `tests/unit/test_persistence_migration.py`. All tests mock the database — no real DB calls in unit tests.

### StrategyBreaker PostgreSQL Migration Tests

```python
class TestStrategyBreakerPersistence:
    """Verify StrategyBreaker save/load round-trips through PostgreSQL."""

    def test_save_load_roundtrip(self):
        """BreakerState saved to DB can be loaded back with identical field values."""
        # Create a breaker, record trades to get it into SCALED state,
        # then create a new StrategyBreaker instance and verify it loads
        # the same state from DB.
        ...

    def test_tripped_state_survives_restart(self):
        """TRIPPED state persists across simulated container restart.

        This is the critical safety property: a new StrategyBreaker instance
        (simulating process restart) must see the TRIPPED status and return
        scale_factor=0.0, not silently reset to ACTIVE.
        """
        ...

    def test_concurrent_reads_do_not_block(self):
        """Multiple get_scale_factor() calls execute without deadlock.

        Simulated with sequential calls (true concurrency tested in integration).
        Verifies the DB queries use no exclusive locks on reads.
        """
        ...

    def test_persist_failure_does_not_crash(self):
        """If DB write fails, the breaker logs an error but does not raise.

        Trading must continue even if persistence is temporarily broken.
        The in-memory state remains correct; next successful persist recovers.
        """
        ...

    def test_load_from_empty_db(self):
        """Fresh database with no rows returns empty state dict (clean start)."""
        ...

    def test_invalid_status_in_db_resets_to_active(self):
        """A corrupted status value in DB is treated as ACTIVE with a warning log."""
        ...
```

### ICAttributionTracker PostgreSQL Migration Tests

```python
class TestICAttributionPersistence:
    """Verify ICAttributionTracker save/load round-trips through PostgreSQL."""

    def test_save_load_roundtrip(self):
        """Observations recorded via record() are retrievable after
        creating a new tracker instance (simulating restart)."""
        ...

    def test_data_persists_across_restart(self):
        """IC computation on a fresh instance matches the original instance
        when both have the same observation history from DB."""
        ...

    def test_window_truncation_in_db(self):
        """Only the most recent 2 * window_size observations are kept per
        collector, matching the current in-memory truncation behavior."""
        ...

    def test_persist_failure_logged_not_raised(self):
        """DB write failure is logged but does not propagate to caller."""
        ...

    def test_load_from_empty_db(self):
        """Empty ic_attribution_data table returns empty collector state."""
        ...
```

## Implementation Details

### StrategyBreaker Changes (`src/quantstack/execution/strategy_breaker.py`)

**What changes:**
1. Replace `_persist()` — instead of writing JSON, upsert all states to `strategy_breaker_states` via `db_conn()`
2. Replace `_load()` — instead of reading JSON, query `strategy_breaker_states` via `db_conn()`
3. Remove `_state_path` and `_DEFAULT_STATE_PATH` — no more file path config
4. Remove the `state_path` constructor parameter (or keep it but ignore it with a deprecation warning)
5. Add `from quantstack.db import db_conn` import

**`_persist()` replacement sketch:**

```python
def _persist(self) -> None:
    """Upsert all breaker states to PostgreSQL. Called under the lock."""
    try:
        with db_conn() as conn:
            for strategy_id, state in self._states.items():
                conn.execute(
                    """
                    INSERT INTO strategy_breaker_states
                        (strategy_id, status, scale_factor, consecutive_losses,
                         peak_equity, current_equity, drawdown_pct, tripped_at,
                         reason, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (strategy_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        scale_factor = EXCLUDED.scale_factor,
                        consecutive_losses = EXCLUDED.consecutive_losses,
                        peak_equity = EXCLUDED.peak_equity,
                        current_equity = EXCLUDED.current_equity,
                        drawdown_pct = EXCLUDED.drawdown_pct,
                        tripped_at = EXCLUDED.tripped_at,
                        reason = EXCLUDED.reason,
                        updated_at = NOW()
                    """,
                    [strategy_id, state.status, state.scale_factor,
                     state.consecutive_losses, state.peak_equity,
                     state.current_equity, state.drawdown_pct,
                     state.tripped_at, state.reason],
                )
    except Exception as exc:
        logger.error(
            f"[BREAKER] Failed to persist state to DB: {exc}"
        )
```

**`_load()` replacement sketch:**

```python
def _load(self) -> None:
    """Load breaker states from PostgreSQL on startup."""
    try:
        with db_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM strategy_breaker_states"
            ).fetchall()

        for row in rows:
            status = row["status"]
            if status not in _VALID_STATUSES:
                logger.warning(
                    f"[BREAKER] Invalid status '{status}' for "
                    f"{row['strategy_id']} in DB — resetting to ACTIVE"
                )
                status = STATUS_ACTIVE

            self._states[row["strategy_id"]] = BreakerState(
                strategy_id=row["strategy_id"],
                status=status,
                consecutive_losses=row["consecutive_losses"],
                peak_equity=row["peak_equity"],
                current_equity=row["current_equity"],
                drawdown_pct=row["drawdown_pct"],
                tripped_at=row["tripped_at"],
                scale_factor=row["scale_factor"],
                reason=row["reason"],
            )

        logger.info(
            f"[BREAKER] Loaded {len(self._states)} strategy states from DB"
        )

        # Log strategies that are tripped or scaled from a previous session
        for sid, state in self._states.items():
            if state.status == STATUS_TRIPPED:
                logger.warning(
                    f"[BREAKER] {sid} is TRIPPED from previous session | "
                    f"reason={state.reason} | tripped_at={state.tripped_at}"
                )
            elif state.status == STATUS_SCALED:
                logger.warning(
                    f"[BREAKER] {sid} is SCALED from previous session | "
                    f"reason={state.reason}"
                )

    except Exception as exc:
        logger.error(f"[BREAKER] Failed to load state from DB: {exc}")
```

**What stays the same:**
- The `_lock` (RLock) for thread safety — DB calls happen under the lock, same as file I/O did
- The in-memory `_states` dict as the primary data structure — DB is the persistence layer, not the query layer for hot-path calls like `get_scale_factor()`
- All public API signatures: `record_trade()`, `check()`, `get_scale_factor()`, `reset()`, `force_trip()`, `force_scale()`, `get_all_states()`
- The `BreakerState` and `BreakerConfig` dataclasses
- The `_evaluate_thresholds()` logic
- The `_copy_state()` defensive copy pattern

**Constructor change:**

```python
def __init__(
    self,
    config: BreakerConfig | None = None,
):
    self._config = config or BreakerConfig()
    self._lock = RLock()
    self._states: dict[str, BreakerState] = {}
    self._load()
    logger.info(
        f"StrategyBreaker initialized | strategies_tracked={len(self._states)} "
        f"| max_dd={self._config.max_drawdown_pct}% "
        f"| consec_loss_limit={self._config.consecutive_loss_limit}"
    )
```

### ICAttributionTracker Changes (`src/quantstack/learning/ic_attribution.py`)

**What changes:**
1. Replace `_persist()` — instead of writing JSON, insert observations to `ic_attribution_data` via `db_conn()`
2. Replace `_load()` — instead of reading JSON, query `ic_attribution_data` grouped by collector
3. Remove `_state_path`, `_DEFAULT_STATE_PATH`, and the `state_path` constructor parameter
4. Add `from quantstack.db import db_conn` import
5. Remove `import json` (no longer needed)

**`_persist()` replacement approach:**

The current `_persist()` writes the entire collector state on every `record()` call. For DB persistence, this is wasteful. Instead, change the strategy:
- On `record()`: insert the single new observation row (not the entire state)
- On window truncation: delete old rows beyond `2 * window_size` per collector

```python
def _persist_observation(
    self, collector: str, signal_value: float, forward_return: float, timestamp: str
) -> None:
    """Insert a single observation to PostgreSQL. Called under lock."""
    try:
        with db_conn() as conn:
            conn.execute(
                """
                INSERT INTO ic_attribution_data
                    (collector, signal_value, forward_return, recorded_at)
                VALUES (%s, %s, %s, %s)
                """,
                [collector, signal_value, forward_return, timestamp],
            )
    except Exception as exc:
        logger.warning(f"[ICAttribution] Failed to persist observation: {exc}")

def _truncate_old_observations(self, collector: str, max_keep: int) -> None:
    """Remove observations beyond the retention window for a collector."""
    try:
        with db_conn() as conn:
            conn.execute(
                """
                DELETE FROM ic_attribution_data
                WHERE collector = %s
                  AND id NOT IN (
                      SELECT id FROM ic_attribution_data
                      WHERE collector = %s
                      ORDER BY recorded_at DESC
                      LIMIT %s
                  )
                """,
                [collector, collector, max_keep],
            )
    except Exception as exc:
        logger.warning(f"[ICAttribution] Failed to truncate old observations: {exc}")
```

**`_load()` replacement sketch:**

```python
def _load(self) -> None:
    """Load persisted observation state from PostgreSQL."""
    try:
        with db_conn() as conn:
            rows = conn.execute(
                """
                SELECT collector, signal_value, forward_return, recorded_at
                FROM ic_attribution_data
                ORDER BY collector, recorded_at ASC
                """
            ).fetchall()

        for row in rows:
            collector = row["collector"]
            state = self._collectors.setdefault(collector, _CollectorState())
            state.observations.append(
                _Observation(
                    signal_value=row["signal_value"],
                    forward_return=row["forward_return"],
                    timestamp=row["recorded_at"].isoformat()
                        if hasattr(row["recorded_at"], "isoformat")
                        else str(row["recorded_at"]),
                )
            )

        logger.info(
            f"[ICAttribution] Loaded state for {len(self._collectors)} collectors from DB"
        )
    except Exception as exc:
        logger.warning(f"[ICAttribution] Failed to load state from DB: {exc}")
```

**What stays the same:**
- The `_lock` for thread safety
- The in-memory `_collectors` dict and `_CollectorState` / `_Observation` dataclasses
- All public API signatures: `record()`, `get_collector_ic()`, `get_report()`, `get_weights()`
- The `_compute_trend()` and `_compute_ic_for_observations()` methods
- The `_classify_status()` helper function
- The `CollectorIC`, `ICAttributionReport` dataclasses

**`record()` method update:**

Replace the `self._persist()` call at the end of `record()` with `self._persist_observation(collector, signal_value, forward_return, ts)`. Replace the list truncation with a call to `self._truncate_old_observations(collector, max_keep)` when the in-memory list exceeds `max_keep`.

### Migration Registration in `db.py`

Add two new migration function calls inside `run_migrations_pg()`:

```python
_migrate_strategy_breaker_pg(conn)
_migrate_ic_attribution_pg(conn)
```

Place them after the existing migration calls (order does not matter since they are independent tables with no foreign keys). Follow the existing pattern: each `_migrate_*_pg` function takes a `PgConnection`, gets the raw cursor, and executes `CREATE TABLE IF NOT EXISTS` in autocommit mode.

## Rollback

Revert the file changes via git. The JSON persistence code is restored. The new DB tables are inert (no consumers) and can be dropped at leisure:

```sql
DROP TABLE IF EXISTS strategy_breaker_states;
DROP TABLE IF EXISTS ic_attribution_data;
```

## Verification Checklist

1. All existing StrategyBreaker unit tests pass unchanged (public API is identical)
2. All existing ICAttributionTracker unit tests pass unchanged
3. New persistence round-trip tests pass
4. TRIPPED state survives simulated restart (the critical safety property)
5. `get_scale_factor()` hot-path performance is unaffected (reads from in-memory dict, not DB)
6. No references to `~/.quantstack/strategy_breakers.json` or `~/.quantstack/ic_attribution.json` remain in the codebase
7. `run_migrations_pg()` creates both new tables without error on a fresh database
