# Section 09: Memory Temporal Decay

**Plan ref:** Item 5.8
**Dependencies:** None (parallelizable in Batch 1)
**Blocks:** Nothing

---

## Background

Memory entries in both PostgreSQL (`agent_memory` table) and `.claude/memory/` markdown files accumulate indefinitely. When injected into agent context, stale entries waste tokens and dilute relevance. Example: the 2026-04-04 EWF market reads flagged as "low-trust" still persist and consume context alongside fresh, actionable data.

The goal is to add temporal decay weighting so that recent entries rank higher in retrieval, and a pruning mechanism that archives entries past their useful lifetime rather than deleting them (recoverable, audit-friendly).

There are two independent surfaces to address:

1. **PostgreSQL `agent_memory` table** -- used by agents at runtime via `Blackboard` class in `src/quantstack/memory/blackboard.py`
2. **Markdown files in `.claude/memory/`** -- used by the Claude Code session layer via the `compact-memory` skill in `.claude/skills/compact-memory/SKILL.md`

---

## Tests (Write First)

All tests go in a new file: `tests/unit/test_memory_decay.py`

```python
# tests/unit/test_memory_decay.py

import pytest
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Temporal decay weighting
# ---------------------------------------------------------------------------

# Test: entry created today has weight ~1.0
# Setup: insert a row with created_at = now(), call the decay weight function
# Assert: weight >= 0.95

# Test: entry created 14 days ago with half_life=14 has weight ~0.5
# Setup: insert row with created_at = now() - 14 days, half_life=14
# Assert: 0.45 <= weight <= 0.55

# Test: entry created 28 days ago with half_life=14 has weight ~0.25
# Setup: insert row with created_at = now() - 28 days, half_life=14
# Assert: 0.20 <= weight <= 0.30

# Test: different categories use different half-lives
# Setup: insert two rows with same created_at (21 days ago), one category="market_regime"
#        (half_life=7), one category="research_finding" (half_life=90)
# Assert: market_regime weight << research_finding weight

# Test: decay weighting changes result ordering (recent != most relevant)
# Setup: insert an old research_finding (90-day half-life, 20 days old) and a
#        recent market_regime (7-day half-life, 5 days old)
# Assert: with decay, the research_finding ranks higher despite being older
#         (because its half-life is longer, so decay_weight is higher)

# ---------------------------------------------------------------------------
# Archival
# ---------------------------------------------------------------------------

# Test: entries older than 3x half_life are archived
# Setup: insert a trade_outcome (half_life=14) created 43 days ago
# Run: the pruning function
# Assert: row no longer in agent_memory, present in agent_memory_archive

# Test: archived entries moved to agent_memory_archive, not deleted
# Setup: same as above
# Assert: SELECT COUNT(*) FROM agent_memory_archive WHERE id = <id> returns 1

# Test: archived entries have archived_at timestamp set
# Assert: archived_at IS NOT NULL and is approximately now()

# Test: active entries (within TTL) are not archived
# Setup: insert a trade_outcome created 10 days ago (within 3 * 14 = 42 day threshold)
# Run: the pruning function
# Assert: row still in agent_memory, not in agent_memory_archive

# ---------------------------------------------------------------------------
# last_accessed_at tracking
# ---------------------------------------------------------------------------

# Test: read_recent updates last_accessed_at for returned rows
# Setup: insert a row, note its last_accessed_at (should be DEFAULT NOW())
# Wait a moment, call read_recent for that symbol
# Assert: last_accessed_at is updated to approximately now()

# Test: read_as_context updates last_accessed_at for returned rows
# Setup: same pattern as above using read_as_context
# Assert: last_accessed_at updated

# Test: entries not accessed in 60+ days are flagged for archival
# Setup: insert row with last_accessed_at = now() - 65 days, created_at = now() - 20 days
# Run: the pruning function (which checks BOTH created_at decay AND last_accessed_at staleness)
# Assert: row is archived despite being within the created_at TTL

# ---------------------------------------------------------------------------
# SQL query correctness
# ---------------------------------------------------------------------------

# Test: decay weight computed in SQL matches Python formula
# Formula: POW(0.5, age_in_days / half_life)
# Setup: insert rows with known ages, retrieve with decay weights
# Assert: SQL-computed weight matches Python math.pow(0.5, age / half_life) within 0.01

# Test: LIMIT applied after decay weighting (not before)
# Setup: insert 20 rows with varied ages, request limit=5
# Assert: the 5 returned rows are the top-5 by decay_weight, not just 5 arbitrary rows

# ---------------------------------------------------------------------------
# Markdown memory (compact-memory skill)
# ---------------------------------------------------------------------------

# Test: compact-memory skill identifies entries past TTL
# (This is a behavioral test of the enhanced skill logic)
# Setup: create a temp .claude/memory/ directory with a file containing entries
#        with dates older than TTL
# Assert: the identification function returns those entries

# Test: expired entries archived to .archive.md file
# Setup: run the archive function on a memory file with expired entries
# Assert: <filename>.archive.md exists and contains the archived entries
# Assert: original file no longer contains the archived entries

# Test: permanent entries (workshop_lessons) never archived
# Setup: workshop_lessons.md with entries older than any TTL
# Run: the archive function
# Assert: no entries moved, no .archive.md created
```

---

## Implementation Details

### Part 1: Schema Migration

**File:** `src/quantstack/db.py` -- modify `_migrate_memory_pg()`

The existing `agent_memory` table schema:

```
agent_memory (
    id          BIGINT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    sim_date    DATE,
    agent       TEXT NOT NULL,
    symbol      TEXT DEFAULT '',
    category    TEXT DEFAULT 'general',
    content_json TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
)
```

Add two columns to `_migrate_memory_pg()` using `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` (idempotent, consistent with all other migrations in this file):

- `last_accessed_at TIMESTAMPTZ DEFAULT NOW()` -- updated on every read
- `archived_at TIMESTAMPTZ DEFAULT NULL` -- set when entry is archived

Create the archive table in the same migration function:

```python
def _migrate_memory_pg(conn: PgConnection) -> None:
    # ... existing CREATE TABLE and CREATE INDEX statements ...

    # New: temporal decay columns
    conn.execute("""
        ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ DEFAULT NOW()
    """)
    conn.execute("""
        ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ DEFAULT NULL
    """)

    # New: archive table (identical schema to agent_memory)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS agent_memory_archive_seq START 1")
    conn.execute(_to_pg("""
        CREATE TABLE IF NOT EXISTS agent_memory_archive (
            id              BIGINT PRIMARY KEY,
            session_id      TEXT NOT NULL,
            sim_date        DATE,
            agent           TEXT NOT NULL,
            symbol          TEXT DEFAULT '',
            category        TEXT DEFAULT 'general',
            content_json    TEXT NOT NULL,
            created_at      TIMESTAMPTZ,
            last_accessed_at TIMESTAMPTZ,
            archived_at     TIMESTAMPTZ DEFAULT NOW()
        )
    """))

    # Index for archive queries (by category for reporting)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS memory_archive_category_idx
        ON agent_memory_archive (category, archived_at DESC)
    """)
```

The archive table intentionally uses the same `id` as the source row (not a new sequence) to maintain traceability.

---

### Part 2: Half-Life Configuration

**File:** `src/quantstack/memory/blackboard.py` -- add a module-level constant

Define the category-to-half-life mapping:

```python
CATEGORY_HALF_LIFE_DAYS: dict[str, int] = {
    "trade_outcome": 14,    # Recent trades most relevant
    "strategy_param": 30,   # Parameters evolve slowly
    "market_regime": 7,     # Regimes shift quickly
    "research_finding": 90, # Foundational knowledge
    "general": 30,          # Default
}

DEFAULT_HALF_LIFE_DAYS = 30
```

This mapping is used in both the SQL decay weight calculation and the pruning threshold (entries older than `3 * half_life` are archived).

---

### Part 3: Modify Read Path with Temporal Decay

**File:** `src/quantstack/memory/blackboard.py` -- modify `read_recent()` and `read_as_context()`

The current `read_recent()` method orders by `created_at DESC` and applies `LIMIT`. This returns the N most recent entries regardless of relevance decay.

Replace with a decay-weighted query. The SQL computes `POW(0.5, age_in_days / half_life)` per row using a CASE expression to look up the half-life by category:

```sql
SELECT *, POW(0.5,
    EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 /
    CASE category
        WHEN 'trade_outcome' THEN 14
        WHEN 'strategy_param' THEN 30
        WHEN 'market_regime' THEN 7
        WHEN 'research_finding' THEN 90
        ELSE 30
    END
) AS decay_weight
FROM agent_memory
WHERE archived_at IS NULL
  AND <existing filters>
ORDER BY decay_weight DESC
LIMIT :limit
```

Key design decisions:
- The `WHERE archived_at IS NULL` filter excludes archived entries from all reads
- The CASE expression keeps the half-life mapping in SQL so the DB can compute weights and apply LIMIT in one pass (no over-fetch)
- Ordering by `decay_weight DESC` means a 1-day-old market_regime entry (weight ~0.91) ranks above a 10-day-old market_regime entry (weight ~0.37), but a 20-day-old research_finding (weight ~0.86) still ranks high

After fetching rows, update `last_accessed_at` for the returned IDs:

```sql
UPDATE agent_memory SET last_accessed_at = NOW() WHERE id = ANY(:ids)
```

This runs as a separate statement after the SELECT. It does not need to be in the same transaction -- eventual consistency on `last_accessed_at` is acceptable.

The `read_as_context()` method delegates to `read_recent()`, so it inherits the decay weighting and access tracking automatically.

---

### Part 4: Weekly Pruning Job

**File:** `src/quantstack/memory/blackboard.py` -- add a new method to `Blackboard`

Add an `archive_stale()` method that the Supervisor graph's scheduled tasks node calls weekly:

```python
def archive_stale(self) -> dict[str, int]:
    """
    Archive entries past their TTL (3x half-life).

    Returns a dict of {category: count_archived} for logging.
    """
```

The method does two things in sequence:

1. **INSERT into archive:** Copy qualifying rows from `agent_memory` to `agent_memory_archive`
2. **DELETE from agent_memory:** Remove the copied rows

The qualification criteria: `created_at < NOW() - INTERVAL '1 day' * (half_life * 3)` per category, OR `last_accessed_at < NOW() - INTERVAL '60 days'` (stale access).

Use a single SQL statement with a CTE for atomicity:

```sql
WITH archived AS (
    INSERT INTO agent_memory_archive
    SELECT id, session_id, sim_date, agent, symbol, category, content_json,
           created_at, last_accessed_at, NOW() as archived_at
    FROM agent_memory
    WHERE archived_at IS NULL
      AND (
          (category = 'trade_outcome'    AND created_at < NOW() - INTERVAL '42 days')
          OR (category = 'strategy_param'  AND created_at < NOW() - INTERVAL '90 days')
          OR (category = 'market_regime'   AND created_at < NOW() - INTERVAL '21 days')
          OR (category = 'research_finding' AND created_at < NOW() - INTERVAL '270 days')
          OR (category = 'general'         AND created_at < NOW() - INTERVAL '90 days')
          OR (last_accessed_at < NOW() - INTERVAL '60 days')
      )
    RETURNING id
)
DELETE FROM agent_memory WHERE id IN (SELECT id FROM archived)
```

Log the count per category after execution. The Supervisor graph calls this method from its `scheduled_tasks` node on a weekly cadence.

---

### Part 5: Supervisor Graph Integration

**File:** `src/quantstack/graphs/supervisor/` -- the scheduled tasks node

Add a memory pruning task to the Supervisor graph's weekly schedule. This is a deterministic node call (no LLM needed):

1. Instantiate `Blackboard` with the graph's DB connection
2. Call `archive_stale()`
3. Log the results as a Langfuse custom event (category counts, total archived)
4. If archival fails, log the error but do not halt the Supervisor cycle -- memory pruning is non-critical

---

### Part 6: Enhanced compact-memory Skill

**File:** `.claude/skills/compact-memory/SKILL.md` -- add a new step for TTL enforcement

Add a step between the existing per-file compaction steps (Steps 1-7) and the final commit (Step 8). This step enforces TTL-based archival on markdown memory files.

TTL rules by memory type:

| Memory type | TTL | Action |
|-------------|-----|--------|
| Market reads (EWF, regime) | 7 days | Archive to `*.archive.md` |
| Session handoffs | 30 days | Archive |
| Strategy states | Monthly validation | Flag for review, don't auto-archive |
| Workshop lessons | Permanent | Never archive |
| Validated principles | Permanent | Never archive |

The skill should:

1. Parse each `.claude/memory/*.md` file for entries with date metadata (either YAML frontmatter `created:` or inline date patterns like `[2026-04-04]`)
2. For entries past their TTL, move them to `<filename>.archive.md` in the same directory, prefixed with `[ARCHIVED <date>]`
3. Files in the "permanent" category (`workshop_lessons.md`, any file containing "validated principles") are skipped entirely
4. For files without date metadata, infer dates from git blame or fall back to file modification time

The archive files are recoverable -- entries can be moved back from `*.archive.md` to the source file if needed.

**Date metadata requirement:** Going forward, all entries written to `.claude/memory/` files should include a `created:` date in their YAML frontmatter or as an inline date marker. For existing files without dates, the skill should add dates inferred from git history on first run.

---

### Part 7: Integration with Context Compaction (Section 02)

After Section 02 (context compaction) is deployed, the temporal decay weighting in `read_as_context()` provides a smooth relevance degradation curve. Stale memories are naturally de-emphasized in retrieval results even before they reach their archival TTL. This means agents see a gradually fading signal rather than a hard cutoff -- entries don't suddenly disappear, they just rank lower over time.

---

## File Change Summary

| File | Change |
|------|--------|
| `src/quantstack/db.py` | Add `last_accessed_at`, `archived_at` columns to `agent_memory`; create `agent_memory_archive` table in `_migrate_memory_pg()` |
| `src/quantstack/memory/blackboard.py` | Add `CATEGORY_HALF_LIFE_DAYS` constant; modify `read_recent()` with decay-weighted SQL and `last_accessed_at` tracking; add `archive_stale()` method |
| `.claude/skills/compact-memory/SKILL.md` | Add TTL enforcement step for markdown memory files |
| `tests/unit/test_memory_decay.py` (new) | All tests listed above |
| Supervisor graph scheduled tasks node | Add weekly `archive_stale()` call |

---

## Rollback Plan

- **Schema columns:** `ALTER TABLE agent_memory DROP COLUMN last_accessed_at, DROP COLUMN archived_at` (data loss on those columns only)
- **Archive table:** `DROP TABLE agent_memory_archive` (archived rows are lost, but originals were already past TTL)
- **Read path:** Revert `read_recent()` to simple `ORDER BY created_at DESC` (one function change)
- **Pruning job:** Remove the `archive_stale()` call from Supervisor scheduled tasks
- **Markdown TTL:** Remove the TTL step from the compact-memory skill; `.archive.md` files remain on disk for manual recovery

---

## Observability

- **Memory store size:** Log total row count in `agent_memory` as a Langfuse metric (tracked by the pruning job)
- **Archival counts:** Log rows archived per category per week
- **Average memory age:** Log the average `created_at` age of active entries to track whether the working set is staying fresh
- **Access patterns:** The `last_accessed_at` column enables future queries like "which categories are read most often" and "which entries have never been accessed"
