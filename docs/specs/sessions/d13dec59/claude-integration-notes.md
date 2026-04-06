# Integration Notes — Opus Review Feedback

## What I'm Integrating

### 1. CRITICAL: Package naming conflict (Integrating)
The existing `src/quantstack/dashboard/` contains a FastAPI web dashboard and `events.py` (imported by `agent_executor.py`). The plan must NOT overwrite this package.

**Decision:** Place the Textual TUI at `src/quantstack/tui/` as a separate package. The existing `dashboard/` package stays untouched.

### 2. Thread worker UI update pattern (Integrating)
The plan needs to specify how `@work(thread=True)` workers push data back to widgets. Will define a `RefreshableWidget` base pattern in Section 1.

### 3. asyncio.Semaphore incompatibility (Integrating)
Replace `asyncio.Semaphore(5)` with `threading.Semaphore(5)` for thread-based workers. Or use Textual worker groups.

### 4. Connection pool size correction (Integrating)
Fix: pool default is 20, not 10. Update analysis accordingly.

### 5. Use PgConnection API (Integrating)
Queries will use `PgConnection.execute()` / `.fetchall()` instead of raw cursors.

### 6. Port existing v1 queries (Integrating)
Call out which of the 45 queries already exist in `scripts/dashboard.py` and should be ported.

### 7. Add Decisions widget (Integrating)
The spec lists `widgets/decisions.py` but the plan omitted it. Adding as part of Overview/Research tabs.

### 8. Separate .tcss file (Integrating)
Use a dedicated `dashboard.tcss` file instead of inline DEFAULT_CSS.

### 9. market_holidays idempotent seeding (Integrating)
Add ON CONFLICT DO NOTHING to holiday inserts.

### 10. Docker health check fallback (Integrating)
Note that `docker compose ps` only works locally. Add TCP port check fallback.

## What I'm NOT Integrating

### 1. "Validate 24-line fit" relaxation
The Overview tab design is already tight at ~20 lines of content. Rather than relaxing the constraint, I'll note it needs prototyping but keep the target. Textual's scrollable containers handle overflow gracefully anyway.

### 2. Query prioritization / phased delivery
The user explicitly chose "all at once" rollout. Not splitting into query phases.

### 3. benchmark_daily population question
The plan already says "migration helper seeds from ohlcv" — this is the approach. No pipeline dependency needed.
