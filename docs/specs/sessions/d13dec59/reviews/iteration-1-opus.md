# Opus Review

**Model:** claude-opus-4-6
**Generated:** 2026-04-03T17:00:00Z

---

## Strengths

1. **Clear structure and section decomposition.** The plan breaks a large feature (45 queries, 6 tabs, 13 sections) into 13 numbered sections with explicit dependency ordering.

2. **Solid query layer design.** Separating queries into per-domain files with typed dataclass returns and uniform error handling is the right abstraction.

3. **Tiered refresh with visibility awareness.** The T1-T4 tier strategy with staggered starts and tab-visibility filtering is well thought out.

4. **Graceful degradation is first-class.** Every query returns a safe default on failure.

5. **Good use of existing infrastructure.** Reusing `pg_conn()` and the existing connection pool rather than introducing asyncpg.

6. **Chart renderers are self-contained.** Custom Unicode rendering with zero external dependencies.

---

## Critical Issues

### 1. Plan overwrites existing `src/quantstack/dashboard/` package

The existing `src/quantstack/dashboard/` contains:
- **`app.py`** -- A FastAPI web dashboard (SSE-based agent event streaming) running as a Docker service on port 8421
- **`events.py`** -- `publish_event()` imported by `agent_executor.py` (production code)

Creating `src/quantstack/dashboard/app.py` with Textual would destroy the FastAPI dashboard and break the agent executor's import.

**Resolution options:**
- (a) Place Textual dashboard at `src/quantstack/tui/`
- (b) Nest under `src/quantstack/dashboard/tui/`
- (c) Explicitly migrate FastAPI to `src/quantstack/dashboard/web/`

### 2. `@work(thread=True)` UI update pattern not specified

Workers in threads cannot directly mutate widget state. Must use `App.call_from_thread()` or post messages. The plan never details this critical pattern.

### 3. `market_holidays` seeding needs idempotency

Plan should specify `ON CONFLICT DO NOTHING` for holiday inserts and where in migration sequence.

### 4. Connection pool size is 20, not 10

Actual code: `maxconn = int(os.getenv("PG_POOL_MAX", "20"))`. Pool pressure analysis is based on incorrect data.

### 5. `asyncio.Semaphore(5)` incompatible with `@work(thread=True)`

Thread workers don't run in asyncio event loop. Need `threading.Semaphore` or Textual worker groups.

---

## Suggestions

1. **Define a base `DBWidget` pattern once** — codify the thread->UI data flow pattern in Section 1.

2. **Use `PgConnection` API, not raw cursors** — `pg_conn()` yields `PgConnection` with retry logic. Don't bypass it.

3. **Port existing v1 queries** — Many of the 45 queries already exist in `scripts/dashboard.py`. Port, don't rewrite.

4. **Add Decisions widget** — Spec lists `widgets/decisions.py` but no plan section covers it.

5. **Use separate `.tcss` file** — For a dashboard this complex, inline CSS will be unmanageable.

6. **Validate 24-line fit** — Listed components may exceed 24 lines with Textual chrome. Relax to "scrollable if needed."

7. **Docker health check fallback** — `docker compose ps` via subprocess won't work inside Docker containers.

---

## Questions for the Author

1. What happens to the existing FastAPI dashboard? Is it replaced, kept alongside, or moved?
2. Is `events.py` (publish_event) being relocated? It's imported by agent_executor.py.
3. What is the target deployment context? Local only, Docker, or both?
4. How are the 45 queries prioritized? Could v1's existing 20 queries ship first?
5. How to mock DB in Textual pilot integration tests?
6. Is `benchmark_daily` populated by existing pipeline or dashboard seeds it?
