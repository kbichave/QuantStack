# Section 10: Dead Letter Queue

## Background

`parse_json_response()` in `src/quantstack/graphs/agent_executor.py` is the single point where every LLM response is parsed into a structured dict for downstream consumption. When parsing fails today, the function returns an empty `{}` fallback, logs a truncated 200-character debug message, and discards the raw LLM output entirely. There is no record of what the agent actually said, how often a given agent fails to produce parseable output, or whether a specific prompt variant is degrading over time.

This section adds a PostgreSQL-backed Dead Letter Queue (DLQ) that captures every parse failure with full context, plus Langfuse-based monitoring with alert thresholds. The DLQ is a diagnostic and observability tool only -- it does not trigger self-healing or automated prompt patching.

## Dependencies

- **section-01-db-migration-and-policy**: The `agent_dlq` table must exist before DLQ writes can succeed. The migration in section 01 creates this table.

No other sections are required. This section does not block any other section except section-14 (integration tests).

## Tests First

All tests go in `tests/unit/test_dead_letter_queue.py`.

```python
# --- DLQ Write Tests ---

# Test: unparseable LLM output -> DLQ row written with agent_name, raw_output, error_type
# Given: parse_json_response receives malformed JSON from agent "daily_planner"
# When: parsing fails
# Then: a row is inserted into agent_dlq with agent_name="daily_planner",
#        raw_output containing the full LLM response, error_type="parse_error"

# Test: DLQ row includes prompt_hash for clustering
# Given: a parse failure occurs
# When: the DLQ row is written
# Then: prompt_hash is a deterministic hash of the prompt text (e.g., SHA-256 truncated),
#        allowing grouping of failures by prompt variant

# Test: DLQ row includes model_used for debugging
# Given: a parse failure from model "claude-3-haiku"
# When: the DLQ row is written
# Then: model_used = "claude-3-haiku"

# Test: parse_json_response still returns fallback after DLQ write (behavior unchanged for caller)
# Given: parse_json_response is called with unparseable output
# When: the DLQ write succeeds (or fails)
# Then: the function still returns {} (or the existing fallback), not an exception
#        The caller's behavior is completely unchanged

# --- DLQ Rate Monitoring Tests ---

# Test: DLQ rate calculation -- 5 failures out of 100 calls = 5% rate
# Given: 100 total parse attempts for agent "daily_planner" in the last 24h,
#        5 of which resulted in DLQ entries
# When: DLQ rate is computed
# Then: rate = 5.0%

# Test: DLQ rate > 5% -> Langfuse warn event emitted
# Given: DLQ rate for agent "entry_scanner" is 6% over 24h rolling window
# When: the monitoring check runs
# Then: a Langfuse event is emitted with level="warn", agent_name, current rate

# Test: DLQ rate > 10% -> Langfuse critical event + outbound notification
# Given: DLQ rate for agent "entry_scanner" is 12% over 24h rolling window
# When: the monitoring check runs
# Then: a Langfuse event with level="critical" is emitted AND an outbound
#        notification (Slack webhook or email) is triggered with payload including
#        agent name, sample of failed messages, rate trend

# Test: DLQ rate query per agent over 24h rolling window
# Given: DLQ entries spanning 48 hours for multiple agents
# When: the rate query runs for a specific agent
# Then: only entries within the last 24h are counted; older entries are excluded
```

## Database Schema

The `agent_dlq` table is created by the migration in section-01. The schema for reference:

```
agent_dlq
  id              SERIAL PRIMARY KEY
  agent_name      TEXT NOT NULL
  graph_name      TEXT NOT NULL
  run_id          TEXT NOT NULL
  input_summary   TEXT              -- truncated input state for context
  raw_output      TEXT              -- the full unparsed LLM output
  error_type      TEXT              -- one of: parse_error, validation_error, timeout, business_rule
  error_detail    TEXT              -- exception message or description of what went wrong
  prompt_hash     TEXT              -- SHA-256 hash of prompt, truncated; clusters failures by prompt variant
  model_used      TEXT              -- LLM model identifier (e.g., "claude-3-haiku")
  created_at      TIMESTAMPTZ DEFAULT NOW()
  resolved_at     TIMESTAMPTZ       -- NULL until manually resolved
  resolution      TEXT              -- one of: manual_override, prompt_fixed, discarded
```

`error_type` values and when each applies:
- `parse_error`: LLM output is not valid JSON or does not match expected structure
- `validation_error`: JSON parses but fails Pydantic validation (future, once node output models exist)
- `timeout`: LLM call timed out before producing a response
- `business_rule`: output is structurally valid but violates a business constraint

## Implementation Details

### File: `src/quantstack/graphs/agent_executor.py`

**Modify `parse_json_response()`** to accept optional context kwargs and write to DLQ on failure.

Current signature (approximate):

```python
def parse_json_response(raw: str) -> dict:
    ...
```

New signature:

```python
def parse_json_response(
    raw: str,
    *,
    agent_name: str = "",
    graph_name: str = "",
    run_id: str = "",
    model_used: str = "",
    prompt_text: str = "",
) -> dict:
    ...
```

Key behaviors:
1. Attempt to parse `raw` as JSON (existing logic, unchanged).
2. On parse failure, before returning the `{}` fallback:
   - Compute `prompt_hash` as `hashlib.sha256(prompt_text.encode()).hexdigest()[:16]` (16-char hex is sufficient for clustering).
   - Compute `input_summary` by truncating the prompt text to 500 characters.
   - Write a row to `agent_dlq` using the `db_conn()` context manager from `src/quantstack/core/db.py`.
   - Wrap the DB write in a try/except so a DLQ write failure never prevents the fallback return. Log the DLQ write failure but do not propagate it.
3. Return the `{}` fallback exactly as before. Callers are unaffected.

**Update all call sites** of `parse_json_response()` to pass the new kwargs. These call sites are in `agent_executor.py` where agent responses are parsed. The agent name, graph name, run ID, and model used are all available in the surrounding execution context.

### File: `src/quantstack/observability/dlq_monitor.py` (NEW)

This module contains the DLQ rate monitoring logic. It is called periodically (e.g., after each graph cycle or on a timer).

Functions to implement:

```python
def compute_dlq_rate(agent_name: str, window_hours: int = 24) -> float:
    """Query agent_dlq for failure count in the rolling window.
    Query a total-attempts counter (or derive from Langfuse traces) for the denominator.
    Return rate as a percentage (0.0 - 100.0).
    """
    ...

def check_dlq_alerts(agent_name: str) -> None:
    """Compute DLQ rate and emit alerts if thresholds are breached.
    - rate > 5%: emit Langfuse warn event
    - rate > 10%: emit Langfuse critical event + trigger outbound notification
    Alert payload: agent_name, current rate, sample of recent failed messages, rate trend.
    """
    ...
```

**Total-attempts denominator**: The DLQ table only records failures. To compute a rate, you need total parse attempts. Two options:
1. Increment a counter (in-memory or DB) on every `parse_json_response()` call. A simple `parse_attempt_counts` table or an in-memory counter flushed periodically.
2. Derive from Langfuse traces -- every agent invocation is already traced, so count traces per agent in the window.

Option 2 is preferred because it avoids a new table and leverages existing observability infrastructure. If Langfuse query latency is too high, fall back to option 1 with a lightweight `parse_attempt_counts` table.

### Outbound Notification

On critical alert (>10% DLQ rate), trigger an outbound notification. The notification mechanism depends on what is already configured in the system:
- If a Slack webhook URL is in env vars, POST to it.
- If email is configured, send an email.
- At minimum, the Langfuse critical event serves as the notification if no outbound channel is configured.

The notification payload includes:
- Agent name and graph name
- Current DLQ rate and trend (rate 24h ago vs now)
- 3 most recent raw_output samples (truncated to 200 chars each)
- Suggested action: "Review prompt for {agent_name}, DLQ rate is {rate}%"

### Dashboard Integration

Surface DLQ metrics on the existing system dashboard (if one exists) or via Langfuse dashboard:
- Per-agent DLQ rate (24h rolling)
- Total DLQ entries (24h)
- Top error types
- Top prompt hashes (most-failing prompt variants)

This is a Langfuse configuration task, not a code task. Create the necessary Langfuse score/metric definitions so dashboards can query them.

## Why No Self-Healing

The self-healing loop (`record_tool_error` -> `bug_fix` task -> AutoResearchClaw) was designed for deterministic tool errors with known fix patterns. Prompt degradation is a symptom with multiple root causes: regime change in market data, data distribution drift, bad deployment, prompt rot over time. Auto-patching prompts that influence capital allocation without human review crosses the wrong automation boundary. The DLQ captures the data needed for a human to diagnose and fix. Revisit automated prompt repair after 60+ days of paper trading data provides enough signal to distinguish fixable patterns from noise.

## Implementation Checklist

1. Verify `agent_dlq` table exists (created by section-01 migration)
2. Modify `parse_json_response()` signature to accept context kwargs
3. Add DLQ write logic inside the parse failure path
4. Update all call sites of `parse_json_response()` to pass context
5. Create `src/quantstack/observability/dlq_monitor.py` with rate computation and alert logic
6. Wire `check_dlq_alerts()` into the post-cycle hook or periodic timer
7. Configure Langfuse metrics/scores for DLQ rate tracking
8. Write and run all tests in `tests/unit/test_dead_letter_queue.py`
