# Section 01: Agent Decision Quality Tracking

## Objective

Extend the existing `agent_quality.py` module with persistent, per-cycle quality scoring backed by PostgreSQL. After each graph cycle, compare each agent's recommendation with the realized outcome and store a granular quality score. Provide a dashboard query for monitoring agent performance over time.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/db.py` | Modify | Add `agent_quality_scores` table to `ensure_tables()` |
| `src/quantstack/learning/agent_quality.py` | Modify | Add DB-backed scoring, rolling aggregation, dashboard query, and degradation alerts |

## Implementation Details

### 1. Database Schema

Add to `ensure_tables()` in `db.py`:

```sql
CREATE TABLE IF NOT EXISTS agent_quality_scores (
    id              SERIAL PRIMARY KEY,
    agent_name      TEXT NOT NULL,
    cycle_id        TEXT NOT NULL,
    recommendation_type TEXT,          -- 'regime_call', 'entry_signal', 'exit_signal', 'sizing', etc.
    recommendation  JSONB,             -- the raw recommendation dict
    outcome         JSONB,             -- the realized outcome dict
    correct_direction BOOLEAN,         -- did the direction match?
    magnitude_accuracy REAL,           -- 0.0-1.0, how close was the magnitude estimate
    timing_label    TEXT,              -- 'early', 'on_time', 'late'
    quality_score   REAL NOT NULL,     -- composite score 0.0-1.0
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_name, cycle_id, recommendation_type)
);
CREATE INDEX IF NOT EXISTS idx_aqs_agent_computed
    ON agent_quality_scores (agent_name, computed_at DESC);
```

### 2. Scoring Logic (`agent_quality.py`)

Add the following functions (do not remove existing functions -- they are consumed elsewhere):

**`record_cycle_quality(agent_name, cycle_id, recommendation_type, recommendation, outcome)`**
- Compute `correct_direction`: compare recommendation direction vs. outcome direction (bool)
- Compute `magnitude_accuracy`: `1.0 - min(1.0, abs(predicted_magnitude - realized_magnitude) / max(abs(realized_magnitude), 1e-6))`
- Compute `timing_label`: compare predicted entry/exit timing vs. actual fill timing. Buckets: "early" (>2 bars before), "on_time" (within 2 bars), "late" (>2 bars after)
- Composite `quality_score`: `0.5 * int(correct_direction) + 0.3 * magnitude_accuracy + 0.2 * timing_score` where timing_score is `{early: 0.5, on_time: 1.0, late: 0.3}`
- Upsert into `agent_quality_scores` (ON CONFLICT on the unique constraint)

**`get_rolling_agent_stats(agent_name, window_days=21)`**
- Query `agent_quality_scores` for the last `window_days` days
- Return: `{"win_rate": float, "avg_quality": float, "total_cycles": int, "trend": "improving"|"stable"|"degrading"}`
- Trend: compare first-half avg vs. second-half avg quality score

**`get_agent_quality_dashboard()`**
- For each distinct `agent_name`: rolling 21-day win rate, avg quality, best/worst agent, quality trend
- Return list of dicts sorted by quality score ascending (worst first, so degraded agents surface)

**`check_degradation_alerts(consecutive_threshold=5, win_rate_floor=0.40)`**
- For each agent, check if the last `consecutive_threshold` cycles all have quality_score below `win_rate_floor`
- Return list of `{"agent_name": str, "consecutive_failures": int, "avg_quality": float}` for degraded agents
- Log a warning for each degraded agent

### 3. Integration Points

- The scoring function will be called from graph node post-processing (wired in section-06)
- The dashboard query will be consumed by the supervisor graph and TUI
- Degradation alerts feed into research task queuing (existing `get_degraded_agents` pattern)

## Test Requirements

File: `tests/unit/learning/test_agent_quality_tracking.py`

1. **test_record_cycle_quality_correct_direction** -- agent predicted long, outcome was positive return, verify correct_direction=True and quality_score > 0.5
2. **test_record_cycle_quality_wrong_direction** -- agent predicted long, outcome was negative, verify correct_direction=False
3. **test_magnitude_accuracy_perfect** -- predicted magnitude matches realized, verify magnitude_accuracy ~1.0
4. **test_magnitude_accuracy_far_off** -- predicted 5% gain, realized 0.5%, verify magnitude_accuracy is low
5. **test_timing_labels** -- verify early/on_time/late classification
6. **test_rolling_stats_window** -- insert 30 days of scores, verify 21-day window only includes recent
7. **test_rolling_stats_trend_detection** -- insert improving scores, verify trend="improving"
8. **test_dashboard_returns_all_agents** -- insert scores for 3 agents, verify dashboard returns all 3
9. **test_degradation_alert_fires** -- 5 consecutive low scores, verify alert returned
10. **test_degradation_alert_does_not_fire** -- mix of good and bad scores, verify no alert
11. **test_upsert_idempotent** -- call record_cycle_quality twice with same (agent, cycle, type), verify single row

## Acceptance Criteria

- [ ] `agent_quality_scores` table is created by `ensure_tables()`
- [ ] `record_cycle_quality` correctly computes and persists all score components
- [ ] `get_rolling_agent_stats` returns accurate rolling statistics with trend detection
- [ ] `get_agent_quality_dashboard` returns a sorted list of all agents with quality metrics
- [ ] `check_degradation_alerts` identifies agents with consecutive failures
- [ ] All 11 unit tests pass
- [ ] Existing `evaluate_agent_quality` and `format_agent_confidence` functions remain unchanged and functional
