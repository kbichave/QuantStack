# Section 11: Agent Decision Quality Tracking

## Purpose

LLM agents (trade_debater, exit_evaluator, etc.) make recommendations that lead to trade outcomes, but there is no feedback loop today. The same prompt produces the same quality regardless of past performance. This section wires SkillTracker to compute per-agent win rates, publishes degradation alerts when an agent's win rate drops below 40%, auto-queues research tasks for investigation, and surfaces per-agent confidence in the daily plan prompt so the planning agent knows which execution agents are currently reliable.

## Dependencies

- **Section 03 (Readpoint Wiring), Wire 4**: SkillTracker must already be wired into `on_trade_close()` in `trade_hooks.py` so that every trade close populates `agent_skills` with prediction accuracy and signal P&L. Without this, there is no data to compute win rates from.
- **Section 06 (EventBus Extension)**: The `AGENT_DEGRADATION` event type must already exist in the `EventType` enum in `src/quantstack/coordination/event_bus.py`.

## Existing Code State

**SkillTracker** (`src/quantstack/learning/skill_tracker.py`, 421 lines) already provides:
- `AgentSkill` dataclass with `prediction_count`, `correct_predictions`, `signal_count`, `winning_signals`, `total_signal_pnl`, and `ic_observations`
- `signal_win_rate` property: `winning_signals / signal_count`
- `prediction_accuracy` property: `correct_predictions / prediction_count`
- `update_agent_skill(agent_id, prediction_correct, signal_pnl)` — updates and persists metrics
- `get_confidence_adjustment(agent_id)` — returns a factor in [0.5, 1.5] combining win rate, signal win rate, ICIR, and IC trend
- `get_all_skills()` — returns all tracked AgentSkill records
- DB tables: `agent_skills`, `agent_ic_observations` (already created in `_ensure_table()`)

**EventBus** (`src/quantstack/coordination/event_bus.py`) already has `STRATEGY_DEMOTED`, `MODEL_DEGRADATION`, `IC_DECAY`, `REGIME_CHANGE`, etc. Section 06 adds `AGENT_DEGRADATION` to this enum.

**Trade hooks** (`src/quantstack/hooks/trade_hooks.py`) currently fires ReflectionManager, ReflexionMemory, CreditAssigner, PromptTuner, and OutcomeTracker on trade close. Section 03 Wire 4 adds SkillTracker to this pipeline.

**Daily plan node** (`src/quantstack/graphs/trading/nodes.py`, `daily_plan` function starting ~line 233) constructs a prompt with market intelligence and regime context. It does not currently include any agent confidence information.

## Tests (Write First)

Test file: `tests/unit/test_agent_quality.py`

### Win rate computation

```python
def test_win_rate_30_trades_18_wins():
    """30 trades with 18 wins should produce win_rate = 0.60."""
    # Setup: create SkillTracker with mock store
    # Call update_agent_skill 30 times (18 with prediction_correct=True, 12 with False)
    # Assert: skill.signal_win_rate == 0.60

def test_rolling_window_old_trades_drop_off():
    """Verify that win rate reflects the rolling window, not all-time."""
    # Setup: record 50 trades — first 20 all wins, last 30 all losses
    # Assert: recent window win rate is much lower than all-time
    # Note: SkillTracker currently uses all-time counters, not rolling.
    # This test documents the gap. The implementation may need a rolling
    # window variant or the alert logic should query recent trades from DB.
```

### Alert threshold

```python
def test_agent_degradation_event_below_40_pct():
    """Win rate < 0.40 should publish AGENT_DEGRADATION event."""
    # Setup: agent with 30 trades, 10 wins (33% win rate)
    # Call the agent quality check function
    # Assert: EventBus.publish called with EventType.AGENT_DEGRADATION
    # Assert: payload contains agent_id, win_rate, trade_count, recent_losses

def test_no_alert_at_40_pct_or_above():
    """Win rate >= 0.40 should not publish any degradation event."""
    # Setup: agent with 30 trades, 12 wins (40% win rate)
    # Call the agent quality check function
    # Assert: EventBus.publish NOT called with AGENT_DEGRADATION

def test_research_task_queued_on_degradation():
    """When AGENT_DEGRADATION fires, a research task should be queued."""
    # Setup: agent with degraded win rate
    # Call the agent quality check function
    # Assert: research_queue INSERT with task_type='agent_prompt_investigation'
    # Assert: context_json includes agent_id, win_rate, recent trade details
```

### Cold-start

```python
def test_cold_start_fewer_than_30_trades():
    """< 30 trades should produce no alert and confidence = 1.0."""
    # Setup: agent with 15 trades, 3 wins (20% win rate — terrible, but too few)
    # Call the agent quality check function
    # Assert: no AGENT_DEGRADATION event
    # Assert: get_confidence_adjustment returns 1.0 or near-default
```

### Daily plan integration

```python
def test_daily_plan_includes_agent_confidence():
    """Daily plan prompt should contain per-agent confidence context."""
    # Setup: mock SkillTracker with 3 agents at varying confidence levels
    # Call the agent confidence formatter
    # Assert: output contains agent names and confidence values
    # Assert: degraded agents are labeled (e.g., "degraded, under investigation")

def test_daily_plan_omits_confidence_when_no_data():
    """With no tracked agents, daily plan should not include confidence section."""
    # Setup: empty SkillTracker
    # Assert: confidence section is absent from the prompt context
```

## Implementation

### 1. Agent quality check function

**File:** `src/quantstack/learning/skill_tracker.py`

Add a method to SkillTracker that evaluates all tracked agents and publishes degradation events. This is the core feedback loop.

```python
def check_agent_quality(self, event_bus: EventBus, min_trades: int = 30, alert_threshold: float = 0.40) -> list[str]:
    """
    Evaluate all agents. Publish AGENT_DEGRADATION for any agent with
    >= min_trades and win_rate < alert_threshold. Queue research task.

    Returns list of degraded agent IDs.
    """
    # For each agent in self._skills:
    #   - Skip if signal_count < min_trades (cold-start)
    #   - Compute win rate from signal_win_rate property
    #   - If win_rate < alert_threshold:
    #     - Publish AGENT_DEGRADATION event with payload:
    #       {agent_id, win_rate, trade_count, recent_losses (last 5 losing trades)}
    #     - Insert into research_queue with task_type='agent_prompt_investigation'
    #       and context including agent_id, win_rate, confidence_adjustment
    #   - Return list of degraded agent_ids
```

The `event_bus` parameter is injected rather than constructed internally to keep the dependency explicit and testable. The `min_trades` and `alert_threshold` parameters are configurable for testing and tuning.

**Why 40% threshold:** Random guessing on direction is ~50%. An agent consistently below 40% is actively harmful — worse than not using it. The 30-trade minimum prevents false alarms from small samples (at 30 trades, a true 50% win rate has a ~95% CI of roughly [32%, 68%], so observing < 40% is meaningful).

### 2. Research task queuing on degradation

When an agent is flagged, the research queue insertion should use `task_type='agent_prompt_investigation'` (not the generic `bug_fix`). The context JSON should include enough information for the research graph to investigate:

- `agent_id`
- `win_rate` (current rolling)
- `confidence_adjustment` (from `get_confidence_adjustment()`)
- `ic` and `icir` (if available)
- `ic_trend` (IMPROVING / STABLE / DECAYING / INSUFFICIENT_DATA)
- `recent_losing_trades` (last 5 trade IDs or summaries)

This follows the same pattern as the existing research queue insert in `trade_hooks.py` (lines 118-144), but with a specific task type so the research graph can dispatch to prompt investigation logic rather than generic bug fixing.

### 3. Daily plan agent confidence context

**File:** `src/quantstack/graphs/trading/nodes.py` (inside the `daily_plan` function)

After the existing market intelligence section construction (~line 236), query SkillTracker for all agent confidences and format them into the prompt.

```python
def format_agent_confidence(skill_tracker: SkillTracker) -> str:
    """
    Format per-agent confidence for inclusion in the daily plan prompt.

    Returns empty string if no agents are tracked (no noise in the prompt).
    Format: "Agent confidence: trade_debater=1.2 (reliable), exit_evaluator=0.7 (degraded, under investigation)."
    """
    # Get all skills from skill_tracker.get_all_skills()
    # Filter to agents with >= 5 predictions (avoid noise from brand-new agents)
    # For each: get_confidence_adjustment(agent_id)
    # Label: >= 1.0 "reliable", 0.8-1.0 "cautious", < 0.8 "degraded, under investigation"
    # Return formatted string, or empty string if no agents qualify
```

This gives the planning agent actionable context. When an agent like `exit_evaluator` is labeled "degraded," the planner can weight that agent's recommendations lower or flag trades that relied heavily on the degraded agent for extra scrutiny.

### 4. Wire the quality check into supervisor batch

**File:** `src/quantstack/graphs/supervisor/nodes.py`

Add `check_agent_quality()` to the daily supervisor batch (runs after market close, alongside existing tasks like `run_ic_computation()`). This is a lightweight check — it reads from in-memory SkillTracker state and only writes to EventBus and research_queue when degradation is detected.

```python
async def run_agent_quality_check(state: SupervisorState) -> dict:
    """Daily agent quality evaluation. Fires AGENT_DEGRADATION events for underperforming agents."""
    # Get or create SkillTracker instance
    # Get or create EventBus instance
    # Call skill_tracker.check_agent_quality(event_bus)
    # Log results
```

### 5. ICIR adjustment fix (from plan Section 9 / ghost module audit Section 2)

The `get_confidence_adjustment()` method in SkillTracker has a known issue: the ICIR adjustment uses `icir * 0.2` which can reach 0.6 for ICIR=3.0 before the outer clamp catches it. The fix is to change the coefficient to 0.15 and apply the min directly:

**File:** `src/quantstack/learning/skill_tracker.py`, line 348

Change:
```python
icir_adj = max(-0.2, min(0.3, skill.icir * 0.2))
```

To:
```python
icir_adj = max(-0.2, min(0.3, skill.icir * 0.15))
```

This ensures ICIR=3.0 produces `min(0.3, 0.45) = 0.3` (the cap), and the intent is clearer: the coefficient 0.15 means ICIR=2.0 gives the maximum boost (0.3), and values above 2.0 are capped rather than overshooting before the clamp.

## Rollback

Disable the quality check by removing the `run_agent_quality_check` call from the supervisor batch. SkillTracker data continues accumulating (no harm), but no degradation events fire and no research tasks are queued. The daily plan confidence context can be independently removed by reverting the prompt addition in `daily_plan`.

## Files Modified

| File | Change |
|------|--------|
| `src/quantstack/learning/skill_tracker.py` | Add `check_agent_quality()` method; fix ICIR coefficient (0.2 to 0.15) |
| `src/quantstack/graphs/trading/nodes.py` | Add agent confidence context to `daily_plan` prompt |
| `src/quantstack/graphs/supervisor/nodes.py` | Add `run_agent_quality_check` to daily supervisor batch |
| `tests/unit/test_agent_quality.py` | New test file with 7 test cases |

## Implementation Checklist

1. Write `tests/unit/test_agent_quality.py` with all 7 test stubs
2. Fix ICIR coefficient in `skill_tracker.py` (line 348: `0.2` to `0.15`)
3. Add `check_agent_quality()` method to `SkillTracker` class
4. Add `format_agent_confidence()` helper and wire into `daily_plan` node
5. Add `run_agent_quality_check` supervisor batch node
6. Run tests, verify all pass
7. Verify EventBus integration by checking that AGENT_DEGRADATION events appear in `loop_events` table after a degraded agent is detected
