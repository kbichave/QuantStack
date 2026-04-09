# Section 06: Integration

**depends_on:** section-02-prompt-ab-testing, section-03-strategy-of-strategies, section-04-research-prioritization, section-05-few-shot-library

## Objective

Wire all P10 meta-learning components into the existing graph nodes and execution paths. Each component was built independently; this section connects them to the live system at the correct hook points.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/graphs/trading/nodes.py` | Modify | Wire agent quality scoring after cycle completion, few-shot injection before agent execution |
| `src/quantstack/graphs/research/nodes.py` | Modify | Wire research priority scoring into queue consumer |
| `src/quantstack/graphs/supervisor/nodes.py` | Modify | Wire meta-allocator weights into fund_manager, add degradation alert checks |
| `src/quantstack/hooks/trade_hooks.py` | Modify | Add post-trade quality scoring hook |
| `src/quantstack/learning/prompt_tuner.py` | Modify | Wire prompt A/B shadow execution alongside existing prompt tuner |

## Implementation Details

### 1. Agent Quality Scoring (Post-Cycle Hook)

**Location:** `src/quantstack/graphs/trading/nodes.py` -- after each agent node completes

Wire `record_cycle_quality()` from section-01 at the end of agent execution nodes:

```python
from quantstack.learning.agent_quality import record_cycle_quality

# After agent produces output in node function:
try:
    record_cycle_quality(
        agent_name=agent_name,
        cycle_id=state.get("cycle_id", ""),
        recommendation_type=recommendation_type,
        recommendation=agent_output,
        outcome=None,  # Outcome filled later by outcome_tracker
    )
except Exception:
    logger.warning(f"Failed to record quality for {agent_name}", exc_info=True)
```

Outcome backfill: when a trade closes, `hooks/trade_hooks.py` updates the `agent_quality_scores` row with the realized outcome and recomputes the quality score.

### 2. Few-Shot Example Injection (Pre-Agent Hook)

**Location:** Agent prompt construction (wherever the agent's system prompt or messages are assembled)

Before each agent call, retrieve and inject relevant examples:

```python
from quantstack.learning.few_shot_library import get_examples, format_examples_for_prompt

examples = get_examples(
    agent_name=agent_name,
    context={"regime": current_regime, "strategy_type": strategy_type},
    max_examples=3,
)
if examples:
    few_shot_block = format_examples_for_prompt(examples, agent_name)
    # Append to system prompt or inject as a user message prefix
```

Guard: if `get_examples` raises or returns empty, proceed without examples. Never block agent execution.

### 3. Prompt A/B Shadow Execution

**Location:** Agent executor path in trading/research graph nodes

When an active variant exists for an agent:

```python
from quantstack.learning.prompt_ab import get_active_variant, record_shadow_result

variant = get_active_variant(agent_name)
production_output = run_agent(production_prompt, context)

if variant:
    try:
        variant_output = run_agent(variant["variant_prompt"], context)
        # Score both outputs using quality scoring
        record_shadow_result(
            agent_name=agent_name,
            variant_id=variant["variant_id"],
            cycle_id=cycle_id,
            production_output=production_output,
            variant_output=variant_output,
            production_quality=score_output(production_output),
            variant_quality=score_output(variant_output),
            input_context=context,
        )
    except Exception:
        logger.warning(f"Shadow variant execution failed for {agent_name}", exc_info=True)
```

**Critical constraint:** Shadow execution must not block or slow the production path. Run variant as a best-effort side-call. If it fails, log and move on.

### 4. Meta-Allocator Integration (Fund Manager)

**Location:** `src/quantstack/graphs/supervisor/nodes.py` or fund_manager node

Replace equal-weight allocation with meta-allocator weights:

```python
from quantstack.learning.meta_allocator import get_current_weights

weights = get_current_weights(regime=current_regime)
if weights:
    # Use meta-allocator weights for position sizing
    for strategy_id, weight in weights.items():
        # Apply weight to strategy allocation
        ...
else:
    # Fallback to equal-weight allocation (existing behavior)
    ...
```

Guard: if `get_current_weights` fails or returns empty, fall back to equal-weight. Log the fallback.

### 5. Research Priority Integration

**Location:** `src/quantstack/graphs/research/nodes.py` -- queue consumer node

Change the queue fetch to order by priority:

```python
# Existing: fetch tasks in FIFO order
# New: fetch tasks ordered by priority_score DESC
with db_conn() as conn:
    rows = conn.execute("""
        SELECT * FROM research_tasks 
        WHERE status = 'pending'
        ORDER BY priority_score DESC NULLS LAST, created_at ASC
        LIMIT %s
    """, [batch_size]).fetchall()
```

Additionally, run priority scoring on newly inserted tasks:

```python
from quantstack.learning.experiment_prioritizer import prioritize_research_queue

# After inserting new research tasks:
prioritize_research_queue(new_tasks, active_strategies, loss_stats, current_regime)
```

### 6. Post-Trade Quality Scoring Hook

**Location:** `src/quantstack/hooks/trade_hooks.py`

Add a hook that fires when a trade closes:

```python
from quantstack.learning.agent_quality import record_cycle_quality
from quantstack.learning.few_shot_library import curate_examples

def on_trade_closed(trade: dict):
    # Backfill agent quality score with realized outcome
    if trade.get("originating_agent") and trade.get("originating_cycle_id"):
        record_cycle_quality(
            agent_name=trade["originating_agent"],
            cycle_id=trade["originating_cycle_id"],
            recommendation_type="trade_outcome",
            recommendation=trade.get("entry_rationale"),
            outcome={"realized_return": trade["realized_pnl_pct"], ...},
        )
    
    # Trigger few-shot curation for agents that performed well
    curate_examples(trade["originating_agent"])
```

### 7. Degradation Alert Integration (Supervisor)

**Location:** `src/quantstack/graphs/supervisor/nodes.py`

Add degradation check to the supervisor health monitoring cycle:

```python
from quantstack.learning.agent_quality import check_degradation_alerts

alerts = check_degradation_alerts()
for alert in alerts:
    # Queue a research task to investigate the degraded agent
    queue_research_task({
        "task_type": "agent_prompt_investigation",
        "agent_name": alert["agent_name"],
        "priority_score": 0.9,  # High priority
        "context": alert,
    })
```

## Test Requirements

File: `tests/unit/learning/test_meta_learning_integration.py`

1. **test_quality_scoring_wired_after_cycle** -- mock agent execution, verify record_cycle_quality is called
2. **test_few_shot_injection_before_agent** -- mock get_examples, verify examples injected into prompt
3. **test_few_shot_failure_does_not_block** -- get_examples raises, verify agent still executes
4. **test_shadow_variant_runs_when_active** -- active variant exists, verify both outputs recorded
5. **test_shadow_failure_does_not_block** -- variant execution fails, verify production output still returned
6. **test_meta_allocator_weights_used** -- weights available, verify fund_manager uses them
7. **test_meta_allocator_fallback_to_equal** -- get_current_weights returns empty, verify equal-weight used
8. **test_research_queue_ordered_by_priority** -- tasks with different priorities, verify processing order
9. **test_trade_close_triggers_quality_backfill** -- trade closes, verify agent quality score updated
10. **test_degradation_alert_queues_research** -- degraded agent detected, verify research task queued

## Acceptance Criteria

- [ ] Agent quality scoring runs after every cycle (non-blocking)
- [ ] Few-shot examples injected into agent prompts when available
- [ ] Prompt A/B shadow execution runs alongside production (non-blocking)
- [ ] Meta-allocator weights replace equal-weight when available, with graceful fallback
- [ ] Research queue ordered by priority score
- [ ] Trade close triggers quality score backfill and few-shot curation
- [ ] Degradation alerts auto-queue investigation research tasks
- [ ] All error paths are guarded: no meta-learning failure blocks trading or research
- [ ] All 10 integration tests pass
