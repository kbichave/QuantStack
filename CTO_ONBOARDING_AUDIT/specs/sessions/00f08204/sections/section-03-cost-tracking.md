# Section 03: Per-Agent Cost Tracking via Langfuse

**Plan Reference:** Item 5.2
**Dependencies:** None
**Blocks:** section-04-tier-reclassification (needs cost data for measurement)

---

## Problem

No per-agent cost aggregation. Langfuse captures raw token counts but no agent-level rollup, no budget enforcement, no alerting. A runaway agent could consume $50+ per cycle undetected.

## Architecture

1. **Metadata enrichment** — tag every LLM call with agent_name, graph_name, cycle_id, llm_tier
2. **Aggregation queries** — roll up costs by agent/graph/day via Langfuse API
3. **Budget enforcement** — max_tokens_budget per agent, graceful exit on breach
4. **Alerting** — Supervisor health_monitor detects cost anomalies

---

## Tests (Write First)

### tests/unit/test_agent_executor.py (extend)

```python
# Test: agent with max_tokens_budget=1000 stops after exceeding budget
# Test: agent with max_tokens_budget=None runs without limit
# Test: budget-exceeded agent returns partial result with flag
# Test: budget accumulates across multiple tool-calling rounds
# Test: budget tracks prompt_tokens + completion_tokens
```

### tests/unit/test_cost_queries.py (new)

```python
# Test: LLM call metadata includes agent_name, graph_name, cycle_id, llm_tier
# Test: cost_by_agent_day returns aggregated costs grouped by agent and date
# Test: cost_anomaly_detection flags agents exceeding 3x 7-day average
```

---

## Implementation

### 1. Metadata Enrichment

In `src/quantstack/observability/instrumentation.py`, ensure `log_llm_call()` accepts and passes `agent_name`, `graph_name`, `cycle_id`, `llm_tier` as Langfuse generation metadata.

In `src/quantstack/graphs/agent_executor.py`, pass these from `AgentConfig` and graph context when invoking the LLM.

### 2. Aggregation Queries

Create `src/quantstack/observability/cost_queries.py`:

- `cost_by_agent_day(start_date, end_date)` → `{agent_name: {date: cost}}`
- `cost_by_cycle(graph_name, start_date, end_date)` → cycle cost breakdowns
- `top_agents_by_cost(days=7, limit=10)` → ranked list
- `detect_cost_anomalies(window_days=7, threshold=3.0)` → agents exceeding threshold

### 3. Budget Enforcement

Add `max_tokens_budget: int | None = None` to `AgentConfig`. In agent_executor tool loop:
- Accumulate `prompt_tokens + completion_tokens` per round
- If cumulative > budget: set `budget_exceeded` flag, return partial result, log Langfuse event
- Downstream nodes check `budget_exceeded` flag

### 4. Alerting

Extend `health_monitor` in Supervisor graph:
- Query `detect_cost_anomalies()` on each health check
- Emit alerts for anomalies
- Trigger kill switch if global daily cap exceeded

---

## Files

| File | Change |
|------|--------|
| `src/quantstack/observability/instrumentation.py` | Modify — add metadata params |
| `src/quantstack/observability/cost_queries.py` | **Create** |
| `src/quantstack/graphs/config.py` | Modify — add max_tokens_budget to AgentConfig |
| `src/quantstack/graphs/agent_executor.py` | Modify — budget enforcement |
| `src/quantstack/graphs/*/config/agents.yaml` (x3) | Modify — add max_tokens_budget |
| `src/quantstack/graphs/supervisor/nodes.py` | Modify — cost alerting |
| `tests/unit/test_cost_queries.py` | **Create** |
