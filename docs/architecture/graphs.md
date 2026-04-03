# LangGraph StateGraph Architecture

QuantStack runs three LangGraph StateGraphs as Docker services, each on its own
cycle cadence. The Trading Graph executes during market hours, the Research Graph
runs after-hours and on weekends, and the Supervisor Graph runs continuously as
a watchdog.

All three graphs share the same agent config system (YAML + hot-reload) and the
same state conventions (TypedDict with `Annotated` reducer fields for append-only
lists).

---

## Trading Graph

**12 nodes. Parallel branches with a mandatory risk gate.**

The trading graph runs one cycle per market session. It begins with a kill-switch
check, fans out into parallel position review and entry scanning, then converges
through a risk gate before executing any orders.

```
START -> safety_check -> [halted?] ---------> END
                      \
                       -> plan_day
                           |                              |
                           |-> position_review            |
                           |       -> execute_exits       |
                           |              -> merge_parallel
                           |                              |
                           |-> entry_scan --------------->|
                                                          v
                                                     risk_sizing
                                                       /       \
                                              [rejected]     [approved]
                                                  |              |
                                                 END      portfolio_review
                                                          analyze_options
                                                          execute_entries
                                                               |
                                                            reflect -> END
```

### Node reference

| Node | LLM Tier | Retry | Purpose |
|------|----------|-------|---------|
| `safety_check` | medium | 0 (fail fast) | Checks kill switch. If halted, routes to END immediately. |
| `plan_day` | medium | 3 | Generates daily trading plan from regime + portfolio context. |
| `position_review` | medium | 3 | Evaluates open positions: HOLD / TRIM / CLOSE decisions. |
| `entry_scan` | heavy | 3 | Scans for new entries, runs trade debater evaluation. |
| `execute_exits` | medium | 2 | Deterministic order execution for exits. No LLM judgment. |
| `execute_entries` | medium | 2 | Deterministic order execution for entries. No LLM judgment. |
| `risk_sizing` | medium | 0 (no retry) | SafetyGate check. **CRITICAL** -- rejection kills the cycle. |
| `merge_parallel` | none | -- | Pure join node, no LLM. Merges position_review and entry_scan branches. |
| `portfolio_review` | heavy | 3 | Fund manager portfolio-level decisions. |
| `analyze_options` | heavy | 3 | Options strategy analysis. |
| `reflect` | medium | 3 | Post-trade reflection, writes to memory. |

### Router functions

**`_safety_check_router`** -- Binary. `halt` routes to END, `continue` routes to
`plan_day`. No partial states.

**`_risk_gate_router`** -- Conservative. ANY rejection in the risk verdicts routes
to `rejected` (END). There is no partial approval: if one candidate fails the
gate, the entire batch is rejected. This is intentional -- the risk gate is law.

### Key files

| File | Contents |
|------|----------|
| `src/quantstack/graphs/trading/graph.py` | `build_trading_graph()`, router functions |
| `src/quantstack/graphs/trading/nodes.py` | Node implementations |
| `src/quantstack/graphs/trading/config/agents.yaml` | Agent personas and tool assignments |

---

## Research Graph

**8 nodes. Linear pipeline with a validation gate.**

The research graph discovers new strategies. It loads context, selects a domain,
generates hypotheses, validates signals, backtests, runs ML experiments, and
registers successful strategies to the database.

```
START -> context_load -> domain_selection -> hypothesis_generation
      -> signal_validation -> [passed?] -> backtest_validation
      -> ml_experiment -> strategy_registration -> knowledge_update -> END
                          |
                          +-> [failed] -> END
```

### Node reference

| Node | LLM Tier | Retry | Type | Purpose |
|------|----------|-------|------|---------|
| `context_load` | heavy | 2 | tool | Loads regime, portfolio, recent research context. |
| `domain_selection` | heavy | 3 | agent | Selects research domain and symbols. |
| `hypothesis_generation` | heavy | 3 | agent | Generates testable trading hypotheses. |
| `signal_validation` | heavy | 2 | tool | Validates signal strength with statistical tests. |
| `backtest_validation` | heavy | 2 | tool | Runs backtest on validated signals. |
| `ml_experiment` | ml | 3 | agent | ML model training (uses `ml_scientist` config). |
| `strategy_registration` | heavy | 2 | tool | Persists validated strategy to DB. |
| `knowledge_update` | heavy | 2 | tool | Updates knowledge base with findings. |

Tool nodes are deterministic (call Python functions, format results). Agent nodes
have LLM reasoning loops with `max_iterations` and `timeout_seconds` from their
YAML config.

### Key files

| File | Contents |
|------|----------|
| `src/quantstack/graphs/research/graph.py` | `build_research_graph()` |
| `src/quantstack/graphs/research/nodes.py` | Node implementations |
| `src/quantstack/graphs/research/config/agents.yaml` | 8 agent configs |

---

## Supervisor Graph

**5 nodes. Strictly linear.**

The supervisor is a watchdog that monitors system health, diagnoses problems,
attempts recovery, manages strategy lifecycle, and runs scheduled maintenance.

```
START -> health_check -> diagnose_issues -> execute_recovery
      -> strategy_lifecycle -> scheduled_tasks -> END
```

### Node reference

| Node | LLM Tier | Retry | Purpose |
|------|----------|-------|---------|
| `health_check` | light | 2 | Checks service health, DB connectivity, API status. |
| `diagnose_issues` | light | 3 | Analyzes health data, identifies root causes. |
| `execute_recovery` | light | 2 | Runs recovery actions (restart services, clear queues). |
| `strategy_lifecycle` | medium | 3 | Promotes/demotes/retires strategies based on performance. |
| `scheduled_tasks` | light | 2 | Runs periodic maintenance (cleanup, aggregation). |

### Key files

| File | Contents |
|------|----------|
| `src/quantstack/graphs/supervisor/graph.py` | `build_supervisor_graph()` |
| `src/quantstack/graphs/supervisor/nodes.py` | Node implementations |
| `src/quantstack/graphs/supervisor/config/agents.yaml` | Agent configs |

---

## State Schemas

All state schemas are defined in `src/quantstack/graphs/state.py` as TypedDicts.
Fields using `Annotated[list, operator.add]` are append-only reducers -- each node
appends to the list rather than replacing it. This is how LangGraph tracks
accumulating decisions and errors across nodes.

### ResearchState

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `cycle_number` | `int` | -- | Current research cycle. |
| `regime` | `str` | -- | Market regime (trending_up, ranging, etc.). |
| `context_summary` | `str` | -- | Loaded context for this cycle. |
| `selected_domain` | `str` | -- | Chosen research domain. |
| `selected_symbols` | `list[str]` | -- | Symbols under research. |
| `hypothesis` | `str` | -- | Generated hypothesis text. |
| `validation_result` | `dict` | -- | Signal validation output. |
| `backtest_id` | `str` | -- | Reference to backtest run. |
| `ml_experiment_id` | `str` | -- | Reference to ML experiment. |
| `registered_strategy_id` | `str` | -- | ID of registered strategy. |
| `errors` | `list[str]` | `operator.add` | Accumulated errors. |
| `decisions` | `list[dict]` | `operator.add` | Accumulated decision records. |

### TradingState

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `cycle_number` | `int` | -- | Current trading cycle. |
| `regime` | `str` | -- | Market regime. |
| `portfolio_context` | `dict` | -- | Current portfolio snapshot. |
| `daily_plan` | `str` | -- | Plan generated by `plan_day`. |
| `position_reviews` | `list[dict]` | -- | HOLD/TRIM/CLOSE decisions per position. |
| `exit_orders` | `list[dict]` | -- | Orders generated by `execute_exits`. |
| `entry_candidates` | `list[dict]` | -- | Candidates from `entry_scan`. |
| `risk_verdicts` | `list[dict]` | -- | SafetyGate verdicts per candidate. |
| `fund_manager_decisions` | `list[dict]` | -- | Portfolio-level decisions. |
| `options_analysis` | `list[dict]` | -- | Options strategy analysis results. |
| `entry_orders` | `list[dict]` | -- | Orders generated by `execute_entries`. |
| `reflection` | `str` | -- | Post-cycle reflection text. |
| `errors` | `list[str]` | `operator.add` | Accumulated errors. |
| `decisions` | `list[dict]` | `operator.add` | Accumulated decision records. |

### SupervisorState

| Field | Type | Reducer | Description |
|-------|------|---------|-------------|
| `cycle_number` | `int` | -- | Current supervisor cycle. |
| `health_status` | `dict` | -- | Health check results. |
| `diagnosed_issues` | `list[dict]` | -- | Issues found by diagnosis. |
| `recovery_actions` | `list[dict]` | -- | Recovery actions taken. |
| `strategy_lifecycle_actions` | `list[dict]` | -- | Promote/demote/retire actions. |
| `scheduled_task_results` | `list[dict]` | -- | Results of scheduled tasks. |
| `errors` | `list[str]` | `operator.add` | Accumulated errors. |

---

## Agent Config System

Agent personas and tool assignments are defined in YAML, not in Python code.
This enables hot-reload without redeployment.

### AgentConfig dataclass

Defined in `src/quantstack/graphs/config.py`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | -- | Agent identifier (matches YAML key). |
| `role` | `str` | -- | Role title shown in prompts. |
| `goal` | `str` | -- | Agent's objective. |
| `backstory` | `str` | -- | Persona context for the LLM. |
| `llm_tier` | `str` | -- | `heavy`, `medium`, `light`, or `ml`. |
| `max_iterations` | `int` | 20 | Max reasoning loop iterations. |
| `timeout_seconds` | `int` | 600 | Hard timeout per invocation. |
| `tools` | `tuple[str]` | -- | Tool names (must exist in TOOL_REGISTRY). |

### YAML format

```yaml
quant_researcher:
  role: "Senior Quantitative Researcher"
  goal: "Discover statistically robust alpha signals"
  backstory: |
    You have 15 years of experience in quantitative finance...
  llm_tier: heavy
  max_iterations: 20
  timeout_seconds: 600
  tools:
    - signal_brief
    - fetch_market_data
```

### Hot-reload

`ConfigWatcher` (in `src/quantstack/graphs/config_watcher.py`) monitors YAML files:

- **Dev mode**: File-system watcher triggers reload on save.
- **Prod mode**: Reloads on SIGHUP signal.

### Startup validation

`validate_tool_references()` runs at startup and checks that every tool name
referenced in every agent's `tools` list exists in `TOOL_REGISTRY`. If a tool
is missing, startup fails with a clear error message naming the agent and the
missing tool.

---

## Adding a New Node

Step-by-step for adding a node to any graph.

### 1. Define the state field

If the node produces new data, add the field to the appropriate state schema in
`src/quantstack/graphs/state.py`. Use `Annotated[list, operator.add]` only for
fields that accumulate across nodes.

### 2. Write the node function

Add the function to the graph's `nodes.py`. Every node function has the signature:

```python
async def my_node(state: TradingState, config: RunnableConfig) -> dict:
    # ... do work ...
    return {"my_new_field": result}
```

Return a dict with only the fields you want to update. LangGraph merges it into
state.

### 3. Create the agent config (if LLM-backed)

Add an entry to the graph's `config/agents.yaml` with role, goal, backstory,
llm_tier, and tools. Run `validate_tool_references()` to confirm tool names.

### 4. Wire the node into the graph

In the graph's `graph.py`, inside the `build_*_graph()` function:

```python
graph.add_node("my_node", my_node)
graph.add_edge("previous_node", "my_node")
graph.add_edge("my_node", "next_node")
```

For conditional routing, use `add_conditional_edges` with a router function.

### 5. Add retry and tier config

Set retry count and LLM tier in the node builder call. Follow the conventions
of the existing graph -- fail-fast (retry=0) for safety-critical nodes,
retry=2-3 for recoverable operations.

### 6. Test

Run the graph in isolation with a mock state to verify the node executes
correctly and state transitions work as expected.
