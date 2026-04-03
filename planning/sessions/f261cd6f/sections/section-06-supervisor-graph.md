# Section 06: Supervisor Graph

## Overview

Build the `SupervisorGraph` as a LangGraph `StateGraph` with 5 sequential nodes, replacing the existing CrewAI `SupervisorCrew`. The supervisor is the simplest of the three graphs (no branching, no parallel edges, no risk gates), making it the ideal first graph to build and validate the LangGraph pattern end-to-end.

The existing supervisor crew (`src/quantstack/crews/supervisor/crew.py`) runs 5 sequential tasks across 3 agents (health_monitor, self_healer, strategy_promoter). The LangGraph replacement preserves identical semantics: a linear `START -> health_check -> diagnose_issues -> execute_recovery -> strategy_lifecycle -> scheduled_tasks -> END` topology.

## Dependencies

- **section-02-llm-provider**: `get_chat_model(tier)` must return a `BaseChatModel` instance for tiers `"light"` and `"medium"`.
- **section-03-agent-config**: `AgentConfig` dataclass, YAML loader, and `ConfigWatcher` must be functional.
- **section-04-state-schemas**: `SupervisorState` TypedDict must be defined.
- **section-05-tool-layer**: Any tools used by supervisor nodes must be available (either as `@tool`-decorated LLM-facing tools or as plain async node-callable functions).

## State Schema (from section-04)

The supervisor graph operates on `SupervisorState`:

```python
class SupervisorState(TypedDict):
    cycle_number: int
    health_status: dict                      # system health check results
    diagnosed_issues: list[dict]             # issues found
    recovery_actions: list[dict]             # actions taken
    strategy_lifecycle_actions: list[dict]   # promotions/retirements
    scheduled_task_results: list[dict]
    errors: Annotated[list[str], operator.add]
```

The `errors` field uses `operator.add` as a reducer, meaning every node that returns an `errors` key appends to the accumulated list. Nodes that have nothing to report must return `{"errors": []}` or omit the key entirely. Returning `None` for `errors` will fail at runtime.

## Tests (Write First)

File: `tests/unit/test_supervisor_graph.py`

```python
"""Unit and integration tests for the SupervisorGraph."""

# Test: build_supervisor_graph() returns CompiledStateGraph
def test_build_supervisor_graph_returns_compiled_graph():
    """build_supervisor_graph(config_watcher, checkpointer) returns a CompiledStateGraph."""

# Test: supervisor graph is linear (5 nodes, no branches)
def test_supervisor_graph_is_linear_with_five_nodes():
    """Graph has exactly 5 nodes: health_check, diagnose_issues, execute_recovery,
    strategy_lifecycle, scheduled_tasks. Edges form a linear chain from START to END."""

# Test: full graph invocation with mock LLM produces valid final state
def test_supervisor_graph_full_invocation_mock_llm():
    """Invoke the compiled graph with a mock ChatModel and verify the final state
    contains all expected fields (health_status, diagnosed_issues, recovery_actions,
    strategy_lifecycle_actions, scheduled_task_results). Errors list should be empty
    on a successful run."""

# Test: graph builder accepts ConfigWatcher and checkpointer
def test_graph_builder_accepts_config_watcher_and_checkpointer():
    """build_supervisor_graph() signature accepts a ConfigWatcher and a checkpointer
    (use MemorySaver for tests)."""

# Test: graph builder reads agent configs from ConfigWatcher
def test_graph_builder_reads_agent_configs():
    """The builder calls config_watcher.get_config() for each agent
    (health_monitor, self_healer, strategy_promoter) and uses the returned
    AgentConfig to configure the node's LLM tier and system prompt."""

# Test: graph builder creates ChatModel instances for each agent tier
def test_graph_builder_creates_chat_models_per_tier():
    """health_monitor and self_healer use tier 'light', strategy_promoter uses
    tier 'medium'. Verify get_chat_model() is called with correct tiers."""

# Test: health_check node returns health_status dict
def test_health_check_node_returns_health_status():
    """health_check node calls system health tools and returns a dict with
    per-service status (healthy/degraded/critical), heartbeat ages, and
    data freshness timestamps."""

# Test: diagnose_issues node returns empty list when all healthy
def test_diagnose_issues_returns_empty_when_healthy():
    """When health_status shows all services healthy, diagnose_issues returns
    an empty diagnosed_issues list."""

# Test: diagnose_issues node returns diagnoses when issues found
def test_diagnose_issues_returns_diagnoses_on_degraded():
    """When health_status contains degraded/critical services, diagnose_issues
    returns a non-empty diagnosed_issues list with per-service diagnosis
    and recommended_actions."""

# Test: execute_recovery node takes recovery actions
def test_execute_recovery_executes_actions():
    """When diagnosed_issues is non-empty, execute_recovery executes the
    recommended actions and returns recovery_actions with action/target/result."""

# Test: execute_recovery is a no-op when no issues
def test_execute_recovery_noop_when_no_issues():
    """When diagnosed_issues is empty, execute_recovery returns an empty
    recovery_actions list."""

# Test: strategy_lifecycle node returns lifecycle decisions
def test_strategy_lifecycle_returns_decisions():
    """strategy_lifecycle queries forward_testing strategies and returns
    decisions (promote/extend/retire/no_change) with reasoning."""

# Test: scheduled_tasks node returns task results
def test_scheduled_tasks_returns_task_results():
    """scheduled_tasks checks due tasks and fires coordination events,
    returning tasks_checked and tasks_fired arrays."""

# Test: node error appends to errors state field
def test_node_error_appends_to_errors():
    """When a node raises an exception, the error is caught and appended
    to the errors list in state rather than crashing the graph."""

# Test: error retry_policy on agent nodes retries up to 2 times
def test_agent_node_retry_policy():
    """Agent nodes (diagnose_issues, strategy_lifecycle) have retry_policy
    with max_attempts=2 and exponential backoff."""

# Test: rebuilding graph with new config produces different node behavior
def test_rebuild_graph_with_new_config():
    """After updating agent configs via ConfigWatcher, rebuilding the graph
    produces nodes that use the updated LLM tier or system prompt."""
```

Use `MemorySaver` (LangGraph's in-memory checkpointer) for all tests instead of `AsyncPostgresSaver`. Mock the LLM with a deterministic `FakeListChatModel` or similar test double from `langchain_core.language_models.fake`.

## Graph Topology

```
START --> health_check --> diagnose_issues --> execute_recovery
      --> strategy_lifecycle --> scheduled_tasks --> END
```

All edges are unconditional. No branching, no conditional routing, no parallel execution.

## File Structure

```
src/quantstack/graphs/supervisor/
    __init__.py              # re-export build_supervisor_graph
    graph.py                 # build_supervisor_graph() function
    nodes.py                 # 5 node functions
    config/
        agents.yaml          # agent profiles (adapted from crews/supervisor/config/)
```

## Agent Configuration (YAML)

File: `src/quantstack/graphs/supervisor/config/agents.yaml`

Adapt the existing CrewAI agent definitions to the new `AgentConfig` format. The three agents map as follows:

| Agent | LLM Tier | Used By Nodes |
|-------|----------|---------------|
| health_monitor | light | health_check, scheduled_tasks |
| self_healer | light | diagnose_issues, execute_recovery |
| strategy_promoter | medium | strategy_lifecycle |

YAML structure per agent:

```yaml
health_monitor:
  role: "System Health Monitor"
  goal: "Detect unhealthy containers, stale heartbeats, unreachable services, and data freshness issues."
  backstory: |
    Each cycle, check heartbeat freshness for trading-graph (max 120s stale) and
    research-graph (max 600s stale), reachability of Langfuse/Ollama/PostgreSQL,
    data freshness for tracked symbols, API rate limit status.
    Classify each finding as healthy, degraded, or critical.
  llm_tier: light
  max_iterations: 10
  timeout_seconds: 120
  tools:
    - get_system_status
    - get_heartbeats
    - check_service_health

self_healer:
  role: "Self-Healing Engineer"
  goal: "Diagnose root causes of system failures and execute recovery actions autonomously."
  backstory: |
    Recovery playbook: stale heartbeat - record issue, let watchdog handle.
    Service down - flag for restart, crews operate degraded. LLM provider failure -
    trigger fallback chain. Database lost - exponential backoff reconnect.
    Data staleness - trigger refresh event. Unrecoverable - activate kill switch.
  llm_tier: light
  max_iterations: 10
  timeout_seconds: 120
  tools:
    - trigger_provider_fallback
    - publish_coordination_event
    - activate_kill_switch

strategy_promoter:
  role: "Strategy Lifecycle Manager"
  goal: "Promote, extend, or retire strategies based on forward-testing performance evidence."
  backstory: |
    For each forward_testing strategy, evaluate: daily P&L, win rate, drawdown,
    trade count, duration in testing, market conditions, similar strategies from
    knowledge base, current portfolio needs. Retire strategies with IS/OOS ratio
    diverged >4x, win rate dropped >20 points from backtest, or regime affinity
    mismatch for >2 weeks.
  llm_tier: medium
  max_iterations: 15
  timeout_seconds: 300
  tools:
    - get_forward_testing_strategies
    - promote_strategy
    - retire_strategy
    - search_knowledge_base
```

Note: The exact tool names must match keys in `TOOL_REGISTRY` (section-05). Update tool names during implementation if the registry uses different keys.

## Node Implementations

File: `src/quantstack/graphs/supervisor/nodes.py`

Each node is an async function with signature `async def node_name(state: SupervisorState) -> dict`. The return dict contains only the state fields the node updates.

### health_check

- **Agent**: health_monitor (light tier)
- **Behavior**: Calls system health tools (heartbeat check, service reachability, data freshness, API rate limits). Returns structured health status.
- **State update**: `{"health_status": {...per-service status...}}`
- **Error handling**: If health tools fail, catch the exception, append to errors, and return a degraded health_status indicating the check itself failed.

### diagnose_issues

- **Agent**: self_healer (light tier)
- **Behavior**: Receives `health_status` from state. If all services are healthy, returns empty list. If any are degraded/critical, uses LLM to reason about root cause and recommend recovery actions from the playbook.
- **State update**: `{"diagnosed_issues": [{"service": ..., "diagnosis": ..., "recommended_action": ...}]}`
- **Key detail**: This is an LLM-backed agent node. The self_healer agent receives the health_status as context in its prompt and reasons about diagnosis. This is NOT a deterministic tool node.

### execute_recovery

- **Agent**: self_healer (light tier)
- **Behavior**: Receives `diagnosed_issues` from state. For each recommended action, executes it (trigger fallback, publish event, activate kill switch). Returns results.
- **State update**: `{"recovery_actions": [{"action": ..., "target": ..., "result": ...}]}`
- **Key detail**: Some recovery actions are tool calls (deterministic), but the agent decides which to execute based on diagnosis context. Mix of LLM reasoning and tool execution.

### strategy_lifecycle

- **Agent**: strategy_promoter (medium tier)
- **Behavior**: Queries DB for strategies in `forward_testing` status. For each, the LLM reasons about promotion/extension/retirement based on performance evidence, market conditions, knowledge base history.
- **State update**: `{"strategy_lifecycle_actions": [{"strategy_id": ..., "decision": ..., "reasoning": ...}]}`
- **Key detail**: This is the heaviest node in the supervisor graph. Uses medium-tier LLM. May make multiple tool calls (query strategies, query knowledge base, execute promotion/retirement).

### scheduled_tasks

- **Agent**: health_monitor (light tier)
- **Behavior**: Checks current time against last-run timestamps in DB for scheduled tasks (weekly community-intel scan, monthly execution audit, 30-min data freshness checks, daily preflight, daily digest). Fires coordination events for due tasks.
- **State update**: `{"scheduled_task_results": [{"task": ..., "was_due": ..., "fired": ...}]}`

## Graph Builder

File: `src/quantstack/graphs/supervisor/graph.py`

```python
def build_supervisor_graph(
    config_watcher: ConfigWatcher,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Build the supervisor pipeline graph.

    Reads agent configs from config_watcher, creates ChatModel instances
    per agent tier, binds them to node functions, and compiles the graph.
    """
```

The builder:

1. Reads agent configs: `config_watcher.get_config("health_monitor")`, etc.
2. Creates ChatModel instances: `get_chat_model("light")` for health_monitor and self_healer, `get_chat_model("medium")` for strategy_promoter.
3. Creates node functions with bound configs. Use closures or `functools.partial` to bind the ChatModel and AgentConfig to each node function. The node functions in `nodes.py` should accept the ChatModel and config as parameters (not just state).
4. Builds the `StateGraph(SupervisorState)`:
   - Adds 5 nodes
   - Adds linear edges: `START -> health_check -> diagnose_issues -> execute_recovery -> strategy_lifecycle -> scheduled_tasks -> END`
5. Compiles with the checkpointer and returns.

## Error Handling

Apply `retry_policy` based on node type:

- **health_check, scheduled_tasks** (tool-heavy, deterministic): Retry once. Failures are usually transient (DB connection, network).
- **diagnose_issues, execute_recovery** (agent, LLM-backed): Retry up to 2 times with exponential backoff. LLM calls can fail transiently (rate limits, timeouts).
- **strategy_lifecycle** (agent, medium-tier LLM): Retry up to 2 times with exponential backoff.

On all retries exhausted, the node should catch the exception and append to the `errors` state field rather than crashing the graph. The runner inspects `errors` in the final state to determine cycle health.

LangGraph's `retry_policy` is set per-node when adding nodes to the graph:

```python
from langgraph.pregel import RetryPolicy

graph.add_node("health_check", health_check_fn, retry=RetryPolicy(max_attempts=2))
graph.add_node("diagnose_issues", diagnose_issues_fn, retry=RetryPolicy(max_attempts=3))
```

Note: `max_attempts` includes the initial attempt, so `max_attempts=3` means 1 initial + 2 retries.

## Integration with Runners (section-11)

The supervisor runner will call `build_supervisor_graph()` each cycle (to pick up config changes via hot-reload), then invoke:

```python
result = await graph.ainvoke(
    {"cycle_number": n, "errors": []},
    config={
        "configurable": {"thread_id": f"supervisor-{date}-cycle-{n}"},
        "callbacks": [langfuse_handler],
    },
)
```

The initial state only needs `cycle_number` and an empty `errors` list. All other fields are populated by nodes.

## Implementation Checklist

1. Write all tests in `tests/unit/test_supervisor_graph.py` (they will fail initially)
2. Create `src/quantstack/graphs/supervisor/config/agents.yaml` adapted from existing CrewAI config
3. Implement node functions in `src/quantstack/graphs/supervisor/nodes.py`
4. Implement graph builder in `src/quantstack/graphs/supervisor/graph.py`
5. Create `src/quantstack/graphs/supervisor/__init__.py` with re-export
6. Run tests, iterate until green
7. Validate LangFuse tracing works by invoking the graph with a callback handler and inspecting trace output
