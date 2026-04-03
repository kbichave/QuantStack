# Section 05: Task Definitions and Crew Workflows

## Overview

This section defines the CrewAI task YAML configs and crew class implementations for all three crews: TradingCrew (11 sequential tasks), ResearchCrew (8 hierarchical tasks), and SupervisorCrew (5 sequential tasks with a built-in scheduler). Each crew class lives in its own module under `src/quantstack/crews/` and reads agent/task definitions from co-located `config/` YAML files.

## Dependencies

- **section-04-agent-definitions** — Agent YAML configs must exist before tasks can reference agents by ID.
- **section-02-llm-providers** — `get_model(tier)` must be available for model string injection into crew constructors.
- **section-03-tool-wrappers** — All `@tool`-decorated functions in `src/quantstack/crewai_tools/` must be importable.
- **section-12-risk-safety** — The programmatic safety boundary must be in place; the `execute_entries` task depends on the safety gate validating LLM risk decisions before order submission.

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/quantstack/crews/trading/config/tasks.yaml` | Create |
| `src/quantstack/crews/trading/crew.py` | Create |
| `src/quantstack/crews/research/config/tasks.yaml` | Create |
| `src/quantstack/crews/research/crew.py` | Create |
| `src/quantstack/crews/supervisor/config/tasks.yaml` | Create |
| `src/quantstack/crews/supervisor/crew.py` | Create |

All paths are relative to the project root `/Users/kshitijbichave/Personal/Trader/`.

---

## Tests (Write First)

All tests go in `tests/unit/test_crew_workflows.py`.

```python
# tests/unit/test_crew_workflows.py

"""
Tests for crew workflow definitions — task YAML validity, crew instantiation,
task ordering, and DAG correctness.
"""

import yaml
import pytest

TRADING_TASKS_PATH = "src/quantstack/crews/trading/config/tasks.yaml"
RESEARCH_TASKS_PATH = "src/quantstack/crews/research/config/tasks.yaml"
SUPERVISOR_TASKS_PATH = "src/quantstack/crews/supervisor/config/tasks.yaml"


# --- YAML validity ---

# Test: tasks.yaml for each crew is valid YAML
def test_trading_tasks_yaml_is_valid():
    """Load trading tasks.yaml and verify it parses without error."""

# Test: tasks.yaml for research crew is valid YAML
def test_research_tasks_yaml_is_valid():
    """Load research tasks.yaml and verify it parses without error."""

# Test: tasks.yaml for supervisor crew is valid YAML
def test_supervisor_tasks_yaml_is_valid():
    """Load supervisor tasks.yaml and verify it parses without error."""


# --- Required fields ---

# Test: each task has required fields: description, expected_output, agent
def test_trading_tasks_have_required_fields():
    """Every task in trading tasks.yaml must have description, expected_output, and agent."""

def test_research_tasks_have_required_fields():
    """Every task in research tasks.yaml must have description, expected_output, and agent."""

def test_supervisor_tasks_have_required_fields():
    """Every task in supervisor tasks.yaml must have description, expected_output, and agent."""


# --- TradingCrew task ordering ---

# Test: TradingCrew tasks are in correct order (safety_check first, persist_state last)
def test_trading_task_order():
    """
    The 11 TradingCrew tasks must appear in this order:
    safety_check, daily_plan, position_review, execute_exits, entry_scan,
    risk_sizing, portfolio_review, options_analysis, execute_entries,
    reflection, persist_state.
    """

# Test: task context dependencies form a valid DAG (no cycles)
def test_trading_task_dag_is_acyclic():
    """
    Parse context references in tasks.yaml. Verify no task references
    a task that comes after it (which would create a cycle in a sequential crew).
    """


# --- Crew instantiation ---

# Test: TradingCrew instantiates without error with mocked LLM
def test_trading_crew_instantiates(mock_llm):
    """
    Import TradingCrew, call its factory with a mock LLM provider,
    and verify the returned Crew object has 11 tasks and 10 agents.
    """

# Test: ResearchCrew instantiates without error with mocked LLM
def test_research_crew_instantiates(mock_llm):
    """
    Import ResearchCrew, call its factory with a mock LLM provider,
    and verify the returned Crew object has 8 tasks and 4 agents.
    """

# Test: SupervisorCrew instantiates without error with mocked LLM
def test_supervisor_crew_instantiates(mock_llm):
    """
    Import SupervisorCrew, call its factory with a mock LLM provider,
    and verify the returned Crew object has 5 tasks and 3 agents.
    """


# --- Async execution flags ---

# Test: position_review task has async_execution=True
def test_position_review_is_async():
    """position_review processes multiple positions in parallel."""

# Test: entry_scan task has async_execution=True
def test_entry_scan_is_async():
    """entry_scan debates multiple candidates in parallel."""


# --- Crew kickoff smoke tests ---

# Test: TradingCrew.kickoff() completes one cycle with mock tools
def test_trading_crew_kickoff_with_mocks(mock_llm, mock_tools):
    """
    Build TradingCrew with mock LLM and mock tools that return canned JSON.
    Call crew.kickoff(inputs={...}) and verify it returns without exception.
    All 11 tasks should execute in sequence.
    """

# Test: ResearchCrew.kickoff() completes one cycle with mock tools
def test_research_crew_kickoff_with_mocks(mock_llm, mock_tools):
    """
    Build ResearchCrew with mock LLM and mock tools.
    Call crew.kickoff(inputs={...}) and verify completion.
    """
```

---

## Implementation Details

### TradingCrew Task Definitions

The TradingCrew runs as a **sequential CrewAI process** with 11 tasks. This mirrors the existing `prompts/trading_loop.md` step-by-step flow where each step depends on the previous step's output.

**File: `src/quantstack/crews/trading/config/tasks.yaml`**

Each task entry follows CrewAI's YAML task format. Key fields per task:

| # | Task ID | Agent | Description Summary | Expected Output | Notes |
|---|---------|-------|---------------------|-----------------|-------|
| 1 | `safety_check` | `executor` | Check system status, kill switch, data freshness. If halted, output a HALT signal that short-circuits the rest of the cycle. | JSON: `{"status": "ok" or "halted", "reason": "..."}` | Must be first. |
| 2 | `daily_plan` | `daily_planner` | Generate daily trading plan with ranked watchlist and exit candidates. Runs once per day (check `loop_iteration_context` DB flag; if already planned today, return cached plan). Inputs: portfolio state, regime, RAG knowledge. | JSON: ranked watchlist array + exit recommendation array + reasoning. | Conditional: skips planning if already done today. |
| 3 | `position_review` | `position_monitor` | Review each open position. Recommend HOLD, TRIM, or CLOSE with reasoning for each. | JSON array: per-position `{"symbol", "action", "reasoning"}`. | `async_execution: true` — reviews multiple positions in parallel. Context: `[daily_plan]`. |
| 4 | `execute_exits` | `executor` | Execute any CLOSE/TRIM recommendations from position_review. | JSON: executed orders array with fill details. | Context: `[position_review]`. |
| 5 | `entry_scan` | `trade_debater` | For each watchlist symbol from daily_plan, conduct bull/bear/risk debate. Produce ENTER or SKIP verdict with reasoning. | JSON array: per-candidate `{"symbol", "verdict", "bull_case", "bear_case", "risk_notes"}`. | `async_execution: true` — parallel debates. Context: `[daily_plan]`. |
| 6 | `risk_sizing` | `risk_analyst` | For all ENTER verdicts from entry_scan, reason about position sizing. Receives full portfolio context (exposure, P&L, volatility, regime, RAG lessons). | JSON array: per-candidate `{"symbol", "recommended_size_pct", "reasoning"}`. | Context: `[entry_scan]`. Output is structured JSON with temperature 0. |
| 7 | `portfolio_review` | `fund_manager` | Review all sized candidates as a batch. Evaluate correlation, concentration, regime coherence, capital allocation. Approve, reject, or modify each. | JSON array: per-candidate `{"symbol", "decision", "modified_size_pct", "reasoning"}`. | Context: `[risk_sizing]`. |
| 8 | `options_analysis` | `options_analyst` | For approved options entries, select structure (spread, condor, straddle) with Greeks validation. | JSON: per-options-candidate `{"symbol", "structure", "legs", "greeks_summary", "reasoning"}`. | Conditional: only runs if approved entries include options domain. Context: `[portfolio_review]`. |
| 9 | `execute_entries` | `executor` | Execute approved entries through broker. Passes through programmatic safety boundary before order submission. | JSON: executed orders array with fill details. | Context: `[portfolio_review, options_analysis]`. |
| 10 | `reflection` | `trade_reflector` | For any positions closed this cycle, classify outcomes (win/loss/scratch), extract lessons, write to RAG knowledge base via `remember_knowledge_tool`. | Markdown summary of lessons learned + confirmation of RAG write. | Context: `[execute_exits]`. |
| 11 | `persist_state` | `executor` | Record heartbeat, write audit trail to DB, update coordination tables. | JSON: `{"heartbeat_written": true, "audit_entries": N}`. | Must be last. Context: none (standalone). |

**Context chaining:** Tasks reference earlier tasks via CrewAI's `context` field. For example, `entry_scan` has `context: [daily_plan]`, meaning it receives the daily_plan task's output as input. This forms a DAG:

```
safety_check -> daily_plan -> position_review -> execute_exits
                          \-> entry_scan -> risk_sizing -> portfolio_review -> options_analysis -> execute_entries
                                                                                              \-> reflection (from execute_exits)
                          persist_state (standalone, runs last)
```

### TradingCrew Class

**File: `src/quantstack/crews/trading/crew.py`**

The crew class is a factory that:
1. Reads `config/agents.yaml` and `config/tasks.yaml` using `CrewBase` decorators
2. Injects model strings from `get_model()` into agent `llm` fields
3. Assigns tool objects to agents based on their tool lists
4. Returns a configured `Crew` instance

```python
# src/quantstack/crews/trading/crew.py

"""TradingCrew — sequential 11-task trading cycle."""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from quantstack.llm.provider import get_model


@CrewBase
class TradingCrew:
    """Sequential trading workflow: safety -> plan -> review -> execute -> reflect."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def executor(self) -> Agent:
        """Trade executor agent."""
        ...

    @agent
    def daily_planner(self) -> Agent:
        """Daily planning agent."""
        ...

    # ... one @agent method per agent defined in agents.yaml ...

    @task
    def safety_check(self) -> Task:
        """First task: system health gate."""
        ...

    @task
    def daily_plan(self) -> Task:
        """Generate or retrieve daily trading plan."""
        ...

    # ... one @task method per task defined in tasks.yaml ...

    @crew
    def crew(self) -> Crew:
        """Assemble the TradingCrew."""
        return Crew(
            agents=self.agents,   # auto-collected by @CrewBase
            tasks=self.tasks,     # auto-collected by @CrewBase
            process=Process.sequential,
            memory=True,
            verbose=True,
        )
```

The `@CrewBase` decorator reads the YAML files and maps `{heavy_model}`, `{medium_model}`, `{light_model}` placeholders to the values provided at kickoff time via `inputs`. The crew factory function called by the runner passes these:

```python
def create_trading_crew() -> Crew:
    """Factory for runner to create a fresh TradingCrew each cycle."""
    trading = TradingCrew()
    return trading.crew()
```

The runner calls `crew.kickoff(inputs={"heavy_model": get_model("heavy"), "medium_model": get_model("medium"), ...})`.

### ResearchCrew Task Definitions

The ResearchCrew runs as a **hierarchical process** with `quant_researcher` as the manager agent. The manager decides which tasks to delegate and in what order, mirroring the current BLITZ mode where the research orchestrator delegates to domain specialists.

**File: `src/quantstack/crews/research/config/tasks.yaml`**

| # | Task ID | Agent | Description Summary | Expected Output |
|---|---------|-------|---------------------|-----------------|
| 1 | `load_context` | `quant_researcher` | Load heartbeat, DB state, memory, cross-domain intel. Always first. | JSON: current system context bundle. |
| 2 | `domain_selection` | `quant_researcher` | Score research domains (investment, swing, options) by portfolio gaps, recent P&L, strategy diversity. Select the domain and symbol(s) to research. | JSON: `{"domain", "symbols", "reasoning"}`. |
| 3 | `hypothesis_generation` | `quant_researcher` | Generate testable hypotheses for selected domain + symbol. Query RAG to check what has been tried before (workshop_lessons, negative results). | JSON array: hypotheses with testable predictions. |
| 4 | `signal_validation` | `strategy_rd` | Run information coefficient (IC) and alpha decay tests on hypothesized signals. | JSON: per-hypothesis IC scores, decay curves, pass/fail reasoning. |
| 5 | `backtest_validation` | `strategy_rd` | Run in-sample backtest, walk-forward, combinatorial CV. The agent reasons about whether results are sufficient — no hardcoded Sharpe threshold. | JSON: backtest results + agent's assessment with reasoning. |
| 6 | `ml_experiment` | `ml_scientist` | Train ML models if hypothesis involves learnable features. Select features, tune hyperparameters, detect concept drift. | JSON: model performance metrics, feature importance, drift assessment. |
| 7 | `strategy_registration` | `quant_researcher` | If validation passes the agent's reasoning threshold, register strategy as `draft` status. | JSON: registered strategy ID or skip reason. |
| 8 | `knowledge_update` | `quant_researcher` | Write discoveries (positive or negative) to RAG knowledge base via `remember_knowledge_tool`. Negative results are tagged as such to prevent re-exploring dead ends. | Confirmation of RAG write with document IDs. |

**Hierarchical process:** The `quant_researcher` agent acts as manager. It can choose to skip tasks (e.g., skip `ml_experiment` if the hypothesis does not involve learnable features) or re-run tasks with different parameters. CrewAI's hierarchical process handles this delegation automatically.

### ResearchCrew Class

**File: `src/quantstack/crews/research/crew.py`**

```python
# src/quantstack/crews/research/crew.py

"""ResearchCrew — hierarchical research with quant_researcher as manager."""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from quantstack.llm.provider import get_model


@CrewBase
class ResearchCrew:
    """Hierarchical research workflow managed by quant_researcher."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def quant_researcher(self) -> Agent:
        """Manager agent for research delegation."""
        ...

    @agent
    def strategy_rd(self) -> Agent:
        """Strategy validation specialist."""
        ...

    @agent
    def ml_scientist(self) -> Agent:
        """ML experiment runner."""
        ...

    @agent
    def community_intel(self) -> Agent:
        """Community intelligence scout."""
        ...

    # ... @task methods for each of the 8 tasks ...

    @crew
    def crew(self) -> Crew:
        """Assemble the ResearchCrew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=self.quant_researcher(),
            memory=True,
            verbose=True,
        )
```

### SupervisorCrew Task Definitions

The SupervisorCrew runs on a slower cadence (every 5 minutes, even on weekends). It monitors system health, handles recovery, manages strategy lifecycle, and runs scheduled tasks that replace the current `scripts/scheduler.py` cron jobs.

**File: `src/quantstack/crews/supervisor/config/tasks.yaml`**

| # | Task ID | Agent | Description Summary | Expected Output |
|---|---------|-------|---------------------|-----------------|
| 1 | `health_check` | `health_monitor` | Check heartbeats of trading-crew and research-crew. Check Langfuse, Ollama, ChromaDB reachability. | JSON: per-service health status. |
| 2 | `diagnose_issues` | `self_healer` | If any service is unhealthy, reason about root cause and recovery action. | JSON: `{"unhealthy_services", "diagnoses", "recommended_actions"}`. |
| 3 | `execute_recovery` | `self_healer` | Take recovery action: restart container via Docker API, flush stale data, switch LLM provider, activate kill switch if unrecoverable. | JSON: actions taken + results. |
| 4 | `strategy_lifecycle` | `strategy_promoter` | Query strategies in `forward_testing` status. For each, reason about whether to promote, extend testing, or retire based on performance evidence, market conditions, and RAG knowledge. No hardcoded thresholds — evidence-based reasoning. | JSON array: per-strategy `{"strategy_id", "decision", "reasoning"}`. |
| 5 | `scheduled_tasks` | `health_monitor` | Check if any scheduled task is due and create coordination events. | JSON: scheduled tasks executed this cycle. |

**Scheduled tasks embedded in task 5:**

The `scheduled_tasks` task replaces `scripts/scheduler.py`. The supervisor runner tracks scheduled task timestamps in PostgreSQL (`loop_iteration_context` table with key prefix `scheduler_*`). Each cycle, the health_monitor agent checks if any task is due:

| Schedule | Task | Implementation |
|----------|------|----------------|
| Weekly (Sunday 19:00 ET) | Community-intel scan | Publish coordination event `research_community_scan` for ResearchCrew to pick up. |
| Monthly | Execution researcher audit | Publish event if 20+ fills exist since last audit. |
| Every 30 min (market hours) | Data freshness check | Check last OHLCV timestamp, trigger refresh if stale. |
| Daily pre-market (9:00 AM ET) | Preflight checks | Verify data sync, Alpaca connection, credit regime. |
| Daily post-market (4:30 PM ET) | Daily digest | Generate digest, compact memory. |

The agent reads the current time and the last-run timestamps from the DB, then decides which scheduled tasks to fire. This is intentionally LLM-reasoned (rather than a cron table) so the agent can adapt — for example, skipping the community-intel scan if the research crew is already overloaded.

### SupervisorCrew Class

**File: `src/quantstack/crews/supervisor/crew.py`**

```python
# src/quantstack/crews/supervisor/crew.py

"""SupervisorCrew — system health, recovery, strategy lifecycle, scheduling."""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from quantstack.llm.provider import get_model


@CrewBase
class SupervisorCrew:
    """Sequential supervisor workflow: health -> diagnose -> recover -> lifecycle -> schedule."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def health_monitor(self) -> Agent:
        """System health monitor."""
        ...

    @agent
    def self_healer(self) -> Agent:
        """Self-healing engineer."""
        ...

    @agent
    def strategy_promoter(self) -> Agent:
        """Strategy lifecycle manager."""
        ...

    # ... @task methods for each of the 5 tasks ...

    @crew
    def crew(self) -> Crew:
        """Assemble the SupervisorCrew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            memory=True,
            verbose=True,
        )
```

---

## Task YAML Structure Reference

Each task in a `tasks.yaml` file follows this structure:

```yaml
safety_check:
  description: >
    Check system status including kill switch state, data freshness,
    and service health. If the system is halted or unhealthy, output
    a HALT signal. Current portfolio state: {portfolio_state}.
  expected_output: >
    JSON object with keys: status ("ok" or "halted"), reason (string),
    data_freshness_ok (boolean), kill_switch_active (boolean).
  agent: executor
```

Variables in `{curly_braces}` are interpolated at kickoff time from the `inputs` dict passed by the runner.

For tasks with context dependencies:

```yaml
entry_scan:
  description: >
    For each symbol on the daily plan watchlist, conduct a rigorous
    bull/bear/risk debate. Consider technical signals, fundamental data,
    market regime, and historical outcomes from the knowledge base.
    Produce an ENTER or SKIP verdict with detailed reasoning.
  expected_output: >
    JSON array where each element has: symbol, verdict ("enter" or "skip"),
    bull_case, bear_case, risk_notes, conviction_score (1-10).
  agent: trade_debater
  context:
    - daily_plan
  async_execution: true
```

For conditional tasks:

```yaml
options_analysis:
  description: >
    For approved entries that are in the options domain, select the optimal
    options structure (vertical spread, iron condor, straddle, etc.).
    Validate Greeks (delta, gamma, theta, vega). If no options entries
    were approved, output an empty result.
  expected_output: >
    JSON array of options structures with legs, strikes, expiry,
    Greeks summary, and max risk. Empty array if no options entries.
  agent: options_analyst
  context:
    - portfolio_review
```

---

## Key Design Decisions

**Sequential vs. hierarchical process:** TradingCrew and SupervisorCrew use `Process.sequential` because their task order is deterministic — safety must come before planning, planning before execution, etc. ResearchCrew uses `Process.hierarchical` because the quant_researcher manager needs flexibility to skip tasks (e.g., skip ML experiment if not applicable) or re-run validation with different parameters.

**Async execution for parallelizable sub-tasks:** `position_review` and `entry_scan` set `async_execution: true` because they process multiple independent items (positions and candidates respectively). CrewAI handles the parallelism internally.

**Context chaining vs. shared state:** Tasks pass data forward via CrewAI's `context` field rather than writing to shared state. This makes the data flow explicit and testable — you can verify the DAG from the YAML alone.

**Conditional tasks:** `options_analysis` and `reflection` handle their own conditionality in their descriptions. The agent returns an empty result if the condition is not met (e.g., no options entries, no positions closed). This avoids complex branching logic in the crew definition.

**Model string injection:** Agent YAML files use `{heavy_model}` / `{medium_model}` / `{light_model}` placeholders. These are resolved at kickoff time when the runner passes them in the `inputs` dict. This keeps the YAML provider-agnostic.

---

## Implementation Checklist

1. Write tests in `tests/unit/test_crew_workflows.py` (stubs above).
2. Create `src/quantstack/crews/trading/config/tasks.yaml` with all 11 task definitions.
3. Create `src/quantstack/crews/trading/crew.py` with `TradingCrew` class using `@CrewBase`.
4. Create `src/quantstack/crews/research/config/tasks.yaml` with all 8 task definitions.
5. Create `src/quantstack/crews/research/crew.py` with `ResearchCrew` class using `@CrewBase` and `Process.hierarchical`.
6. Create `src/quantstack/crews/supervisor/config/tasks.yaml` with all 5 task definitions.
7. Create `src/quantstack/crews/supervisor/crew.py` with `SupervisorCrew` class using `@CrewBase`.
8. Verify all task `agent` references match agent IDs defined in section-04 agent YAML configs.
9. Verify all `context` references form a valid DAG (no forward references in sequential crews).
10. Run `uv run pytest tests/unit/test_crew_workflows.py` — all tests pass.
