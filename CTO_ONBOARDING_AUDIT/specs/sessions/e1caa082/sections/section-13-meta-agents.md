# Section 13: Metacognitive Self-Modification (Meta Agents)

## Overview

This section implements four meta agents that continuously improve the system's own prompts, thresholds, tool bindings, and architecture. These agents run as supervisor graph nodes on varying cadences (weekly, monthly, quarterly) and are fully autonomous — no human review gate. Safety is enforced through protected file allowlists, regression test gates, auto-revert on Sharpe decline, and git audit trails.

This is the capstone of Phase 10. It transforms QuantStack from a system that trades into a system that improves how it trades.

## Dependencies

- **Section 10 (Knowledge Graph):** The prompt optimizer and architecture critic query the knowledge graph for experiment history and factor overlap when evaluating agent performance.
- **Section 11 (Consensus Validation):** Meta agents must be aware of the consensus subgraph when optimizing trading graph prompts (bull/bear/arbiter agents are optimization targets).
- **Section 12 (Governance):** Meta agents target the NEW agent hierarchy (CIO + strategy agents at Haiku tier), not the old flat Sonnet model. Governance must be in place before meta agents come online, otherwise prompt optimization targets the wrong agent structure.
- **Section 01 (DB Migrations):** The `meta_optimizations` table must exist.
- **Section 05 (Event Bus):** The `META_OPTIMIZATION_APPLIED` event type must be registered.

## Problem

Agent prompts and thresholds are static. The hypothesis_critique threshold (0.7), backtest Sharpe gate (0.5), IC gate (0.02) were chosen once and never updated based on outcomes. Prompt text was written once and not optimized for the system's evolving needs. There is no feedback loop from trade outcomes back to the prompts and parameters that generated the trade decisions.

## Tests First

All tests use pytest with existing fixtures from `tests/unit/conftest.py`.

### Unit Tests: `tests/unit/test_meta_agents.py`

```python
# Test: meta_prompt_optimizer generates <= 3 prompt variants per week per agent
# Test: A/B split applies new prompt to 50% of cycles
# Test: auto-revert triggers when 30-day Sharpe declines >10% after prompt change
# Test: protected file allowlist blocks modification of risk_gate.py
# Test: protected file allowlist blocks modification of kill_switch.py
# Test: protected file allowlist blocks modification of db.py
# Test: all meta changes committed with "meta:" prefix
# Test: regression test failure triggers auto-revert
```

### Unit Tests: `tests/unit/test_threshold_tuner.py`

```python
# Test: threshold lowered by 0.05 when false rejection rate > 20%
# Test: threshold raised by 0.05 when false acceptance rate > 30%
# Test: threshold never drops below floor (e.g., hypothesis_critique >= 0.4)
# Test: threshold never exceeds ceiling (e.g., hypothesis_critique <= 0.9)
# Test: threshold unchanged when both rates within acceptable range
# Test: thresholds read from thresholds.yaml, not hardcoded
```

### Unit Tests: `tests/unit/test_tool_selector.py`

```python
# Test: never-used tools removed from agent binding
# Test: frequently-searched deferred tools promoted to always-loaded
# Test: tool changes written to agents.yaml
```

### Unit Tests: `tests/unit/test_architecture_critic.py`

```python
# Test: bottleneck identified as node with highest token_cost / alpha ratio
# Test: recursive modification: critic modifies optimizer prompt when optimizer underperforms
# Test: quarterly cadence enforced (no run within 90 days of last)
```

## Design

### Prerequisite: Threshold Extraction

Before any meta agent can tune thresholds, hardcoded thresholds scattered throughout node logic must be extracted into a centralized `thresholds.yaml` config file. This is a refactoring step that must happen first within this section.

Thresholds to extract include:

- `hypothesis_critique` confidence gate (currently 0.7 in research nodes)
- Backtest Sharpe gate (currently 0.5 in backtest validation)
- OOS IC gate (currently 0.02 in overnight autoresearch and research nodes)
- Any other numeric gates in graph node logic that control strategy acceptance/rejection

The config loader reads `thresholds.yaml` at startup and exposes values via a simple accessor. All node logic references thresholds through this accessor, never through hardcoded literals.

File: `src/quantstack/meta/config.py` — loads `thresholds.yaml`, provides `get_threshold(name: str) -> float`, tracks version for audit.

File: `src/quantstack/meta/thresholds.yaml` — YAML file with structure:

```yaml
hypothesis_critique:
  value: 0.7
  floor: 0.4
  ceiling: 0.9
backtest_sharpe:
  value: 0.5
  floor: 0.2
  ceiling: 1.0
oos_ic:
  value: 0.02
  floor: 0.005
  ceiling: 0.05
```

### Meta Agent 1: Prompt Optimizer (weekly)

**Cadence:** Weekly, runs as a supervisor graph node.

**Input:** Agent outcomes over trailing 30 days from `skill_tracker`, `outcome_tracker`, and `autoresearch_experiments` tables. For each task agent, compares outcomes before/after the most recent prompt change (if any). Identifies patterns: which agents underperform, what hypothesis types they miss.

**Action:** Uses DSPy's MIPROv2 optimizer to search for better instructions. The optimization metric is strategy Sharpe improvement over 30 days. Generates up to 3 prompt variants per agent per week. Deploys the most promising variant via A/B split: 50% of cycles use the new prompt, 50% use the current prompt. After 1 week, evaluates. If improved: adopt permanently. If not: revert.

**Output:** Updated agent prompt written to `agents.yaml` (or a `prompt_versions/` directory), committed to git with `meta:` prefix.

**DSPy integration:** The optimizer wraps the agent's prompt as a DSPy module signature, defines the metric function (Sharpe delta), and runs MIPROv2 with a small search budget (~$2 per optimization run, ~$8/month total). The DSPy dependency is isolated to this module — no other part of the codebase imports DSPy.

**A/B split mechanism:** A simple flag on the agent config (`prompt_variant: "A"` or `"B"`) read at cycle start. The supervisor node randomly assigns variant per cycle. Results tagged with variant ID in Langfuse traces for comparison.

### Meta Agent 2: Threshold Tuner (monthly)

**Cadence:** Monthly, runs as a supervisor graph node.

**Input:** 30-day outcome data. For each gate threshold, computes:
- False rejection rate: strategies that were rejected by the gate but would have succeeded (measured by paper-trading rejected candidates for 30 days post-rejection).
- False acceptance rate: strategies that passed the gate but subsequently failed (negative Sharpe over 30 days post-acceptance).

**Action:**
- If false rejection rate > 20%: lower threshold by 0.05.
- If false acceptance rate > 30%: raise threshold by 0.05.
- Threshold unchanged when both rates are within acceptable ranges.

**Constraints — hard floors and ceilings:**
- `hypothesis_critique`: [0.4, 0.9]
- `backtest_sharpe`: [0.2, 1.0]
- `oos_ic`: [0.005, 0.05]

Thresholds never adjust beyond these bounds. The tuner reads from and writes to `thresholds.yaml` via the config module. Changes are committed with `meta:` prefix and logged to `meta_optimizations` table.

### Meta Agent 3: Tool Selector (weekly)

**Cadence:** Weekly, runs as a supervisor graph node.

**Input:** Tool usage data from `tool_health` table combined with agent outcome data.

**Analysis per agent:**
- Tools in the agent's binding that are never invoked (waste of prompt cache space and context window).
- Tools NOT in the agent's binding that the agent searches for via deferred loading (should be promoted to always-loaded).
- Tools correlated with successful outcomes (boost priority in binding order).

**Action:** Updates `agents.yaml` tool bindings. Adds high-demand deferred tools to the always-loaded set. Removes never-used tools from the binding. Changes committed with `meta:` prefix.

### Meta Agent 4: Architecture Critic (quarterly)

**Cadence:** Quarterly (every 90 days), runs as a supervisor graph node. Cadence enforced by checking `meta_optimizations` for the last `architecture_critic` run — if fewer than 90 days have passed, the node is a no-op.

**Input:**
- Portfolio Sharpe vs. benchmark (SPY) over 90 days.
- Per-node latency and token consumption from Langfuse traces.
- Strategy win rates by domain (swing, investment, options, mean_reversion).

**Analysis:** Identifies the bottleneck node — the node with the highest token cost per unit of alpha generated, or the longest latency on the critical path. Proposes improvement: is the bottleneck caused by a bad prompt (refer to prompt_optimizer)? A missing tool (refer to tool_selector)? A structural graph issue (propose graph restructuring)?

**Recursive self-modification:** If `meta_prompt_optimizer`'s suggestions do not improve performance over 2 consecutive weeks, the architecture critic modifies the optimizer's own prompt. This is the recursive self-modification loop — the system can improve the thing that improves it. The critic targets the optimizer's DSPy metric function and search parameters, not just the prompt text.

### Guardrails

All meta agent actions pass through a guardrails module before taking effect.

**Protected file allowlist** (`src/quantstack/meta/guardrails.py`):

Meta agents may only modify files matching an explicit allowlist. The following files are NEVER modifiable by meta agents:
- `src/quantstack/execution/risk_gate.py`
- `src/quantstack/execution/kill_switch.py`
- `src/quantstack/db.py`
- Anything in `src/quantstack/execution/` (entire execution layer)

The allowlist check runs before any git commit. If a meta agent attempts to modify a protected file, the commit is blocked and the change is reverted.

**Regression test gate:** Before any meta change goes live, the full regression test suite runs. If any test fails, the change is auto-reverted. This includes existing risk_gate tests, kill_switch tests, and all new Phase 10 tests.

**Sharpe monitoring:** A 30-day rolling Sharpe is tracked. If Sharpe declines more than 10% after a meta change (compared to the 30-day Sharpe before the change), the change is automatically reverted to the previous version.

**Commit audit trail:** All meta agent changes are committed to git with a `meta:` prefix in the commit message (e.g., `meta: prompt_optimizer updated swing_agent prompt variant B`). This provides a complete audit trail of what changed, when, and why.

**Rate limits:** Maximum 3 prompt variants per week per agent. This prevents churn and gives each variant enough time to be evaluated.

## Files to Create/Modify

```
src/quantstack/meta/
  __init__.py                 # NEW: package init
  prompt_optimizer.py         # NEW: DSPy-based prompt optimization (weekly)
  threshold_tuner.py          # NEW: gate threshold adjustment (monthly)
  tool_selector.py            # NEW: agent tool binding optimization (weekly)
  architecture_critic.py      # NEW: quarterly bottleneck analysis + recursive modification
  guardrails.py               # NEW: file allowlist, auto-revert, regression runner
  config.py                   # NEW: thresholds.yaml loader, get_threshold(), version tracking
  thresholds.yaml             # NEW: centralized threshold config with floors/ceilings

src/quantstack/graphs/supervisor/
  nodes.py                    # MODIFY: add meta agent scheduling nodes (weekly/monthly/quarterly)
  graph.py                    # MODIFY: wire meta nodes into supervisor graph

src/quantstack/graphs/research/
  nodes.py                    # MODIFY: replace hardcoded thresholds with get_threshold() calls

src/quantstack/graphs/trading/
  config/agents.yaml          # MODIFY: add prompt_variant field for A/B testing
```

## Implementation Notes

**DSPy dependency:** Add `dspy-ai` to `pyproject.toml` dependencies. DSPy is only imported in `prompt_optimizer.py` — it is not a system-wide dependency. If DSPy is unavailable at runtime, the prompt optimizer logs a warning and skips its cycle (graceful degradation).

**Meta optimization logging:** Every meta agent action is logged to the `meta_optimizations` table (created in section-01-db-migrations) with: agent_id, change_type (prompt/threshold/tool/architecture), before_value, after_value, metric_before, metric_after, status (applied/reverted), timestamp.

**Supervisor graph wiring:** Each meta agent is a conditional node in the supervisor graph. The supervisor checks cadence (weekly/monthly/quarterly) at each cycle and only routes to the meta node when the cadence window has elapsed. This uses existing supervisor scheduling patterns — no new scheduling infrastructure needed.

**Order of implementation within this section:**
1. Threshold extraction (`config.py` + `thresholds.yaml` + refactor node logic)
2. Guardrails module (`guardrails.py`)
3. Threshold tuner (`threshold_tuner.py`) — simplest meta agent, good first test
4. Tool selector (`tool_selector.py`)
5. Prompt optimizer (`prompt_optimizer.py`) — depends on DSPy setup
6. Architecture critic (`architecture_critic.py`) — depends on all others being operational
7. Supervisor graph wiring

**Cost:** ~$2 per prompt optimization run (weekly) + negligible for threshold tuner and tool selector. Architecture critic uses Sonnet for analysis (~$0.15 per quarterly run). Total meta agent cost: ~$8-10/month.
