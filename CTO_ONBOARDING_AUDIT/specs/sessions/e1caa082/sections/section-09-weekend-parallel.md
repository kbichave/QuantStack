# Section 09: Weekend Parallel Research (AR-5)

## Overview

The weekend is 56 hours of idle compute (Friday 20:00 ET to Monday 04:00 ET). Currently, the only weekend activity is a single AutoResearchClaw run on Sunday evening. This section builds a parallel research coordinator that launches four independent research streams via LangGraph's `Send` API, each exploring a different alpha domain simultaneously. Results merge Monday morning into prioritized research tasks for the week ahead.

**Dependency:** section-04 (Budget Discipline) must be complete. The weekend runner uses per-experiment budget tracking and the experiment prioritization formula from that section.

---

## Tests First

### Unit Tests (`tests/unit/test_weekend_runner.py`)

```python
# --- Weekend Runner Coordination ---

# Test: weekend runner spawns exactly 4 parallel streams
def test_weekend_runner_spawns_four_streams(): ...

# Test: each stream has isolated state (no cross-contamination)
def test_stream_state_isolation(): ...

# Test: results merge via reducer into weekend_research_results list
def test_results_merge_via_reducer(): ...

# Test: synthesis node produces prioritized research tasks from 4 stream results
def test_synthesis_produces_prioritized_tasks(): ...

# Test: runner operates from Friday 20:00 to Monday 04:00 ET
def test_runner_time_window(): ...

# Test: individual stream failure doesn't crash other streams
def test_stream_failure_isolation(): ...
```

### Unit Tests (Per Stream)

```python
# --- Factor Mining Stream ---

# Test: factor_mining stream extracts testable factor from mock paper reference
def test_factor_mining_extracts_factor(): ...

# --- Regime Research Stream ---

# Test: regime_research stream labels regimes from historical data
def test_regime_research_labels_regimes(): ...

# --- Cross-Asset Signals Stream ---

# Test: cross_asset_signals stream computes lead-lag correlation
def test_cross_asset_lead_lag(): ...

# --- Portfolio Construction Stream ---

# Test: portfolio_construction stream compares risk parity vs equal weight
def test_portfolio_construction_comparison(): ...
```

---

## Architecture

### Four Parallel Research Streams

The weekend runner is a LangGraph StateGraph that uses the `Send` API to fan out work to four independent subgraphs. Each subgraph is a complete research pipeline focused on one alpha domain.

```
weekend_runner (parent graph)
  |
  |--> Send("factor_mining", stream_state)
  |--> Send("regime_research", stream_state)
  |--> Send("cross_asset_signals", stream_state)
  |--> Send("portfolio_construction", stream_state)
  |
  |--- wait for all ---
  |
  |--> synthesis_node (Sonnet, merges all 4 stream results)
  |
  |--> write prioritized research_queue tasks
```

Each `Send` target is an independent subgraph with its own state slice. There is no shared mutable state between streams. Results merge via a list reducer on the parent state's `weekend_research_results` field.

### Stream Descriptions

**1. Factor Mining Stream** (`streams/factor_mining.py`)

Takes academic paper references (from knowledge graph seeds, arxiv queries via web search tools) and extracts testable factor definitions. Computes IC on universe symbols. Logs results to `autoresearch_experiments`.

- Haiku for paper parsing and factor extraction
- Deterministic IC computation
- Output: list of `{factor_name, definition, ic, source_paper}` dicts

**2. Regime Research Stream** (`streams/regime_research.py`)

Takes historical price data across multiple timeframes (daily, weekly, monthly). Labels regimes using the existing RegimeDetector. Tests regime-conditional allocation rules (e.g., "in trending_up, overweight momentum; in ranging, overweight mean_reversion"). Computes regime transition probabilities.

- Mostly deterministic with Haiku for transition hypothesis generation
- Output: list of `{regime_rule, backtest_sharpe, transition_matrix}` dicts

**3. Cross-Asset Signal Stream** (`streams/cross_asset_signals.py`)

Takes bond yields, FX rates, commodity prices from the existing data acquisition pipeline. Computes lead-lag relationships at various lags (1-21 days). Identifies cross-asset signals that predict equity returns.

- Purely deterministic: correlation analysis, Granger causality tests
- No LLM usage
- Output: list of `{signal_name, lead_asset, lag_days, correlation, p_value}` dicts

**4. Portfolio Construction Stream** (`streams/portfolio_construction.py`)

Takes existing strategies from the strategies table. Tests alternative portfolio optimizers against the current equal-weight allocation:
- Risk parity
- Black-Litterman
- Hierarchical Risk Parity (HRP)

Compares portfolio-level Sharpe, max drawdown, and turnover.

- Mostly deterministic (optimization math)
- Output: list of `{optimizer_name, sharpe, max_dd, turnover, vs_equal_weight}` dicts

### State Isolation

Each stream receives its own state via `Send`. The parent graph defines a `WeekendResearchState`:

```python
class StreamResult(BaseModel):
    stream_name: str
    findings: list[dict]       # stream-specific result dicts
    experiments_run: int
    cost_usd: float
    errors: list[str]

class WeekendResearchState(BaseModel):
    start_time: datetime
    end_time: datetime
    budget_remaining: float = 50.0   # ~$50/weekend
    weekend_research_results: Annotated[list[StreamResult], operator.add]  # reducer
    synthesis_tasks: list[dict] = []
```

The `operator.add` reducer on `weekend_research_results` collects results from all four streams without cross-contamination.

### Monday Synthesis

At Monday 04:00 ET (or when all streams complete, whichever comes first), a synthesis node runs with Sonnet. It reviews all four stream results and creates prioritized research tasks by looking for cross-domain patterns. For example:

- Factor mining found momentum works in trending regime + regime research found we're entering a trending regime = activate momentum strategies
- Cross-asset signals found bond yields leading equities by 5 days + portfolio construction found risk parity outperforms = combine into a macro-aware risk parity strategy

The synthesis node writes tasks to the `research_queue` table with source `"weekend_synthesis"`. These are picked up by the research graph's `context_load` node during Monday's first cycle.

---

## Budget

Total budget is approximately $50 per weekend (user confirmed budget flexibility). The breakdown:

- Factor mining: ~$5 (Haiku for paper parsing, ~50 papers)
- Regime research: ~$3 (Haiku for transition hypotheses, mostly deterministic)
- Cross-asset signals: ~$0 (purely deterministic)
- Portfolio construction: ~$1 (minor Haiku calls for interpretation)
- Monday synthesis: ~$0.15 (one Sonnet call)
- Headroom: ~$41

Most compute is deterministic (backtests, correlations, IC calculations). LLM usage is primarily Haiku for hypothesis generation in the factor mining and regime research streams.

Budget tracking uses the same per-experiment mechanism from section-04. Each stream tracks its own cumulative cost. The parent graph monitors total cost across all streams and can halt individual streams if the total approaches the $50 ceiling.

---

## Scheduling

The weekend runner is triggered by the scheduler at Friday 20:00 ET. It runs continuously until Monday 04:00 ET (or budget exhaustion).

Modify `scripts/scheduler.py` to add a Friday 20:00 cron entry that launches the weekend runner. The runner should be a separate Docker service or a subprocess managed by the scheduler, similar to how the overnight runner (section-07) is scheduled.

The runner must be idempotent on restart: if the system crashes and restarts mid-weekend, completed experiments are already persisted to DB and are not re-run. The runner reads cumulative cost from DB and resumes from where it left off.

---

## Error Handling

- **Individual stream failure:** If one stream crashes (e.g., data fetch timeout, LLM error), the other three continue. The parent graph's `Send` fan-out handles this by collecting whatever results are available when the time window closes. The failed stream's `StreamResult` includes the error in its `errors` list.
- **Total runner crash:** On restart, read persisted state from DB. Each stream logs completed experiments individually. Resume from last checkpoint.
- **Budget exhaustion:** If total cost across all streams hits $47.50 (95% of $50), halt all streams gracefully and proceed to synthesis with whatever results exist.

---

## Files to Create/Modify

```
src/quantstack/research/
  weekend_runner.py                  # NEW: 56-hour parallel research coordinator
  streams/
    __init__.py                      # NEW: streams package
    factor_mining.py                 # NEW: academic paper -> testable factors
    regime_research.py               # NEW: regime labeling + conditional allocation
    cross_asset_signals.py           # NEW: lead-lag cross-asset analysis
    portfolio_construction.py        # NEW: alternative optimizer comparison

scripts/
  scheduler.py                       # MODIFY: add Friday 20:00 weekend launch

tests/unit/
  test_weekend_runner.py             # NEW: all unit tests above
```

---

## Implementation Notes

- The `Send` API is LangGraph's mechanism for dynamic fan-out. Each `Send` call specifies a target node name and a state payload. The target node runs as an independent branch. Results are collected via reducers on the parent state.
- All four streams should import shared utilities from `src/quantstack/core/` for IC computation, regime detection, and backtesting. Do not duplicate this logic.
- The synthesis node (Sonnet) should have access to the knowledge graph tools (from section-10, when available) to check novelty of weekend findings before creating research tasks. If the knowledge graph is not yet built, skip the novelty check and create all tasks.
- No GPU is needed for weekend research. All ML experiments requiring training are deferred to the overnight loop (section-07). Weekend research focuses on breadth (many hypotheses across domains) over depth (model training).
- Stream subgraphs should be defined in their respective files and imported by `weekend_runner.py`. Each stream file exports a `build_<stream_name>_graph()` function that returns a compiled `StateGraph`.
