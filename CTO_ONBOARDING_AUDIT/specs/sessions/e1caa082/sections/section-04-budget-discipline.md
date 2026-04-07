# Section 04: Experiment Budget Discipline (AR-9)

## Purpose

Research agents currently have no cost constraints. A single research cycle can consume unlimited tokens if hypothesis generation retries repeatedly or ML experiment nodes train multiple models. There is no mechanism to compare experiment value against compute cost, and no graceful degradation when budget is exhausted.

This section introduces per-cycle token and dollar budgets on both ResearchState and TradingState, a conditional routing mechanism that synthesizes partial results when budget is exhausted, a prioritization formula that ranks experiments by expected value per compute dollar, and a 3-window patience protocol that prevents premature hypothesis rejection.

## Dependencies

- **section-01-db-migrations**: Budget fields are added to graph state (in-memory Pydantic models), not database tables, so no direct DB dependency. However, the experiment_prioritizer reads from the knowledge graph tables (kg_nodes) when computing novelty_score — until section-10 (knowledge graph) is live, novelty_score defaults to 1.0 for all hypotheses.
- **section-05-event-bus-extensions**: No new event types are introduced by this section. No dependency.

## What This Section Blocks

- **section-07-overnight-autoresearch**: The overnight runner references the patience protocol (3-window validation) and per-experiment budget. Those mechanisms are defined here.
- **section-08-feature-factory**: Feature screening uses IC thresholds that the patience protocol governs.
- **section-09-weekend-parallel**: Weekend streams inherit the prioritization formula for experiment selection.

---

## Tests First

All tests go in the files listed below. Write test stubs with docstrings describing the assertion, then implement.

### `tests/unit/test_budget_discipline.py`

```python
"""Budget tracking on graph state and exhaustion routing."""


def test_research_state_default_budget():
    """ResearchState initializes with token_budget_remaining=50_000 and cost_budget_remaining=0.50."""


def test_trading_state_default_budget():
    """TradingState initializes with token_budget_remaining=30_000 and cost_budget_remaining=0.20."""


def test_budget_check_continue_when_remaining_exceeds_estimate():
    """budget_check returns 'continue' when remaining > estimated_next_cost."""


def test_budget_check_synthesize_when_remaining_below_estimate():
    """budget_check returns 'synthesize' when remaining < estimated_next_cost."""


def test_budget_check_synthesize_when_remaining_zero():
    """budget_check returns 'synthesize' when remaining == 0."""


def test_synthesize_partial_results_produces_summary():
    """synthesize_partial_results node returns a dict with 'summary' key
    that references the partial state accumulated so far."""


def test_budget_fields_compatible_with_extra_forbid():
    """Constructing ResearchState with the new budget fields does not raise
    ValidationError, confirming compatibility with extra='forbid'."""
```

### `tests/unit/test_experiment_prioritizer.py`

```python
"""Experiment prioritization formula and queue sorting."""


def test_high_ic_cheap_ranked_above_low_ic_expensive():
    """An experiment with expected_IC=0.05, cost=5000 tokens ranks above one
    with expected_IC=0.02, cost=30000 tokens."""


def test_novelty_score_novel_hypothesis():
    """novelty_score returns 1.0 when no similar hypothesis exists in knowledge graph."""


def test_novelty_score_known_hypothesis():
    """novelty_score returns 0.1 when a similar hypothesis exists in knowledge graph."""


def test_cold_start_all_novel():
    """When KG is empty, all hypotheses get novelty_score=1.0 and are ranked
    by regime_fit / compute_cost."""


def test_prioritizer_empty_queue():
    """prioritize_experiments returns an empty list when given an empty queue."""
```

### `tests/unit/test_patience_protocol.py`

```python
"""3-window backtest patience protocol."""


def test_rejected_only_when_all_three_windows_fail():
    """A hypothesis is rejected only when it fails gates in ALL 3 windows."""


def test_provisional_when_two_of_three_pass():
    """A hypothesis is marked 'provisional' when exactly 2 of 3 windows pass."""


def test_fully_accepted_when_three_of_three_pass():
    """A hypothesis is fully accepted when all 3 windows pass."""


def test_three_windows_are_full_recent_stressed():
    """The 3 windows are: full historical period, recent 12 months,
    and a configurable stressed period (default: 2020-03 to 2020-06)."""


def test_stressed_period_configurable():
    """The stressed period start/end dates are configurable, not hardcoded."""
```

---

## Implementation Details

### 1. Add Budget Fields to Graph State

**File:** `src/quantstack/graphs/state.py`

Add four new fields to `ResearchState`:

```python
class ResearchState(BaseModel):
    # ... existing fields ...

    # Budget discipline (AR-9)
    token_budget_remaining: int = 50_000   # per-cycle token cap
    cost_budget_remaining: float = 0.50    # per-cycle dollar cap
    tokens_consumed: int = 0
    cost_consumed: float = 0.0
```

Add the same pattern to `TradingState` with different defaults:

```python
class TradingState(BaseModel):
    # ... existing fields ...

    # Budget discipline (AR-9)
    token_budget_remaining: int = 30_000
    cost_budget_remaining: float = 0.20
    tokens_consumed: int = 0
    cost_consumed: float = 0.0
```

**Migration note:** Both models use `ConfigDict(extra="forbid")`. Adding new fields with defaults is backward-compatible for new cycles. However, in-flight LangGraph checkpoints serialized before the change will fail deserialization because they lack the new keys. This requires a clean restart: stop all graph services, clear the checkpoint store (or let it expire), and restart. This is a one-time cost.

**Scope clarification:** These per-cycle budgets apply to the daytime interactive research and trading graphs only. The overnight autoresearch runner (section-07) uses its own per-experiment budget of ~$0.10, independent of these per-cycle limits. The two budget systems do not interact.

### 2. Budget Check Conditional Edge

**File:** `src/quantstack/graphs/research/graph.py` (and corresponding `nodes.py`)

After each LLM-calling node in the research graph, add a `budget_check` conditional edge. The check is a pure function (no LLM call):

```python
def budget_check(state: ResearchState) -> str:
    """Route to 'continue' or 'synthesize' based on remaining budget.

    Estimates the cost of the next node based on hypothesis complexity:
    - Simple rule-based hypothesis: ~5,000 tokens
    - ML model experiment: ~30,000 tokens
    Uses the larger estimate when ambiguous.
    """
    ...
```

The function compares `state.token_budget_remaining` and `state.cost_budget_remaining` against the estimated cost of the next node. If either is insufficient, return `"synthesize"` to route to the partial-results terminal node.

**Budget tracking mechanism:** LangFuse callbacks already track token usage per LLM call. After each LLM-calling node returns, a wrapper (in the node function or a graph middleware) extracts the token count from the LangFuse callback metadata, computes the dollar cost using model-specific rates, and updates `tokens_consumed`, `cost_consumed`, `token_budget_remaining`, and `cost_budget_remaining` in the returned state dict.

Model-specific cost rates (per 1M tokens, input/output averaged for simplicity):

| Model | Rate |
|-------|------|
| Haiku | ~$0.75/MTok |
| Sonnet | ~$9.00/MTok |
| Opus | ~$45.00/MTok |

These rates should live in `src/quantstack/llm/config.py` as a dict keyed by model tier, not hardcoded in the budget check.

### 3. Synthesize Partial Results Node

**File:** `src/quantstack/graphs/research/nodes.py` (add new node factory), `src/quantstack/graphs/research/graph.py` (wire the node)

When budget is exhausted, the graph routes to a `synthesize_partial_results` terminal node instead of aborting. This node:

1. Reads whatever state has been accumulated so far (context_summary, hypothesis, any partial validation_results).
2. Produces a structured summary: what was accomplished, what was deferred, how much budget was consumed.
3. Logs the deferred work as a `research_queue` entry so it can be picked up in the next cycle or overnight.
4. Returns the summary in `decisions` and routes to END.

This is a Haiku call (cheap) or even deterministic (template-based) depending on how much context needs summarizing. Recommend deterministic for budget-exhaustion cases to avoid spending more tokens.

**Graph wiring:** Add conditional edges after `hypothesis_generation`, `signal_validation`, `backtest_validation`, and `ml_experiment` nodes. Each routes to `synthesize_partial_results` if budget_check returns `"synthesize"`, otherwise continues to the next node in the pipeline.

### 4. Experiment Prioritization Formula

**File:** `src/quantstack/learning/experiment_prioritizer.py` (new file)

The prioritizer ranks competing experiments before the research graph selects which to validate:

```python
def compute_priority(
    expected_ic: float,
    regime_fit: float,
    novelty_score: float,
    estimated_compute_cost: int,
) -> float:
    """Rank experiments by expected value per compute dollar.

    priority = (expected_ic * regime_fit * novelty_score) / estimated_compute_cost

    Parameters
    ----------
    expected_ic : float
        Prior from similar strategies in knowledge graph, or 0.03 default
        for novel hypotheses.
    regime_fit : float
        How well the hypothesis matches current regime (from RegimeDetector).
        Range [0.0, 1.0].
    novelty_score : float
        1.0 if not in knowledge graph, 0.1 if similar hypothesis tested before.
    estimated_compute_cost : int
        Token estimate. Simple rule = 5,000. ML model = 30,000.
    """
    ...


def prioritize_experiments(
    experiments: list[dict],
    current_regime: str,
) -> list[dict]:
    """Sort experiment queue by priority (descending).

    Each experiment dict must contain: 'hypothesis', 'complexity'
    ('simple' or 'ml'), and optionally 'expected_ic'.

    Uses knowledge graph for novelty check when available (falls back
    to novelty_score=1.0 when KG is not yet built).
    """
    ...
```

The research graph's `context_load` node calls `prioritize_experiments` to sort the experiment queue before selecting which hypotheses to validate in the current cycle. The budget then determines how many experiments can actually run.

**Knowledge graph integration:** `novelty_score` calls `check_hypothesis_novelty` from section-10 when available. Before section-10 is implemented, it returns 1.0 for all hypotheses (everything is treated as novel). This is handled via a try/import guard or a feature flag, not a stub.

### 5. Patience Protocol (3-Window Validation)

**File:** `src/quantstack/core/backtesting.py` (new file, since it does not exist yet)

The existing backtest validation in the research graph uses a single time window. The patience protocol changes this to 3 mandatory windows:

1. **Full historical period** (e.g., 2020-01-01 to present)
2. **Recent period** (last 12 months from today)
3. **Stressed period** (configurable; default: 2020-03-01 to 2020-06-30, the COVID crash)

```python
@dataclass
class PatienceConfig:
    """Configuration for multi-window validation."""
    full_start: str = "2020-01-01"
    recent_months: int = 12
    stressed_start: str = "2020-03-01"
    stressed_end: str = "2020-06-30"


@dataclass
class WindowResult:
    """Result from a single validation window."""
    window_name: str       # "full", "recent", "stressed"
    passed: bool
    sharpe: float
    max_drawdown: float
    ic: float


def evaluate_patience(results: list[WindowResult]) -> str:
    """Determine hypothesis status from multi-window results.

    Returns
    -------
    'accepted' if all 3 windows pass.
    'provisional' if exactly 2 of 3 pass (lower confidence, smaller position sizing).
    'rejected' if fewer than 2 pass.
    """
    ...
```

**Provisional strategies:** When a hypothesis passes 2/3 windows, it is registered with status `'provisional'` rather than `'draft'`. Provisional strategies receive a position sizing scalar of 0.5x (via the existing `FORWARD_TESTING_SIZE_SCALAR` mechanism). This prevents premature rejection of strategies that work in specific regimes while limiting exposure to unproven edge cases.

**Integration point:** The research graph's `backtest_validation` node calls the patience protocol instead of a single-window evaluation. The `morning_validator` in section-07 also uses this protocol for overnight winner validation.

---

## Files Summary

| File | Action | What Changes |
|------|--------|-------------|
| `src/quantstack/graphs/state.py` | MODIFY | Add 4 budget fields to ResearchState, 4 to TradingState |
| `src/quantstack/graphs/research/nodes.py` | MODIFY | Add budget tracking after LLM nodes, add synthesize_partial_results node |
| `src/quantstack/graphs/research/graph.py` | MODIFY | Wire budget_check conditional edges, synthesize_partial_results routing |
| `src/quantstack/learning/experiment_prioritizer.py` | CREATE | Prioritization formula, queue sorting, novelty score lookup |
| `src/quantstack/core/backtesting.py` | CREATE | PatienceConfig, WindowResult, evaluate_patience, multi-window runner |
| `src/quantstack/llm/config.py` | MODIFY | Add model cost rates dict for budget computation |
| `tests/unit/test_budget_discipline.py` | CREATE | 7 test stubs |
| `tests/unit/test_experiment_prioritizer.py` | CREATE | 5 test stubs |
| `tests/unit/test_patience_protocol.py` | CREATE | 5 test stubs |

---

## Key Design Decisions

1. **Per-cycle budget, not per-day.** Per-cycle maps directly to graph state, which is the natural unit of execution. The overnight autoresearch (section-07) has its own budget mechanism (nightly $10 ceiling). The two do not interact.

2. **Synthesize on exhaustion, do not abort.** When budget runs out, the system summarizes partial results and defers remaining work. This captures value from tokens already spent rather than discarding it. Deferred work re-enters via research_queue.

3. **Deterministic budget_check, not LLM-based.** The budget check is a pure comparison function. Using an LLM to decide whether to continue would itself consume budget and introduce unpredictability.

4. **3 validation windows enforced by graph structure.** The patience protocol is called by the backtest_validation node, not left to agent judgment. Agents cannot skip windows or override the protocol.

5. **Provisional status for 2/3 pass.** Rather than binary accept/reject, strategies that pass most but not all windows get reduced position sizing. This prevents throwing away strategies that may work in the current regime even if they failed historically in a different regime.

6. **Novelty score degrades gracefully.** Before the knowledge graph (section-10) exists, all hypotheses are treated as novel (score=1.0). This means prioritization still works on the other three factors (expected_ic, regime_fit, compute_cost), just without deduplication benefit.
