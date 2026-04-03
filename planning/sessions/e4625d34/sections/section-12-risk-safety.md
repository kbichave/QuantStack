# Section 12: Risk & Safety — Programmatic Safety Boundary + LLM-Reasoned Risk

## Overview

This section implements the defense-in-depth risk architecture for the CrewAI migration. The core idea: LLM agents reason about risk decisions (position sizing, strategy promotion, trade approval), but a programmatic safety boundary prevents catastrophic outcomes from hallucination, prompt injection, or stochastic variance.

The existing `src/quantstack/execution/risk_gate.py` is preserved and widened to serve as an outer envelope. The LLM's reasoning is the binding constraint in normal operation; the programmatic gate only fires when the LLM produces something unreasonable.

**Dependencies:** Section 03 (tool wrappers — `risk_tools.py` and `get_portfolio_context_tool` must exist).

**Blocks:** Section 05 (crew workflows consume the safety gate), Section 13 (testing validates risk behavior).

---

## Tests First

These tests go in `tests/unit/test_crewai_risk_safety.py`.

```python
# --- Programmatic Safety Gate ---

# Test: safety gate rejects position size > 15% of equity
def test_safety_gate_rejects_position_above_15_pct():
    """Given an LLM recommendation of 20% position size and equity of $25K,
    the safety gate must reject with a clear violation message.
    Input: {"symbol": "AAPL", "recommended_size_pct": 20, "reasoning": "..."}
    Assert: verdict.approved is False, violation rule is 'max_position_size'."""

# Test: safety gate rejects when daily loss exceeds 3%
def test_safety_gate_rejects_daily_loss_above_3_pct():
    """Given daily P&L of -$800 on $25K equity (3.2%), the safety gate must
    reject any new trade regardless of LLM recommendation.
    Assert: verdict.approved is False, violation rule is 'daily_loss_halt'."""

# Test: safety gate rejects when ADV < 200K
def test_safety_gate_rejects_low_adv():
    """Given a symbol with ADV of 150K, the safety gate must reject regardless
    of LLM conviction level.
    Assert: verdict.approved is False, violation rule is 'min_liquidity'."""

# Test: safety gate rejects when gross exposure > 200%
def test_safety_gate_rejects_gross_exposure_above_200_pct():
    """Given current gross exposure of 190% and a proposed trade adding 15%,
    the safety gate must reject.
    Assert: verdict.approved is False, violation rule is 'max_gross_exposure'."""

# Test: safety gate rejects when options premium at risk > 10% of equity
def test_safety_gate_rejects_options_premium_above_10_pct():
    """Given total options premium at risk of $2,800 on $25K equity (11.2%),
    the safety gate must reject the options trade.
    Assert: verdict.approved is False, violation rule is 'max_options_premium'."""

# Test: safety gate PASSES valid LLM recommendations
def test_safety_gate_passes_valid_recommendation():
    """Given an LLM recommendation of 8% position size, ADV of 5M, daily loss
    of -0.5%, and gross exposure of 80%, the safety gate approves.
    Assert: verdict.approved is True."""

# --- Risk Decision Format ---

# Test: risk decision output is valid JSON with required fields
def test_risk_decision_json_schema():
    """The RiskDecision dataclass serializes to JSON containing at minimum:
    symbol, recommended_size_pct, reasoning, confidence.
    Assert: all required keys present, types correct."""

# Test: risk decision uses temperature 0
def test_risk_agent_temperature_zero():
    """The risk_analyst agent definition in agents.yaml specifies temperature=0
    (or the crew passes temperature=0 to the LLM for risk tasks).
    Assert: agent config includes temperature setting of 0."""

# --- Kill Switch and Daily Halt Persistence ---

# Test: daily loss halt persists across process restarts (DB sentinel)
def test_daily_halt_persists_via_db():
    """When the safety gate triggers a daily halt, it writes a sentinel to the
    DB (not just in-memory). A new RiskGate instance on the same day reads the
    sentinel and starts in halted state.
    Assert: new_gate.is_halted() is True after sentinel write."""

# Test: kill switch check runs before every crew cycle
def test_kill_switch_checked_before_cycle():
    """The safety_check task (first task in TradingCrew) calls
    get_system_status_tool. If status is 'halted', the entire cycle is skipped.
    Assert: when system_status returns halted, no subsequent tasks execute."""

# --- Context Bundle Completeness ---

# Test: risk agent receives complete context bundle
def test_risk_context_bundle_completeness():
    """The get_portfolio_context_tool returns a JSON object containing all
    required fields for risk reasoning.
    Required fields: total_equity, cash_available, positions (list with symbol,
    quantity, current_price, unrealized_pnl), daily_pnl, gross_exposure_pct,
    net_exposure_pct, current_regime, vix_level.
    Assert: all keys present in returned JSON."""
```

---

## Implementation Details

### 12.1 Philosophy: LLM Reasons, Code Guards

The current `risk_gate.py` has ~20 hardcoded numeric thresholds (position < 10%, daily loss < 2%, ADV > 500K, etc.). In the CrewAI system, the LLM risk agent reasons about whether a trade is appropriate given full context. But a programmatic safety boundary prevents the LLM from doing anything catastrophic.

The key distinction:

- **LLM reasoning** (binding in normal operation): "Given this portfolio has 60% exposure, VIX at 22, and this is a momentum strategy in a ranging regime, I recommend a 5% position size."
- **Programmatic gate** (fires only on anomalies): "The LLM recommended 40% position size. This exceeds the 15% hard limit. REJECTED."

In practice, the LLM's decisions will always be well within the programmatic limits. The gate exists for the tail risk of hallucination or prompt injection.

### 12.2 Updated Safety Gate Limits

Create a new module `src/quantstack/crews/risk/safety_gate.py` that wraps the existing `RiskGate` with widened CrewAI-specific outer limits. The existing `risk_gate.py` is NOT modified — it continues to serve the legacy system. The new module calls into it with adjusted `RiskLimits`.

**Hard outer limits (non-negotiable):**

| Rule | Limit | Rationale |
|------|-------|-----------|
| Max position size | 15% of equity per symbol | Wider than current 10% — LLM should self-limit to 5-10% |
| Daily loss halt | -3% of equity | Wider than current 2% — deterministic, persists via DB sentinel |
| Min liquidity | 200,000 ADV | Tighter than current 500K to give LLM more universe, but still safe |
| Max gross exposure | 200% of equity | Wider than current 150% — allows LLM to use leverage if reasoned |
| Max options premium at risk | 10% of equity | Total book, not per-position |
| Kill switch | DB write halts everything | Supervisor agent or manual trigger |

The module structure:

```python
# src/quantstack/crews/risk/safety_gate.py

@dataclass
class SafetyGateLimits:
    """Outer envelope limits for LLM-reasoned risk decisions.

    These are intentionally wider than what a reasonable LLM would recommend.
    They exist to catch hallucination, not to constrain normal operation.
    """
    max_position_pct: float = 0.15
    daily_loss_halt_pct: float = 0.03
    min_adv: int = 200_000
    max_gross_exposure_pct: float = 2.00
    max_options_premium_pct: float = 0.10


@dataclass
class RiskDecision:
    """Structured output from the LLM risk agent.

    Parsed from the agent's JSON response. Every field is validated
    before the decision reaches execution.
    """
    symbol: str
    recommended_size_pct: float
    reasoning: str
    confidence: float  # 0.0 to 1.0
    approved: bool = True


class SafetyGate:
    """Programmatic safety boundary around LLM risk decisions.

    Validates every RiskDecision against hard outer limits before
    allowing execution. Logs all gate triggers to Langfuse.
    """

    def validate(self, decision: RiskDecision, portfolio_context: dict) -> RiskVerdict:
        """Check an LLM risk decision against hard safety limits.

        Returns RiskVerdict (reuses existing risk_gate.py data model).
        On pass: the LLM's recommendation flows through unchanged.
        On fail: logged as safety boundary trigger, trade rejected.
        """
```

The `validate` method checks each limit sequentially, collects violations, and returns early on daily halt (same pattern as the existing `RiskGate.check`). It delegates to the existing `RiskGate` for the actual portfolio state queries (position lookups, exposure calculations) but applies the widened limits.

### 12.3 Risk Agent Context Bundle

The `get_portfolio_context_tool` (from Section 03, `risk_tools.py`) returns a comprehensive JSON context for the risk agent to reason over. The tool assembles data from multiple sources:

```python
# In src/quantstack/crewai_tools/risk_tools.py

@tool("Get Portfolio Risk Context")
def get_portfolio_context_tool() -> str:
    """Return full portfolio context for risk reasoning.

    Includes: equity, cash, positions with P&L, daily P&L, exposure
    percentages, current regime, VIX level, and sector breakdown.
    The risk agent uses this to reason about position sizing.
    """
```

The returned JSON structure:

```json
{
  "total_equity": 25000.0,
  "cash_available": 12000.0,
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 50,
      "current_price": 185.0,
      "unrealized_pnl": 250.0,
      "weight_pct": 37.0
    }
  ],
  "daily_pnl": -150.0,
  "daily_pnl_pct": -0.006,
  "gross_exposure_pct": 0.74,
  "net_exposure_pct": 0.74,
  "current_regime": "trending_up",
  "volatility_regime": "normal",
  "vix_level": 18.5,
  "sector_exposure": {"technology": 0.37, "healthcare": 0.20},
  "options_premium_at_risk": 500.0,
  "options_premium_pct": 0.02
}
```

This is all the data the existing `risk_gate.py` uses internally, but surfaced as readable context for the LLM agent.

### 12.4 Risk Decision Flow

The end-to-end flow for every trade:

1. **Trade debater** produces an ENTER/SKIP verdict with reasoning.
2. **Risk agent** receives the ENTER candidate plus the full portfolio context bundle. It reasons about position sizing at temperature 0 and produces structured JSON output (`RiskDecision`).
3. **Safety gate** validates the `RiskDecision` against hard outer limits. If any limit is breached, the trade is rejected and logged to Langfuse as a safety boundary trigger.
4. **Fund manager** reviews the batch of approved, sized candidates for portfolio-level coherence.
5. **Executor** sends the order to the broker.

The safety gate sits between steps 2 and 3. It is called programmatically in the crew workflow — not by an agent, not by an LLM. It is pure Python validation.

### 12.5 Daily Loss Halt (Deterministic)

The daily loss halt is the one risk check that is NOT LLM-reasoned. It is fully deterministic:

- Every cycle, before any tasks run, the `safety_check` task reads daily P&L from the portfolio.
- If `abs(daily_loss) / equity >= 0.03`, the system writes a halt sentinel to the database (`loop_iteration_context` table with key `daily_halt_YYYY-MM-DD`).
- The sentinel persists across process restarts — a crashed and restarted container will not resume trading on a halted day.
- The sentinel auto-expires at midnight (next trading day).
- The existing file-based sentinel in `risk_gate.py` (`~/.quantstack/DAILY_HALT_ACTIVE`) is preserved as a secondary mechanism. The DB sentinel is the primary, file-based is fallback.

### 12.6 Kill Switch

The kill switch is a coordination mechanism:

- Any agent (typically the supervisor's `health_monitor`) can write `kill_switch=active` to the database via `get_system_status_tool` / a dedicated coordination tool.
- The `safety_check` task (first task in every TradingCrew cycle) reads the kill switch state. If active, the entire cycle is skipped — no trades, no analysis, just a heartbeat write and sleep.
- Manual activation: `python -c "from quantstack.mcp.tools._impl import set_system_status; set_system_status('halted')"`
- Manual deactivation requires explicit human action (by design — the system should not be able to un-halt itself from a kill switch).

### 12.7 Strategy Promotion Reasoning

The `strategy_promoter` agent in SupervisorCrew reasons about strategy lifecycle transitions. It receives:

- Strategy definition (entry/exit rules, regime affinity, economic mechanism)
- Forward testing performance (daily P&L, win rate, drawdown, trade count)
- Duration in forward testing
- Market conditions during testing period
- Similar strategies' historical performance (from RAG)
- Current portfolio needs (domain gaps, diversity)

The agent produces a structured recommendation (promote / extend / retire) with full reasoning. There are no hardcoded Sharpe thresholds or minimum trade counts — the agent reasons from evidence.

However, the programmatic safety boundary applies here too: a strategy cannot be promoted to live if it has fewer than 5 forward-testing trades (absolute minimum to have any statistical signal). This is a sanity check, not a decision criterion.

### 12.8 Temperature 0 for Risk Decisions

All risk-related LLM calls use temperature 0 for maximum consistency. This is configured in two places:

1. **Agent definition** (`crews/trading/config/agents.yaml`): The `risk_analyst` agent's `llm` field includes temperature configuration.
2. **Task definition** (`crews/trading/config/tasks.yaml`): The `risk_sizing` task specifies structured JSON output format.

CrewAI supports temperature via the LLM string (e.g., `bedrock/anthropic.claude-sonnet-4-20250514-v1:0` with `temperature=0` passed as a parameter to the `LLM` class).

### 12.9 Structured JSON Output

Risk decisions must be parseable, not free-text. The `risk_sizing` task's `expected_output` field specifies the JSON schema:

```yaml
risk_sizing:
  description: >
    For each ENTER candidate, reason about appropriate position size given
    the full portfolio context. Consider: volatility, conviction, correlation
    with existing positions, regime alignment, available capital, and
    lessons from similar past trades.

    Return a JSON array of risk decisions.
  expected_output: >
    A JSON array where each element has:
    - symbol (string)
    - recommended_size_pct (float, 0-100)
    - dollar_amount (float)
    - reasoning (string, 2-3 sentences)
    - confidence (float, 0.0-1.0)
  agent: risk_analyst
```

The crew workflow parses this JSON output and feeds each decision through the `SafetyGate.validate()` method before passing to the fund manager.

### 12.10 Capital Boundaries

Physical limits that no software can override:

- Total paper account: $25K ($20K equity, $5K options allocation)
- Alpaca paper mode enforced by `ALPACA_PAPER=true` environment variable
- Broker API rejects orders exceeding buying power (ultimate backstop)
- The programmatic safety gate adds inner limits within the capital boundary

### 12.11 Langfuse Audit Trail

Every safety gate evaluation is logged to Langfuse:

- **PASS events**: decision JSON, portfolio context snapshot, which limits were checked
- **FAIL events**: decision JSON, portfolio context snapshot, specific violation(s), full LLM reasoning that produced the rejected decision
- **Daily halt triggers**: timestamp, daily P&L at trigger, positions at time of halt
- **Kill switch activations**: who triggered it (agent ID or manual), reason

This creates a complete audit trail for every risk decision, enabling post-hoc analysis of whether the LLM's reasoning was sound and whether the safety gate fired appropriately.

---

## File Paths

| File | Action | Purpose |
|------|--------|---------|
| `src/quantstack/crews/risk/__init__.py` | Create | Package init |
| `src/quantstack/crews/risk/safety_gate.py` | Create | SafetyGate class, SafetyGateLimits, RiskDecision dataclass |
| `src/quantstack/crewai_tools/risk_tools.py` | Create (Section 03) | `get_portfolio_context_tool` — assembles context bundle |
| `src/quantstack/execution/risk_gate.py` | Preserve (no changes) | Existing risk gate — used as reference and fallback |
| `tests/unit/test_crewai_risk_safety.py` | Create | All tests listed above |

---

## Dependencies on Other Sections

- **Section 03 (Tool Wrappers):** `risk_tools.py` with `get_portfolio_context_tool` must exist. The safety gate consumes its output format.
- **Section 04 (Agent Definitions):** The `risk_analyst` agent YAML must specify temperature 0 and structured JSON output.
- **Section 05 (Crew Workflows):** The `risk_sizing` and `safety_check` tasks must call the safety gate programmatically between LLM reasoning and execution.
- **Section 08 (Observability):** Langfuse tracing must be initialized before safety gate events can be logged.
