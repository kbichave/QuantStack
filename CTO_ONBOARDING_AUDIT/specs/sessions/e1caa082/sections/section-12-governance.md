# Section 12: Hierarchical Governance (AR-4)

## Overview

This section implements a three-tier governance system that separates strategic decisions (what to trade) from tactical execution (how to trade), reducing daily token costs from $150-450 to ~$10 while adding principled oversight. A CIO Agent (Sonnet, once/day) produces a DailyMandate, four Strategy Agents (Haiku, every 5 minutes) execute within that mandate, and the existing Risk Gate (deterministic, per-trade) enforces position sizing and exposure limits unchanged.

The mandate is not advisory -- it is enforced by code as a hard gate in the trading graph, running before the risk gate. No trade can proceed that violates the active mandate.

## Dependencies

- **section-01-db-migrations**: The `daily_mandates` table must exist before the CIO agent can persist mandates. The table schema is defined in section-01 but the governance module is the primary consumer.
- **section-05-event-bus-extensions**: The `MANDATE_ISSUED` event type must be registered in the `EventType` enum so the CIO agent can announce mandates to all consumers. The event bus is poll-based and append-only; no architectural changes needed.
- **section-06-prompt-caching**: Prompt caching must be enabled for the governance tier to hit the $10/day target. Without caching, Haiku strategy agents running 78 cycles/day would exceed budget on system prompt tokens alone.

## Data Model

### DailyMandate Schema

The CIO agent produces this structure once per day. It is persisted to the `daily_mandates` table and read by the mandate_check gate on every trade evaluation.

```python
@dataclass
class DailyMandate:
    """Daily trading directives from the CIO agent.

    Persisted to daily_mandates table. Read by mandate_check gate.
    Immutable once published -- a new mandate replaces, never edits.
    """
    mandate_id: str              # UUID
    date: str                    # YYYY-MM-DD
    regime_assessment: str       # e.g. "trending_up", "ranging", "unknown"
    allowed_sectors: list[str]   # sectors where new entries are permitted
    blocked_sectors: list[str]   # sectors where no new entries allowed
    max_new_positions: int       # cap on new positions for the day
    max_daily_notional: float    # max total $ of new entries today
    strategy_directives: dict[str, str]  # strategy_id -> "active"|"reduce"|"pause"|"exit"
    risk_overrides: dict         # optional: tighten/relax specific risk params
    focus_areas: list[str]       # what research should prioritize today
    reasoning: str               # audit trail for CIO's rationale
    created_at: datetime         # TIMESTAMPTZ
```

### Conservative Default Mandate

If no mandate exists for today by 09:30 ET (CIO failure, LLM timeout, Bedrock outage), a deterministic fallback activates:

```python
def _default_mandate(today: str) -> DailyMandate:
    """Conservative fallback -- no new entries, existing positions in monitor mode.

    Generated deterministically (no LLM). Ensures the system always has
    valid constraints even when the CIO agent fails.
    """
    # max_new_positions=0: no new entries
    # strategy_directives: all set to "pause"
    # Does NOT liquidate existing positions
    # Does NOT override stop-losses (they still fire)
```

The default mandate degrades gracefully: the system pauses new entries without touching existing positions. Stop-losses and existing exit logic continue to function.

### Database Table

The `daily_mandates` table (created in section-01-db-migrations) stores one row per day:

- `mandate_id` UUID primary key
- `date` DATE with unique constraint (one mandate per day)
- `regime_assessment` TEXT
- `allowed_sectors` JSONB
- `blocked_sectors` JSONB
- `max_new_positions` INTEGER
- `max_daily_notional` NUMERIC
- `strategy_directives` JSONB
- `risk_overrides` JSONB
- `focus_areas` JSONB
- `reasoning` TEXT
- `created_at` TIMESTAMPTZ DEFAULT NOW()
- ON CONFLICT (date) DO UPDATE for idempotent writes

## Tests

Tests go in two files. Write these first; implementation follows.

### Unit Tests (`tests/unit/test_governance.py`)

```python
"""Unit tests for hierarchical governance.

Tests cover: CIO mandate generation, mandate_check hard gate,
conservative default fallback, tier routing, kill switch precedence.
"""
import pytest
from datetime import datetime, date
from unittest.mock import AsyncMock, MagicMock, patch


# --- CIO Agent ---

def test_cio_produces_valid_daily_mandate():
    """CIO agent output must include all required DailyMandate fields:
    date, regime_assessment, allowed_sectors, blocked_sectors,
    max_new_positions, max_daily_notional, strategy_directives,
    risk_overrides, focus_areas, reasoning.
    """

def test_cio_agent_uses_sonnet_tier():
    """CIO agent must resolve to Sonnet-class model, not Haiku.
    Verify via get_model_for_role("governance") or equivalent tier lookup.
    """

def test_strategy_agents_use_haiku_tier():
    """All four strategy agents (swing, investment, options, mean_reversion)
    must resolve to Haiku-class model. Verify via agents.yaml config or
    tier resolution.
    """


# --- Mandate Check Gate ---

def test_mandate_check_rejects_trade_in_blocked_sector():
    """A trade in a sector listed in blocked_sectors must be rejected.
    E.g., mandate blocks "Technology", trade proposes AAPL -> reject.
    """

def test_mandate_check_rejects_when_max_new_positions_reached():
    """If max_new_positions=3 and 3 new positions already opened today,
    the next entry must be rejected regardless of sector/notional.
    """

def test_mandate_check_rejects_when_max_daily_notional_exceeded():
    """If max_daily_notional=50000 and $48000 already entered today,
    a $5000 trade must be rejected (would push total to $53000).
    """

def test_mandate_check_approves_trade_within_all_constraints():
    """A trade in an allowed sector, under position cap, under notional
    limit, with strategy_directive="active" must be approved.
    """

def test_mandate_check_is_hard_gate():
    """mandate_check returns a binary pass/fail. It does NOT return a
    score or advisory. A failing trade must not proceed to risk_gate.
    """

def test_mandate_check_runs_before_risk_gate():
    """In the trading graph edge ordering, mandate_check must execute
    BEFORE risk_gate. Verify by inspecting graph edges or by confirming
    a mandate-rejected trade never reaches risk_gate.check().
    """


# --- Conservative Default ---

def test_conservative_default_mandate_activates_when_no_mandate_by_0930():
    """If no mandate row exists for today's date at 09:30 ET,
    _default_mandate() must be called and its result used.
    """

def test_default_mandate_has_zero_new_positions():
    """The default mandate must set max_new_positions=0.
    No new entries permitted when CIO is unavailable.
    """

def test_default_mandate_does_not_liquidate_existing_positions():
    """strategy_directives in the default mandate must be "pause"
    (not "exit"). Existing positions remain open with their
    existing stop-losses intact.
    """


# --- Kill Switch Precedence ---

def test_kill_switch_overrides_mandate():
    """When kill switch is active, no trade executes even if the
    mandate says "active" and all constraints are met.
    Kill switch > mandate > risk gate in precedence.
    """
```

### Integration Tests (`tests/integration/test_governance.py`)

```python
"""Integration tests for the full governance flow.

Requires a running PostgreSQL instance with daily_mandates table.
"""
import pytest


def test_full_mandate_flow():
    """End-to-end: CIO agent produces mandate -> persisted to DB ->
    strategy agent reads it -> mandate_check enforces constraints.
    Uses mock LLM responses for CIO and strategy agents.
    """

def test_mandate_published_via_event_bus():
    """After CIO produces a mandate, a MANDATE_ISSUED event must be
    published to the event bus with mandate_id and key directives.
    """

def test_token_cost_with_governance_lower_than_without():
    """Over a simulated trading day (78 cycles), total LLM token cost
    with governance (1 Sonnet CIO + 4 Haiku strategy agents) must be
    less than 50% of the cost without governance (all agents at Sonnet).
    Measure via mock token counters.
    """
```

## Implementation

### File: `src/quantstack/governance/__init__.py`

New package. Exports `DailyMandate`, `mandate_check`, `get_active_mandate`.

### File: `src/quantstack/governance/mandate.py`

Contains:

- `DailyMandate` dataclass (schema shown above)
- `_default_mandate(today: str) -> DailyMandate` -- deterministic conservative fallback
- `persist_mandate(mandate: DailyMandate) -> None` -- writes to `daily_mandates` table using `db_conn()` context manager with ON CONFLICT DO UPDATE
- `get_active_mandate(today: str | None = None) -> DailyMandate` -- reads today's mandate from DB. If no row exists and current time is past 09:30 ET, returns `_default_mandate(today)`. If before 09:30 ET and no mandate yet, returns None (CIO hasn't run yet, which is expected)

### File: `src/quantstack/governance/cio_agent.py`

Contains:

- `async def generate_daily_mandate() -> DailyMandate` -- the CIO agent function

The CIO agent is a single LLM call (Sonnet tier) that receives:

1. **System prompt**: Role as Chief Investment Officer. Instructions to produce a DailyMandate JSON. Emphasis on regime awareness and capital preservation.
2. **Context payload** (assembled before the LLM call, all deterministic):
   - Overnight autoresearch winners (query `autoresearch_experiments` WHERE status='winner' AND night_date=yesterday)
   - Current regime from RegimeDetector (`classify_regime()`)
   - Portfolio state from `portfolio_state.py` (open positions, total exposure, unrealized P&L)
   - Knowledge graph factor crowding alerts (if section-10 is implemented; graceful skip if not)
   - Meta agent reports (if section-13 is implemented; graceful skip if not)
3. **Output parsing**: Parse the LLM response as JSON into a `DailyMandate`. If parsing fails, log the error and fall through to `_default_mandate()`.

Model resolution: add a `"governance"` tier alias in `src/quantstack/llm/provider.py` that maps to Sonnet. This keeps governance model selection explicit and auditable.

```python
# In provider.py TIER_ALIASES:
TIER_ALIASES["governance"] = "heavy"  # Sonnet
TIER_ALIASES["strategy"] = "light"    # Haiku
```

### File: `src/quantstack/governance/mandate_check.py`

Contains:

- `def mandate_check(symbol: str, sector: str, side: str, notional: float, strategy_id: str) -> MandateVerdict`

`MandateVerdict` is a simple dataclass:

```python
@dataclass
class MandateVerdict:
    """Result of mandate compliance check."""
    approved: bool
    rejection_reason: str | None = None
```

The check logic (all deterministic, no LLM):

1. Load active mandate via `get_active_mandate()`. If None (before 09:30, CIO hasn't run), approve (system is in pre-mandate window).
2. Check sector: if `sector in mandate.blocked_sectors` -> reject with reason.
3. Check position count: query `fills` for today's new entries count. If >= `mandate.max_new_positions` -> reject.
4. Check notional: query `fills` for today's cumulative entry notional. If current + proposed > `mandate.max_daily_notional` -> reject.
5. Check strategy directive: if `mandate.strategy_directives.get(strategy_id) in ("pause", "exit")` -> reject for that specific strategy.
6. All checks pass -> approve.

This function does NOT modify the risk gate. It runs before the risk gate as an additional filter. The risk gate remains immutable and unchanged.

### File: `src/quantstack/graphs/supervisor/nodes.py` (MODIFY)

Add a CIO scheduling node:

```python
def make_cio_mandate(llm, config, tools=None):
    """Create the CIO mandate generation node.

    Scheduled at 09:00 ET daily. Produces a DailyMandate and publishes
    a MANDATE_ISSUED event to the event bus.
    """
    # Calls generate_daily_mandate()
    # Publishes Event(event_type=EventType.MANDATE_ISSUED, ...)
    # Returns updated SupervisorState with mandate_id
```

Wire this node into the supervisor graph (`src/quantstack/graphs/supervisor/graph.py`) as a daily-cadence node. The supervisor graph already has scheduling logic for health checks; the CIO node follows the same pattern with a daily cadence gated by time-of-day check.

### File: `src/quantstack/graphs/trading/nodes.py` (MODIFY)

Insert `mandate_check` into the trading pipeline between `entry_scan` and `risk_sizing`. Specifically:

- After entry_scan produces a proposed trade (symbol, side, quantity, price), compute notional and sector
- Call `mandate_check(symbol, sector, side, notional, strategy_id)`
- If rejected: log the rejection with reason, skip to reflect node (do not reach risk_gate)
- If approved: proceed to risk_sizing as before

This is a new conditional edge in the trading graph. The graph builder (`src/quantstack/graphs/trading/graph.py`) needs a new edge from entry_scan output to a mandate_check node, then conditional routing to either risk_sizing (approved) or reflect (rejected).

### File: `src/quantstack/graphs/trading/graph.py` (MODIFY)

Update the graph structure. Current flow relevant to this change:

```
entry_scan -> merge_parallel -> risk_sizing -> [SafetyGate] -> ...
```

New flow:

```
entry_scan -> merge_parallel -> mandate_check -> risk_sizing -> [SafetyGate] -> ...
                                    |-> [rejected] -> reflect -> END
```

Add the `mandate_check` node and conditional edge. The mandate_check node is deterministic (no LLM, no tools), so it does not need retry policy or error threshold handling.

### File: `src/quantstack/graphs/trading/config/agents.yaml` (MODIFY)

Update strategy agent configurations to use Haiku tier:

```yaml
# Each strategy agent explicitly set to "light" (Haiku) tier
swing_agent:
  tier: light
  # ... existing tool bindings

investment_agent:
  tier: light

options_agent:
  tier: light

mean_reversion_agent:
  tier: light
```

### File: `src/quantstack/llm/provider.py` (MODIFY)

Add tier aliases for governance:

```python
TIER_ALIASES["governance"] = "heavy"   # CIO uses Sonnet
TIER_ALIASES["strategy"] = "light"     # Strategy agents use Haiku
```

## Execution Order

The mandate_check gate sits in the trading pipeline as follows (full pipeline context):

```
START -> data_refresh -> safety_check -> [halted?] -> END
                                      -> market_intel -> plan_day
                          |-> position_review -> execute_exits -> merge_parallel
                          |-> entry_scan --------------------->  merge_parallel
                      -> mandate_check -> [approved?]
                                           |-> risk_sizing -> [SafetyGate] -> ...
                                           |-> [rejected] -> reflect -> END
```

Precedence chain for trade approval: **kill_switch > mandate_check > risk_gate**. If the kill switch is active, no processing occurs. If the mandate rejects, the trade never reaches risk_gate. The risk gate remains the final arbiter of position sizing and exposure limits.

## Token Cost Model

| Component | Model | Frequency | Cost/call | Daily Cost |
|-----------|-------|-----------|-----------|------------|
| CIO Agent | Sonnet | 1x/day at 09:00 ET | ~$0.15 | $0.15 |
| Swing Agent | Haiku | 78 cycles/day (every 5 min, 6.5 hrs) | ~$0.001 | $0.08 |
| Investment Agent | Haiku | 78 cycles/day | ~$0.001 | $0.08 |
| Options Agent | Haiku | 78 cycles/day | ~$0.001 | $0.08 |
| Mean Reversion Agent | Haiku | 78 cycles/day | ~$0.001 | $0.08 |
| Risk Gate | Deterministic | per-trade | $0 | $0 |
| **Total Governance** | | | | **~$0.47/day** |

Plus ~$3-5/day for overnight research and supervisor monitoring, total system target is ~$10/day. Prompt caching (section-06) reduces the strategy agent costs by ~80%, bringing the actual figure closer to $0.10/day for strategy agents.

## Failure Modes

1. **CIO agent fails (LLM error, timeout)**: Conservative default mandate activates at 09:30 ET. No new entries, existing positions monitored. System degrades to "hold and wait" mode. Recovery: CIO retries on next supervisor cycle or the following morning.

2. **Mandate parsing fails (malformed JSON from LLM)**: Same as CIO failure -- fall through to `_default_mandate()`. Log the raw LLM output for debugging.

3. **Database unavailable**: `get_active_mandate()` fails. The mandate_check function should catch the DB error and fall through to conservative behavior (reject the trade). A trade not happening is always safer than a trade happening without mandate validation.

4. **Stale mandate (CIO ran but regime changed)**: The daily mandate is not updated intraday by default. If a `REGIME_CHANGE` event fires from the supervisor graph, a future enhancement could trigger intraday mandate refresh. For now, the conservative approach is acceptable: the risk gate still enforces position limits regardless of mandate staleness.

5. **Strategy agent ignores mandate**: Impossible by design. mandate_check is a code-level gate, not a prompt instruction. Even if the Haiku agent hallucinates a trade in a blocked sector, mandate_check will reject it deterministically.

## Key Design Decisions

- **Daily mandate, not per-trade CIO**: A per-trade Sonnet call would add ~$0.15/trade latency and cost. Daily is sufficient because regime changes are not intraday events for swing/investment timeframes. If intraday regime shifts matter, a REGIME_CHANGE event can trigger an ad-hoc mandate refresh in a future iteration.

- **Code enforcement, not prompt enforcement**: If the mandate were prompt-only, a hallucinating Haiku agent could bypass it. Code enforcement makes the mandate as hard as the risk gate. This is the same design pattern used by the existing risk gate and kill switch.

- **4 strategy agents by domain**: Maps to the 4 trading domains (swing, investment, options, mean_reversion). Each agent gets domain-specific tools and context, keeping Haiku's limited context window focused on relevant information.

- **Governance before meta-agents**: This section must be implemented before section-13 (meta-agents) because meta-agents need to target the new agent hierarchy (CIO + strategy agents), not the old flat model where all agents run at Sonnet tier.
