# Section 04: Authority Matrix

## Objective

Define and enforce ceilings on every autonomous decision, ensuring no agent can exceed predefined limits without human review. This is a safety layer that complements the risk gate.

## Files to Create

### `src/quantstack/autonomous/authority_matrix.py`

Authority definitions, ceiling enforcement, and escalation logging.

## Files to Modify

### `src/quantstack/execution/risk_gate.py`

Add authority ceiling checks as an additional gate layer.

### `src/quantstack/autonomous/__init__.py`

Export `AuthorityMatrix`, `AuthorityDecision`.

## Implementation Details

### AuthorityRule Dataclass

```python
@dataclass(frozen=True)
class AuthorityRule:
    decision_type: str           # e.g., "open_position", "promote_strategy"
    authorized_agents: list[str] # which agents can make this decision
    ceiling: float               # numeric ceiling value
    ceiling_unit: str            # e.g., "pct_portfolio", "count_per_day", "pct_relative"
    escalation: str              # what happens on breach: "reject", "flag_for_review"
    description: str
```

### Default Authority Rules

| Decision Type | Ceiling | Unit | Escalation |
|---------------|---------|------|------------|
| `max_single_position` | 5.0 | pct_portfolio | reject |
| `max_daily_new_positions` | 3 | count_per_day | reject |
| `max_strategy_promotions_weekly` | 1 | count_per_week | flag_for_review |
| `max_signal_weight_change` | 10.0 | pct_relative | reject |
| `max_daily_loss_tolerance` | 2.0 | pct_portfolio | reject (triggers kill switch) |

### AuthorityMatrix Class

```python
class AuthorityMatrix:
    def __init__(self, rules: list[AuthorityRule] | None = None): ...

    def check(
        self,
        decision_type: str,
        agent_name: str,
        proposed_value: float,
        context: dict[str, Any] | None = None,
    ) -> AuthorityDecision: ...

    def log_decision(self, decision: AuthorityDecision) -> None: ...
```

### AuthorityDecision Dataclass

```python
@dataclass
class AuthorityDecision:
    decision_type: str
    agent_name: str
    proposed_value: float
    ceiling: float
    approved: bool
    escalated: bool
    reason: str
    timestamp: datetime
```

### Ceiling Tracking

For rate-limited ceilings (`count_per_day`, `count_per_week`):
- Query the `authority_decisions` DB table to count approved decisions in the window
- Do NOT rely on in-memory counters (they reset on restart)

### Risk Gate Integration

Add to `RiskGate.check()`:
- After existing risk checks pass, call `AuthorityMatrix.check("max_single_position", ...)` with the proposed position size as % of portfolio
- If authority check fails, return `RiskVerdict(approved=False, reason=authority_decision.reason)`
- The authority check is additive — it can only reject, never override a risk gate rejection

### Escalation Logging

When `escalation == "flag_for_review"`:
- Insert into `authority_escalations` table: decision details, context, timestamp
- Log at WARNING level
- The system continues operating within ceiling (does not block)

### DB Schema

```sql
CREATE TABLE IF NOT EXISTS authority_decisions (
    id SERIAL PRIMARY KEY,
    decision_type TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    proposed_value DOUBLE PRECISION,
    ceiling DOUBLE PRECISION,
    approved BOOLEAN NOT NULL,
    escalated BOOLEAN DEFAULT FALSE,
    reason TEXT,
    context JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS authority_escalations (
    id SERIAL PRIMARY KEY,
    decision_id INTEGER REFERENCES authority_decisions(id),
    reviewed BOOLEAN DEFAULT FALSE,
    reviewed_at TIMESTAMPTZ,
    resolution TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Test Requirements

- `tests/unit/autonomous/test_authority_matrix.py`:
  - Test approval when under ceiling
  - Test rejection when over ceiling
  - Test rate-limited ceilings (mock 3 approved positions today, 4th rejected)
  - Test escalation path (flag_for_review does not reject, but logs)
  - Test unknown decision type raises clear error
  - Test unauthorized agent is rejected regardless of value
  - Test risk gate integration: authority rejection after risk approval

## Acceptance Criteria

1. Every autonomous decision is logged in `authority_decisions` table with full context
2. Ceiling breach always results in either rejection or escalation — never silent approval
3. Rate-limited ceilings use DB state, not in-memory counters
4. Authority matrix is additive to risk gate — never weakens existing checks
5. Default rules match the plan exactly (5% position, 3/day, 1 promotion/week, 10% weight change)
6. Rules are configurable via constructor parameter (not hardcoded constants)
