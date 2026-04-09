# Section 07: Safety Gates

## Objective

Implement the safety constraints from plan section 6 to ensure RL models cannot bypass risk management, cannot deploy without paper validation, and have bounded influence on trading decisions. These are hard gates, not suggestions.

## Dependencies

- **section-06-signal-integration**: RL signal must flow through synthesis before safety gates apply

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/finrl/config.py` | **Modify** | Add safety gate configuration fields |
| `src/quantstack/finrl/promotion.py` | **Modify** | Add 30-day paper validation hard gate to `PromotionGate` |
| `src/quantstack/execution/risk_gate.py` | **Modify** | Ensure RL-originated signals pass through risk gate like all other signals |
| `src/quantstack/signal_engine/synthesis.py` | **Modify** | Enforce RL signal weight cap and position change limits |

## Implementation Details

### Safety Constraint 1: RL Signal Weight Cap (plan section 6)

In `synthesis.py`, enforce that the RL voter never exceeds weight 0.15 regardless of regime profile configuration. Add a hard cap check:

```python
MAX_RL_WEIGHT = 0.15

# After loading regime weight profile:
if weights.get("rl", 0) > MAX_RL_WEIGHT:
    excess = weights["rl"] - MAX_RL_WEIGHT
    weights["rl"] = MAX_RL_WEIGHT
    # Redistribute excess proportionally to other voters
```

This prevents accidental configuration from giving RL too much influence.

### Safety Constraint 2: Maximum Position Change (plan section 6)

The 10% maximum position change per step is enforced at the environment level (section-01). Add a second enforcement point at the execution layer:

In `risk_gate.py`, add a check that rejects position changes exceeding 10% of portfolio value when the signal source is RL:

```python
def _check_rl_position_change(self, signal: dict, portfolio_value: float) -> bool:
    """Reject RL-driven position changes exceeding 10% of portfolio."""
    if signal.get("source") != "rl":
        return True  # Non-RL signals use standard position limits
    change_pct = abs(signal.get("position_change", 0)) / max(portfolio_value, 1)
    return change_pct <= 0.10
```

### Safety Constraint 3: 30-Day Paper Validation (plan section 6)

Add a hard gate to `PromotionGate.evaluate()` that checks shadow duration:

```python
def _check_paper_duration(self, shadow_start: datetime) -> PromotionCheckResult:
    """Model must have 30 calendar days in shadow before promotion."""
    days_in_shadow = (datetime.utcnow() - shadow_start).days
    min_days = 30
    passed = days_in_shadow >= min_days
    return PromotionCheckResult(
        name="paper_duration",
        passed=passed,
        value=float(days_in_shadow),
        threshold=float(min_days),
        message=f"{days_in_shadow} days in shadow {'>=  ' if passed else '<'} {min_days} required",
    )
```

This check runs before all other promotion checks. If the model hasn't been in shadow for 30 days, promotion is immediately rejected regardless of performance.

### Safety Constraint 4: Turnover Penalty (plan section 6)

Already enforced in the `PortfolioOptEnv` reward function at 20bps/day. Add a config field to make it tunable:

```python
# In FinRLConfig:
rl_turnover_penalty_bps: float = 20.0  # basis points per unit turnover per day
```

### Safety Constraint 5: Feature Flag Default Off (plan section 6)

Already implemented in section-06. Verify that `rl_signal_enabled()` defaults to `False` and that the env var `FEEDBACK_RL_SIGNAL` is not set in any default configuration.

### Config Additions

Add to `FinRLConfig`:
```python
# Safety gates
max_rl_signal_weight: float = 0.15
max_rl_position_change_pct: float = 0.10
min_paper_days: int = 30
rl_turnover_penalty_bps: float = 20.0
```

### Risk Gate Integration

The plan states: "RL outputs are ADVISORY, same as all other signals." The existing risk gate in `risk_gate.py` already validates all trade signals. Verify that:
1. RL-sourced signals are tagged with `source: "rl"` in their signal dict
2. Risk gate does not have any special bypass for RL signals
3. All existing risk checks (position limits, drawdown limits, correlation checks) apply to RL signals

If any bypass exists, remove it. If RL signals are not tagged, add tagging in the RL collector.

## Test Requirements

1. **Weight cap**: RL weight never exceeds 0.15 even if config is misconfigured
2. **Position change gate**: RL signal requesting 15% position change is rejected by risk gate
3. **Position change gate passthrough**: Non-RL signal requesting 15% position change is not affected by RL-specific check
4. **30-day gate**: Model in shadow for 15 days fails promotion
5. **30-day gate pass**: Model in shadow for 31 days passes the duration check
6. **Feature flag**: RL signal returns empty when flag is off
7. **Risk gate passthrough**: RL signal passes through standard risk gate checks (not bypassed)
8. **Turnover penalty config**: Penalty value is read from config, not hardcoded

## Acceptance Criteria

- [ ] RL signal weight is hard-capped at 0.15 in synthesis
- [ ] Risk gate rejects RL-driven position changes > 10% of portfolio
- [ ] 30-day paper validation is a hard gate in `PromotionGate` (runs first, blocks on failure)
- [ ] Turnover penalty is configurable via `FinRLConfig`
- [ ] Feature flag defaults to `False` in all environments
- [ ] No special bypasses exist for RL signals in risk gate
- [ ] All safety config fields are in `FinRLConfig` with documented defaults
