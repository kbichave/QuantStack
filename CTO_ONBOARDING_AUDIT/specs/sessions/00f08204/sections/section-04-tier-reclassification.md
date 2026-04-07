# Section 04: Agent Tier Reclassification

**Plan Reference:** Item 5.3
**Dependencies:** section-01 (consolidated LLM config), section-03 (cost queries for measurement)
**Blocks:** None

---

## Problem

Despite `llm_tier` fields in agents.yaml, naming-convention-based fallback logic causes some agents to default to Sonnet. Need explicit tier enforcement with zero silent defaults.

## Tests (Write First)

### tests/unit/test_llm_provider.py (extend)

```python
# Test: get_chat_model with unrecognized tier logs WARNING (not silent default)
# Test: get_chat_model never returns a model via naming-convention fallback
```

### tests/unit/test_agent_configs.py (extend)

```python
# Test: every agent in all 3 agents.yaml has explicit llm_tier field
# Test: all tier values are in {"heavy", "medium", "light"}
```

---

## Implementation

### Step 1: Audit via Langfuse

Create `scripts/audit_agent_tiers.py` — query Langfuse for actual model IDs per agent over 7 days, compare against agents.yaml config. Document mismatches.

### Step 2: Remove Naming-Convention Fallback

In `src/quantstack/llm/provider.py`:
- Add `VALID_TIERS = {"heavy", "medium", "light", "embedding", "bulk"}`
- Raise ValueError on unrecognized tier (log WARNING first)
- Delete any `if "researcher" in name: tier = "heavy"` patterns

### Step 3: Correct Tier Assignments

**Heavy (8):** quant_researcher, ml_scientist, strategy_rd, trade_debater, fund_manager, options_analyst, domain_researcher, execution_researcher
**Medium (12):** hypothesis_critic, community_intel, daily_planner, position_monitor, exit_evaluator, earnings_analyst, market_intel, trade_reflector, executor, self_healer, portfolio_risk_monitor, strategy_promoter
**Light (1):** health_monitor

### Step 4: Measure Impact

7-day cost comparison before/after using cost queries from section-03.

---

## Rollback

Revert agents.yaml tier changes only. Never rollback the naming-convention removal.

## Files

| File | Change |
|------|--------|
| `src/quantstack/llm/provider.py` | Modify — remove fallback, add VALID_TIERS |
| `src/quantstack/graphs/*/config/agents.yaml` (x3) | Modify — correct tiers |
| `scripts/audit_agent_tiers.py` | **Create** |
| `scripts/validate_agent_tiers.py` | **Create** |
