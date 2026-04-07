# Section 05: Per-Agent Temperature Config

**Plan Reference:** Item 5.4
**Dependencies:** None
**Blocks:** None

---

## Problem

All 21 agents use `temperature=0.0`. Research agents that benefit from diverse hypothesis generation are constrained; execution agents are already correct at 0.0.

## Tests (Write First)

### tests/unit/test_agent_executor.py (extend)

```python
# Test: agent with temperature=0.7 passes temperature to get_chat_model
# Test: agent with temperature=None uses default (0.0)
```

### tests/unit/test_llm_provider.py (extend)

```python
# Test: get_chat_model(tier, temperature=0.5) creates LLM with temperature=0.5
# Test: get_chat_model(tier) without temperature kwarg still works (backward compat)
```

### tests/unit/test_agent_configs.py (extend)

```python
# Test: AgentConfig accepts temperature field from YAML
# Test: research agents have temperature > 0.0
# Test: executor/fund_manager have temperature == 0.0
```

---

## Implementation

### Step 1: Update AgentConfig

Add `temperature: float | None = None` to AgentConfig in `src/quantstack/graphs/config.py`.

### Step 2: Update get_chat_model() Signature

```python
def get_chat_model(
    tier: str,
    thinking: dict | None = None,
    temperature: float | None = None,  # NEW
) -> BaseChatModel:
```

`temperature=None` preserves backward compatibility for ~30+ existing call sites.

### Step 3: Wire in Agent Executor

Pass `config.temperature` through to `get_chat_model()` when instantiating agent LLM.

### Step 4: Configure Per-Agent Temperatures

| Temperature | Agents | Rationale |
|-------------|--------|-----------|
| 0.7 | quant_researcher, hypothesis_critic | Ideation diversity (hypothesis_critic at medium tier + 0.7 is intentional for diverse scoring) |
| 0.3-0.5 | trade_debater, community_intel, ml_scientist, strategy_rd, domain_researcher | Moderate diversity |
| 0.1 | daily_planner, position_monitor, market_intel, trade_reflector | Mostly deterministic |
| 0.0 | executor, fund_manager, exit_evaluator, options_analyst, all supervisor agents | Determinism required |

---

## Note for Implementer

This section and section-03 both modify AgentConfig. Implement both schema changes together.

## Files

| File | Change |
|------|--------|
| `src/quantstack/graphs/config.py` | Modify — add temperature to AgentConfig |
| `src/quantstack/llm/provider.py` | Modify — temperature parameter |
| `src/quantstack/graphs/agent_executor.py` | Modify — pass temperature |
| `src/quantstack/graphs/*/config/agents.yaml` (x3) | Modify — per-agent temps |
