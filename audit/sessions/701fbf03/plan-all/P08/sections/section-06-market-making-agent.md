# Section 06: Market-Making Agent

**Depends on:** section-01 (vol arb strategy), section-04 (condor harvesting)

## Objective

Add an `options_market_maker` agent to the trading graph that orchestrates vol strategy selection, identifies mispricings, generates trade proposals, and monitors existing vol positions. This is the LLM-facing integration layer that ties the strategy engines (sections 01-04) to the existing agent infrastructure.

## Files to Modify

### `src/quantstack/graphs/trading/config/agents.yaml`

Add new agent definition:

```yaml
options_market_maker:
  role: "Options Market-Making and Volatility Strategy Specialist"
  goal: "Identify volatility mispricings, select optimal vol strategy based on regime, generate trade proposals through risk gate, and monitor existing vol positions for management actions."
  backstory: |
    You are the volatility desk. Each cycle you:
    1. Compute IV surface for universe symbols with active options chains.
    2. Identify vol mispricings: IV vs realized vol divergence > threshold.
    3. Select strategy based on regime:
       - Ranging + high IV rank -> iron condor harvesting
       - Trending + IV overpriced -> vol arb (sell vol)
       - High realized vol -> gamma scalping
       - Correlation rich -> dispersion trading
    4. Generate trade proposals and submit through risk gate.
    5. Monitor existing vol positions: check management triggers (roll, profit take, hedge).

    Hard limits:
    - Max portfolio vega: $5,000 (configurable via risk limits)
    - Max single-strike concentration: 20% of options allocation
    - Max delta exposure before hedge: $500
    - Never trade illiquid options (bid-ask spread > 10% of mid)

    Before proposing any trade, query the knowledge base for lessons from similar past vol trades.
    Check the trading calendar for events that affect vol strategies (FOMC, earnings, holidays).
  llm_tier: heavy
  max_iterations: 15
  timeout_seconds: 300
  tools:
    - compute_greeks
    - get_iv_surface
    - score_trade_structure
    - simulate_trade_outcome
    - signal_brief
    - fetch_portfolio
    - fetch_options_chain
    - search_knowledge_base
    - check_risk_limits
    - get_trading_calendar
    - price_option
    - compute_implied_vol
  always_loaded_tools:
    - compute_greeks
    - get_iv_surface
    - fetch_portfolio
    - check_risk_limits
    - search_knowledge_base
```

### `src/quantstack/graphs/trading/nodes.py`

Add new node function `options_market_maker_node`:

**Logic flow:**

1. **Surface scan**: For each symbol in universe with options chains, compute/fetch IV surface metrics (ATM IV, IV rank, skew).
2. **Mispricing detection**: Compare IV to realized vol. Flag symbols where `abs(iv - rv) / iv > 0.10`.
3. **Strategy selection** (regime-based):
   - `regime.trend_regime == "SIDEWAYS" and iv_rank > 50` -> `CondorHarvestingStrategy`
   - `iv_rank > 50 and iv > rv` -> `VolArbStrategy` (sell vol direction)
   - `rv > iv by threshold` -> `GammaScalpingStrategy`
   - Dispersion only when correlation data available and spread is significant
4. **Trade proposal generation**: Call selected strategy's `on_bar` with current `MarketState`.
5. **Position management**: For each open vol position, call the strategy's `compute_management_action` and `should_exit`. Generate exit signals or management orders.
6. **Risk limit check**: Verify portfolio vega < $5,000, single-strike concentration < 20%.

### `src/quantstack/graphs/trading/graph.py` (or equivalent graph definition)

Wire `options_market_maker_node` into the trading graph:
- Execute after `daily_planner` (needs regime context)
- Execute before `fund_manager` (proposals need approval)
- Parallel with `options_analyst` (independent specialization)

## Files to Create

### `src/quantstack/tools/langchain/vol_strategy_tools.py`

LLM-facing tools for the market-making agent:

1. **`@tool get_vol_mispricings`**: Scan universe for IV vs RV divergences. Returns ranked list of symbols with spread, z-score, suggested strategy.
2. **`@tool get_vol_position_status`**: For each open vol position, return current Greeks, P&L, management action recommendation, and exit triggers.
3. **`@tool propose_vol_trade`**: Given symbol + strategy type, compute entry parameters and return structured trade proposal.

Register all tools in `src/quantstack/tools/registry.py`.

## Implementation Details

- The agent is an LLM agent, not pure deterministic code. It uses the strategy engines from sections 01-04 as computation backends, but the LLM reasons about which symbols to prioritize and how to size across vol strategies.
- The node must respect existing portfolio state. If the portfolio already has significant vega exposure, the agent should prioritize management over new entries.
- Tool registration follows existing pattern: `@tool` decorator in `langchain/` module, string name in `TOOL_REGISTRY`.
- The agent config binds tools by string name. Only use tools that already exist or are created in this section.
- `simulate_trade_outcome` and `score_trade_structure` are existing tools from P06 that the agent reuses.

## Test Requirements

File: `tests/unit/graphs/test_market_maker_node.py`

1. **Strategy selection by regime**: SIDEWAYS + high IV -> condor; trending + IV rich -> vol arb; high RV -> gamma scalp.
2. **Mispricing detection**: Symbols with IV-RV divergence > threshold are flagged; those below are skipped.
3. **Vega limit enforcement**: Agent does not propose new vol trades when portfolio vega > $5,000.
4. **Position management**: Open vol position with profit > 50% triggers close recommendation.
5. **Illiquidity filter**: Options with bid-ask spread > 10% of mid are excluded.
6. **Tool registration**: All 3 new tools resolve from `TOOL_REGISTRY`.

## Acceptance Criteria

- [ ] Agent definition in `agents.yaml` with correct role, tools, and backstory
- [ ] Node function implements 6-step logic flow (scan, detect, select, propose, manage, check)
- [ ] Three new LLM-facing tools registered in tool registry
- [ ] Node wired into trading graph at correct position
- [ ] Vega and concentration limits enforced before proposing trades
- [ ] All 6 unit tests pass
- [ ] Agent reuses existing tools (`compute_greeks`, `get_iv_surface`, etc.) -- no duplication
