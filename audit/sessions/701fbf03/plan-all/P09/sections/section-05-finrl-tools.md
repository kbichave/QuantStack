# Section 05: FinRL Tool Implementation

## Objective

Implement the 8 core FinRL LangChain tools that are currently stubbed in `finrl_tools.py`. These tools bridge the LangGraph agents to the FinRL training, evaluation, and inference infrastructure built in sections 01-04.

## Dependencies

- **section-04-training-infra**: Trainer, registry, and checkpoint infrastructure must be functional

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/tools/langchain/finrl_tools.py` | **Modify** | Replace stub implementations with real logic |
| `src/quantstack/tools/_shared.py` | **Modify** | Add FinRL-specific shared helpers if needed |

## Implementation Details

### Tool Implementations

All tools currently return `{"error": "Tool pending implementation", "status": "not_available"}`. Replace each with working implementations that delegate to the `quantstack.finrl` module.

**1. `finrl_create_environment`**
- Validate `env_type` against known types: `stock_trading`, `portfolio`, `execution`, `sizing`, `alpha_selection`
- For `portfolio` type: instantiate `PortfolioOptEnv` with data from `FinRLDataAdapter`
- For `execution` type: instantiate `ExecutionEnv`
- For `sizing` type: instantiate `SizingEnv`
- For `alpha_selection` type: instantiate `AlphaSelectionEnv`
- For `stock_trading`: use FinRL's built-in `StockTradingEnv` via data adapter
- Store env instance in a session-level cache (dict keyed by env_id)
- Return: `{"env_id": str, "env_type": str, "obs_shape": list, "action_shape": list}`

**2. `finrl_train_model`**
- Look up cached env by `env_id`
- Create `FinRLTrainer` and call `trainer.train()`
- Register model in `ModelRegistry` with shadow status
- Return: `{"model_id": str, "checkpoint_path": str, "training_time_s": float, "metrics": dict}`

**3. `finrl_train_ensemble`**
- Look up cached env, create train/val split
- Call `trainer.train_ensemble()`
- Register winning model
- Return: `{"model_id": str, "winner_algo": str, "per_algo_sharpe": dict}`

**4. `finrl_evaluate_model`**
- Look up model from registry, load checkpoint
- Create test environment with specified date range
- Call `trainer.evaluate()`
- Update eval metrics in registry
- Return: `{"model_id": str, "sharpe": float, "max_dd": float, "total_return": float, "win_rate": float}`

**5. `finrl_predict`**
- Look up model from registry, load checkpoint
- Construct observation from current market data (via DataStore/Alpaca) or use provided `current_state`
- Call `trainer.predict()`
- Tag prediction with `[SHADOW]` if model status is shadow
- Return: `{"action": str, "confidence": float, "shadow": bool, "model_id": str}`

**6. `finrl_list_models`**
- Query `ModelRegistry.list_models()` with optional filters
- Return: `{"models": list[dict], "count": int}`

**7. `finrl_compare_models`**
- For each model_id: load model, evaluate on same test period
- Build comparison table
- Return: `{"comparison": dict[str, dict], "best_model_id": str}`

**8. `finrl_promote_model`**
- Load model metadata and shadow performance data
- Run `PromotionGate.evaluate()`
- If passes: update status to "live" in registry
- Return: `{"model_id": str, "promoted": bool, "checks": list[dict]}`

### Tools left as stubs (per plan)

The following 3 tools are monitoring/utility and can remain stubs:
- `finrl_get_model_status` - defer to later iteration
- `finrl_screen_stocks` - ML screening, not core RL
- `finrl_screen_options` - options screening, not core RL

### Environment cache

Use a module-level dict `_env_cache: dict[str, gym.Env] = {}` to store created environments by ID. Environment IDs are generated as `f"{env_type}_{uuid4().hex[:8]}"`. Cache is cleared on process restart.

### Error handling

All tools must:
- Never raise to the LLM agent (catch and return error JSON)
- Log errors with `logger.error()` including context
- Return structured error: `{"error": str, "status": "failed"}`

## Test Requirements

1. **Create environment**: Each env_type creates the correct class
2. **Train model**: Training produces a checkpoint file on disk
3. **Evaluate model**: Evaluation returns valid Sharpe/drawdown metrics
4. **Predict**: Prediction returns valid action and confidence
5. **List models**: Returns empty list when no models, populated list after training
6. **Compare models**: Returns comparison table with correct structure
7. **Promote model**: Passes promotion gate with sufficient shadow data
8. **Error handling**: Invalid model_id returns error JSON, does not raise

## Acceptance Criteria

- [ ] 8 core tools return real results instead of stub errors
- [ ] All tools follow the error handling pattern (never raise, return JSON)
- [ ] Environment cache stores and retrieves environments by ID
- [ ] Training produces checkpoints registered in the model registry
- [ ] Predictions are tagged `[SHADOW]` when model is in shadow status
- [ ] Promotion runs through `PromotionGate` statistical checks
- [ ] Tools are importable and compatible with the existing `TOOL_REGISTRY`
