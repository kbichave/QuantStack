# Section 08: Unit Tests

## Objective

Comprehensive test suite covering all P09 components: environments, training infrastructure, FinRL tools, signal integration, and safety gates. Tests must run without GPU, without database, and without network access (all external dependencies mocked).

## Dependencies

- **section-01, section-02, section-03**: All three environment classes
- **section-05**: FinRL tool implementations
- **section-07**: Safety gates and promotion checks

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `tests/unit/finrl/test_portfolio_opt_env.py` | **Create** | PortfolioOptEnv environment tests |
| `tests/unit/finrl/test_execution_env.py` | **Create** | ExecutionEnv environment tests (including TWAP updates) |
| `tests/unit/finrl/test_strategy_select_env.py` | **Create** | StrategySelectEnv environment tests |
| `tests/unit/finrl/test_training_infra.py` | **Create** | Walk-forward, early stopping, checkpoint tests |
| `tests/unit/finrl/test_finrl_tools.py` | **Create** | Tool implementation integration tests |
| `tests/unit/finrl/test_rl_signal_collector.py` | **Create** | RL signal collector and synthesis integration |
| `tests/unit/finrl/test_safety_gates.py` | **Create** | Safety gate enforcement tests |
| `tests/unit/finrl/__init__.py` | **Create** | Package init |

## Implementation Details

### Environment Tests (plan section 8, bullet 1)

**test_portfolio_opt_env.py**:
```python
def test_reset_observation_shape():
    """Observation shape matches observation_space after reset."""

def test_step_valid_action():
    """Step with valid action returns (obs, reward, done, truncated, info)."""

def test_softmax_normalization():
    """Action weights sum to 1.0 after softmax."""

def test_position_change_clamp():
    """Position change is clamped to max_position_change (10%)."""

def test_reward_formula():
    """Reward = daily_return - vol_penalty - turnover_penalty."""

def test_episode_termination():
    """Episode terminates when data is exhausted."""

def test_synthetic_mode():
    """Works without real data (synthetic returns)."""
```

**test_execution_env.py**:
```python
def test_twap_reward():
    """Reward is negative implementation shortfall vs TWAP."""

def test_observation_shape_updated():
    """Obs shape includes recent fill features."""

def test_completion_bonus():
    """100% fill gets no time penalty."""

def test_discrete_action_space():
    """Action space is Discrete(5) for DQN compatibility."""

def test_existing_behavior_preserved():
    """Existing execution fraction mapping works."""
```

**test_strategy_select_env.py**:
```python
def test_observation_shape():
    """Obs dim = 4 + 4*n_strategies + 4."""

def test_sharpe_reward():
    """Reward is rolling 21-day Sharpe after warmup."""

def test_regime_transitions():
    """All 4 regimes visited over 1000 steps."""

def test_replay_mode():
    """Uses provided strategy_returns when available."""
```

### Training Infrastructure Tests (plan section 8, bullet 2)

**test_training_infra.py**:
```python
def test_walk_forward_data_split():
    """Train/val windows are correctly sliced from data."""

def test_early_stopping_sharpe_degradation():
    """Training stops after 3 consecutive Sharpe drops."""

def test_early_stopping_nan():
    """Training stops immediately on NaN loss."""

def test_checkpoint_saved():
    """Checkpoint files exist at specified intervals."""

def test_training_run_db_tracking():
    """Training run is recorded in rl_training_runs table."""

def test_best_checkpoint_selection():
    """Returns checkpoint with highest validation Sharpe."""

def test_short_training_convergence():
    """100 episodes of training shows increasing reward (plan section 8)."""
```

Use a simple environment (e.g., `SizingEnv` with synthetic data) for training tests to keep them fast. Mock the database with an in-memory SQLite or mock objects.

### Tool Tests

**test_finrl_tools.py**:
```python
def test_create_environment_returns_env_id():
    """finrl_create_environment returns valid env_id."""

def test_train_model_produces_checkpoint():
    """finrl_train_model produces a checkpoint file."""

def test_predict_returns_action():
    """finrl_predict returns action and confidence."""

def test_predict_shadow_tag():
    """Shadow model predictions are tagged [SHADOW]."""

def test_list_models_empty():
    """finrl_list_models returns empty list initially."""

def test_promote_insufficient_shadow():
    """Promotion fails with < 30 days in shadow."""

def test_error_handling_invalid_model():
    """Invalid model_id returns error JSON, does not raise."""
```

### Signal Collector Tests

**test_rl_signal_collector.py**:
```python
def test_flag_disabled_returns_empty():
    """Returns {} when FEEDBACK_RL_SIGNAL is not set."""

def test_no_model_returns_empty():
    """Returns {} when no RL model is registered."""

def test_shadow_model_signal():
    """Returns signal with rl_shadow=True for shadow model."""

def test_direction_mapping():
    """confidence > 0.55 = bullish, < 0.45 = bearish, else neutral."""

def test_weight_redistribution():
    """Weights sum to 1.0 when RL signal is absent."""
```

### Safety Gate Tests (plan section 8, bullet 4)

**test_safety_gates.py**:
```python
def test_rl_weight_cap():
    """RL weight never exceeds 0.15."""

def test_risk_gate_filters_rl():
    """Risk gate rejects RL position change > 10% (plan section 8)."""

def test_risk_gate_passes_valid_rl():
    """Risk gate accepts RL position change <= 10%."""

def test_30_day_paper_gate():
    """Model with < 30 days shadow fails promotion."""

def test_30_day_paper_gate_pass():
    """Model with >= 30 days shadow passes duration check."""

def test_no_rl_bypass_in_risk_gate():
    """RL signals are not exempt from any standard risk check."""

def test_turnover_penalty_configurable():
    """Penalty reads from FinRLConfig, not hardcoded."""
```

### Inference Test (plan section 8, bullet 3)

Include in `test_finrl_tools.py`:
```python
def test_predict_action_in_action_space():
    """Predicted action is valid within the environment's action space."""
```

Train a tiny model (100 timesteps) on a simple env, then verify predict output is within bounds.

### Test Fixtures

Create shared fixtures in `conftest.py` or at the top of each test file:
- `synthetic_ohlcv()` - generates FinRL-format DataFrame with realistic price series
- `tmp_checkpoint_dir()` - temporary directory for model checkpoints (cleaned up after test)
- `mock_model_registry()` - in-memory mock of ModelRegistry
- `mock_store()` - mock DataStore for collector tests

## Test Requirements

All tests must:
1. Run without GPU (CPU-only training with minimal timesteps)
2. Run without database (mock or in-memory)
3. Run without network (mock external API calls)
4. Complete in under 60 seconds total
5. Use `pytest` with standard assertions
6. Use `tmp_path` fixture for file-based tests

## Acceptance Criteria

- [ ] All environment tests pass: shape validation, reward formulas, termination
- [ ] Training tests pass: walk-forward, early stopping, checkpointing
- [ ] Tool tests pass: create, train, evaluate, predict, list, promote
- [ ] Signal collector tests pass: flag gating, direction mapping, weight redistribution
- [ ] Safety gate tests pass: weight cap, position limit, 30-day gate, risk gate
- [ ] All tests run without GPU, database, or network
- [ ] Total test suite runs in under 60 seconds
- [ ] `uv run pytest tests/unit/finrl/` passes with zero failures
