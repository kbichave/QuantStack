# P09 TDD Plan: Reinforcement Learning Pipeline

## 1. Portfolio Optimization Environment

```python
# tests/unit/rl/test_portfolio_opt_env.py

class TestPortfolioOptEnv:
    def test_observation_space_shape(self):
        """State space shape is (n_assets, n_features) as defined."""

    def test_action_space_shape(self):
        """Action space is continuous Box with n_assets dimensions."""

    def test_action_softmax_normalization(self):
        """Actions normalized via softmax to sum to 1.0."""

    def test_reward_risk_adjusted(self):
        """Reward = daily_return - 0.5 * daily_return^2 / target_vol^2 - turnover_penalty."""

    def test_step_advances_time(self):
        """Each step advances the environment by one time period."""

    def test_episode_terminates_at_end_of_data(self):
        """Episode ends when historical data is exhausted."""

    def test_reset_returns_valid_initial_state(self):
        """Reset produces observation within observation_space bounds."""


class TestPortfolioOptEdgeCases:
    def test_single_asset_portfolio(self):
        """Environment works with n_assets=1 (trivial allocation)."""

    def test_zero_volume_day_handled(self):
        """Days with zero volume produce valid state (no NaN/inf)."""

    def test_negative_returns_produce_valid_reward(self):
        """Large drawdown day does not produce NaN reward."""
```

## 2. Order Execution Environment

```python
# tests/unit/rl/test_execution_opt_env.py

class TestExecutionOptEnv:
    def test_observation_space_shape(self):
        """State includes remaining_qty, time_remaining, spread, volume_profile, recent_fills."""

    def test_action_space_discrete(self):
        """Action space has 5 discrete options (0%, 10%, 25%, 50%, 100%)."""

    def test_reward_is_negative_shortfall(self):
        """Reward = negative implementation shortfall vs TWAP benchmark."""

    def test_episode_ends_when_fully_filled(self):
        """Episode terminates when remaining_qty reaches 0."""

    def test_episode_ends_when_time_expires(self):
        """Episode terminates when time_remaining reaches 0."""

    def test_remaining_qty_decreases_after_fill(self):
        """After a non-zero action with fill, remaining_qty decreases."""


class TestExecutionOptEdgeCases:
    def test_zero_remaining_qty_no_action(self):
        """Action on zero remaining qty is a no-op."""

    def test_wide_spread_penalizes_reward(self):
        """Execution during wide spread results in higher shortfall."""
```

## 3. Strategy Selection Environment

```python
# tests/unit/rl/test_strategy_select_env.py

class TestStrategySelectEnv:
    def test_observation_includes_regime_and_ic(self):
        """State space includes regime indicator, per-strategy IC, vol level."""

    def test_action_space_continuous_softmax(self):
        """Action is continuous allocation per strategy, summing to 1.0."""

    def test_reward_is_rolling_sharpe(self):
        """Reward = 21-day rolling portfolio Sharpe ratio."""

    def test_regime_change_reflected_in_state(self):
        """State correctly reflects regime transitions."""
```

## 4. Training Infrastructure

```python
# tests/unit/rl/test_training.py

class TestWalkForwardValidation:
    def test_train_window_excludes_buffer(self):
        """Train window is T-252 to T-21 (excludes 21-day buffer)."""

    def test_validation_window_is_last_month(self):
        """Validation window is T-21 to T."""

    def test_checkpoint_saved_every_n_steps(self):
        """Checkpoint files created every 10K training steps."""


class TestEarlyStopping:
    def test_stops_on_sharpe_degradation(self):
        """Training stops after 3 consecutive checkpoints with declining val Sharpe."""

    def test_stops_on_max_steps(self):
        """Training stops at 500K steps regardless of performance."""

    def test_stops_on_nan_loss(self):
        """Training halts immediately on NaN in loss function."""

    def test_continues_on_improving_sharpe(self):
        """Training continues when validation Sharpe is improving."""


class TestModelRegistryIntegration:
    def test_model_stored_with_correct_type(self):
        """RL model registered with type 'rl_ppo', 'rl_sac', or 'rl_dqn'."""

    def test_metadata_includes_training_info(self):
        """Registry entry includes env_name, episodes, final_sharpe, action_space_dim."""

    def test_versioning_follows_ab_path(self):
        """RL models follow same A/B promotion pipeline as ML models."""
```

## 5. FinRL Tool Implementation

```python
# tests/unit/rl/test_finrl_tools.py

class TestFinrlTrainModel:
    def test_train_returns_result_with_metrics(self):
        """finrl_train_model returns training_result with loss curve, final Sharpe."""

    def test_invalid_env_name_raises(self):
        """Unknown env_name raises ValueError."""

    def test_invalid_model_type_raises(self):
        """Unsupported model_type raises ValueError."""


class TestFinrlPredict:
    def test_predict_returns_valid_action(self):
        """finrl_predict returns action within environment's action space."""

    def test_predict_with_nonexistent_model_raises(self):
        """Prediction with unknown model_id raises ModelNotFoundError."""


class TestFinrlEvaluate:
    def test_evaluate_returns_performance_metrics(self):
        """finrl_evaluate returns Sharpe, max drawdown, total return."""

    def test_evaluate_on_unseen_data(self):
        """Evaluation uses specified eval_window, not training data."""


class TestFinrlEnsemble:
    def test_ensemble_combines_predictions(self):
        """finrl_ensemble with weights produces weighted average action."""

    def test_ensemble_weights_must_sum_to_one(self):
        """Weights that don't sum to 1.0 raise ValueError."""


class TestFinrlPromote:
    def test_promote_updates_model_status(self):
        """finrl_promote moves model from candidate to production."""

    def test_promote_requires_30_day_paper(self):
        """Promotion rejected if model has < 30 days paper validation."""
```

## 6. Safety Constraints

```python
# tests/unit/rl/test_rl_safety.py

class TestRLSafety:
    def test_rl_signal_weight_capped(self):
        """RL collector weight in synthesis is 0.15 (same as ML)."""

    def test_max_position_change_per_step(self):
        """Single step cannot change allocation by more than 10%."""

    def test_turnover_penalty_in_reward(self):
        """Turnover > threshold reduces reward by penalty factor."""

    def test_feature_flag_default_disabled(self):
        """rl_signal_enabled() returns False by default."""

    def test_risk_gate_filters_rl_recommendations(self):
        """RL action that violates risk limits is rejected by risk gate."""

    def test_no_live_without_paper_validation(self):
        """RL model without 30 days of paper results cannot influence live trades."""
```
