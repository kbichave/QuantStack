# Section 04: Training Infrastructure

## Objective

Extend the existing `FinRLTrainer` with walk-forward validation, early stopping, checkpoint management, and model registry integration. Add the database schema for `rl_training_runs` and `rl_model_checkpoints` tables. This section transforms the trainer from a simple train-and-save wrapper into a production training pipeline.

## Dependencies

- **section-01-portfolio-opt-env**: `PortfolioOptEnv` must exist for walk-forward training

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/finrl/trainer.py` | **Modify** | Add walk-forward validation, early stopping, checkpoint callbacks |
| `src/quantstack/finrl/model_registry.py` | **Modify** | Add `rl_training_runs` and `rl_model_checkpoints` table support |
| `src/quantstack/finrl/config.py` | **Modify** | Add training infrastructure config fields |

## Implementation Details

### Walk-Forward Validation (plan section 4.1)

Add `train_walk_forward()` method to `FinRLTrainer`:

```
Train window: T-252 to T-21 (1 year minus 1 month buffer)
Validation window: T-21 to T (last month)
Retrain: weekly (overnight, on finrl-worker container)
Checkpoint: every 10K steps to models/rl/ directory
```

**Method signature**:
```python
def train_walk_forward(
    self,
    env_cls: type[gym.Env],
    data: pd.DataFrame,
    algorithm: str = "ppo",
    train_window: int = 252,
    val_window: int = 21,
    total_timesteps: int = 500_000,
    checkpoint_interval: int = 10_000,
    model_name: str | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> TrainResult:
```

Walk-forward splits the data into train/val using the specified windows, trains on the train portion, and evaluates on the val portion. The trainer creates the environment instance from `env_cls` with the appropriate data slice.

### Early Stopping (plan section 4.2)

Add `EarlyStoppingCallback` (extends `stable_baselines3.common.callbacks.BaseCallback`):

Stop training when:
1. Validation Sharpe degrades for 3 consecutive checkpoints
2. Max training steps reached (default 500K)
3. NaN detected in loss function

**Implementation**:
- Custom SB3 callback that evaluates the model on validation env every `checkpoint_interval` steps
- Tracks validation Sharpe at each checkpoint
- Maintains a counter of consecutive Sharpe degradations
- Saves checkpoint to disk at each evaluation
- Raises `StopIteration` (via `return False` from `_on_step`) when early stopping triggers

### Checkpoint Management

Add checkpoint tracking to the training loop:
- Save model every `checkpoint_interval` steps (default 10K) to `{checkpoint_base_path}/{model_id}/step_{N}/`
- Save metadata JSON alongside each checkpoint: step number, training reward, validation Sharpe
- Register checkpoints in `rl_model_checkpoints` table

### Config Additions

Add to `FinRLConfig`:
```python
# Walk-forward defaults
wf_train_window: int = 252
wf_val_window: int = 21
wf_checkpoint_interval: int = 10_000
wf_max_timesteps: int = 500_000
wf_early_stop_patience: int = 3  # consecutive Sharpe degradations
```

### Database Schema (plan section 7)

Add two new tables via `ModelRegistry`:

**`rl_training_runs`**:
```sql
CREATE TABLE IF NOT EXISTS rl_training_runs (
    id              VARCHAR PRIMARY KEY,
    env_name        VARCHAR NOT NULL,
    model_type      VARCHAR NOT NULL,
    start_time      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time        TIMESTAMP,
    episodes        INTEGER,
    final_sharpe    FLOAT,
    status          VARCHAR DEFAULT 'running',
    config_json     TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**`rl_model_checkpoints`**:
```sql
CREATE TABLE IF NOT EXISTS rl_model_checkpoints (
    id              VARCHAR PRIMARY KEY,
    training_run_id VARCHAR REFERENCES rl_training_runs(id),
    step            INTEGER NOT NULL,
    sharpe          FLOAT,
    file_path       VARCHAR NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Add methods to `ModelRegistry`:
- `create_training_run(env_name, model_type, config) -> run_id`
- `update_training_run(run_id, status, final_sharpe, end_time)`
- `save_checkpoint(run_id, step, sharpe, file_path) -> checkpoint_id`
- `get_best_checkpoint(run_id) -> checkpoint_dict`

### Model Registry Integration (plan section 4.3)

Store RL models with type prefixes: `rl_ppo`, `rl_sac`, `rl_dqn`. Metadata includes:
- Environment name (portfolio_opt, execution, strategy_select)
- Training episodes count
- Final validation Sharpe
- Action space dimensionality

Use the same A/B promotion path as ML models via existing `PromotionGate`.

## Test Requirements

1. **Walk-forward split**: Verify train/val data windows are correctly sliced
2. **Early stopping**: Training stops after 3 consecutive Sharpe degradations (mock env with declining reward)
3. **NaN detection**: Training stops immediately when NaN loss is detected
4. **Checkpoint save**: Checkpoints are saved at specified intervals with correct metadata
5. **DB schema**: Tables are created without errors; CRUD operations work
6. **Training run tracking**: Start/end times and status transitions are recorded
7. **Best checkpoint selection**: Returns checkpoint with highest validation Sharpe

## Acceptance Criteria

- [ ] `train_walk_forward()` method correctly splits data into train/val windows
- [ ] `EarlyStoppingCallback` stops training on Sharpe degradation (patience=3), max steps, or NaN
- [ ] Checkpoints saved every 10K steps with metadata JSON
- [ ] `rl_training_runs` and `rl_model_checkpoints` tables created and functional
- [ ] Training runs are tracked with start/end times and status
- [ ] Config fields for walk-forward parameters are in `FinRLConfig`
- [ ] RL models stored with `rl_` prefix in model type field
