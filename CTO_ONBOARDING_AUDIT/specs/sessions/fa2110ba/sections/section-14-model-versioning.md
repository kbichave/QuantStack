# Section 14: Model Versioning + Champion/Challenger

## Overview

No model versioning exists today. Models are trained ad-hoc by `src/quantstack/ml/trainer.py`, saved as flat files (`models/{symbol}_latest.joblib`), and loaded by the ML signal collector (`src/quantstack/signal_engine/collectors/ml_signal.py`) via `_load_latest_training_result()`. There is no way to compare model versions, run shadow evaluations, safely roll back, or even know which model version is currently driving signals. A bad model silently replaces a good one the moment training completes.

This section introduces a `model_registry` table, a `model_shadow_predictions` table, and a champion/challenger workflow. New models enter as challengers, run in shadow for a minimum of 30 trading days, and only promote to champion if they demonstrably outperform. The ML collector is modified to load the current champion from the registry instead of whatever file happens to exist on disk.

**Dependencies:** None. This section is parallelizable (Batch 1). However, it integrates naturally with Section 13 (drift detection) which can trigger retraining that produces new challengers.

**Key files touched:**

| File | Action |
|------|--------|
| `src/quantstack/db.py` | Add `model_registry` and `model_shadow_predictions` table creation |
| `src/quantstack/ml/model_registry.py` | **New file** — registry CRUD, promotion logic |
| `src/quantstack/ml/trainer.py` | Register new models as challengers after training |
| `src/quantstack/signal_engine/collectors/ml_signal.py` | Load champion from registry; run challenger shadow inference |
| `src/quantstack/graphs/supervisor/nodes.py` | New batch node `run_model_promotion_check()` |
| `tests/unit/test_model_versioning.py` | All tests for this section |

---

## Tests (Write First)

File: `tests/unit/test_model_versioning.py`

### Model Registry Tests

```python
class TestModelRegistry:
    """Tests for model_registry CRUD and versioning logic."""

    def test_register_new_model_auto_increments_version(self, mock_settings):
        """Register two models for the same strategy.
        First should get version=1, second version=2.
        Uses mock DB via mock_settings fixture."""

    def test_query_champion_returns_correct_version(self, mock_settings):
        """Register a model, promote it to champion.
        query_champion(strategy_id) returns that model's version."""

    def test_retire_old_model_changes_status(self, mock_settings):
        """Retire a model. Status changes to 'retired', retired_at is set."""

    def test_register_model_stores_all_metadata(self, mock_settings):
        """Verify train_date, features_hash, hyperparams, backtest metrics,
        and model_path are all persisted correctly."""

    def test_only_one_champion_per_strategy(self, mock_settings):
        """Promoting a new champion retires the old one.
        At most one champion per strategy_id at any time."""
```

### Shadow Mode Tests

```python
class TestShadowMode:
    """Tests for challenger shadow predictions."""

    def test_challenger_predictions_logged_to_shadow_table(self, mock_settings):
        """Run shadow inference for a challenger model.
        Verify prediction row written to model_shadow_predictions
        with model_id, symbol, date, prediction."""

    def test_champion_drives_real_signals(self, mock_settings):
        """When both champion and challenger exist, the dict returned
        by collect_ml_signal uses champion's prediction for
        ml_prediction/ml_direction/ml_confidence.
        Challenger prediction only appears in shadow table."""
```

### Promotion Tests

```python
class TestModelPromotion:
    """Tests for champion/challenger promotion criteria."""

    def test_promote_when_all_criteria_met(self, mock_settings):
        """Challenger with IC improvement > 0.005,
        Sharpe improvement > 0.15, and max DD <= 1.1x champion DD.
        All three met -> promote. Old champion becomes retired."""

    def test_no_promotion_when_ic_criterion_not_met(self, mock_settings):
        """Challenger IC only 0.002 better than champion.
        No promotion despite Sharpe and DD passing."""

    def test_no_promotion_when_sharpe_criterion_not_met(self, mock_settings):
        """Challenger Sharpe only 0.10 better.
        No promotion."""

    def test_no_promotion_when_dd_regresses(self, mock_settings):
        """Challenger max DD is 1.2x champion DD.
        No promotion."""

    def test_retire_challenger_after_60_days_without_promotion(self, mock_settings):
        """Challenger has been in shadow for 60 trading days
        and never met promotion criteria. Status set to retired."""

    def test_promotion_publishes_model_trained_event(self, mock_settings):
        """On successful promotion, a MODEL_TRAINED event is published
        to the EventBus with the new champion's model_id and strategy_id."""
```

### Cold-Start Tests

```python
class TestModelVersioningColdStart:
    """Tests for graceful degradation when registry is empty."""

    def test_no_champion_falls_back_to_disk_loading(self, mock_settings):
        """No champion in registry for a strategy.
        ML collector falls back to _load_latest_training_result()
        (existing disk-based path). Signals still produced."""

    def test_challenger_with_fewer_than_30_days_not_eligible(self, mock_settings):
        """Challenger registered 15 days ago.
        Promotion check skips it — not enough shadow data."""
```

---

## Database Schema

### `model_registry` table

Add to `src/quantstack/db.py` in the table creation section, following the existing `CREATE TABLE IF NOT EXISTS` pattern:

```sql
CREATE TABLE IF NOT EXISTS model_registry (
    model_id        TEXT PRIMARY KEY,
    strategy_id     TEXT NOT NULL,
    version         INTEGER NOT NULL,
    train_date      DATE NOT NULL,
    train_data_range TEXT,
    features_hash   TEXT,
    hyperparams     JSONB DEFAULT '{}',
    backtest_sharpe FLOAT,
    backtest_ic     FLOAT,
    backtest_max_dd FLOAT,
    model_path      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'challenger',
    promoted_at     TIMESTAMPTZ,
    retired_at      TIMESTAMPTZ,
    shadow_start    DATE,
    shadow_ic       FLOAT,
    shadow_sharpe   FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (strategy_id, version)
);
CREATE INDEX IF NOT EXISTS idx_model_registry_strategy_status
    ON model_registry (strategy_id, status);
```

Status values: `champion`, `challenger`, `retired`.

### `model_shadow_predictions` table

```sql
CREATE TABLE IF NOT EXISTS model_shadow_predictions (
    id              SERIAL PRIMARY KEY,
    model_id        TEXT NOT NULL REFERENCES model_registry(model_id),
    symbol          TEXT NOT NULL,
    prediction_date DATE NOT NULL,
    prediction      FLOAT NOT NULL,
    realized_return FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_shadow_preds_model_date
    ON model_shadow_predictions (model_id, prediction_date);
```

The `realized_return` column is backfilled after the forward period elapses (same pattern as ICAttributionTracker's `forward_return`).

---

## Implementation Details

### 1. Model Registry Module

**New file:** `src/quantstack/ml/model_registry.py`

This module provides CRUD operations for the `model_registry` table and the promotion decision logic. All DB access uses `db_conn()` context managers.

Key functions (signatures only):

```python
from dataclasses import dataclass
from datetime import date, datetime

@dataclass
class ModelVersion:
    """Represents a registered model version."""
    model_id: str
    strategy_id: str
    version: int
    train_date: date
    train_data_range: str
    features_hash: str
    hyperparams: dict
    backtest_sharpe: float
    backtest_ic: float
    backtest_max_dd: float
    model_path: str
    status: str  # champion | challenger | retired
    promoted_at: datetime | None
    retired_at: datetime | None
    shadow_start: date | None
    shadow_ic: float | None
    shadow_sharpe: float | None
    created_at: datetime

def register_model(
    strategy_id: str,
    train_date: date,
    train_data_range: str,
    features_hash: str,
    hyperparams: dict,
    backtest_sharpe: float,
    backtest_ic: float,
    backtest_max_dd: float,
    model_path: str,
) -> ModelVersion:
    """Register a new model as challenger. Version auto-increments per strategy.
    Sets shadow_start to today."""

def query_champion(strategy_id: str) -> ModelVersion | None:
    """Return the current champion for a strategy, or None."""

def promote_challenger(model_id: str) -> None:
    """Promote challenger to champion. Retire the current champion for
    the same strategy. Set promoted_at. Publish MODEL_TRAINED event."""

def retire_model(model_id: str) -> None:
    """Set status to retired, set retired_at timestamp."""

def get_challengers_for_review() -> list[ModelVersion]:
    """Return all challengers with >= 30 trading days of shadow data."""

def get_stale_challengers(max_shadow_days: int = 60) -> list[ModelVersion]:
    """Return challengers that have been in shadow > max_shadow_days
    without promotion. These should be retired."""
```

**Version auto-increment logic:** Query `SELECT COALESCE(MAX(version), 0) + 1 FROM model_registry WHERE strategy_id = %s`. Use this as the new version number.

**Features hash:** `hashlib.sha256(json.dumps(sorted(feature_list)).encode()).hexdigest()[:16]`. This lets you detect when the feature set changes between model versions.

**Model file storage convention:** `~/.quantstack/models/{strategy_id}/v{version}/model.pkl`. Include a `metadata.json` alongside containing training config for reproducibility. The `model_path` column stores the absolute path.

### 2. Promotion Criteria

The promotion check runs as a supervisor batch node (daily after market close). For each challenger with >= 30 trading days of shadow data:

1. Compute challenger's shadow IC and shadow Sharpe from `model_shadow_predictions` (requires `realized_return` to be backfilled).
2. Compare to the current champion's shadow metrics (or backtest metrics if no shadow data for champion).
3. **All three criteria must be met simultaneously:**
   - Challenger IC > champion IC + 0.005 (meaningful predictive improvement)
   - Challenger Sharpe > champion Sharpe + 0.15 (meaningful risk-adjusted improvement)
   - Challenger max drawdown <= 1.1 x champion max drawdown (no drawdown regression)
4. If all met: call `promote_challenger()`, which retires the old champion and publishes `MODEL_TRAINED` event.
5. If not met and challenger has been in shadow > 60 trading days: retire the challenger. It had its chance.

The thresholds (0.005 IC, 0.15 Sharpe, 1.1x DD) are intentionally conservative. False promotions (replacing a good model with a worse one) are more expensive than missed promotions (keeping a slightly suboptimal model for another training cycle).

### 3. ML Collector Integration

Modify `src/quantstack/signal_engine/collectors/ml_signal.py`:

**Replace `_load_latest_training_result()`** with a registry-aware loading sequence:

```python
def _load_model_for_symbol(symbol: str, strategy_id: str) -> tuple[Any, str | None]:
    """Load champion model from registry. Returns (model, challenger_model_id).

    Falls back to disk-based loading if no champion is registered
    (cold-start / backward compatibility).

    If a challenger exists, also returns its model_id so the caller
    can run shadow inference.
    """
```

The function:
1. Calls `query_champion(strategy_id)` to get the champion `ModelVersion`.
2. If found, loads the model from `model_path` via joblib.
3. Also checks for an active challenger. If one exists, returns its `model_id` so the caller can run shadow inference.
4. If no champion found, falls back to `_load_latest_training_result()` (existing behavior).

**Shadow inference:** After computing the champion's prediction (which drives the real signal output), if a challenger exists:
1. Load the challenger model from its `model_path`.
2. Run inference on the same feature set.
3. Write the prediction to `model_shadow_predictions` with `realized_return=NULL` (backfilled later).
4. Do NOT include challenger predictions in the returned signal dict.

The shadow inference should be wrapped in a try/except — a failing challenger must never impact the champion's signal production.

### 4. Trainer Integration

Modify `src/quantstack/ml/trainer.py`:

After a successful training run (when `TrainingResult` is produced and saved to disk), call `register_model()` to register it as a challenger:

```python
# After saving model to disk:
from quantstack.ml.model_registry import register_model

register_model(
    strategy_id=strategy_id,
    train_date=date.today(),
    train_data_range=f"{train_start} to {train_end}",
    features_hash=compute_features_hash(feature_columns),
    hyperparams=training_config.to_dict(),
    backtest_sharpe=result.sharpe,
    backtest_ic=result.ic,
    backtest_max_dd=result.max_drawdown,
    model_path=str(model_save_path),
)
```

This means every trained model automatically enters the shadow pipeline. No manual intervention needed.

### 5. Supervisor Batch Node

Add `run_model_promotion_check()` to `src/quantstack/graphs/supervisor/nodes.py`. Schedule: daily after market close.

Logic:
1. Call `get_challengers_for_review()` — challengers with >= 30 days shadow data.
2. For each, compute shadow IC and Sharpe from `model_shadow_predictions`.
3. Compare against champion metrics. Promote or skip.
4. Call `get_stale_challengers(60)` — retire any that exceeded the 60-day window.
5. Log a summary: "Model promotion check: {n} challengers reviewed, {promoted} promoted, {retired} retired."

### 6. Realized Return Backfill

The `model_shadow_predictions.realized_return` column needs to be populated once the forward period elapses. Add a hook in `trade_hooks.py::on_daily_close()` (or a lightweight supervisor batch) that:

1. Queries shadow predictions where `realized_return IS NULL` and `prediction_date <= today - 5` (assuming 5-day forward returns).
2. Computes the actual 5-day return from OHLCV data.
3. Updates the `realized_return` column.

This is the same backfill pattern used by ICAttributionTracker for `forward_return`.

---

## Rollback Path

Set no config flag (model versioning is always-on for data collection). Rollback: the ML collector's `_load_model_for_symbol()` falls back to `_load_latest_training_result()` when no champion exists in the registry. To fully disable, revert the ML collector changes — it returns to pure disk-based model loading. The `model_registry` and `model_shadow_predictions` tables remain but go stale.

---

## Cold-Start Behavior

| Condition | Behavior |
|-----------|----------|
| No champion registered for a strategy | ML collector falls back to `_load_latest_training_result()` (disk-based). Existing behavior preserved. |
| Challenger has < 30 trading days of shadow data | Promotion check skips it. Challenger continues accumulating shadow predictions. |
| No challenger exists | Shadow inference step is skipped entirely. No performance cost. |
| `model_shadow_predictions` has no backfilled returns yet | Promotion check cannot compute shadow IC/Sharpe. Challenger waits. |

---

## EventBus Integration

Uses the existing `MODEL_TRAINED` event type (already in `EventType` enum). Published when a challenger is promoted to champion. Payload:

```python
{
    "model_id": "abc123",
    "strategy_id": "swing_momentum_AAPL",
    "version": 3,
    "shadow_ic": 0.045,
    "shadow_sharpe": 1.82,
    "previous_champion_version": 2,
}
```

The research graph polls for `MODEL_TRAINED` events to log model lifecycle transitions.
