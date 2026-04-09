# Section 03: Transformer Time Series Forecaster

## Objective

Add transformer-based time series forecasting using PatchTST (primary) and Chronos (zero-shot fallback). These models capture long-range temporal dependencies that tree-based models miss. They run as additional signal sources alongside existing LightGBM/XGBoost, not replacements.

## Files to Create/Modify

### New Files

- **`src/quantstack/ml/transformers/__init__.py`** — Package init.
- **`src/quantstack/ml/transformers/forecaster.py`** — PatchTST/Chronos training and inference.
- **`src/quantstack/ml/transformers/config.py`** — Training configuration dataclass.

### Modified Files

- None (standalone module; signal collector integration is section-04).

## Implementation Details

### `src/quantstack/ml/transformers/config.py`

```
@dataclass
class TransformerTrainingConfig:
    model_type: Literal["patchtst", "chronos"] = "patchtst"
    horizon: int = 5                    # 5-day ahead return prediction
    input_size: int = 20                # number of input features
    patch_len: int = 16                 # PatchTST patch length
    stride: int = 8                     # PatchTST stride
    n_heads: int = 4                    # attention heads
    d_model: int = 64                   # model dimension
    max_epochs: int = 50                # training epochs
    batch_size: int = 32                # CPU-friendly batch size
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    checkpoint_dir: str = "models/transformers/"
    use_gpu: bool = False               # GPU optional via CUDA flag
```

### `src/quantstack/ml/transformers/forecaster.py`

```
class TransformerForecaster:
    """PatchTST / Chronos forecaster for financial time series."""

    def __init__(self, config: TransformerTrainingConfig):
        ...

    def prepare_data(self, df: pd.DataFrame, symbol: str) -> tuple:
        """Prepare OHLCV + technical indicators into neuralforecast format.
        
        Input: DataFrame with OHLCV + 20 feature columns.
        Output: neuralforecast-compatible DataFrame with unique_id, ds, y, exogenous cols.
        Walk-forward split: train on all but last `horizon` days, validate on last `horizon`.
        """

    def train(self, df: pd.DataFrame, symbol: str) -> TransformerTrainResult:
        """Train PatchTST model with walk-forward validation.
        
        Steps:
        1. Prepare data into neuralforecast format
        2. Instantiate PatchTST from neuralforecast.models
        3. Fit with early stopping
        4. Evaluate on held-out walk-forward fold
        5. Save checkpoint to config.checkpoint_dir/{symbol}/
        
        Returns TransformerTrainResult with metrics and checkpoint path.
        """

    def train_chronos_fallback(self, df: pd.DataFrame, symbol: str) -> TransformerTrainResult:
        """Use pre-trained Chronos for zero-shot forecasting (no training needed).
        
        Chronos is Amazon's pre-trained foundation model — it produces forecasts
        without any fine-tuning. Used when PatchTST training data is insufficient
        (< 252 trading days).
        """

    def predict(self, df: pd.DataFrame, symbol: str) -> TransformerPrediction:
        """Run inference using latest checkpoint.
        
        Returns:
            TransformerPrediction with:
                predicted_return: float — predicted 5-day return
                direction_confidence: float — abs(predicted_return) normalized
                model_type: str — "patchtst" or "chronos"
        """

    def load_checkpoint(self, symbol: str) -> None:
        """Load model from checkpoint_dir/{symbol}/latest.ckpt."""
```

```
@dataclass
class TransformerTrainResult:
    symbol: str
    model_type: str
    mse: float
    directional_accuracy: float
    checkpoint_path: str
    train_samples: int
    epochs_trained: int

@dataclass
class TransformerPrediction:
    predicted_return: float
    direction_confidence: float
    model_type: str
    horizon: int
```

### Key Design Decisions

1. **neuralforecast library** for PatchTST — mature, maintained, handles training loops.
2. **CPU-first**: `batch_size=32`, `use_gpu=False` by default. GPU via env `CUDA_VISIBLE_DEVICES`.
3. **Chronos fallback** for symbols with insufficient history — zero-shot, no training needed.
4. **Walk-forward validation** reuses the same temporal discipline as P03 tree models.
5. **Checkpoints saved per-symbol** to `models/transformers/{symbol}/`.

## Dependencies

- **PyPI**: `neuralforecast` (Nixtla — PatchTST implementation), `chronos-forecasting` (Amazon — pre-trained foundation model)
- **Internal**: `quantstack.core.features.*` for feature engineering, `quantstack.data.storage.DataStore` for OHLCV data

## Test Requirements

### `tests/unit/ml/test_transformer_forecaster.py`

1. **Synthetic trend test**: Generate synthetic uptrend data, train PatchTST, verify predicted return is positive.
2. **Config validation**: Invalid config values (horizon=0, batch_size=0) raise `ValueError`.
3. **Chronos fallback**: When training data < 252 rows, Chronos is used automatically.
4. **Checkpoint save/load**: Train, save, load, predict — verify predictions match.
5. **Data preparation**: Verify neuralforecast-format DataFrame has correct columns (`unique_id`, `ds`, `y`).
6. **CPU inference**: Verify prediction runs without GPU (mock CUDA unavailable).

## Acceptance Criteria

- [ ] PatchTST trains on OHLCV + features and produces 5-day return predictions
- [ ] Chronos fallback works for symbols with < 252 days of data
- [ ] Checkpoints are saved/loaded per-symbol
- [ ] Walk-forward validation produces MSE and directional accuracy metrics
- [ ] All inference runs on CPU without GPU requirement
- [ ] All unit tests pass
