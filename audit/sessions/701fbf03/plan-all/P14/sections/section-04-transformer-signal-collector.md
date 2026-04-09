# Section 04: Transformer Signal Collector

## Objective

Create a signal engine collector that loads the trained transformer model, runs inference on current features, and outputs a predicted 5-day return signal for the synthesis engine.

## Dependencies

- **section-03-transformer-forecaster** — requires `TransformerForecaster` and `TransformerPrediction`

## Files to Create/Modify

### New Files

- **`src/quantstack/signal_engine/collectors/transformer_signal.py`** — Transformer signal collector following the standard collector interface.

### Modified Files

- **`src/quantstack/signal_engine/collectors/__init__.py`** — Register the new collector.
- **`src/quantstack/signal_engine/synthesis.py`** — Add `transformer` to weight profiles with weight 0.10, adjusted by IC.

## Implementation Details

### `src/quantstack/signal_engine/collectors/transformer_signal.py`

Follow the same pattern as `ml_signal.py` — async wrapper around sync inference via `asyncio.to_thread`, never raises, returns `{}` on any failure.

```
async def collect_transformer_signal(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Run transformer inference for *symbol* using the latest checkpoint.

    Returns a dict with keys:
        transformer_prediction    : float — predicted 5-day return
        transformer_direction     : str — "bullish" (>0.005), "bearish" (<-0.005), "neutral"
        transformer_confidence    : float 0-1 — direction confidence
        transformer_model_type    : str — "patchtst" or "chronos"
        transformer_horizon       : int — prediction horizon in days (5)
        transformer_model_age_days: int | None — days since checkpoint was saved

    Returns {} if no checkpoint exists or data is insufficient.
    """
```

Implementation steps:
1. Load latest checkpoint for symbol from `models/transformers/{symbol}/`.
2. Fetch recent OHLCV + features from `store` (need at least `input_size` days of history).
3. Run `TransformerForecaster.predict()`.
4. Map predicted return to direction string and confidence.
5. Return dict or `{}` on any failure.

### Synthesis Integration

Add `transformer` to `_WEIGHT_PROFILES` in `synthesis.py`:

- Starting weight: 0.10 across all regimes
- Weight is stolen proportionally from existing voters (each reduced slightly)
- Like `ml`, the transformer weight is redistributed to other voters when no transformer signal is available
- IC-based adjustment: track transformer signal IC separately; if IC < 0.02 after 60 days, reduce weight to 0.03

### Collector Registration

Add to `__init__.py` collector registry so the synthesis engine discovers it automatically.

## Test Requirements

### `tests/unit/signal_engine/test_transformer_signal.py`

1. **Happy path**: Mock `TransformerForecaster.predict()` to return a known prediction; verify collector output dict has all expected keys.
2. **No checkpoint**: When no checkpoint exists, returns `{}` without raising.
3. **Data insufficient**: When store returns < required history, returns `{}`.
4. **Direction mapping**: predicted_return > 0.005 -> "bullish", < -0.005 -> "bearish", else "neutral".
5. **Never raises**: Inject exception in `predict()`, verify returns `{}` and logs warning.

## Acceptance Criteria

- [ ] Collector follows the standard async interface (`async def collect_transformer_signal(symbol, store) -> dict`)
- [ ] Returns all documented keys when a checkpoint exists
- [ ] Returns `{}` gracefully when no checkpoint or insufficient data
- [ ] Never raises exceptions — all failures caught and logged
- [ ] Registered in collector `__init__.py`
- [ ] Synthesis weights updated to include `transformer` at 0.10
- [ ] All unit tests pass
