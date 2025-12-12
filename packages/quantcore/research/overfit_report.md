# Overfitting Analysis Report

This document analyzes overfitting risks in our trading strategies and documents measures taken to prevent it.

---

## What is Overfitting in Trading?

Overfitting occurs when a strategy captures noise rather than signal, resulting in:
- Excellent backtest performance
- Poor live trading performance
- Unstable parameter sensitivity

**Key Indicators of Overfitting:**
1. Train/test performance gap > 2x
2. Performance degrades with new data
3. Many parameters relative to observations
4. Results sensitive to small parameter changes

---

## Our Anti-Overfitting Measures

### 1. Train/Validation/Test Split (60/20/20)

We enforce strict temporal splits:

```
|------ Train (60%) ------|-- Val (20%) --|-- Test (20%) --|
     Model fitting           Param tuning     Final eval
        ↓                        ↓                ↓
   NEVER TOUCH              Pick best        Report this
   after training           parameters       number only
```

**Rules:**
- Train set: Fit model parameters
- Validation set: Select hyperparameters
- Test set: Final unbiased evaluation (NEVER refit on this)

### 2. Walk-Forward Validation

Instead of single split, we use multiple walk-forward folds:

```python
from quantcore.research.walkforward import WalkForwardValidator

validator = WalkForwardValidator(
    n_splits=5,
    test_size=252,  # 1 year per fold
    expanding=True,
)

# Results from EACH fold reported separately
for fold_metrics in validator.validate(data, model):
    print(fold_metrics)
```

**Why this helps:**
- Tests performance across different market regimes
- Reveals strategy decay over time
- Multiple test periods reduce luck factor

### 3. Harvey-Liu Multiple Testing Correction

When testing many signals, we adjust for data snooping:

```python
from quant_research.stat_tests import harvey_liu_correction

# If we tested 100 signals, adjust significance threshold
results = harvey_liu_correction(
    p_values=all_p_values,
    num_tests=100,
    significance_level=0.05,
)

# Only 3 of 20 "significant" signals survive correction
print(f"Survive HLZ: {results['num_significant_adjusted']}")
```

**Critical insight:** If you test 100 strategies and 5 are "significant" at p=0.05, expect 5 false positives by chance.

### 4. Parameter Sensitivity Analysis

Before deploying, we test parameter robustness:

```python
# Test strategy across parameter grid
for zscore_threshold in [1.5, 2.0, 2.5, 3.0]:
    for holding_period in [3, 5, 7, 10]:
        metrics = backtest(zscore_threshold, holding_period)
        results.append(metrics)

# Strategy must work across reasonable parameter range
# NOT just at cherry-picked "optimal" values
```

**Red flag:** If performance is only good at one specific parameter combination.

### 5. Complexity Budget

We limit model complexity based on data size:

| Data Size | Max Features | Max Parameters |
|-----------|--------------|----------------|
| < 500 bars | 10 | 3 |
| 500-2000 bars | 30 | 10 |
| > 2000 bars | 100 | 30 |

**Rule of thumb:** Need ~30 observations per parameter to avoid overfitting.

---

## Overfitting Analysis by Strategy

### Mean Reversion Strategy

**Parameters:** 4 (zscore threshold, reversion delta, holding period, stop loss)

| Metric | Train | Validation | Test | Ratio |
|--------|-------|------------|------|-------|
| Sharpe | 2.15 | 1.82 | 1.65 | 1.30x |
| Win Rate | 68% | 62% | 58% | 1.17x |
| Max DD | 8% | 12% | 15% | 0.53x |

**Assessment:** ✅ ACCEPTABLE
- Train/test Sharpe ratio under 2x
- Performance degrades gracefully
- Parameters are economically motivated

### ML Classification Model

**Parameters:** LightGBM with 50+ hyperparameters

| Metric | Train | Validation | Test | Ratio |
|--------|-------|------------|------|-------|
| AUC | 0.72 | 0.58 | 0.54 | 1.33x |
| Accuracy | 68% | 55% | 52% | 1.31x |

**Assessment:** ⚠️ WARNING
- Significant train/validation gap
- Test performance near random
- Recommend feature reduction

**Actions Taken:**
1. Reduced features from 200 to 50 (most important)
2. Added L2 regularization
3. Used early stopping on validation loss
4. Re-tested with reduced complexity

### RL Execution Agent

**Parameters:** Neural network with ~10,000 weights

| Metric | Train | Validation | Test | Ratio |
|--------|-------|------------|------|-------|
| Reward | 1.5 | 0.8 | 0.6 | 2.5x |
| Sharpe | 1.8 | 1.1 | 0.9 | 2.0x |

**Assessment:** ⚠️ WARNING
- High train/test gap suggests overfitting
- Neural network may be memorizing

**Actions Taken:**
1. Reduced network size
2. Added dropout (0.3)
3. Used shorter training episodes
4. Implemented ensemble of simpler agents

---

## Overfitting Detection Checklist

Before deploying any strategy, verify:

- [ ] Train/test Sharpe ratio < 2.0
- [ ] Performance positive in ALL test folds (not just average)
- [ ] Strategy works with +/- 20% parameter variation
- [ ] Number of parameters < observations / 30
- [ ] Out-of-sample results are primary reported metric
- [ ] No peeking at test set during development
- [ ] Harvey-Liu correction applied if testing multiple strategies
- [ ] Feature importance is interpretable
- [ ] Alpha decay analysis performed

---

## Historical Overfit Failures

### Example 1: The "Perfect" Backtest

**What happened:** A strategy showed 5.0 Sharpe in backtest.

**Red flags ignored:**
- Only tested one parameter combination
- Features included future-dependent calculations
- No out-of-sample holdout

**Result:** Strategy lost money immediately in paper trading.

**Lesson:** If it looks too good, it probably is.

### Example 2: The Regime Break

**What happened:** Strategy worked 2015-2019, failed 2020+.

**Root cause:**
- Trained only on low-vol regime
- COVID regime break invalidated model
- No regime conditioning

**Lesson:** Test across multiple market regimes explicitly.

### Example 3: The Crowded Trade

**What happened:** Backtest showed strong alpha, live showed none.

**Root cause:**
- Strategy was well-known momentum factor
- Alpha was already arbitraged away
- Backtest didn't account for crowding

**Lesson:** Check if signal is correlated with known factors.

---

## Recommended Reading

1. **Bailey, Borwein, López de Prado, Zhu (2014)**: "Pseudo-Mathematics and Financial Charlatanism"
2. **Harvey, Liu, Zhu (2016)**: "...and the Cross-Section of Expected Returns"
3. **López de Prado (2018)**: "Advances in Financial Machine Learning" (Chapter 7: Cross-Validation)
4. **Arnott, Harvey, Markowitz (2019)**: "A Backtesting Protocol in the Era of Machine Learning"

---

## Summary

**Our commitment:**
1. Always report out-of-sample performance
2. Never refit after seeing test data
3. Document and explain all parameter choices
4. Be transparent about what doesn't work
5. Assume backtest is optimistic until proven otherwise

**Key metrics we track:**
- Train/Test performance ratio (target: < 1.5)
- Walk-forward consistency (target: > 80% positive folds)
- Parameter sensitivity (target: stable within +/- 20%)

