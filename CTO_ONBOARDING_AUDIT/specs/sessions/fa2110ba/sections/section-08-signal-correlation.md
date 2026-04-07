# Section 8: Signal Correlation Tracking

## Overview

QuantStack runs 22 signal collectors independently with static weights, but many are highly correlated (e.g., technical RSI and ML direction often > 0.7 correlated). When correlated signals are weighted independently, effective independent signal count may be 10-12 rather than 22, inflating conviction and overstating confidence. This section adds a weekly supervisor batch that computes pairwise Spearman correlations, applies continuous penalties to redundant signals, and tracks effective independent signal count via eigenvalue decomposition.

**Dependencies:**
- Section 07 (IC weight adjustment): The correlation penalty stacks on top of the IC factor introduced in Section 07. The final effective weight formula becomes: `static_weight * ic_factor * correlation_penalty`. Section 07 must be complete so the synthesis integration point exists.

**Blocks:** Nothing directly. Section 16 (config flags integration) references this section's kill-switch flag.

**Kill-switch flag:** `FEEDBACK_CORRELATION_PENALTY` (env var, default `false`). When false, `correlation_penalty` always returns 1.0 -- the weekly correlation computation still runs and stores data, but penalties are not applied to synthesis weights.

---

## Tests First

File: `tests/unit/test_signal_correlation.py`

### Correlation Matrix Computation

```python
class TestCorrelationMatrix:
    """Pairwise Spearman correlation across collectors from 63 trading days of signal data."""

    def test_pairwise_spearman_computed_correctly(self):
        """Given known signal vectors for 3 collectors over 63 days,
        verify pairwise Spearman correlations match scipy.stats.spearmanr output."""

    def test_effective_independent_signal_count_via_eigenvalues(self):
        """Given a correlation matrix with known structure (e.g., two groups of
        perfectly correlated signals), eigenvalue count > 0.1 should equal the
        number of independent groups."""

    def test_insufficient_data_returns_identity_matrix(self):
        """With < 63 days of signal data, correlation matrix should be identity
        (no correlations detected) and no penalties applied."""
```

### Continuous Correlation Penalty

```python
class TestCorrelationPenalty:
    """Penalty formula: max(0.2, 1.0 - max(0.0, abs(corr) - 0.5) * 2.0)
    No penalty below 0.5 correlation. Linear increase from 0.5 to 0.75.
    Floor at 0.2x weight for correlations >= 0.9."""

    def test_low_correlation_no_penalty(self):
        """corr=0.4 should produce penalty = 1.0 (no penalty)."""
        # max(0.2, 1.0 - max(0.0, 0.4 - 0.5) * 2.0) = max(0.2, 1.0 - 0.0) = 1.0

    def test_moderate_correlation_partial_penalty(self):
        """corr=0.6 should produce penalty = 0.8."""
        # max(0.2, 1.0 - max(0.0, 0.6 - 0.5) * 2.0) = max(0.2, 1.0 - 0.2) = 0.8

    def test_high_correlation_strong_penalty(self):
        """corr=0.8 should produce penalty = 0.4."""
        # max(0.2, 1.0 - max(0.0, 0.8 - 0.5) * 2.0) = max(0.2, 1.0 - 0.6) = 0.4

    def test_very_high_correlation_hits_floor(self):
        """corr=0.95 should produce penalty = 0.2 (floor)."""
        # max(0.2, 1.0 - max(0.0, 0.95 - 0.5) * 2.0) = max(0.2, 1.0 - 0.9) = 0.2

    def test_weaker_signal_gets_penalty_not_stronger(self):
        """When two collectors are correlated, the one with lower IC receives the
        penalty. The stronger signal (higher IC) keeps penalty = 1.0."""
        # Given collector_a (IC=0.05) correlated at 0.7 with collector_b (IC=0.02),
        # collector_b gets the penalty, collector_a is unpenalized.

    def test_penalty_symmetric_for_equal_ic(self):
        """When two equally-IC collectors are correlated, both receive the penalty
        (or a consistent tie-breaking rule is applied)."""
```

### Config Flag

```python
class TestCorrelationConfigFlag:
    """FEEDBACK_CORRELATION_PENALTY env var controls whether penalties are applied."""

    def test_flag_false_penalty_always_one(self):
        """With FEEDBACK_CORRELATION_PENALTY=false, all correlation penalties
        should return 1.0 regardless of actual correlation values."""

    def test_flag_true_penalties_applied(self):
        """With FEEDBACK_CORRELATION_PENALTY=true, computed penalties should
        be applied to collector weights."""
```

---

## Implementation Details

### Supervisor Batch Node

Create a new supervisor batch node: `run_signal_correlation()`.

**Schedule:** Weekly, Friday after market close. This aligns with the weekly weight rebalancing cadence from Section 07 -- correlation penalties are computed on the same schedule as IC factor updates, so both feed into the next week's synthesis weights simultaneously.

**File:** `src/quantstack/graphs/supervisor/nodes.py` (add new node function)

### Correlation Computation Logic

Create a helper module for the computation: `src/quantstack/signal_engine/correlation.py`

Core function signature:

```python
def compute_signal_correlations(
    signal_data: dict[str, list[float]],
    ic_data: dict[str, float],
    min_observations: int = 63,
) -> CorrelationResult:
    """Compute pairwise Spearman correlations and derive penalties.

    Args:
        signal_data: Mapping of collector_name -> list of daily signal values
                     (most recent last). At least 63 trading days (~3 months).
                     Sourced from the signals table or ICAttributionTracker records.
        ic_data: Mapping of collector_name -> current rolling IC value.
                 Used to determine which signal in a correlated pair gets penalized.
        min_observations: Minimum number of signal observations required per collector.
                          Below this, the collector is excluded from correlation analysis.

    Returns:
        CorrelationResult with correlation_matrix, penalties, and effective_signal_count.
    """
```

The function should:

1. Filter to collectors with `>= min_observations` signal values.
2. Compute pairwise Spearman rank correlation using `scipy.stats.spearmanr` for each collector pair. Build an NxN correlation matrix.
3. For each correlated pair (abs(corr) > 0.5), identify the weaker signal (lower IC from `ic_data`). Apply the continuous penalty to the weaker signal only: `penalty = max(0.2, 1.0 - max(0.0, abs(corr) - 0.5) * 2.0)`.
4. When a collector is correlated with multiple others, take the minimum penalty (most penalized) across all its correlated pairs. This prevents a collector from accumulating separate penalties that compound below the floor.
5. Compute effective independent signal count: eigenvalue decomposition of the correlation matrix, count eigenvalues > 0.1.
6. Return a result containing the correlation matrix, per-collector penalties, and effective signal count.

### Penalty Formula Rationale

The formula `max(0.2, 1.0 - max(0.0, abs(corr) - 0.5) * 2.0)` is designed with three properties:

- **No penalty below 0.5:** With 63-day sample sizes, the standard error of Spearman correlation is approximately 0.13, so correlations below 0.5 are within noise range and should not trigger penalties.
- **Linear ramp 0.5 to 0.75+:** Smooth degradation avoids cliff effects where a collector oscillates between penalized and unpenalized on successive weeks due to sampling noise.
- **Floor at 0.2:** Even highly correlated signals retain 20% weight. Complete removal (0.0) is too aggressive because correlations can be regime-dependent -- two signals correlated in trending markets may diverge in ranging markets.

### Penalty Assignment: Weaker Signal Gets Penalized

When two collectors are correlated, the penalty goes to the one with **lower IC** (from the rolling 21-day IC used in Section 07). The rationale: if both signals carry the same information, keep the one that is more predictive. The stronger signal retains its full weight; the weaker one is discounted.

If IC values are equal (or both missing due to cold-start), apply the penalty to both collectors. This is conservative but fair -- neither has demonstrated superiority.

### Effective Independent Signal Count

Eigenvalue decomposition of the NxN Spearman correlation matrix reveals the true dimensionality of the signal space. Count eigenvalues greater than 0.1 (a threshold that filters out noise dimensions). This count is logged for operational awareness and can be surfaced in the daily plan as context: "Effective independent signals: 11 out of 22."

This metric does not directly affect sizing or conviction -- it is informational. Over time, if independent signal count trends downward, it suggests the collector suite is becoming more homogeneous and new orthogonal signals should be researched.

### Integration with Section 07 (IC Weight Adjustment)

The correlation penalties integrate into the same weight adjustment pipeline from Section 07. The final effective weight per collector becomes:

```
final_weight = static_weight * ic_factor * correlation_penalty
```

In the synthesis code (modified in Section 07 to accept `ic_adjustments`), extend the adjustment dict to include correlation penalties. Two approaches:

1. **Combined dict:** The caller pre-multiplies `ic_factor * correlation_penalty` and passes a single `adjustments: dict[str, float]` to synthesis. Simpler integration.
2. **Separate dicts:** Pass both `ic_adjustments` and `correlation_adjustments` to synthesis, which multiplies them internally. Better for logging and debugging.

Approach 1 (combined dict) is preferred for simplicity. The calling code in the signal engine entry point reads both IC factors (computed daily, cached weekly per Section 07) and correlation penalties (computed weekly by this section), multiplies them per collector, and passes the result.

### DB Storage

New table: `signal_correlation_matrix`

Columns:
- `date DATE` -- the Friday date when correlation was computed
- `collector_a TEXT` -- first collector name
- `collector_b TEXT` -- second collector name
- `correlation FLOAT` -- Spearman correlation value
- `action_taken TEXT` -- "none", "penalty_applied_to_a", "penalty_applied_to_b"
- `penalty_value FLOAT` -- the penalty applied (1.0 if none)
- `created_at TIMESTAMP DEFAULT NOW()`

Primary key: `(date, collector_a, collector_b)`.

Migration: `CREATE TABLE IF NOT EXISTS signal_correlation_matrix (...)` in `db.py`, following the existing pattern.

Additionally, log a summary row or separate metadata:
- Effective independent signal count
- Total collectors analyzed
- Number of pairs penalized

### Cold-Start Behavior

With fewer than 63 trading days of signal data per collector, that collector is excluded from correlation analysis. In practice, this means:

- First 63 trading days after system start: no correlation penalties at all (all penalties = 1.0)
- New collectors added later: excluded from correlation matrix until they accumulate 63 days of data, then included in the next weekly computation
- The config flag `FEEDBACK_CORRELATION_PENALTY=false` provides an additional safety layer during cold-start

### Supervisor Node Wiring

The `run_signal_correlation()` node must be:

1. Registered in the supervisor graph's batch schedule for Friday post-market
2. Given access to the signals table (or ICAttributionTracker) for historical signal values
3. Given access to the latest IC data (for penalty assignment)
4. Able to write to `signal_correlation_matrix` table
5. Able to write to the correlation penalty cache that synthesis reads

The node function signature:

```python
async def run_signal_correlation(state: SupervisorState) -> SupervisorState:
    """Weekly signal correlation analysis and penalty computation.

    Reads 63 trading days of per-collector signal values, computes pairwise
    Spearman correlations, applies penalties to redundant signals, and stores
    results for synthesis consumption.
    """
```

### Logging

The node should log:
- Full correlation matrix summary (pairs with abs(corr) > 0.3, for visibility)
- Penalized collectors with their penalty values
- Effective independent signal count vs total
- Any collectors excluded due to insufficient data

### Rollback

Set `FEEDBACK_CORRELATION_PENALTY=false`. All correlation penalties revert to 1.0. The weekly correlation computation continues running and storing data in `signal_correlation_matrix` (useful for analysis), but no penalties flow into synthesis weights. No data migration or cleanup required.

---

## Files Modified

| File | Change |
|------|--------|
| `src/quantstack/signal_engine/correlation.py` (new) | `compute_signal_correlations()` helper with Spearman matrix, penalty computation, eigenvalue analysis |
| `src/quantstack/graphs/supervisor/nodes.py` | Add `run_signal_correlation()` batch node |
| `src/quantstack/db.py` | Add `signal_correlation_matrix` table creation in schema init |
| Signal engine entry point (e.g., `engine.py` or graph node calling synthesis) | Merge correlation penalties with IC factors before passing to synthesizer |
| `tests/unit/test_signal_correlation.py` (new) | All tests listed above |

---

## Checklist

- [ ] Create `correlation.py` with `compute_signal_correlations()` function
- [ ] Implement pairwise Spearman correlation matrix computation
- [ ] Implement continuous penalty formula with 0.5 threshold and 0.2 floor
- [ ] Implement weaker-signal-gets-penalty logic using IC data
- [ ] Implement effective independent signal count via eigenvalue decomposition
- [ ] Create `signal_correlation_matrix` table in DB schema
- [ ] Add `run_signal_correlation()` supervisor batch node (Friday schedule)
- [ ] Wire config flag `FEEDBACK_CORRELATION_PENALTY` (default false)
- [ ] Integrate correlation penalties with IC factors in synthesis entry point
- [ ] Handle cold-start (< 63 days data -> no penalties)
- [ ] Add logging for correlation summary, penalized collectors, effective signal count
- [ ] Write all tests and verify they pass
