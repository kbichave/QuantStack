# P01: Signal Statistical Rigor

**Objective:** Prove that signals predict returns. Implement IC tracking, confidence intervals, decay modeling, and correlation analysis so the system knows which signals work.

**Scope:** signal_engine/, learning/, tools/langchain/qc_research_tools.py

**Depends on:** None

**Enables:** P05 (Adaptive Synthesis), P10 (Meta-Learning)

**Effort estimate:** 1-2 weeks

---

## What Changes

The Quant Scientist audit's most damning finding: "No signal has ever been validated against forward returns" (QS-S1). This is the single most important phase — everything downstream (position sizing, strategy selection, execution) depends on knowing which signals work.

### 1.1 IC Computation & Tracking (QS-S1)

**What:** Daily rank correlation of each signal value vs 1/5/21-day forward returns.

**Implementation:**
```python
# New: src/quantstack/learning/ic_tracker.py
class ICTracker:
    def compute_daily_ic(self, signal_name: str, symbol: str) -> ICResult:
        """Rank correlation of signal[-1] vs forward_return[1d/5d/21d]."""
        # Spearman rank correlation
        # Store in signal_ic table (already exists, currently empty)
    
    def get_rolling_ic(self, signal_name: str, window: int = 63) -> float:
        """63-day rolling average IC."""
    
    def get_ic_stability(self, signal_name: str) -> float:
        """Std dev of daily IC — lower is better."""
    
    def get_ic_by_regime(self, signal_name: str) -> dict[str, float]:
        """IC breakdown by regime type."""
```

**Gate:** If rolling_63d_IC < 0.02 for any collector → disable from synthesis. Alert if IC negative for >5 consecutive days.

**Files:**
- New: `src/quantstack/learning/ic_tracker.py`
- `src/quantstack/tools/langchain/qc_research_tools.py` — implement `compute_information_coefficient()` (currently stubbed)
- `src/quantstack/signal_engine/engine.py` — add IC computation after synthesis
- `scripts/scheduler.py` — add daily IC computation job

### 1.2 Signal Confidence Intervals (QS-S2)

**What:** Bootstrap confidence intervals on conviction scores.

**Implementation:**
- Compute bootstrap CI from collector agreement distribution
- High agreement (8/10 collectors bullish) → narrow interval [0.70, 0.80]
- Low agreement (5 bullish, 5 bearish) → wide interval [0.20, 0.80]
- Add `uncertainty_estimate` field to `SignalBrief`
- Propagate to position sizing: `size = base_size * (1 - confidence_width)`

**Files:**
- `src/quantstack/signal_engine/brief.py` — add `uncertainty_estimate` field
- `src/quantstack/signal_engine/synthesis.py` — compute bootstrap CI
- `src/quantstack/execution/risk_gate.py` — use uncertainty in sizing

### 1.3 Signal Decay Modeling (QS-S3)

**What:** Exponential decay on cached signals based on per-collector half-life.

**Implementation:**
```python
# Per-collector half-lives (calibrate from IC decay curves)
HALF_LIVES = {
    "technical": 15,      # minutes — fast-moving
    "momentum": 30,       # minutes
    "volume": 20,         # minutes
    "sentiment": 120,     # minutes — news decays slower
    "fundamentals": 1440, # minutes (1 day) — slow-moving
    "macro": 10080,       # minutes (7 days) — very slow
    "ml_signal": 60,      # minutes — model predictions
}

effective_conviction = conviction * exp(-age_minutes / half_life)
```

**Files:**
- `src/quantstack/signal_engine/cache.py` — add decay computation on cache read
- `src/quantstack/signal_engine/engine.py` — store signal timestamps per collector

### 1.4 Signal Correlation Analysis (QS-S5)

**What:** Weekly pairwise correlation matrix between all signals. Identify redundant signals.

**Implementation:**
- Compute 63-day rolling pairwise correlation matrix
- If `corr(signal_A, signal_B) > 0.7`: halve weight of weaker IC signal
- Report effective signal count = eigenvalues > 0.1 of correlation matrix
- Expected: 27 raw collectors → ~12-15 effective independent signals

**Files:**
- New: `src/quantstack/learning/signal_correlation.py`
- `src/quantstack/signal_engine/synthesis.py` — apply correlation-adjusted weights
- `scripts/scheduler.py` — add weekly correlation computation job

### 1.5 Conflict Detection (QS-S9)

**What:** When signals disagree strongly, reduce exposure instead of blending.

**Implementation:**
- `if max_signal - min_signal > 0.5: flag as "conflicting"`
- When conflicting: cap conviction at 0.3 regardless of weighted average
- Log conflict frequency per symbol per day

**Files:**
- `src/quantstack/signal_engine/synthesis.py` — add conflict detection post-synthesis

## Tests

| Test | What It Verifies |
|------|-----------------|
| `test_ic_computation` | Spearman rank correlation computed correctly |
| `test_ic_gate_disables_signal` | Collector with IC < 0.02 removed from synthesis |
| `test_confidence_interval_width` | High agreement → narrow CI, low agreement → wide CI |
| `test_signal_decay` | 59-min-old signal has lower effective conviction than 1-min-old |
| `test_correlation_weight_adjustment` | Highly correlated signals get halved weights |
| `test_conflict_detection` | Conflicting signals cap conviction at 0.3 |

## Acceptance Criteria

1. `signal_ic` table populated with daily IC for all 16 active collectors
2. Signals with IC < 0.02 for 21+ days auto-disabled in synthesis
3. `SignalBrief.uncertainty_estimate` populated for every brief
4. Cache reads apply exponential decay based on signal age
5. Weekly correlation report generated, redundant signals weight-adjusted

## Risk

| Risk | Severity | Mitigation |
|------|----------|-----------|
| IC tracking requires sufficient trade history | High | Start with paper trades; need 60+ days of data per symbol |
| Decay half-lives miscalibrated | Medium | Start conservative (shorter half-lives), calibrate quarterly |
| Correlation matrix unstable with few symbols | Medium | Require 30+ data points per pair before adjusting weights |

## References

- CTO Audit: QS-S1 through QS-S9
- CTO Audit: Loop 3 (IC Degradation → Weight Adjustment)
