# P00 TDD Plan: Wire Learning Modules

**Testing framework:** pytest (existing codebase)
**Test locations:** `tests/unit/`, `tests/integration/`
**Existing infrastructure:** `tests/unit/test_learning_wiring.py` (781 lines of existing fixtures and test patterns)
**Fixtures:** DB mocking via `monkeypatch`, `MagicMock` for learning module instances

---

## Section 1: Flip Feature Flag Defaults

```python
# Test: all 3 feedback flags return True with no environment variables set
# Test: setting FEEDBACK_REGIME_AFFINITY_SIZING=false overrides default to False
# Test: setting FEEDBACK_SKILL_CONFIDENCE=false overrides default to False
# Test: setting FEEDBACK_IC_DRIVEN_WEIGHTS=false overrides default to False
# Test: empty string env var ('') treated as false (not truthy)
```

**File:** `tests/unit/test_feedback_flags.py` (extend existing TestFeatureFlags class in test_learning_wiring.py, or new file)

---

## Section 2: Heuristic Trade Quality Scorer

```python
# Test: profitable trade (pnl > 0) → thesis_accuracy > 0.7
# Test: small loss trade (-1% to 0%) → thesis_accuracy between 0.3 and 0.7
# Test: large loss trade (< -3%) → thesis_accuracy ≤ 0.3
# Test: stop-loss triggered → risk_management < 0.5
# Test: normal loss (no stop) → risk_management > 0.7
# Test: hold_days matching target → timing_quality > 0.8
# Test: hold_days far from target → timing_quality < 0.5
# Test: no target provided → timing_quality defaults to 0.6
# Test: position_size within 20% of target → sizing_quality > 0.8
# Test: no target provided → sizing_quality defaults to 0.6
# Test: output conforms to TradeQualityScore schema (all 6 dimensions + justification)
# Test: all scores are in [0.0, 1.0] range
# Test: overall_score is weighted average of other 5 dimensions
```

**File:** `tests/unit/test_trade_evaluator_heuristic.py`

---

## Section 3: Wire TradeEvaluator into Trade Close Hook

```python
# Test: on_trade_close writes a trade_quality_scores row
# Test: when openevals evaluator raises RuntimeError, heuristic fallback is used and row still written
# Test: when both evaluators fail, trade close processing completes without error
# Test: trade_quality_scores row has correct trade_id foreign key
# Test: scored_at timestamp is within 1 second of call time
# Test: model_used field reflects 'heuristic' when fallback used
```

**File:** `tests/unit/test_trade_hooks_evaluator.py` (or extend existing hook tests)

---

## Section 4: Fix Silent Exception Swallowing

```python
# Test: IC weight lookup failure produces logger.warning with exception message
# Test: after IC failure, static weights are still returned (not None/empty)
# Test: synthesis output is identical whether IC fails or IC returns None (both fallback to static)
```

**File:** `tests/unit/test_synthesis_ic_fallback.py` (or extend existing synthesis tests)

---

## Section 5: Observability for Feedback Loops

```python
# Test: wire2 activation produces structured log with strategy, affinity, before/after signal values
# Test: wire4 activation produces structured log with agent_id, adjustment, before/after signal values
# Test: wire6 scoring produces structured log with trade_id, method, overall_score
# Test: learning_loop_health() returns dict with keys for each loop's 24h count
# Test: learning_loop_health() returns 0 counts when tables are empty
# Test: learning_loop_health() handles DB failure gracefully (returns error dict, not exception)
```

**File:** `tests/unit/test_learning_observability.py`

---

## Section 6: Integration Test

```python
# Test: full trade close → all hooks fire → learning modules update → next cycle uses updated values
#   - Simulate trade entry then close
#   - Verify strategies.regime_affinity updated (apply_learning fired)
#   - Verify trade_quality_scores has new row
#   - Verify strategy_breaker_states reflects updated drawdown (for loss case)
#   - Verify ic_attribution_data has new rows per collector
#   - On simulated next risk_sizing call, affinity and skill adjustments are applied

# Test: concurrent trade closes for same strategy don't lose updates
#   - Close 2 trades simultaneously for same strategy_id
#   - Both should be reflected in final regime_affinity value
#   - If race condition detected, document need for SELECT FOR UPDATE
```

**File:** `tests/integration/test_learning_loop_e2e.py`
**Note:** Integration tests may need a real PostgreSQL test instance or transaction-based isolation.

---

## Test Execution Order

For TDD, write and verify tests in this order:

1. **Section 1 tests** — Verify flag defaults (fast, no dependencies)
2. **Section 2 tests** — Verify heuristic scorer outputs (pure function, no DB)
3. **Section 4 tests** — Verify logging fix (mock-based, fast)
4. **Section 5 tests** — Verify observability functions (requires DB mocks)
5. **Section 3 tests** — Verify hook wiring (requires Section 2 implementation)
6. **Section 6 tests** — Verify end-to-end (requires all other sections)
