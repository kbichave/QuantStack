# P00 Implementation Plan: Wire Learning Modules (Revised)

## 1. Background

QuantStack is an autonomous trading system built on LangGraph with 5 learning modules that track trade outcomes, agent skill, signal predictive power, strategy health, and trade quality. These modules were implemented during a CTO audit.

**Post-review finding:** Codebase verification confirms that **all 6 core wires are already implemented**, not just 4 as initially assessed. The remaining work is narrower than originally scoped:

| Wire | Status | Evidence |
|------|--------|----------|
| 1: OutcomeTracker → strategy selection | COMPLETE | `nodes.py:287-314` (daily_plan affinity context) |
| 2: OutcomeTracker → position sizing | COMPLETE (flag-gated) | `nodes.py:742-746` (affinity × signal_value, floor 0.1) |
| 3a: StrategyBreaker → execute_entries | COMPLETE | `nodes.py:1163-1195` (TRIPPED=reject, SCALED=0.5x) |
| 3b: StrategyBreaker ← on_trade_close | COMPLETE | `trade_hooks.py:226-236` (record_trade called) |
| 4 L1: SkillTracker → sizing | COMPLETE (flag-gated) | `nodes.py:718-752` (skill_adjustments × signal_value) |
| 5a: ICAttribution ← on_trade_close | COMPLETE | `trade_hooks.py:238-260` (per-collector recording) |
| 5b: ICAttribution → synthesis weights | COMPLETE (flag-gated) | `synthesis.py:530-544` (IC weight lookup + fallback) |
| 6 consumer: TradeEvaluator → daily_plan | COMPLETE | `nodes.py:264-285` (quality scores in LLM prompt) |
| 6 producer: TradeEvaluator scoring | **MISSING** | Not called in any hook |
| Auto-trigger: apply_learning | COMPLETE | `trade_hooks.py:429` (in `_on_trade_fill`) |

**Actual remaining work:**
1. Flip 3 feature flag defaults from `false` to `true`
2. Build heuristic trade quality scorer + wire into trade close hook
3. Fix silent exception swallowing in synthesis IC fallback
4. Add observability for feedback loop health
5. Add integration test for end-to-end learning loop
6. Wire 4 Layer 2 (synthesis-level agent confidence) — interview requested both layers; only L1 exists

## 2. Anti-Goals

- **Do NOT re-implement Wires 2, 4 L1, or 5b** — they already exist and work. Verify with tests, don't rewrite.
- **Do NOT modify StrategyBreaker thresholds** — Wire 3 is fully operational with tuned parameters.
- **Do NOT implement P05 adaptive synthesis features** — Ensemble A/B, transition dampening, vol sub-regimes belong to P05.
- **Do NOT remove feature flags** — Flip defaults to `true` but keep flags as kill switches.
- **Do NOT add the collector-to-agent mapping for synthesis-level confidence** — The agent_id mismatch (debate_verdict names vs. clean agent names) makes this fragile. Defer Wire 4 Layer 2 to a follow-up after the agent_id naming is standardized.

## 3. Implementation Sections

### Section 1: Flip Feature Flag Defaults (config/feedback_flags.py)

Change the default return value for three flag functions from `false` to `true`:
- `regime_affinity_sizing_enabled()` → `true`
- `skill_confidence_enabled()` → `true`
- `ic_driven_weights_enabled()` → `true`

**Compound multiplication awareness:** With all three active, a strategy with affinity=0.3 and skill_adj=0.5 gets signal_value × 0.15. The effective floor is 0.05 (affinity floor 0.1 × skill floor 0.5). This is aggressive but acceptable in paper mode. Document the minimum effective multiplier in a code comment at the multiplication site.

**Rollback:** Each flag independently disableable via environment variable.

### Section 2: Heuristic Trade Quality Scorer (performance/trade_evaluator.py)

Add `score_trade_heuristic()` that produces a `TradeQualityScore` without LLM dependency:

```python
def score_trade_heuristic(
    realized_pnl_pct: float,
    hold_days: int,
    position_size_pct: float,
    target_size_pct: float | None = None,
    target_hold_days: int | None = None,
    had_stop_loss: bool = False,
) -> dict:
    """Rule-based trade quality scoring. Fallback when openevals unavailable."""
```

Scoring heuristics (all 0.0-1.0):
- **execution_quality:** 0.7 baseline (no slippage data available in hook context — intentional simplification)
- **thesis_accuracy:** Derived from P&L outcome. This conflates outcome with thesis quality (documented as intentional debt — LLM evaluator provides the nuanced version)
- **risk_management:** 1.0 if loss within normal range, penalize if stop-loss triggered or loss > 3%
- **timing_quality:** Compare hold duration to target (if available), else 0.6 baseline
- **sizing_quality:** Compare actual to target size (if available), else 0.6 baseline
- **overall_score:** Weighted average (thesis 0.30, risk 0.25, execution 0.15, timing 0.15, sizing 0.15)
- **justification:** Auto-generated string describing the P&L-based assessment

**Why no OHLC lookup:** The hook context doesn't have reference prices. Adding a DB query would add latency and failure modes to what should be a fast, reliable fallback. Accept the limitation.

### Section 3: Wire TradeEvaluator into Trade Close Hook (trade_hooks.py)

Add trade quality scoring to `on_trade_close()` after the existing learning module calls:

1. Import `create_trade_evaluator` and `score_trade_heuristic` at module level
2. Try LLM evaluator first (call, not import — import at top of file, catch runtime errors)
3. On any runtime failure (API error, missing key, rate limit), fall back to heuristic
4. Write result to `trade_quality_scores` table
5. Entire block wrapped in try/except with `logger.warning` — scoring failure must never block trade close

**Import approach:** Module-level imports for both evaluator functions. Runtime try/except around the actual `create_trade_evaluator()()` call, not the import. This follows the CLAUDE.md import guidelines.

### Section 4: Fix Silent Exception Swallowing (synthesis.py)

**Current (line 543-544):**
```python
except Exception:
    pass  # fall back to static weights silently
```

**Fix:** Replace bare `except Exception: pass` with logged warning:
```python
except Exception as exc:
    logger.warning("IC-driven weight lookup failed, using static: %s", exc)
```

This applies to the Wire 5b IC weight lookup at `synthesis.py:530-544`. The same pattern should be checked across all learning module integration points.

### Section 5: Observability for Feedback Loops

Add structured logging at each feedback loop activation point so operators can verify loops are functioning:

1. **Wire 2 (nodes.py:742-746):** Log when affinity scaling is applied: `logger.info("wire2_affinity | strategy=%s affinity=%.2f signal_before=%.4f signal_after=%.4f", ...)`
2. **Wire 4 (nodes.py:748-752):** Log skill adjustment: `logger.info("wire4_skill | agent=%s adjustment=%.2f signal_before=%.4f signal_after=%.4f", ...)`
3. **Wire 5b (synthesis.py:530-544):** Already has `logger.debug` — change to `logger.info` on first activation per cycle
4. **Wire 6 producer (new):** Log scoring method and result: `logger.info("wire6_score | trade=%s method=%s overall=%.2f", ...)`

Add a health query helper:

```python
def learning_loop_health() -> dict:
    """Return counts of feedback loop activations in last 24h. Used by supervisor graph."""
```

Query `strategy_outcomes`, `strategy_breaker_states`, `ic_attribution_data`, `agent_skills`, `trade_quality_scores` for recent row counts.

### Section 6: Integration Test

Add `tests/integration/test_learning_loop_e2e.py`:

Test the full path: simulate trade close → verify all hooks fire → verify learning modules update → verify next cycle uses updated values.

Key assertions:
- After trade close, `strategies.regime_affinity` has been updated (apply_learning fired)
- After trade close, `trade_quality_scores` has a new row (evaluator fired)
- After trade close with loss, `strategy_breaker_states` reflects updated drawdown
- After trade close, `ic_attribution_data` has new rows per collector
- On next risk_sizing call, affinity and skill adjustments are applied to signal values

**Concurrency note:** Test concurrent trade closes for the same strategy to verify `apply_learning` doesn't lose updates. If it does, add `SELECT FOR UPDATE` in `OutcomeTracker.apply_learning()`.

### Section 7: Tests

| Test | Section | What it verifies |
|------|---------|-----------------|
| `test_flags_default_true` | 1 | All 3 feedback flags return true with no env vars set |
| `test_flags_env_override` | 1 | Setting env var to 'false' disables the flag |
| `test_evaluator_heuristic_profitable` | 2 | Profitable trade → thesis_accuracy > 0.7 |
| `test_evaluator_heuristic_loss` | 2 | Loss trade → thesis_accuracy < 0.5 |
| `test_evaluator_heuristic_stop_loss` | 2 | Stop-loss trade → risk_management < 0.5 |
| `test_evaluator_called_on_close` | 3 | on_trade_close writes trade_quality_scores row |
| `test_evaluator_openevals_fallback` | 3 | openevals runtime failure → heuristic used, row still written |
| `test_evaluator_failure_nonfatal` | 3 | Both evaluators fail → trade close still succeeds |
| `test_ic_fallback_logged` | 4 | IC weight failure produces warning log, not silent pass |
| `test_loop_health_query` | 5 | learning_loop_health returns recent activation counts |
| `test_e2e_learning_loop` | 6 | Full path: close → hooks → update → next cycle sees changes |
| `test_concurrent_apply_learning` | 6 | Two concurrent closes for same strategy → both updates preserved |

## 4. Dependency Order

```
Section 1 (flags) ───────────────────────────────┐
Section 2 (heuristic scorer) → Section 3 (hook)──┤
Section 4 (fix silent catch) ─────────────────────┤──→ Section 7 (tests)
Section 5 (observability) ────────────────────────┤
Section 6 (integration test) ─────────────────────┘
```

Sections 1, 2, 4, 5 are independent. Section 3 depends on Section 2. Sections 6 and 7 depend on all others.

## 5. Deferred Work

- **Wire 4 Layer 2** (synthesis-level agent confidence): Deferred until agent_id naming is standardized. Currently, SkillTracker data uses debate_verdict names while the proposed mapping uses clean names like "ml_scientist". This mismatch would produce no-data lookups for most agents.
- **Staged flag rollout monitoring:** The plan enables all flags at once (user decision for paper mode). If the system moves to live trading, add a staged rollout protocol with per-flag Sharpe/drawdown monitoring.

## 6. Rollback Plan

| Component | Rollback |
|-----------|----------|
| Wire 2 | `FEEDBACK_REGIME_AFFINITY_SIZING=false` |
| Wire 4 | `FEEDBACK_SKILL_CONFIDENCE=false` |
| Wire 5b | `FEEDBACK_IC_DRIVEN_WEIGHTS=false` |
| Wire 6 producer | Evaluator failure is non-fatal (try/except) — no action needed |
| Observability | Logging only — zero runtime impact |

No data migrations. No schema changes. No API changes.
