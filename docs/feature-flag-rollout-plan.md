# Feature Flag Rollout Plan

**Created:** 2026-04-08
**Applies to:** P05–P15 implementation (42 new modules, ~25 DB tables, 18 flags)

---

## Why Not Enable Everything at Once

1. **Empty tables.** IC-driven weights, meta-strategy allocation, Sharpe demotion, and regime affinity all query tables that start at 0 rows. They'll produce garbage weights, divide-by-zero, or silently no-op — and you won't know which.

2. **Feedback loops amplify each other.** If IC gate disables a collector AND correlation penalty downweights it AND drift detection triggers a retrain simultaneously, you get cascading signal collapse. Each loop changes state that the next loop reacts to. Observe each in isolation before stacking.

3. **A/B tests need a control group.** Ensemble A/B and prompt A/B require baseline data from the current method. Enable them on day one and there's no baseline to compare against.

4. **Authority gate + meta allocation without calibration.** The authority matrix has hardcoded ceilings (max position size, max daily trades). If meta-strategy allocation pushes capital toward one strategy while authority gate clips it, you get silent position caps with no understanding of why allocation differs from intent. Neither has been calibrated against your actual portfolio size.

5. **Missing deps = silent fallbacks.** Conformal prediction and transformer flags will "work" but route to linear regression / residual fallbacks. You'd think you're running advanced ML but you're actually running toy models.

6. **Multi-asset with no data providers.** `MULTI_ASSET_ENABLED` enables crypto/futures code paths but there are zero data adapters for those asset classes yet. Any non-equity symbol will error immediately.

---

## Rollout Tiers

### Tier 1 — Day 1 (data collection and gating)

Enable these immediately after DB migrations run. They collect data, gate bad signals, and build the foundation that Tier 2 reads from.

```bash
# .env additions
FEEDBACK_IC_GATE=true                    # P01 §1.1 — auto-disable collectors with rolling 63d IC < 0.02
FEEDBACK_SIGNAL_DECAY=true               # P01 §1.3 — exponential decay on stale cached signals
FEEDBACK_CORRELATION_PENALTY=true        # P00 §8  — penalize redundant correlated signals
FEEDBACK_DRIFT_DETECTION=true            # P00 §13 — concept drift detection + auto-retrain trigger
FEEDBACK_RESEARCH_PRIORITY_SCORING=true  # P10 §4  — score-ranked research queue (replaces FIFO)
FEEDBACK_LOOPS_ENABLED=true              # P15 §2  — closed feedback loops (trade→research, IC→weights)
```

**What to monitor:** Langfuse traces for IC gate activations, signal decay curves, drift alerts. Verify `research_candidates` table is populating with scores.

**Exit criteria for Tier 2:** At least 7 days of IC history in `signal_ic_scores`, non-empty `agent_quality_scores`, and at least 50 trade outcomes logged.

---

### Tier 2 — Day 7+ (loops that read accumulated data)

Enable after Tier 1 tables have at least one week of data. These flags consume what Tier 1 collected.

```bash
# .env additions
FEEDBACK_IC_DRIVEN_WEIGHTS=true              # P05 §5.1 — IC-driven regime-conditioned signal weights
FEEDBACK_SIGNAL_CI=true                      # P01 §1.2 — bootstrap confidence intervals on conviction
FEEDBACK_SHARPE_DEMOTION=true                # P00 §12  — demote strategies where live Sharpe diverges from backtest
FEEDBACK_TRANSITION_SIGNAL_DAMPENING=true    # P05 §5.2 — halve signal score during regime transitions
FEEDBACK_TRANSITION_POSITION_SIZING=true     # P05 §4   — halve position size during regime transitions
FEEDBACK_PROMPT_AB_TESTING=true              # P10 §2   — A/B test prompt variants per agent
AUTHORITY_GATE_ENABLED=true                  # P15 §3   — decision ceilings (caps autonomous risk)
```

**What to monitor:**
- Compare signal weights before/after IC-driven mode (query `signal_ic_scores` vs old EWMA blend)
- Check `prompt_ab_results` table is recording variants and outcomes
- Verify authority gate is logging decisions to `authority_decisions`, not silently clipping

**Enable one at a time, not all seven at once.** Suggested order:
1. `FEEDBACK_IC_DRIVEN_WEIGHTS` — highest impact, most data available
2. `AUTHORITY_GATE_ENABLED` — safety cap, low risk
3. `FEEDBACK_SHARPE_DEMOTION` — requires backtest vs live comparison data
4. `FEEDBACK_TRANSITION_SIGNAL_DAMPENING` + `FEEDBACK_TRANSITION_POSITION_SIZING` — pair these
5. `FEEDBACK_SIGNAL_CI` — refinement
6. `FEEDBACK_PROMPT_AB_TESTING` — needs enough agent calls for statistical significance

**Exit criteria for Tier 3:** 2+ weeks of stable operation, no cascading signal collapses, authority gate ceilings calibrated to actual portfolio size.

---

### Tier 3 — Day 21+ (advanced features, optional deps required)

These either require optional Python packages, need extensive calibration, or unlock new asset classes with no existing data pipeline.

```bash
# .env additions
FEEDBACK_REGIME_AFFINITY_SIZING=true         # P00   — scale position by regime affinity score
FEEDBACK_SKILL_CONFIDENCE=true               # P00   — adjust conviction by agent skill tracker
FEEDBACK_META_STRATEGY_ALLOCATION=true       # P10 §3 — meta-model for cross-strategy capital allocation
FEEDBACK_ENSEMBLE_AB_TEST=true               # P05 §5.4 — A/B test ensemble aggregation methods
FEEDBACK_CONFORMAL_PREDICTION=true           # P14 §1 — conformal prediction intervals (requires mapie)
FEEDBACK_TRANSFORMER_FORECAST=true           # P14 §2 — transformer signal collector (requires torch)
MULTI_ASSET_ENABLED=true                     # P12 §1 — crypto/futures/forex trading
```

**Prerequisites before enabling:**

| Flag | Prerequisite |
|------|-------------|
| `FEEDBACK_CONFORMAL_PREDICTION` | `pip install mapie` — without it, falls back to residual method (not useful) |
| `FEEDBACK_TRANSFORMER_FORECAST` | `pip install torch` — without it, falls back to linear regression |
| `MULTI_ASSET_ENABLED` | Data adapters for crypto/futures must be built and tested first |
| `FEEDBACK_META_STRATEGY_ALLOCATION` | Needs 50+ strategy performance records in `research_candidates` |
| `FEEDBACK_ENSEMBLE_AB_TEST` | Needs 2+ weeks of baseline from current `ENSEMBLE_ACTIVE_METHOD` |

**String flag (adjust anytime):**

```bash
ENSEMBLE_ACTIVE_METHOD=weighted_avg   # options: weighted_avg | weighted_median | trimmed_mean
```

Only change this after `FEEDBACK_ENSEMBLE_AB_TEST` has run long enough to show a winner.

---

## Rollout Checklist

```
[ ] Code committed and pushed
[ ] ./start.sh — Docker services running (postgres, langfuse, ollama, graphs)
[ ] DB migrations applied (verify schema_migrations table exists)
[ ] Tier 1 flags added to .env
[ ] Restart graph services to pick up new env vars
[ ] Monitor Langfuse for 24h — no error spikes
[ ] Verify Tier 1 tables populating (signal_ic_scores, research_candidates, agent_quality_scores)
[ ] Day 7: Enable Tier 2 flags one at a time
[ ] Day 7–14: Monitor each Tier 2 flag for 48h before enabling the next
[ ] Day 21: Install optional deps, enable Tier 3 flags
[ ] Calibrate authority gate ceilings to actual portfolio size
[ ] Run full backtest with all flags enabled to validate no regressions
```

---

## Rollback

Any flag can be set back to `false` and services restarted. All flags are safe-off by default. Data collection continues regardless of flag state — disabling a flag stops the feedback loop from acting, not from collecting.

If cascading signal collapse is observed (multiple collectors disabled simultaneously), set all Tier 2 flags to `false`, let Tier 1 stabilize for 48h, then re-enable one at a time.
