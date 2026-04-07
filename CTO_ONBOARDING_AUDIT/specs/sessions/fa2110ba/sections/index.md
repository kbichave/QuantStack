<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest tests/unit/ -x -q
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-persistence-migration
section-02-ghost-module-audit
section-03-readpoint-wiring
section-04-failure-taxonomy
section-05-loss-aggregation
section-06-eventbus-extension
section-07-ic-weight-adjustment
section-08-signal-correlation
section-09-conflict-resolution
section-10-conviction-multiplicative
section-11-agent-quality
section-12-sharpe-demotion
section-13-drift-detection
section-14-model-versioning
section-15-regime-transitions
section-16-config-flags-integration
END_MANIFEST -->

# Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-persistence-migration | - | 02, 03 | Yes |
| section-02-ghost-module-audit | 01 | 03 | No |
| section-03-readpoint-wiring | 02 | 04, 05, 07, 08, 11 | No |
| section-04-failure-taxonomy | 03 | 05 | No |
| section-05-loss-aggregation | 04 | - | No |
| section-06-eventbus-extension | - | 07, 09, 11, 12 | Yes |
| section-07-ic-weight-adjustment | 03, 06 | 08 | No |
| section-08-signal-correlation | 07 | - | No |
| section-09-conflict-resolution | 06 | - | Yes |
| section-10-conviction-multiplicative | - | - | Yes |
| section-11-agent-quality | 03, 06 | - | Yes |
| section-12-sharpe-demotion | 06 | - | Yes |
| section-13-drift-detection | - | - | Yes |
| section-14-model-versioning | - | - | Yes |
| section-15-regime-transitions | - | - | Yes |
| section-16-config-flags-integration | 07, 08, 10, 12, 13, 15 | - | No |

## Execution Order (Batches)

1. **Batch 1** (no dependencies): section-01-persistence-migration, section-06-eventbus-extension, section-10-conviction-multiplicative, section-13-drift-detection, section-14-model-versioning, section-15-regime-transitions
2. **Batch 2** (after batch 1): section-02-ghost-module-audit, section-09-conflict-resolution, section-12-sharpe-demotion
3. **Batch 3** (after batch 2): section-03-readpoint-wiring, section-11-agent-quality
4. **Batch 4** (after batch 3): section-04-failure-taxonomy, section-07-ic-weight-adjustment
5. **Batch 5** (after batch 4): section-05-loss-aggregation, section-08-signal-correlation
6. **Batch 6** (final): section-16-config-flags-integration

## Section Summaries

### section-01-persistence-migration
Migrate StrategyBreaker from JSON to PostgreSQL (`strategy_breaker_states` table). Migrate ICAttributionTracker from JSON to PostgreSQL (`ic_attribution_data` table). Critical for Docker container safety.

### section-02-ghost-module-audit
Audit and fix OutcomeTracker affinity formula (20-trade halflife exponential decay, 0.15 step), SkillTracker ICIR adjustment, verify ExpectancyEngine deprecation, verify scipy dependency for ICAttribution.

### section-03-readpoint-wiring
Wire all 6 ghost module readpoints: get_regime_strategies(), StrategyBreaker in risk_sizing + execute_entries, SkillTracker in trade hooks, ICAttribution in signal engine, trade quality in daily_plan.

### section-04-failure-taxonomy
FailureMode enum, hybrid rule-based + async LLM classifier, strategy_outcomes.failure_mode column, research queue priority enhancement.

### section-05-loss-aggregation
New supervisor batch node `run_loss_aggregation()` at 16:30 ET. Aggregate by failure mode/strategy/symbol. Top 3 patterns → targeted research tasks. `loss_aggregation` table.

### section-06-eventbus-extension
Add SIGNAL_DEGRADATION, SIGNAL_CONFLICT, AGENT_DEGRADATION event types to EventType enum and EventBus.

### section-07-ic-weight-adjustment
Continuous sigmoid IC factor function. Static weights as priors × IC factor. Weekly weight rebalancing. SIGNAL_DEGRADATION event publishing. Weight floor check.

### section-08-signal-correlation
Weekly `run_signal_correlation()` supervisor batch. Pairwise Spearman matrix. Continuous correlation penalty. Effective independent signal count via eigenvalues.

### section-09-conflict-resolution
Signal conflict detection (max-min spread > 0.5). Conviction cap at 0.3. SIGNAL_CONFLICT event publishing.

### section-10-conviction-multiplicative
Convert 6 additive conviction rules to multiplicative factors. ADX, stability, timeframe, regime agreement, ML confirmation, data quality. Log factors for calibration.

### section-11-agent-quality
Wire SkillTracker for per-agent win rate. AGENT_DEGRADATION event on < 40% win rate. Auto-research task queuing. Daily plan agent confidence context.

### section-12-sharpe-demotion
Rolling 21-day live Sharpe computation. Auto-demotion gate (< 50% of backtest for 21 days). 0.25× sizing via force_scale(). STRATEGY_DEMOTED event.

### section-13-drift-detection
Extend DriftDetector with IC-based drift (daily), label drift via KS test (weekly), interaction drift via adversarial validation (monthly). Auto-retrain decision tree with 20-day cooldown.

### section-14-model-versioning
`model_registry` table, `model_shadow_predictions` table. Champion/challenger workflow. 30-day minimum shadow, 60-day retirement. ML collector loads from registry.

### section-15-regime-transitions
Expose filtered transition probabilities from HMM predict_proba(). Sizing response tiers. Vol-conditioned sub-regimes. Minimum tradeable size floor. Degraded mode when HMM unavailable.

### section-16-config-flags-integration
Wire all 6 kill-switch env vars (FEEDBACK_IC_WEIGHT_ADJUSTMENT, etc.). Cold-start behavior for each feedback loop. Integration test: verify each flag independently disables its adjustment.
