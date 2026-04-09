# P10 TDD Plan: Meta-Learning & Self-Improvement

## 1. Agent Decision Quality Tracking

```python
# tests/unit/learning/test_agent_quality.py

class TestAgentQualityScoring:
    def test_correct_direction_scored_1(self):
        """Agent recommends BUY, price goes up -> direction score = 1."""

    def test_incorrect_direction_scored_0(self):
        """Agent recommends BUY, price goes down -> direction score = 0."""

    def test_magnitude_accuracy_scaled_0_to_1(self):
        """Magnitude accuracy is 0-1 based on predicted vs actual move size."""

    def test_timing_classification(self):
        """Timing scored as early/on-time/late based on entry vs optimal."""

    def test_composite_quality_score(self):
        """Overall quality score combines direction, magnitude, timing."""


class TestAgentQualityAggregation:
    def test_rolling_21d_win_rate(self):
        """Win rate computed over rolling 21-day window per agent."""

    def test_alert_on_sustained_low_win_rate(self):
        """Alert triggered when win rate < 40% for 5 consecutive cycles."""

    def test_no_alert_on_brief_dip(self):
        """Win rate dip for fewer than 5 cycles does not trigger alert."""


class TestAgentQualityDashboard:
    def test_dashboard_returns_per_agent_metrics(self):
        """get_agent_quality_dashboard returns win rate and trend per agent."""

    def test_dashboard_identifies_best_worst(self):
        """Dashboard highlights best and worst performing agents."""


class TestAgentQualityEdgeCases:
    def test_no_outcomes_yet(self):
        """Agent with zero recorded outcomes returns empty metrics, not error."""

    def test_flat_price_outcome(self):
        """Zero price change -> direction is neither correct nor incorrect; handled gracefully."""
```

## 2. Prompt A/B Testing

```python
# tests/unit/learning/test_prompt_ab_testing.py

class TestVariantManagement:
    def test_create_variant(self):
        """New prompt variant stored with agent_name, variant_id, status='active'."""

    def test_max_variants_per_agent(self):
        """Cannot create more than 3 active variants per agent."""

    def test_variant_status_lifecycle(self):
        """Variant transitions: active -> evaluating -> promoted/rejected."""


class TestShadowExecution:
    def test_shadow_runs_both_production_and_variant(self):
        """Shadow mode executes production prompt and variant, records both outputs."""

    def test_shadow_output_does_not_affect_live(self):
        """Variant output is recorded but not used for live decisions."""

    def test_quality_scores_recorded_for_both(self):
        """Both production and variant outputs get quality scores."""


class TestPromotion:
    def test_promote_on_statistical_significance(self):
        """Variant promoted when quality improvement has p < 0.05."""

    def test_no_promote_before_minimum_period(self):
        """Variant cannot be promoted with fewer than 2 weeks of data."""

    def test_rejected_variant_deactivated(self):
        """Variant that fails significance test is set to 'rejected' status."""
```

## 3. Strategy-of-Strategies Meta-Allocator

```python
# tests/unit/learning/test_meta_allocator.py

class TestMetaAllocator:
    def test_weights_sum_to_one(self):
        """Meta-allocator output weights for all strategies sum to 1.0."""

    def test_weights_non_negative(self):
        """No strategy receives negative weight allocation."""

    def test_regime_influences_allocation(self):
        """Different regime inputs produce different weight distributions."""

    def test_high_ic_strategy_gets_more_weight(self):
        """Strategy with higher rolling IC receives larger allocation."""

    def test_retrain_updates_model(self):
        """Monthly retrain on realized performance changes model coefficients."""


class TestMetaAllocatorEdgeCases:
    def test_single_active_strategy(self):
        """With only one strategy, it receives 100% weight."""

    def test_all_strategies_negative_ic(self):
        """All negative IC -> allocator still produces valid weights (possibly equal-weight fallback)."""

    def test_missing_ic_data_for_strategy(self):
        """Strategy with no IC data falls back to default weight."""
```

## 4. Research Prioritization

```python
# tests/unit/learning/test_research_priority.py

class TestPriorityScoring:
    def test_higher_alpha_uplift_ranks_higher(self):
        """Research item with higher expected alpha uplift scores above others."""

    def test_portfolio_gap_boosts_priority(self):
        """Underexplored asset/strategy gets portfolio_gap=1.0 boost."""

    def test_failure_frequency_boosts_priority(self):
        """High failure count for a strategy type increases priority."""

    def test_staleness_decay_function(self):
        """Items not investigated recently get increasing priority over time."""

    def test_combined_score_is_weighted_sum(self):
        """Final priority = weighted combination of all factors."""


class TestPriorityQueue:
    def test_queue_ordered_by_priority(self):
        """Research queue returns items in descending priority order."""

    def test_new_item_inserted_at_correct_position(self):
        """Newly queued item lands at position matching its priority score."""

    def test_empty_queue_returns_none(self):
        """Empty priority queue returns None, not error."""
```

## 5. Few-Shot Example Library

```python
# tests/unit/learning/test_few_shot_library.py

class TestFewShotCuration:
    def test_auto_extract_top_10_pct(self):
        """Examples auto-extracted from top 10% quality-scored outputs."""

    def test_max_3_examples_per_context(self):
        """No more than 3 examples injected per agent per context type."""

    def test_manual_curation_flag(self):
        """Gold-standard examples marked manually are always preferred."""


class TestFewShotInjection:
    def test_examples_injected_into_prompt(self):
        """Retrieved examples appear in agent prompt before main instruction."""

    def test_context_matching(self):
        """Examples selected based on current regime and strategy type match."""

    def test_no_examples_available_gracefully(self):
        """When no matching examples exist, prompt built without examples (no error)."""

    def test_example_quality_threshold(self):
        """Only examples above minimum quality score are candidates for injection."""
```
