# Section 06: Synthesis Integration

## Objective

Wire all four alt-data collectors into the SignalEngine's collector orchestration and the RuleBasedSynthesizer's weight profiles. Alt-data signals start at 0.05 (5%) total weight, distributed across whichever collectors return data. IC tracking is automatic via existing infrastructure.

## Dependencies

Requires section-05 (data providers wired up) so collectors can fetch data through the provider framework.

## Files to Modify

### `src/quantstack/signal_engine/engine.py`

1. **Add imports** for the four new collectors at the top of the file:
   ```python
   from quantstack.signal_engine.collectors.congressional import collect_congressional
   from quantstack.signal_engine.collectors.web_traffic import collect_web_traffic
   from quantstack.signal_engine.collectors.job_postings import collect_job_postings
   from quantstack.signal_engine.collectors.patents import collect_patents
   ```

2. **Register in `_run_collectors`** — Add to the `collector_map` dict under a new "Phase 8" comment block:
   ```python
   # Phase 8 collectors (P11) — alternative data signals
   "congressional": collect_congressional(symbol, self._store),
   "web_traffic": collect_web_traffic(symbol, self._store),
   "job_postings": collect_job_postings(symbol, self._store),
   "patents": collect_patents(symbol, self._store),
   ```

3. **Update `_build_brief`** — Extract alt-data outputs and pass to synthesizer:
   ```python
   alt_data = {
       "congressional": outputs.get("congressional", {}),
       "web_traffic": outputs.get("web_traffic", {}),
       "job_postings": outputs.get("job_postings", {}),
       "patents": outputs.get("patents", {}),
   }
   ```
   Pass `alt_data` to synthesizer so it can incorporate the signals.

### `src/quantstack/signal_engine/synthesis.py`

1. **Add `"alt_data"` key to all weight profiles** in `_WEIGHT_PROFILES`. Initial weight = 0.05 across all regimes. Steal proportionally from existing weights (reduce each by ~0.7% to free up 5%). Example for `trending_up`:
   ```python
   "trending_up": {
       "trend": 0.33, "rsi": 0.09, "macd": 0.19, "bb": 0.05,
       "sentiment": 0.10, "ml": 0.14, "flow": 0.05, "alt_data": 0.05,
   },
   ```
   Apply same pattern to all 10 profiles (6 base + 4 vol-conditioned). Weights must sum to 1.0.

2. **Add `_DEFAULT_WEIGHTS["alt_data"] = 0.05`** and adjust others proportionally.

3. **Alt-data score aggregation** — In the synthesis logic, compute a single `alt_data_score` from whichever collectors returned data:
   ```python
   def _aggregate_alt_data(self, alt_data: dict) -> float:
       scores = []
       weights = []
       for key, data in alt_data.items():
           score_key = f"{key}_signal_score"  # e.g., "congress_signal_score"
           if score_key in data:
               scores.append(data[score_key])
               weights.append(data.get("confidence", 0.3))
       if not scores:
           return 0.0  # No alt-data available → neutral
       total_w = sum(weights)
       return sum(s * w for s, w in zip(scores, weights)) / total_w
   ```
   This weighted average becomes the `alt_data` voter in synthesis.

4. **Redistribution when no alt-data** — If `_aggregate_alt_data` returns 0.0 (no collectors returned data), redistribute the 0.05 weight proportionally to other voters (same pattern used for ML weight redistribution).

### `src/quantstack/signal_engine/brief.py` (if exists) or `src/quantstack/shared/schemas.py`

Add optional alt-data fields to `SymbolBrief` or `SignalBrief`:
- `alt_data_score: float | None` — aggregated alt-data signal.
- `alt_data_sources: list[str]` — which collectors contributed (e.g., `["congressional", "patents"]`).

## Test Requirements

### `tests/unit/signal_engine/test_synthesis_alt_data.py`

1. **test_weight_profiles_sum_to_one** — All profiles in `_WEIGHT_PROFILES` sum to 1.0 (within float tolerance).
2. **test_alt_data_aggregation_single_source** — Only congressional returns data → uses that score.
3. **test_alt_data_aggregation_multiple_sources** — Congressional + patents → weighted average by confidence.
4. **test_alt_data_aggregation_empty** — No alt-data → returns 0.0.
5. **test_alt_data_weight_redistribution** — No alt-data → 0.05 weight redistributed to others.
6. **test_synthesis_with_alt_data** — Full synthesis run with mocked alt-data → alt_data_score appears in brief.
7. **test_synthesis_without_alt_data** — Full synthesis run with empty alt-data → brief still valid, no alt_data_score.
8. **test_collector_registration** — All four new collectors present in `SignalEngine._run_collectors` collector_map.

## Acceptance Criteria

- [ ] All weight profiles sum to 1.0 after alt_data addition.
- [ ] Alt-data weight starts at 0.05 (5%) — never more without IC evidence.
- [ ] When no alt-data collectors return data, weight is redistributed (not wasted).
- [ ] Alt-data aggregation handles 1, 2, 3, or 4 sources gracefully.
- [ ] SignalBrief/SymbolBrief schema updated with optional alt-data fields.
- [ ] All four collectors registered in engine's `_run_collectors`.
- [ ] All 8 unit tests pass.
- [ ] Existing synthesis tests still pass (no regression from weight changes).
