# Section 03: Job Postings Collector

## Objective

Build a signal collector that ingests job posting data and produces a directional signal based on hiring trends. Hiring surges (>20% YoY) indicate growth investment; layoff signals (posting drops + role type shifts) indicate contraction. Expected IC 0.02-0.04 with a 3-6 month lead time — this is a slow, confirmatory signal.

## Files to Create

### `src/quantstack/signal_engine/collectors/job_postings.py`

New collector module following the `async collect_*(symbol, store) -> dict` pattern.

**Implementation details:**

1. **Data source** — Thinknum API (primary) or free alternative (Glassdoor API). Requires `THINKNUM_API_KEY` env var. If no key, collector returns `{}` gracefully (this is a MEDIUM priority, optional source).

2. **Role classification** — `classify_role(title: str) -> str`: maps job titles to categories:
   - `"engineering"`: software engineer, data scientist, ML engineer, SRE, etc.
   - `"sales"`: account executive, sales rep, business development, etc.
   - `"operations"`: supply chain, logistics, warehouse, etc.
   - `"executive"`: VP, Director, C-suite titles.
   - `"other"`: unclassified.
   
   Uses keyword matching (not ML) — fast and deterministic.

3. **Signal computation** — `compute_job_postings_signal(current_count: int, historical_counts: list[tuple[str, int]], role_breakdown: dict[str, int]) -> dict[str, Any]`:
   - `yoy_growth`: (current_count - count_12m_ago) / count_12m_ago. Capped at [-1.0, 5.0] to avoid division artifacts.
   - `hiring_surge`: yoy_growth > 0.20.
   - `layoff_signal`: yoy_growth < -0.15 AND engineering postings declining.
   - `engineering_ratio`: engineering postings / total. Rising ratio → R&D investment (bullish for tech).
   - `sales_expansion`: sales postings growing faster than total → revenue push.
   - `job_postings_signal_score` in [-1.0, 1.0]:
     - Base: `tanh(yoy_growth) * 0.5`
     - Engineering boost: `+0.2` if engineering_ratio > 0.4 and growing.
     - Sales expansion boost: `+0.15` if sales growth > total growth.
     - Layoff penalty: `-0.3` if layoff_signal detected.
   - Confidence: 0.4 (slow signal, low IC — never high confidence).
   - Return dict: `job_postings_signal_score`, `yoy_growth`, `hiring_surge`, `layoff_signal`, `engineering_ratio`, `role_breakdown`, `confidence`.

4. **Entry point** — `async def collect_job_postings(symbol: str, store: DataStore) -> dict[str, Any]`:
   - Check freshness: `check_freshness(symbol, "job_postings", max_days=10)` (weekly updates + buffer).
   - Fetch current + 12 months historical from API.
   - On failure, return `{}`.

## Files to Modify

### `src/quantstack/signal_engine/staleness.py`

Add entry to `STALENESS_THRESHOLDS`:
```python
"job_postings": 10,
```

## Test Requirements

### `tests/unit/signal_engine/test_job_postings_collector.py`

1. **test_compute_signal_hiring_surge** — 40% YoY growth → positive score, `hiring_surge=True`.
2. **test_compute_signal_layoff** — -25% YoY with engineering decline → negative score, `layoff_signal=True`.
3. **test_compute_signal_flat** — 5% growth → small positive, no surge/layoff flags.
4. **test_engineering_ratio_boost** — High engineering ratio + growth → score boosted.
5. **test_sales_expansion_boost** — Sales growing faster than total → score boosted.
6. **test_classify_role_engineering** — "Senior Software Engineer" → "engineering".
7. **test_classify_role_sales** — "Account Executive - Enterprise" → "sales".
8. **test_classify_role_other** — "Office Manager" → "other".
9. **test_yoy_growth_cap** — Division by near-zero historical count → capped, not infinity.
10. **test_api_failure** — Mock httpx timeout → returns `{}`.
11. **test_missing_api_key** — No `THINKNUM_API_KEY` → returns `{}`.

## Acceptance Criteria

- [ ] `collect_job_postings(symbol, store)` returns signal dict or `{}` on failure.
- [ ] Role classification is deterministic keyword matching, not ML.
- [ ] YoY growth is capped to prevent division artifacts.
- [ ] Confidence never exceeds 0.4 (this is a slow, confirmatory signal).
- [ ] Signal computation is pure and independently testable.
- [ ] All 11 unit tests pass.
- [ ] Staleness threshold registered at 10 days.
