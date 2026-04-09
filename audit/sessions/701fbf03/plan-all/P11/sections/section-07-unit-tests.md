# Section 07: Unit Tests (Cross-Cutting)

## Objective

Ensure all four collectors handle edge cases, API failures, and graceful degradation correctly. This section covers integration-level tests that span multiple collectors and verify the overall alt-data subsystem works end-to-end — complementing the per-collector unit tests defined in sections 01-04.

## Dependencies

Requires section-01 through section-04 (all four collectors must exist).

## Files to Create

### `tests/unit/signal_engine/test_alt_data_integration.py`

Cross-cutting tests that verify the alt-data subsystem as a whole.

**Test cases:**

1. **test_all_collectors_return_empty_on_failure** — Mock all four APIs to raise `httpx.TimeoutException`. Verify all return `{}`, no exceptions propagate to caller.

2. **test_all_collectors_return_empty_on_missing_keys** — Unset all API key env vars. Verify all collectors return `{}` with appropriate warning logs (not errors).

3. **test_collector_contract_return_type** — Each collector returns either `dict` or empty `dict`. Never `None`, never raises. Verify by calling each with a mock store and a random symbol.

4. **test_collector_contract_signal_score_range** — When collectors return data, the `*_signal_score` value is always in [-1.0, 1.0]. Test with extreme inputs (massive buy counts, zero traffic, 10000 patents).

5. **test_collector_contract_confidence_range** — When collectors return data, `confidence` is always in [0.0, 1.0].

6. **test_staleness_thresholds_registered** — Verify that `STALENESS_THRESHOLDS` in `staleness.py` contains entries for all four alt-data sources: `congressional_trades`, `web_traffic`, `job_postings`, `patent_filings`.

7. **test_concurrent_collector_isolation** — Run all four collectors concurrently via `asyncio.gather`. One raises an exception mid-flight. Verify the other three still complete and return valid results.

8. **test_signal_score_deterministic** — Same input → same output. Call each pure computation function twice with identical inputs, verify identical outputs (no random state, no time-dependency in the signal math).

9. **test_api_response_malformed_json** — Mock each API to return HTTP 200 with invalid JSON body. Verify collectors catch the parse error and return `{}`.

10. **test_api_response_empty_results** — Mock each API to return HTTP 200 with valid JSON but empty results array. Verify collectors return `{}` or a zero-score dict (not crash).

11. **test_api_response_partial_fields** — Mock each API to return data with some expected fields missing. Verify collectors handle gracefully (use defaults or return `{}`).

12. **test_collector_timeout_respected** — Mock each API with a 30-second sleep. Verify the collector's internal timeout fires and returns `{}` within the expected window (8-10 seconds).

### `tests/unit/signal_engine/conftest.py` (update if exists, create if not)

Add shared fixtures:

1. **`mock_store`** — A minimal `DataStore` mock that returns empty DataFrames and basic company overview data.
2. **`mock_httpx_client`** — Patchable httpx async client for API mocking.
3. **`sample_congressional_response`** — Realistic Quiver API response fixture (5 transactions, mixed buy/sell).
4. **`sample_similarweb_response`** — Realistic SimilarWeb response fixture (12 months of traffic data).
5. **`sample_thinknum_response`** — Realistic Thinknum response fixture (50 job postings across roles).
6. **`sample_patentsview_response`** — Realistic PatentsView response fixture (20 patents with CPC codes).

## Test Requirements

All 12 integration tests must pass. Fixtures must be realistic enough to exercise the full computation paths but deterministic (no randomness, no real API calls).

## Acceptance Criteria

- [ ] All 12 integration tests pass.
- [ ] No test makes real HTTP calls — all APIs fully mocked.
- [ ] Tests verify the collector contract: returns `dict`, scores in [-1, 1], confidence in [0, 1].
- [ ] Tests verify graceful degradation: timeouts, missing keys, malformed responses, empty results.
- [ ] Tests verify isolation: one collector failure does not affect others.
- [ ] Tests verify determinism: same input → same output.
- [ ] Shared fixtures provide realistic API response samples for all four sources.
- [ ] `conftest.py` fixtures are reusable by per-collector test files (sections 01-04).
