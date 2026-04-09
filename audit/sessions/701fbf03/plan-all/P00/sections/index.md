<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-flag-defaults
section-02-heuristic-scorer
section-03-hook-wiring  depends_on:section-02-heuristic-scorer
section-04-fix-silent-catch
section-05-observability
section-06-integration-test  depends_on:section-01-flag-defaults,section-02-heuristic-scorer,section-03-hook-wiring,section-04-fix-silent-catch,section-05-observability
section-07-unit-tests  depends_on:section-01-flag-defaults,section-02-heuristic-scorer,section-03-hook-wiring,section-04-fix-silent-catch,section-05-observability
END_MANIFEST -->

# P00 Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-flag-defaults | - | 06, 07 | Yes |
| section-02-heuristic-scorer | - | 03, 06, 07 | Yes |
| section-03-hook-wiring | 02 | 06, 07 | No |
| section-04-fix-silent-catch | - | 06, 07 | Yes |
| section-05-observability | - | 06, 07 | Yes |
| section-06-integration-test | 01-05 | - | No |
| section-07-unit-tests | 01-05 | - | No |

## Execution Order

1. section-01-flag-defaults, section-02-heuristic-scorer, section-04-fix-silent-catch, section-05-observability (parallel, no dependencies)
2. section-03-hook-wiring (after 02)
3. section-06-integration-test, section-07-unit-tests (parallel, after all implementation sections)

## Section Summaries

### section-01-flag-defaults
Flip 3 feature flag defaults from false to true in feedback_flags.py. Add compound multiplication comment.

### section-02-heuristic-scorer
Add score_trade_heuristic() to trade_evaluator.py. Rule-based TradeQualityScore without LLM dependency.

### section-03-hook-wiring
Wire trade evaluator (LLM + heuristic fallback) into on_trade_close hook. Module-level imports, runtime try/except.

### section-04-fix-silent-catch
Replace except Exception: pass with logged warning in synthesis.py IC weight fallback.

### section-05-observability
Add structured logging at Wire 2/4/5b/6 activation points. Add learning_loop_health() query helper.

### section-06-integration-test
End-to-end test: trade close → hooks → learning updates → next cycle sees changes. Concurrency test.

### section-07-unit-tests
Unit tests for flags, heuristic scorer, hook wiring, logging, and observability.
