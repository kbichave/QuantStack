# Section 07: Unit Tests

**depends_on:** section-01-agent-quality-tracking, section-02-prompt-ab-testing, section-03-strategy-of-strategies, section-04-research-prioritization, section-05-few-shot-library

## Objective

Comprehensive test suite for all P10 meta-learning modules. Each section specifies its own test cases; this section consolidates them into test files, defines shared fixtures, and adds cross-module tests that validate the meta-learning pipeline end-to-end.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `tests/unit/learning/test_agent_quality_tracking.py` | Create | Tests from section-01 (11 tests) |
| `tests/unit/learning/test_prompt_ab.py` | Create | Tests from section-02 (10 tests) |
| `tests/unit/learning/test_meta_allocator.py` | Create | Tests from section-03 (10 tests) |
| `tests/unit/learning/test_research_prioritization.py` | Create | Tests from section-04 (10 tests) |
| `tests/unit/learning/test_few_shot_library.py` | Create | Tests from section-05 (13 tests) |
| `tests/unit/learning/test_meta_learning_integration.py` | Create | Tests from section-06 (10 tests) |
| `tests/unit/learning/test_meta_learning_e2e.py` | Create | Cross-module end-to-end tests (this section) |
| `tests/unit/learning/conftest.py` | Modify | Shared fixtures for P10 tests |

## Implementation Details

### 1. Shared Fixtures (`conftest.py`)

Add the following shared fixtures:

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_db_conn():
    """Mock database connection for all P10 tests.
    
    Returns a context manager that yields a mock connection.
    Tests that need specific query results override execute().fetchall().
    """
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    with patch("quantstack.db.db_conn", return_value=mock_conn):
        yield mock_conn

@pytest.fixture
def sample_agent_quality_scores():
    """30 days of sample quality scores for 3 agents."""
    # Generate realistic score sequences for testing
    ...

@pytest.fixture
def sample_strategies():
    """5 sample StrategyFeatures for meta-allocator tests."""
    from quantstack.learning.meta_allocator import StrategyFeatures
    return [
        StrategyFeatures(
            strategy_id="AAPL_momentum",
            rolling_ic_21d=0.05,
            rolling_sharpe_21d=1.2,
            vol_contribution=0.15,
            correlation_avg=0.3,
            regime_fit=0.9,
            days_active=45,
        ),
        # ... 4 more with varying characteristics
    ]

@pytest.fixture
def sample_research_tasks():
    """10 sample research tasks with varying characteristics."""
    return [
        {"task_id": "t1", "task_type": "new_strategy", "symbol": "AAPL", "hypothesis": "momentum works"},
        {"task_id": "t2", "task_type": "bug_fix", "symbol": "NVDA", "hypothesis": "fix entry timing"},
        # ... etc.
    ]
```

### 2. End-to-End Pipeline Tests (`test_meta_learning_e2e.py`)

These tests validate that the meta-learning components work together as a pipeline:

**test_quality_to_few_shot_pipeline**
- Record 20 cycle quality scores for an agent (varying quality)
- Run few-shot curation
- Verify only top-quality outputs become few-shot examples
- Retrieve examples for a matching context
- Verify the retrieved examples have quality scores in the top percentile

**test_quality_to_degradation_to_research_pipeline**
- Record 5 consecutive low-quality scores for an agent
- Run degradation alert check
- Verify an alert is generated
- Verify the alert would produce a valid research task with high priority

**test_prompt_ab_lifecycle_e2e**
- Create a variant for an agent
- Activate it
- Record 30 shadow results (variant consistently better)
- Evaluate the variant
- Verify it recommends promotion
- Promote it
- Verify status is 'promoted'

**test_meta_allocator_regime_shift**
- Compute weights for regime="trending_up"
- Change regime to "ranging"
- Recompute weights
- Verify momentum strategies lost weight and mean_reversion gained weight

**test_research_priority_adapts_to_losses**
- Start with a balanced priority queue
- Simulate 10 losses on a specific symbol
- Re-prioritize
- Verify tasks related to that symbol moved up in priority

**test_full_cycle_meta_learning**
- Simulate a complete trading cycle:
  1. Agent executes with few-shot examples
  2. Trade opens and closes
  3. Quality score recorded
  4. Few-shot library updated
  5. Meta-allocator recomputed
  6. Research queue re-prioritized
- Verify all components received and processed data correctly

### 3. Test Execution Notes

All tests should:
- Use `mock_db_conn` fixture to avoid needing a real database
- Use `pytest.mark.parametrize` where appropriate (e.g., testing multiple regime/strategy combinations)
- Be independent: no test depends on another test's state
- Run in < 2 seconds each (no real I/O)
- Use `loguru` capture or `caplog` to verify warning/error logging where specified

### 4. Test Count Summary

| File | Test Count |
|------|-----------|
| `test_agent_quality_tracking.py` | 11 |
| `test_prompt_ab.py` | 10 |
| `test_meta_allocator.py` | 10 |
| `test_research_prioritization.py` | 10 |
| `test_few_shot_library.py` | 13 |
| `test_meta_learning_integration.py` | 10 |
| `test_meta_learning_e2e.py` | 6 |
| **Total** | **70** |

## Test Requirements

All 70 tests must pass with `uv run pytest tests/unit/learning/test_*meta* tests/unit/learning/test_agent_quality* tests/unit/learning/test_prompt_ab* tests/unit/learning/test_research_prior* tests/unit/learning/test_few_shot*`.

## Acceptance Criteria

- [ ] All 70 tests pass
- [ ] No test requires a running database (all mocked)
- [ ] No test takes longer than 2 seconds
- [ ] Shared fixtures in `conftest.py` reduce duplication across test files
- [ ] End-to-end tests validate cross-module data flow
- [ ] Test coverage for error/edge cases: empty inputs, insufficient data, database failures, missing columns
- [ ] All existing tests in `tests/unit/learning/` continue to pass (no regressions)
