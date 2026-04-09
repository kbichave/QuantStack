# Section 11: Unit Tests

## Objective

Comprehensive test coverage for all P15 subsystems. Each section specifies its own tests, but this section defines the integration tests that verify subsystems work together, plus any cross-cutting test infrastructure.

**Depends on:** section-01-operating-modes, section-03-loop-verifier, section-04-authority-matrix, section-05-reconciler, section-08-benchmarks

## Files to Create

### `tests/unit/autonomous/__init__.py`

Package init for autonomous test module.

### `tests/unit/autonomous/conftest.py`

Shared fixtures for P15 tests (mock DB, mock broker, mock clock).

### `tests/unit/config/test_operating_modes.py`

Tests for operating mode detection (detailed in Section 01).

### `tests/unit/autonomous/test_loop_verifier.py`

Tests for loop verification (detailed in Section 03).

### `tests/unit/autonomous/test_authority_matrix.py`

Tests for authority matrix (detailed in Section 04).

### `tests/unit/execution/test_reconciler.py`

Tests for position reconciliation (detailed in Section 05).

### `tests/unit/autonomous/test_benchmarks.py`

Tests for benchmark tracking (detailed in Section 08).

### `tests/unit/autonomous/test_mode_manager.py`

Tests for mode manager (detailed in Section 02).

### `tests/unit/autonomous/test_health_dashboard.py`

Tests for health dashboard (detailed in Section 06).

### `tests/unit/autonomous/test_alerting.py`

Tests for alert manager (detailed in Section 07).

### `tests/unit/autonomous/test_weekly_report.py`

Tests for weekly report (detailed in Section 07).

### `tests/unit/autonomous/test_kill_switch_layers.py`

Tests for kill switch layers (detailed in Section 09).

### `tests/unit/autonomous/test_burn_in.py`

Tests for burn-in protocol (detailed in Section 10).

## Implementation Details

### Shared Fixtures (`conftest.py`)

```python
@pytest.fixture
def mock_db():
    """In-memory mock for db_conn() that returns predefined query results."""
    ...

@pytest.fixture
def mock_broker():
    """Mock broker with configurable positions and P&L."""
    ...

@pytest.fixture
def mock_clock():
    """Freezable clock for testing time-dependent behavior."""
    ...

@pytest.fixture
def sample_portfolio_state():
    """Realistic portfolio state with 5 positions across strategies."""
    ...

@pytest.fixture
def loop_verifier(mock_db):
    """LoopVerifier wired to mock DB."""
    ...

@pytest.fixture
def authority_matrix(mock_db):
    """AuthorityMatrix wired to mock DB."""
    ...

@pytest.fixture
def reconciler(mock_broker, mock_db):
    """PositionReconciler wired to mock broker and DB."""
    ...
```

### Integration Test Scenarios

These verify that subsystems compose correctly:

1. **Mode transition triggers reconciliation**
   - Simulate MARKET_HOURS → EXTENDED_HOURS transition
   - Verify reconciliation hook fires
   - Verify reconciliation report is logged

2. **Authority matrix integrated with risk gate**
   - Submit a trade that passes risk gate but exceeds authority ceiling
   - Verify the trade is rejected with authority reason

3. **Broken loop triggers alert**
   - Mock a loop as broken for 48h
   - Run health dashboard → alert manager pipeline
   - Verify Discord alert would fire

4. **Reconciliation mismatch resets burn-in counter**
   - Run burn-in with 5 clean days
   - Inject mismatch on day 6
   - Verify consecutive-day counter resets to 0

5. **Full health dashboard with mixed status**
   - Some loops healthy, some stale
   - One reconciliation mismatch
   - One underperforming agent
   - Verify `overall_status == "degraded"` (not critical, not healthy)

### Test Naming Convention

All tests follow `test_{subsystem}_{scenario}_{expected_outcome}` pattern:
- `test_authority_matrix_over_ceiling_rejected`
- `test_reconciler_phantom_system_position_removed`
- `test_mode_detection_holiday_returns_overnight_weekend`

### Coverage Targets

- All public methods have at least 1 happy-path and 1 edge-case test
- All error paths (missing data, DB errors, API failures) have tests
- No test depends on real time, network, or database

## Test Requirements

- All tests run with `uv run pytest tests/unit/autonomous/ tests/unit/config/test_operating_modes.py tests/unit/execution/test_reconciler.py`
- No test takes more than 1 second
- No test requires network access, Docker, or a running database
- Tests are deterministic (no random, no real clock)

## Acceptance Criteria

1. Every P15 module has corresponding test file with meaningful coverage
2. Shared fixtures eliminate test boilerplate (no copy-paste mock setup)
3. Integration test scenarios cover the 5 critical cross-subsystem interactions listed above
4. All tests pass in CI without external dependencies
5. `conftest.py` fixtures are documented with docstrings explaining what they provide
6. Test files mirror the source module structure (`src/quantstack/autonomous/X.py` → `tests/unit/autonomous/test_X.py`)
