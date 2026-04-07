# Section 07: Overnight Autoresearch

## Background

QuantStack currently runs research only when a Claude session is active or when AutoResearchClaw fires on its weekly Sunday schedule. This produces roughly 12 hypotheses per week. Meanwhile, 8 overnight hours across 7 nights (56 hours/week) go completely unused. The overnight autoresearch system reclaims this idle time by running a tight hypothesis-backtest loop from 20:00 to 04:00 ET every night, targeting 96+ experiments per night as a conservative lower bound (actual throughput is likely hundreds, since most backtests complete in under 10 seconds).

At 04:00 ET, a morning validation pipeline picks up the "winners" from the night, runs deeper multi-window backtests using Sonnet, and registers passing strategies as `status='draft'` in the strategies table for daytime trading consideration.

## Dependencies

- **section-01-db-migrations**: The `autoresearch_experiments` table must exist before the runner can log experiments.
- **section-02-tool-lifecycle**: Tool health data informs which tools are available for experiment evaluation.
- **section-03-error-driven-research**: Loss-driven research tasks feed into the overnight queue as `hypothesis_source='error_driven'`.
- **section-04-budget-discipline**: The morning validator uses the 3-window patience protocol for deeper validation. The overnight per-experiment budget is independent of the daytime per-cycle budget.
- **section-05-event-bus-extensions**: The `EXPERIMENT_COMPLETED` event type must be registered for publishing results.

## Tests First

File: `tests/unit/test_overnight_runner.py`

```python
"""Tests for the overnight autoresearch runner."""

# Test: runner starts at 20:00 ET and stops at 04:00 ET
def test_runner_respects_operating_window():
    """The run loop only executes experiments between 20:00 and 04:00 ET.
    Outside this window, the runner exits gracefully."""
    ...

# Test: runner halts when cumulative cost reaches $9.50 (leaves headroom)
def test_runner_halts_at_budget_ceiling():
    """When cumulative nightly spend hits $9.50, the runner stops
    generating new experiments, leaving $0.50 headroom for the
    morning validator's Sonnet calls."""
    ...

# Test: budget tracking persists to DB (survives simulated crash)
def test_budget_persisted_to_db():
    """The nightly_budget row in autoresearch_experiments (keyed by
    night_date) is updated after each experiment. On simulated crash
    and restart, the cumulative cost is read from DB, not reset to zero."""
    ...

# Test: runner resumes from last cumulative cost after restart
def test_runner_resumes_after_crash():
    """Pre-populate DB with a partial night's experiments. Runner
    picks up cumulative cost and experiment count, does not re-run
    completed experiments (idempotent via experiment_id)."""
    ...

# Test: experiment with OOS IC > 0.02 marked as "winner"
def test_winner_threshold():
    """An experiment whose out-of-sample IC exceeds 0.02 on the
    purged holdout is tagged status='winner'."""
    ...

# Test: experiment with OOS IC <= 0.02 marked as "tested" (not winner)
def test_non_winner_marked_tested():
    """Experiments at or below OOS IC 0.02 are status='tested'."""
    ...

# Test: experiment exceeding 5-minute timeout is killed and logged
def test_experiment_timeout():
    """If a backtest or ML training exceeds 5 minutes wall-clock,
    the runner kills the experiment, logs status='timeout' with
    duration_seconds, and moves to the next experiment."""
    ...

# Test: experiments run back-to-back (no artificial sleep between)
def test_no_sleep_between_experiments():
    """Verify the runner loop has no sleep/delay between experiments.
    The 5-minute budget is a timeout, not a sleep interval."""
    ...

# Test: experiment_id is unique (no duplicates on restart)
def test_experiment_id_uniqueness():
    """Each experiment gets a UUID. After crash and restart, new
    experiments get new IDs. No collisions with prior experiments."""
    ...
```

File: `tests/unit/test_morning_validator.py`

```python
"""Tests for the morning validation pipeline."""

# Test: morning validator runs at 04:00 unconditionally
def test_validator_runs_unconditionally():
    """Even if only 5 experiments completed overnight, the morning
    validator runs at 04:00 and processes whatever winners exist."""
    ...

# Test: morning validator processes all winners from overnight
def test_validator_processes_all_winners():
    """Query autoresearch_experiments for tonight's status='winner'
    rows. Each gets the full validation pipeline."""
    ...

# Test: morning validator uses 3-window patience protocol
def test_validator_uses_patience_protocol():
    """Each winner is backtested across 3 windows: full historical,
    recent 12 months, and stressed period. Rejection requires
    failure in all 3 windows."""
    ...

# Test: passing winner registered as status='draft' in strategies
def test_passing_winner_registered():
    """A winner that passes the 3-window patience protocol is
    inserted into the strategies table with status='draft'."""
    ...

# Test: failing winner logged with rejection reason
def test_failing_winner_logged():
    """A winner that fails validation gets status='rejected' and
    a rejection_reason in autoresearch_experiments."""
    ...

# Test: morning validator handles zero winners gracefully
def test_zero_winners():
    """When no experiments achieved winner status overnight, the
    morning validator logs this fact and exits cleanly."""
    ...
```

## Implementation Details

### New Module: `src/quantstack/research/overnight_runner.py`

This is the core overnight loop. It runs as a scheduled job (launched by `scripts/scheduler.py` at 20:00 ET nightly) or as a standalone Docker service.

**Responsibilities:**

1. **Time window enforcement**: Run only between 20:00 and 04:00 ET. Check `datetime.now(tz=ET)` at the top of each iteration. Exit the loop when outside the window.

2. **Hypothesis generation**: Call Haiku to generate a trading hypothesis. Each hypothesis is a JSON structure containing `entry_rules`, `exit_rules`, and `parameters`. Cost per hypothesis: ~$0.0008 (500 tokens at Haiku rates). The hypothesis source can be:
   - `"haiku_generated"` -- novel ideas from the LLM
   - `"error_driven"` -- from `research_queue` tasks generated by the loss analyzer (section-03)
   - `"feature_factory"` -- from feature candidates with high IC (section-08, when available)

3. **Knowledge graph deduplication** (optional, after section-10 is built): Before running a backtest, call `check_hypothesis_novelty()`. If the hypothesis is redundant (>0.85 cosine similarity to a previously tested hypothesis in the same regime), skip it and generate a new one. Before section-10 exists, skip this step entirely.

4. **Backtest execution**: Run the hypothesis through the existing backtest infrastructure with a 5-minute wall-clock timeout. Most backtests complete in under 10 seconds (CPU-bound on ~5 years of daily OHLCV data). The timeout is a safety net for ML experiments that might involve model training.

5. **Scoring**: Compute out-of-sample IC using Spearman rank correlation on a purged holdout. The holdout is the last 20% of the data, with a purge gap equal to the strategy's holding period to prevent information leakage from overlapping windows.

6. **Logging**: Write every experiment to the `autoresearch_experiments` table. The table schema (from section-01-db-migrations):

```python
@dataclass
class AutoresearchExperiment:
    experiment_id: str          # UUID
    night_date: str             # YYYY-MM-DD
    hypothesis: str             # JSON: entry_rules, exit_rules, parameters
    hypothesis_source: str      # "haiku_generated", "error_driven", "feature_factory"
    oos_ic: float | None        # Out-of-sample information coefficient
    sharpe: float | None        # Out-of-sample Sharpe ratio
    cost_tokens: int            # Tokens consumed for this experiment
    cost_usd: float             # Dollar cost for this experiment
    duration_seconds: int       # Wall-clock time for this experiment
    status: str                 # "tested", "winner", "validated", "rejected", "timeout"
    rejection_reason: str | None
    created_at: datetime
```

7. **Budget tracking**: Maintain cumulative nightly cost in a dedicated `nightly_budget` row in the same table, keyed by `night_date`. After each experiment, update the cumulative cost. Halt at $9.50 to leave $0.50 headroom for the morning validator's Sonnet calls.

8. **Event publishing**: After each experiment, publish an `EXPERIMENT_COMPLETED` event to the event bus with payload `{experiment_id, status, oos_ic}`.

**Crash recovery**: On startup, the runner queries the DB for the current `night_date`'s cumulative cost and completed experiment IDs. It resumes from the last cumulative cost without re-running completed experiments. The morning validator at 04:00 runs unconditionally -- even if only a handful of experiments completed, it validates whatever winners exist.

**Key function signatures:**

```python
async def run_overnight_loop() -> None:
    """Main entry point. Loops until 04:00 ET or budget exhaustion."""
    ...

async def run_single_experiment(
    hypothesis: dict,
    source: str,
    night_date: str,
) -> AutoresearchExperiment:
    """Generate, backtest, score, and log a single experiment.
    Raises TimeoutError if backtest exceeds 5 minutes."""
    ...

def generate_hypothesis() -> dict:
    """Call Haiku to produce a hypothesis JSON.
    Returns dict with entry_rules, exit_rules, parameters."""
    ...

def score_experiment(backtest_result: dict) -> tuple[float, float]:
    """Compute OOS IC and Sharpe from backtest output.
    Returns (oos_ic, sharpe)."""
    ...

def get_nightly_budget_state(night_date: str) -> tuple[float, set[str]]:
    """Read cumulative cost and completed experiment IDs from DB.
    Returns (cumulative_cost_usd, set_of_experiment_ids)."""
    ...
```

### New Module: `src/quantstack/research/morning_validator.py`

Runs at 04:00 ET as a separate scheduled job. Evaluates all `status='winner'` experiments from the current night.

**Validation pipeline per winner:**

1. Run the full 3-window patience protocol (from section-04-budget-discipline):
   - Window 1: Full historical period (e.g., 2020-2025)
   - Window 2: Recent 12 months
   - Window 3: Stressed period (configurable, default: 2020-03 to 2020-06 for COVID crash)

2. Use Sonnet for deeper regime-fit analysis. Sonnet evaluates: does this strategy's edge persist across the regime transitions in the test windows? Cost: ~$0.05-0.10 per winner (Sonnet with cached system prompt).

3. Decision:
   - Passes all 3 windows: register as `status='draft'` in the strategies table. Update experiment to `status='validated'`.
   - Passes 2/3 windows: register as `status='draft'` with a `provisional` flag and reduced confidence (lower position sizing). Update experiment to `status='validated'` with a note.
   - Fails all 3 windows: update experiment to `status='rejected'` with `rejection_reason`.

4. Log the rejection reason in `autoresearch_experiments` for downstream learning (the knowledge graph in section-10 will ingest these).

**Key function signatures:**

```python
async def run_morning_validation(night_date: str) -> list[dict]:
    """Validate all winners from the given night.
    Returns list of validation outcomes."""
    ...

async def validate_winner(experiment: AutoresearchExperiment) -> dict:
    """Run 3-window patience protocol + Sonnet regime analysis.
    Returns {status, rejection_reason, strategy_id}."""
    ...

def register_draft_strategy(experiment: AutoresearchExperiment, validation: dict) -> str:
    """Insert into strategies table with status='draft'.
    Returns the new strategy_id."""
    ...
```

### Modified File: `scripts/scheduler.py`

Add two new scheduled jobs:

- **Overnight autoresearch**: Trigger at 20:00 ET daily (Mon-Sun). Calls `run_overnight_loop()`. This replaces the idle overnight window. AutoResearchClaw's existing `bug_fix` and `ml_arch_search` tasks move to a separate weekly slot (section-14-autoresclaw-upgrades handles this).

- **Morning validation**: Trigger at 04:00 ET daily (Mon-Sun). Calls `run_morning_validation()`.

The scheduler already uses APScheduler with `CronTrigger`. The new jobs follow the exact same pattern as existing jobs (`data_refresh` at 08:00, `pnl_attribution` at 16:10).

### Modified File: `docker-compose.yml`

Add an `overnight-research` service definition. This service:
- Uses the same image as the research-graph service
- Mounts the same volumes (source code, models, data, logs, memory)
- Sets `OVERNIGHT_MODE=true` environment variable
- The entrypoint runs `python -m quantstack.research.overnight_runner`
- Restart policy: `unless-stopped` (auto-restart on crash for crash recovery)

Alternatively, the overnight runner can be triggered by the existing scheduler service. The choice depends on isolation preference: a dedicated service provides better resource isolation and independent restarts; using the scheduler is simpler operationally. The scheduler approach is recommended for the initial implementation because the overnight runner is a single-threaded sequential loop with low resource requirements.

## Budget Arithmetic

| Component | Unit Cost | Volume/Night | Nightly Total |
|-----------|-----------|--------------|---------------|
| Haiku hypothesis generation | ~$0.0008/hypothesis | ~100-500 | $0.08-0.40 |
| Backtest execution | $0 (CPU only) | ~100-500 | $0.00 |
| OOS scoring | $0 (CPU only) | ~100-500 | $0.00 |
| Morning validation (Sonnet) | ~$0.05-0.10/winner | ~5-50 winners | $0.25-5.00 |
| **Total** | | | **$0.33-5.40** |

The $10/night ceiling is generous for this workload. The binding constraint is wall-clock time, not LLM cost. At ~10 seconds per experiment, 8 hours yields ~2,880 experiments. At ~60 seconds per experiment (ML-heavy), ~480. The 96-per-night figure in the plan is a conservative lower bound assuming every experiment takes the full 5-minute timeout, which is unrealistic.

## Failure Modes

- **Haiku API outage**: The runner retries with exponential backoff (3 attempts, then skips to next experiment). If Haiku is down for the entire night, zero experiments complete. The morning validator handles zero winners gracefully.
- **Database connection loss**: The runner uses `db_conn()` context managers. On connection failure, the current experiment is lost but the runner retries on the next iteration. Cumulative budget state may be slightly stale (off by one experiment's cost).
- **Runaway backtest**: The 5-minute timeout kills stuck backtests. The experiment is logged with `status='timeout'`.
- **Budget tracking drift**: If the runner crashes between experiment completion and budget update, the resumed runner underestimates cumulative cost by one experiment (~$0.001). This is negligible against the $10 ceiling.
- **Morning validator fails**: If Sonnet is unavailable at 04:00, winners remain in `status='winner'` and are picked up on the next morning run. No data loss.

## Scope Boundaries

- The overnight runner does NOT modify the risk gate, kill switch, or any execution logic.
- The overnight runner does NOT place trades. It only generates and validates hypotheses, registering passing ones as `status='draft'` strategies for daytime trading consideration.
- The per-experiment budget (~$0.10 max) is independent of the daytime per-cycle budget from section-04. The two systems do not interact.
- Knowledge graph deduplication (checking for redundant hypotheses) is a soft dependency on section-10. The runner works without it -- it just may re-test similar hypotheses until the knowledge graph is available.
