# P15 TDD Plan: Autonomous Fund Integration

**Testing framework:** pytest (existing codebase)
**Test locations:** `tests/unit/autonomous/`, `tests/unit/execution/`, `tests/integration/`
**Fixtures:** DB mocking via `monkeypatch`, `MagicMock` for broker API, `freezegun` for time-based mode testing
**Key principle:** P15 is integration/hardening -- tests focus on wiring, ceilings, and failure modes, not new algorithms

---

## Section 1: Multi-Mode Scheduler (`config/operating_modes.py`)

```python
# tests/unit/autonomous/test_operating_modes.py

import pytest
from datetime import datetime
from unittest.mock import patch


class TestModeDetection:
    """Tests for automatic operating mode detection from clock + calendar."""

    def test_market_hours_weekday(self):
        """Monday 10:00 ET -> MARKET_HOURS."""

    def test_extended_hours_after_close(self):
        """Tuesday 17:00 ET -> EXTENDED_HOURS."""

    def test_extended_hours_premarket(self):
        """Wednesday 07:00 ET -> EXTENDED_HOURS."""

    def test_overnight_late_night(self):
        """Thursday 23:00 ET -> OVERNIGHT_WEEKEND."""

    def test_weekend_saturday(self):
        """Saturday 14:00 ET -> OVERNIGHT_WEEKEND."""

    def test_weekend_sunday(self):
        """Sunday 10:00 ET -> OVERNIGHT_WEEKEND."""

    def test_market_holiday_is_overnight(self):
        """July 4th Friday 11:00 ET (holiday) -> OVERNIGHT_WEEKEND."""

    def test_boundary_930_is_market(self):
        """Exactly 09:30:00 ET Monday -> MARKET_HOURS (inclusive)."""

    def test_boundary_1600_is_extended(self):
        """Exactly 16:00:00 ET Monday -> EXTENDED_HOURS (market close boundary)."""

    def test_boundary_2000_is_overnight(self):
        """Exactly 20:00:00 ET Monday -> OVERNIGHT_WEEKEND."""

    def test_boundary_0400_is_extended(self):
        """Exactly 04:00:00 ET Tuesday -> EXTENDED_HOURS (premarket start)."""


class TestSchedulerModeIntegration:
    """Tests for scheduler graph activation per mode."""

    def test_market_hours_activates_trading_graph(self):
        """MARKET_HOURS mode activates trading graph with full priority."""

    def test_market_hours_has_lightweight_research(self):
        """MARKET_HOURS mode runs research at reduced compute allocation."""

    def test_overnight_activates_full_research(self):
        """OVERNIGHT_WEEKEND mode gives research full compute budget."""

    def test_overnight_disables_equity_trading(self):
        """OVERNIGHT_WEEKEND mode disables equity trading graph."""

    def test_extended_hours_position_monitoring_only(self):
        """EXTENDED_HOURS mode activates monitoring but not new entries."""

    def test_mode_transition_fires_hooks(self):
        """Transition from MARKET_HOURS to EXTENDED_HOURS fires EOD reconciliation hook."""

    def test_mode_transition_extended_to_overnight_fires_sync(self):
        """Transition from EXTENDED_HOURS to OVERNIGHT_WEEKEND fires data sync hook."""
```

---

## Section 2: Feedback Loop Verifier (`autonomous/loop_verifier.py`)

```python
# tests/unit/autonomous/test_loop_verifier.py

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock


class TestLoopVerifier:
    """Tests for feedback loop health monitoring."""

    def test_trade_outcome_loop_detected(self, mock_db):
        """Trade closed in last 24h and research queue updated -> loop 1 healthy."""

    def test_trade_outcome_loop_stale(self, mock_db):
        """Trade closed 36h ago, no research queue update -> loop 1 stale."""

    def test_trade_outcome_loop_broken(self, mock_db):
        """No trade close and no research update in 72h -> loop 1 broken."""

    def test_realized_cost_loop_healthy(self, mock_db):
        """TCA feedback written in last 24h + cost model updated -> loop 2 healthy."""

    def test_ic_degradation_loop_healthy(self, mock_db):
        """IC dropped below 0.02 and weight was adjusted -> loop 3 healthy."""

    def test_live_performance_loop_healthy(self, mock_db):
        """3 losing weeks detected and strategy demoted -> loop 4 healthy."""

    def test_agent_quality_loop_healthy(self, mock_db):
        """Agent win rate < 40% and few-shot injected -> loop 5 healthy."""

    def test_no_trades_recently_loop_not_broken(self, mock_db):
        """No trades in 24h is not a broken loop -- loop simply did not trigger."""

    def test_all_loops_healthy_returns_clean_health(self, mock_db):
        """All 5 loops closed recently: overall status is 'all_healthy'."""

    def test_get_loop_health_returns_required_fields(self, mock_db):
        """Each loop entry has last_triggered, last_behavior_change, status."""

    def test_db_failure_returns_error_status(self, mock_db):
        """DB connection failure: returns status='error' per loop, not exception."""


class TestLoopHealthDashboard:
    """Tests for loop health dashboard output."""

    def test_healthy_status_within_24h(self, mock_db):
        """Loop closed 12h ago -> status='healthy'."""

    def test_stale_status_24_to_48h(self, mock_db):
        """Loop closed 30h ago -> status='stale'."""

    def test_broken_status_beyond_48h(self, mock_db):
        """Loop closed 60h ago -> status='broken'."""

    def test_alert_fired_for_broken_loop(self, mock_db):
        """Broken loop triggers alert (verify alert function called)."""
```

---

## Section 3: Decision Authority Matrix (`autonomous/authority_matrix.py`)

```python
# tests/unit/autonomous/test_authority_matrix.py

import pytest


class TestAuthorityCeilings:
    """Tests for decision ceiling enforcement."""

    def test_single_position_max_5pct(self):
        """Position request at 6% of portfolio -> rejected."""

    def test_single_position_at_5pct_accepted(self):
        """Position request at exactly 5% of portfolio -> accepted."""

    def test_max_daily_new_positions_3(self, mock_db):
        """4th new position in same day -> rejected."""

    def test_first_3_positions_accepted(self, mock_db):
        """Positions 1, 2, 3 in same day -> all accepted."""

    def test_max_strategy_promotion_1_per_week(self, mock_db):
        """2nd strategy promotion in same week -> rejected."""

    def test_signal_weight_change_max_10pct_relative(self):
        """Weight change from 0.20 to 0.25 (25% relative change) -> rejected."""

    def test_signal_weight_change_within_limit(self):
        """Weight change from 0.20 to 0.22 (10% relative change) -> accepted."""

    def test_zero_portfolio_value_handled(self):
        """Portfolio value = 0: position sizing does not divide by zero."""


class TestAuthorityCeilingEscalation:
    """Tests for ceiling breach escalation."""

    def test_ceiling_breach_logged_with_context(self, mock_db):
        """Rejected request is logged with agent, decision_type, requested_value, ceiling."""

    def test_ceiling_breach_not_executed(self, mock_db):
        """After rejection, position is NOT opened -- verify no trade record."""

    def test_ceiling_breach_flagged_for_human(self, mock_db):
        """Rejected request creates human_review_queue entry."""

    def test_system_continues_within_ceiling(self, mock_db):
        """After rejection, subsequent within-ceiling requests still processed."""


class TestAuthorityDecisionLogging:
    """Tests for mandatory audit trail on autonomous decisions."""

    def test_every_decision_logged(self, mock_db):
        """Accepted decision creates log entry with agent, inputs, ceiling_applied."""

    def test_rejected_decision_logged(self, mock_db):
        """Rejected decision creates log entry with rejection_reason."""
```

---

## Section 4: Position Reconciliation (`execution/reconciler.py`)

```python
# tests/unit/execution/test_reconciler.py

import pytest
from unittest.mock import MagicMock


class TestPositionReconciler:
    """Tests for system vs broker position reconciliation."""

    def test_matching_positions_pass(self, mock_db, mock_broker):
        """System and broker agree on all positions: no alerts, no corrections."""

    def test_notional_mismatch_within_tolerance(self, mock_db, mock_broker):
        """$80 notional difference: within $100 tolerance, pass without alert."""

    def test_notional_mismatch_beyond_tolerance(self, mock_db, mock_broker):
        """$150 notional difference: alert triggered, system adjusted to match broker."""

    def test_broker_has_position_system_doesnt(self, mock_db, mock_broker):
        """Phantom position in broker: alert, system state updated to include it."""

    def test_system_has_position_broker_doesnt(self, mock_db, mock_broker):
        """Stale position in system: alert, system state updated to remove it."""

    def test_broker_is_source_of_truth(self, mock_db, mock_broker):
        """On mismatch, system state always adjusted to match broker, not vice versa."""

    def test_reconciliation_logged(self, mock_db, mock_broker):
        """Every reconciliation run creates a log entry with timestamp and results."""

    def test_broker_api_failure_does_not_crash(self, mock_db, mock_broker):
        """Broker API timeout: log error, skip reconciliation, alert."""

    def test_empty_portfolio_both_sides(self, mock_db, mock_broker):
        """No positions on either side: pass cleanly."""


class TestPnLReconciliation:
    """Tests for P&L discrepancy detection."""

    def test_pnl_within_1pct_passes(self, mock_db, mock_broker):
        """System P&L $1000, broker P&L $1005: within 1%, no alert."""

    def test_pnl_beyond_1pct_alerts(self, mock_db, mock_broker):
        """System P&L $1000, broker P&L $1020 (2% off): alert triggered."""

    def test_zero_pnl_both_sides_passes(self, mock_db, mock_broker):
        """Both report $0 P&L: pass."""

    def test_negative_pnl_reconciled_correctly(self, mock_db, mock_broker):
        """Losses reconciled with same tolerance rules as gains."""


class TestReconciliationSchedule:
    """Tests for reconciliation trigger timing."""

    def test_runs_on_mode_transition(self):
        """Mode change from MARKET_HOURS to EXTENDED_HOURS triggers reconciliation."""

    def test_runs_every_4_hours(self):
        """Reconciliation scheduled at 4-hour intervals within a mode."""
```

---

## Section 5: Health & Safety Dashboard

```python
# tests/unit/autonomous/test_dashboard.py

import pytest
from unittest.mock import MagicMock


class TestUnifiedStatus:
    """Tests for supervisor health dashboard output."""

    def test_portfolio_section_present(self, mock_db):
        """Dashboard includes positions, unrealized_pnl, greeks_exposure, margin_util."""

    def test_signals_section_present(self, mock_db):
        """Dashboard includes per-collector IC health and synthesis weights."""

    def test_agents_section_present(self, mock_db):
        """Dashboard includes per-agent quality_score, win_rate, last_cycle_time."""

    def test_research_section_present(self, mock_db):
        """Dashboard includes pipeline_depth, hypothesis_velocity, lifecycle_counts."""

    def test_system_section_present(self, mock_db):
        """Dashboard includes db_health, api_rate_limits, container_status."""

    def test_empty_db_returns_defaults(self, mock_db):
        """No data in DB: dashboard returns zero/empty defaults, not errors."""


class TestAlerting:
    """Tests for Discord webhook alert triggers."""

    def test_kill_switch_triggers_alert(self):
        """Kill switch engaged: Discord alert fires with context."""

    def test_drawdown_5pct_triggers_alert(self, mock_db):
        """Drawdown exceeds 5% from peak: alert triggered."""

    def test_drawdown_4pct_no_alert(self, mock_db):
        """Drawdown at 4%: no alert (below threshold)."""

    def test_agent_win_rate_below_30_triggers(self, mock_db):
        """Agent win rate < 30% for 5 cycles: alert triggered."""

    def test_data_staleness_triggers_alert(self, mock_db):
        """Price data older than configured threshold: alert triggered."""

    def test_reconciliation_mismatch_triggers_alert(self, mock_db):
        """Reconciliation found mismatch: alert triggered."""

    def test_broken_loop_triggers_alert(self, mock_db):
        """Feedback loop broken >48h: alert triggered."""

    def test_discord_webhook_failure_does_not_crash(self):
        """Discord API failure: log error, system continues operating."""


class TestWeeklyReport:
    """Tests for automated Sunday 20:00 ET report."""

    def test_report_includes_sharpe(self, mock_db):
        """Weekly report contains weekly Sharpe ratio."""

    def test_report_includes_drawdown(self, mock_db):
        """Weekly report contains max drawdown."""

    def test_report_includes_top3_winners_losers(self, mock_db):
        """Weekly report lists top 3 winners and top 3 losers."""

    def test_report_includes_research_velocity(self, mock_db):
        """Weekly report has hypotheses generated, validated, rejected counts."""

    def test_report_with_no_trades_still_generates(self, mock_db):
        """Zero trades in the week: report generates with zeros, not errors."""
```

---

## Section 6: Performance Benchmarking (`autonomous/benchmarks.py`)

```python
# tests/unit/autonomous/test_benchmarks.py

import pytest
import numpy as np


class TestBenchmarkTracking:
    """Tests for SPY / 60-40 / equal-weight benchmarking."""

    def test_information_ratio_computed(self, mock_db):
        """IR = (portfolio_return - benchmark_return) / tracking_error. Verify formula."""

    def test_tracking_error_computed(self, mock_db):
        """Tracking error is std of (portfolio_returns - benchmark_returns)."""

    def test_alpha_and_beta_computed(self, mock_db):
        """CAPM regression produces alpha and beta values."""

    def test_rolling_windows_1w_1m_3m_ytd(self, mock_db):
        """All four rolling windows computed without error."""

    def test_zero_benchmark_return_handled(self):
        """Benchmark return of 0 does not cause division by zero in IR."""

    def test_insufficient_data_returns_none(self, mock_db):
        """Less than 5 days of data: return None for metrics, not error."""


class TestReturnAttribution:
    """Tests for signal/execution/timing/risk alpha decomposition."""

    def test_attribution_sums_to_total_return(self, mock_db):
        """signal_alpha + execution_alpha + timing_alpha + risk_management = total_return."""

    def test_risk_management_positive_when_gate_saves_money(self, mock_db):
        """Risk gate rejected a trade that would have lost money: risk_management > 0."""

    def test_no_trades_gives_zero_attribution(self, mock_db):
        """Zero trades: all attribution components are 0."""
```

---

## Section 7: Disaster Recovery

```python
# tests/unit/autonomous/test_disaster_recovery.py

import pytest


class TestKillSwitchLayering:
    """Tests for 4-layer kill switch hierarchy."""

    def test_layer1_per_position_stop_loss(self, mock_db, mock_broker):
        """Position hits stop loss: position closed, other positions unaffected."""

    def test_layer2_portfolio_drawdown_breaker(self, mock_db):
        """Portfolio drawdown exceeds threshold: all new entries halted."""

    def test_layer3_agent_kill_switch(self, mock_db):
        """Supervisor kills agent: that agent's cycle stops, others continue."""

    def test_layer4_system_kill_switch(self, mock_db):
        """System kill switch: everything stops, no new trades, no research."""

    def test_layer4_auto_on_critical_failure(self, mock_db):
        """Critical system failure (e.g., DB down) triggers layer 4 automatically."""

    def test_kill_switch_recovery(self, mock_db):
        """After kill switch cleared, system resumes normal operation."""


class TestGracefulDegradation:
    """Tests for partial system failure handling."""

    def test_research_down_trading_continues(self):
        """Research graph offline: trading graph still operates."""

    def test_trading_down_kills_switch_engages(self):
        """Trading graph offline: kill switch engaged automatically."""

    def test_db_reconnect_after_transient_failure(self, mock_db):
        """DB connection drops and recovers: system resumes without manual intervention."""
```

---

## Section 8: Burn-In Protocol

```python
# tests/unit/autonomous/test_burn_in.py

import pytest
from unittest.mock import MagicMock


class TestBurnInValidation:
    """Tests for 7-day burn-in protocol checks."""

    def test_all_loops_closed_at_least_once(self, mock_db):
        """Verify check: all 5 feedback loops have >= 1 closure in 7 days."""

    def test_fails_if_any_loop_never_closed(self, mock_db):
        """Loop 3 never closed in 7 days: burn-in fails."""

    def test_no_bug_kill_switch_triggers(self, mock_db):
        """Kill switch triggered by bug (not real risk): burn-in fails."""

    def test_real_risk_kill_switch_ok(self, mock_db):
        """Kill switch triggered by real market risk: burn-in still passes."""

    def test_reconciliation_7_consecutive_days(self, mock_db):
        """Reconciliation within tolerance for all 7 days: passes."""

    def test_reconciliation_miss_on_day_4_fails(self, mock_db):
        """Reconciliation mismatch on day 4: burn-in fails."""

    def test_weekly_report_generated(self, mock_db):
        """At least 1 weekly report generated successfully during burn-in."""

    def test_strategy_lifecycle_exercised(self, mock_db):
        """At least 1 strategy promoted and 1 managed through lifecycle."""

    def test_go_live_checklist_all_items(self):
        """Go-live checklist returns list of all required items with pass/fail status."""
```

---

## Section 9: Integration Tests

```python
# tests/integration/test_autonomous_fund_e2e.py

import pytest


class TestModeTransitionEndToEnd:
    """End-to-end: mode transitions with proper hook firing."""

    def test_market_to_extended_transition(self, mock_db, mock_broker):
        """
        1. System in MARKET_HOURS mode
        2. Clock hits 16:00 ET
        3. Mode transitions to EXTENDED_HOURS
        4. EOD reconciliation fires
        5. Trading graph scales down to monitoring only
        6. Verify no new entry orders accepted
        """

    def test_overnight_to_market_transition(self, mock_db, mock_broker):
        """
        1. System in OVERNIGHT_WEEKEND mode (Sunday night)
        2. Clock hits 09:30 ET Monday
        3. Mode transitions to MARKET_HOURS
        4. Trading graph activated with full priority
        5. Research compute reduced
        """


class TestAutonomousDecisionEndToEnd:
    """End-to-end: autonomous decision with ceiling enforcement."""

    def test_within_ceiling_trade_executes(self, mock_db, mock_broker):
        """
        1. Agent requests 3% position (within 5% ceiling)
        2. Authority matrix approves
        3. Risk gate clears
        4. Trade executes
        5. Decision logged with full audit trail
        """

    def test_beyond_ceiling_escalated(self, mock_db, mock_broker):
        """
        1. Agent requests 7% position (above 5% ceiling)
        2. Authority matrix rejects
        3. Request logged and flagged for human review
        4. No trade executed
        5. Agent continues with other decisions
        """


class TestReconciliationRecoveryEndToEnd:
    """End-to-end: mismatch detection and correction."""

    def test_position_mismatch_corrected(self, mock_db, mock_broker):
        """
        1. System says 100 shares AAPL, broker says 95 shares AAPL
        2. Reconciler detects mismatch ($500+ notional at ~$200/share)
        3. Alert fired
        4. System state updated to 95 shares (broker is truth)
        5. Mismatch logged for investigation
        """
```

---

## Test Execution Order

For TDD, write and verify tests in this order:

1. **Section 1 tests** -- Mode detection (pure time logic, no DB, fast)
2. **Section 3 tests** -- Authority matrix ceilings (pure business rules, fast)
3. **Section 2 tests** -- Loop verifier (DB mocks, checks timestamps)
4. **Section 4 tests** -- Reconciler (broker mock + DB mock)
5. **Section 5 tests** -- Dashboard (aggregates from DB, mocked)
6. **Section 6 tests** -- Benchmarks (numerical, synthetic data)
7. **Section 7 tests** -- Disaster recovery (mock system components)
8. **Section 8 tests** -- Burn-in validation (composite checks)
9. **Section 9 tests** -- End-to-end integration (requires all sections)
