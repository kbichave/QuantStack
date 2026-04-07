"""Unit tests for health metrics collector and Prometheus gauge updates."""

import logging
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# Stub psycopg modules to prevent import chain failures
_PSYCOPG_MODULES = [
    "psycopg", "psycopg.rows", "psycopg.types", "psycopg.types.json",
    "psycopg_pool",
]
for _mod_name in _PSYCOPG_MODULES:
    if _mod_name not in sys.modules:
        _stub = ModuleType(_mod_name)
        if _mod_name == "psycopg":
            _stub.Connection = MagicMock
        elif _mod_name == "psycopg.types.json":
            _stub.set_json_loads = MagicMock()
            _stub.Jsonb = MagicMock
        elif _mod_name == "psycopg_pool":
            _stub.ConnectionPool = MagicMock
        elif _mod_name == "psycopg.rows":
            _stub.dict_row = MagicMock()
        sys.modules[_mod_name] = _stub


# ---------------------------------------------------------------------------
# collect_health_metrics() tests
# ---------------------------------------------------------------------------

class TestCollectHealthMetrics:
    """Tests for the collect_health_metrics() function."""

    @pytest.fixture(autouse=True)
    def _patch_db(self):
        """Patch db_conn so no real DB is needed."""
        self.mock_conn = MagicMock()
        self.mock_ctx = MagicMock()
        self.mock_ctx.__enter__ = MagicMock(return_value=self.mock_conn)
        self.mock_ctx.__exit__ = MagicMock(return_value=False)

        # Import the module first to ensure it's loaded, then patch
        from quantstack.graphs.supervisor import health_metrics
        with patch.object(health_metrics, "db_conn", return_value=self.mock_ctx):
            yield

    @pytest.mark.asyncio
    async def test_cycle_success_rate_computed(self):
        from quantstack.graphs.supervisor.health_metrics import collect_health_metrics

        self.mock_conn.execute.return_value.fetchone.side_effect = [
            (7, 10),   # trading cycle success
            (3,),      # trading error count
            (9, 10),   # research cycle success
            (1,),      # research error count
            (5,),      # strategy gen 7d
            (12,),     # research queue depth
        ]
        result = await collect_health_metrics()
        assert result["trading_cycle_success_rate"] == pytest.approx(0.70)

    @pytest.mark.asyncio
    async def test_zero_success_rate_when_no_checkpoints(self):
        from quantstack.graphs.supervisor.health_metrics import collect_health_metrics

        self.mock_conn.execute.return_value.fetchone.side_effect = [
            (0, 0),    # trading: no checkpoints
            (0,),      # trading error count
            (0, 0),    # research: no checkpoints
            (0,),      # research error count
            (0,),      # strategy gen 7d
            (0,),      # research queue depth
        ]
        result = await collect_health_metrics()
        assert result["trading_cycle_success_rate"] == 0.0
        assert result["research_cycle_success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_error_count_for_latest_cycle(self):
        from quantstack.graphs.supervisor.health_metrics import collect_health_metrics

        self.mock_conn.execute.return_value.fetchone.side_effect = [
            (8, 10),   # trading success
            (3,),      # trading: 3 errors in latest cycle
            (10, 10),  # research success
            (0,),      # research: 0 errors
            (5,),      # strategy gen 7d
            (0,),      # research queue
        ]
        result = await collect_health_metrics()
        assert result["trading_cycle_error_count"] == 3
        assert result["research_cycle_error_count"] == 0

    @pytest.mark.asyncio
    async def test_strategy_generation_7d(self):
        from quantstack.graphs.supervisor.health_metrics import collect_health_metrics

        self.mock_conn.execute.return_value.fetchone.side_effect = [
            (10, 10), (0,),  # trading
            (10, 10), (0,),  # research
            (5,),            # 5 strategies in 7d
            (0,),            # queue
        ]
        result = await collect_health_metrics()
        assert result["strategy_generation_7d"] == 5

    @pytest.mark.asyncio
    async def test_research_queue_depth(self):
        from quantstack.graphs.supervisor.health_metrics import collect_health_metrics

        self.mock_conn.execute.return_value.fetchone.side_effect = [
            (10, 10), (0,),  # trading
            (10, 10), (0,),  # research
            (0,),            # strategy gen
            (12,),           # 12 pending research tasks
        ]
        result = await collect_health_metrics()
        assert result["research_queue_depth"] == 12


# ---------------------------------------------------------------------------
# Prometheus gauge update tests
# ---------------------------------------------------------------------------

class TestHealthMetricsGauges:
    """Verify that Prometheus gauge setter functions update correctly."""

    def test_cycle_success_rate_gauge(self):
        from quantstack.observability.metrics import record_cycle_success_rate

        # Call should not raise
        record_cycle_success_rate("trading", 0.85)

    def test_cycle_error_count_gauge(self):
        from quantstack.observability.metrics import record_cycle_error_count

        record_cycle_error_count("research", 2)

    def test_strategy_generation_gauge(self):
        from quantstack.observability.metrics import record_strategy_generation

        record_strategy_generation(3)

    def test_research_queue_depth_gauge(self):
        from quantstack.observability.metrics import record_research_queue_depth

        record_research_queue_depth(42)


# ---------------------------------------------------------------------------
# Alert rule YAML validation
# ---------------------------------------------------------------------------

class TestAlertRuleExpressions:
    """Validate that alert rules contain expected PromQL expressions."""

    @pytest.fixture(autouse=True)
    def _load_alerts(self):
        import os
        alerts_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "config", "grafana", "provisioning", "alerting", "alerts.yaml",
        )
        alerts_path = os.path.abspath(alerts_path)
        with open(alerts_path) as f:
            self.alerts_content = f.read()

    def test_success_rate_alert_expression(self):
        assert "quantstack_cycle_success_rate < 0.70" in self.alerts_content

    def test_error_count_alert_expression(self):
        assert "quantstack_cycle_error_count > 3" in self.alerts_content

    def test_no_strategies_alert_expression(self):
        assert "quantstack_strategy_generation_7d == 0" in self.alerts_content

    def test_queue_backlog_alert_expression(self):
        assert "quantstack_research_queue_depth > 50" in self.alerts_content
