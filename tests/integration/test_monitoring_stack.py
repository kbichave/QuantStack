"""Integration tests for the container monitoring stack (cAdvisor + Prometheus + Grafana).

Requires running Docker Compose services.
Run with: pytest -m integration tests/integration/test_monitoring_stack.py
"""

import pytest
import requests

pytestmark = pytest.mark.integration

PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"


def test_prometheus_scrapes_cadvisor():
    """Prometheus successfully scrapes cAdvisor metrics."""
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": 'up{job="cadvisor"}'},
        timeout=5,
    )
    result = resp.json()["data"]["result"]
    assert len(result) > 0
    assert result[0]["value"][1] == "1"


def test_prometheus_scrapes_trading_graph():
    """Prometheus successfully scrapes trading-graph application metrics."""
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": 'up{job="trading-graph"}'},
        timeout=5,
    )
    result = resp.json()["data"]["result"]
    assert len(result) > 0
    assert result[0]["value"][1] == "1"


def test_grafana_prometheus_datasource_configured():
    """Grafana has a Prometheus datasource provisioned."""
    resp = requests.get(f"{GRAFANA_URL}/api/datasources", timeout=5)
    datasources = resp.json()
    names = [ds["name"] for ds in datasources]
    assert "Prometheus" in names


def test_oom_alert_rule_exists():
    """Grafana has an OOMKilled alert rule provisioned."""
    resp = requests.get(
        f"{GRAFANA_URL}/api/v1/provisioning/alert-rules", timeout=5
    )
    rules = resp.json()
    titles = [r["title"] for r in rules]
    assert any("OOM" in t for t in titles)


def test_memory_warning_alert_rule_exists():
    """Grafana has a memory warning alert rule provisioned."""
    resp = requests.get(
        f"{GRAFANA_URL}/api/v1/provisioning/alert-rules", timeout=5
    )
    rules = resp.json()
    titles = [r["title"] for r in rules]
    assert any("Memory" in t and "warning" in t.lower() for t in titles)


def test_alert_list_panel_on_dashboard():
    """Dashboard has an alert list panel at the top."""
    resp = requests.get(
        f"{GRAFANA_URL}/api/search", params={"query": "QuantStack"}, timeout=5
    )
    dashboards = resp.json()
    assert len(dashboards) > 0
    uid = dashboards[0]["uid"]
    dash_resp = requests.get(f"{GRAFANA_URL}/api/dashboards/uid/{uid}", timeout=5)
    panels = dash_resp.json()["dashboard"]["panels"]
    alert_panels = [p for p in panels if p.get("type") == "alertlist"]
    assert len(alert_panels) >= 1


def test_alert_history_panel_on_dashboard():
    """Dashboard has an alert history / state-timeline panel."""
    resp = requests.get(
        f"{GRAFANA_URL}/api/search", params={"query": "QuantStack"}, timeout=5
    )
    dashboards = resp.json()
    uid = dashboards[0]["uid"]
    dash_resp = requests.get(f"{GRAFANA_URL}/api/dashboards/uid/{uid}", timeout=5)
    panels = dash_resp.json()["dashboard"]["panels"]
    state_panels = [
        p for p in panels if p.get("type") in ("state-timeline", "alertlist")
    ]
    assert len(state_panels) >= 2
