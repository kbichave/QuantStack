# Section 6: Container Monitoring Stack

## Overview

Add cAdvisor and Prometheus to the Docker Compose stack so that container-level resource metrics (CPU, memory, OOM events, restarts) are collected and queryable. Wire Prometheus as a Grafana datasource alongside Loki (from section-05). Define PromQL alert rules for memory pressure, OOM kills, and container restarts. Build a Grafana dashboard with alert panels at the top and operational panels below.

This section depends on **section-05-log-aggregation** because it shares the Grafana service defined there. Grafana is added to `docker-compose.yml` in section-05; this section extends it with a Prometheus datasource and PromQL alert rules.

## Background

QuantStack runs 3 LangGraph graph services (trading-graph, research-graph, supervisor-graph) plus supporting infrastructure (postgres, langfuse, ollama, finrl-worker) as Docker Compose services. All graph containers have `mem_limit` set in `docker-compose.yml`, but nothing monitors actual usage or detects OOM kills. The application already exposes Prometheus metrics at `GET /metrics` on each graph service (trades, risk rejections, latency, NAV, P&L, kill switch state) via `src/quantstack/observability/metrics.py` using `prometheus_client`. No Prometheus server currently scrapes these endpoints.

## Tests

All tests are integration tests requiring Docker Compose services to be running. Mark with `@pytest.mark.integration`.

```python
# tests/integration/test_monitoring_stack.py

import pytest
import requests

pytestmark = pytest.mark.integration

PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"


def test_prometheus_scrapes_cadvisor():
    """Prometheus successfully scrapes cAdvisor metrics."""
    # Query: up{job="cadvisor"} == 1
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": 'up{job="cadvisor"}'},
    )
    result = resp.json()["data"]["result"]
    assert len(result) > 0
    assert result[0]["value"][1] == "1"


def test_prometheus_scrapes_trading_graph():
    """Prometheus successfully scrapes trading-graph application metrics."""
    # Query: up{job="trading-graph"} == 1
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": 'up{job="trading-graph"}'},
    )
    result = resp.json()["data"]["result"]
    assert len(result) > 0
    assert result[0]["value"][1] == "1"


def test_grafana_prometheus_datasource_configured():
    """Grafana has a Prometheus datasource provisioned."""
    resp = requests.get(f"{GRAFANA_URL}/api/datasources")
    datasources = resp.json()
    names = [ds["name"] for ds in datasources]
    assert "Prometheus" in names


def test_oom_alert_rule_exists():
    """Grafana has an OOMKilled alert rule provisioned."""
    resp = requests.get(f"{GRAFANA_URL}/api/v1/provisioning/alert-rules")
    rules = resp.json()
    titles = [r["title"] for r in rules]
    assert any("OOM" in t for t in titles)


def test_memory_warning_alert_rule_exists():
    """Grafana has a memory warning alert rule provisioned."""
    resp = requests.get(f"{GRAFANA_URL}/api/v1/provisioning/alert-rules")
    rules = resp.json()
    titles = [r["title"] for r in rules]
    assert any("Memory" in t and "warning" in t.lower() for t in titles)


def test_alert_list_panel_on_dashboard():
    """Dashboard has an alert list panel at the top."""
    resp = requests.get(
        f"{GRAFANA_URL}/api/search", params={"query": "QuantStack"}
    )
    dashboards = resp.json()
    assert len(dashboards) > 0
    uid = dashboards[0]["uid"]
    dash_resp = requests.get(f"{GRAFANA_URL}/api/dashboards/uid/{uid}")
    panels = dash_resp.json()["dashboard"]["panels"]
    # First panel should be alert-oriented
    alert_panels = [p for p in panels if p.get("type") == "alertlist"]
    assert len(alert_panels) >= 1


def test_alert_history_panel_on_dashboard():
    """Dashboard has an alert history/state-timeline panel."""
    resp = requests.get(
        f"{GRAFANA_URL}/api/search", params={"query": "QuantStack"}
    )
    dashboards = resp.json()
    uid = dashboards[0]["uid"]
    dash_resp = requests.get(f"{GRAFANA_URL}/api/dashboards/uid/{uid}")
    panels = dash_resp.json()["dashboard"]["panels"]
    state_panels = [
        p for p in panels if p.get("type") in ("state-timeline", "alertlist")
    ]
    assert len(state_panels) >= 2  # alert list + alert history
```

## Implementation

### 6.1 cAdvisor Service

Add to `docker-compose.yml`:

```yaml
cadvisor:
  image: gcr.io/cadvisor/cadvisor:v0.49.1
  container_name: quantstack-cadvisor
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock:ro
    - /sys:/sys:ro
    - /var/lib/docker:/var/lib/docker:ro
  ports:
    - "127.0.0.1:8080:8080"
  mem_limit: 128m
  memswap_limit: 128m
  restart: unless-stopped
  networks:
    - quantstack-net
  logging:
    driver: json-file
    options:
      max-size: "10m"
      max-file: "3"
```

cAdvisor provides container-level CPU, memory (working set + cache), network, disk I/O, and `container_oom_events_total`. Port 8080 is bound to localhost only -- it is scraped by Prometheus internally, not exposed to the network.

Platform note: on macOS Docker Desktop, cAdvisor runs inside the Linux VM. Some metrics (particularly CPU) may behave differently than on a native Linux host. Memory metrics work correctly. Document this in a comment in `docker-compose.yml`.

### 6.2 Prometheus Service

Add to `docker-compose.yml`:

```yaml
prometheus:
  image: prom/prometheus:v2.53.0
  container_name: quantstack-prometheus
  command:
    - "--config.file=/etc/prometheus/prometheus.yml"
    - "--storage.tsdb.retention.time=15d"
    - "--web.enable-lifecycle"
  volumes:
    - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    - prometheus-data:/prometheus
  ports:
    - "127.0.0.1:9090:9090"
  mem_limit: 256m
  memswap_limit: 256m
  restart: unless-stopped
  depends_on:
    cadvisor:
      condition: service_started
  networks:
    - quantstack-net
  logging:
    driver: json-file
    options:
      max-size: "10m"
      max-file: "3"
```

Add `prometheus-data` to the `volumes:` section at the bottom of `docker-compose.yml`:

```yaml
volumes:
  # ... existing volumes ...
  prometheus-data:
```

Retention is 15 days. Container metrics are high-cardinality; 15 days is sufficient for operational monitoring. Application-level metrics (trades, P&L) have the same retention -- acceptable since Langfuse stores the authoritative event history.

### 6.3 Prometheus Scrape Configuration

Create `config/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: cadvisor
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: trading-graph
    static_configs:
      - targets: ['trading-graph:8000']
    metrics_path: /metrics

  - job_name: research-graph
    static_configs:
      - targets: ['research-graph:8000']
    metrics_path: /metrics

  - job_name: supervisor-graph
    static_configs:
      - targets: ['supervisor-graph:8000']
    metrics_path: /metrics
```

All targets use Docker Compose service names (DNS resolution within the `quantstack-net` network). Port 8000 is the FastAPI HTTP port that each graph service exposes for health checks and the `/metrics` endpoint.

### 6.4 Grafana Prometheus Datasource

Extend the datasources provisioning file created in section-05. The file at `config/grafana/provisioning/datasources/datasources.yaml` should contain both Loki and Prometheus:

```yaml
apiVersion: 1

datasources:
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    isDefault: false

  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

Prometheus is the default datasource since most operational alerting uses PromQL.

### 6.5 PromQL Alert Rules

Add to `config/grafana/provisioning/alerting/alerts.yaml` (extending the LogQL rules from section-05):

```yaml
# PromQL alert rules (container monitoring)
- name: Container Memory Warning
  condition: B
  data:
    - refId: A
      datasourceUid: prometheus
      model:
        expr: >
          container_memory_working_set_bytes{name=~"quantstack-.*-graph"}
          / container_spec_memory_limit_bytes{name=~"quantstack-.*-graph"}
          > 0.80
    - refId: B
      datasourceUid: __expr__
      model:
        type: reduce
        expression: A
        reducer: last
  for: 2m
  labels:
    severity: warning

- name: Container Memory Critical
  condition: B
  data:
    - refId: A
      datasourceUid: prometheus
      model:
        expr: >
          container_memory_working_set_bytes{name=~"quantstack-.*-graph"}
          / container_spec_memory_limit_bytes{name=~"quantstack-.*-graph"}
          > 0.90
    - refId: B
      datasourceUid: __expr__
      model:
        type: reduce
        expression: A
        reducer: last
  for: 1m
  labels:
    severity: critical

- name: OOMKilled
  condition: B
  data:
    - refId: A
      datasourceUid: prometheus
      model:
        expr: >
          increase(container_oom_events_total{name=~"quantstack-.*-graph|quantstack-postgres|quantstack-langfuse"}[5m]) > 0
    - refId: B
      datasourceUid: __expr__
      model:
        type: reduce
        expression: A
        reducer: last
  for: 0m
  labels:
    severity: critical

- name: Container Restart
  condition: B
  data:
    - refId: A
      datasourceUid: prometheus
      model:
        expr: >
          changes(container_start_time_seconds{name=~"quantstack-.*"}[5m]) > 0
    - refId: B
      datasourceUid: __expr__
      model:
        type: reduce
        expression: A
        reducer: last
  for: 0m
  labels:
    severity: warning
```

The choice of `container_memory_working_set_bytes` over `container_memory_usage_bytes` is deliberate: working set excludes reclaimable cache and is the metric the kernel OOM killer evaluates. Using `usage_bytes` would cause false alerts when containers have large but reclaimable page caches.

Container name filters use `quantstack-` prefix to match the `container_name` values set in `docker-compose.yml`. OOM monitoring covers graph services plus postgres and langfuse (the stateful services where data loss would be most damaging).

### 6.6 Grafana Dashboard

Create `config/grafana/provisioning/dashboards/quantstack.json`. This is a Grafana dashboard JSON model. The key structural decisions:

**Panel layout (top to bottom):**

1. **Active Alerts panel** (type: `alertlist`) -- shows all currently firing alerts with severity badges. This is the first thing visible when opening the dashboard.
2. **Alert History panel** (type: `state-timeline`) -- time series of alert state transitions over the last 24 hours, showing when alerts fired and resolved.
3. **Kill Switch Status** (type: `stat`) -- single stat panel showing `quantstack_kill_switch_active`. Color mapping: 0 = green ("INACTIVE"), 1 = red ("ACTIVE").
4. **Container Memory Usage** (type: `bargauge`) -- `container_memory_working_set_bytes / container_spec_memory_limit_bytes * 100` per service. Thresholds: green < 70%, yellow 70-85%, red > 85%.
5. **Container CPU Usage** (type: `timeseries`) -- `rate(container_cpu_usage_seconds_total[5m])` per service.
6. **OOM Events** (type: `stat`) -- `sum(container_oom_events_total)`. Should be 0. Red if > 0.
7. **Trading Metrics row:**
   - Trades executed: `sum(quantstack_trades_executed_total)`
   - Risk rejections: `sum(quantstack_risk_rejections_total)`
   - Agent latency p50/p95: `histogram_quantile(0.5, quantstack_agent_latency_seconds_bucket)` / `histogram_quantile(0.95, ...)`
   - Portfolio NAV: `quantstack_portfolio_nav_dollars`
   - Daily P&L: `quantstack_daily_pnl_dollars`
8. **Log Volume by Level** (type: `timeseries`, Loki datasource) -- `sum by (level) (rate({job=~".*-graph"} | json | __error__="" [5m]))`. Shows ERROR/WARNING/INFO rates over time.

The dashboard JSON should be auto-provisioned via `config/grafana/provisioning/dashboards/dashboards.yaml`:

```yaml
apiVersion: 1

providers:
  - name: QuantStack
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
      foldersFromFilesStructure: false
```

The full `quantstack.json` file will be ~300-500 lines of Grafana JSON. It should be generated once and then maintained by hand. The structural decisions above are the spec; the implementer builds the JSON to match.

### 6.7 Memory Budget

New services added by this section:

| Service | Memory Limit |
|---------|-------------|
| cAdvisor | 128 MB |
| Prometheus | 256 MB |
| **Total new** | **384 MB** |

Grafana (256 MB) is added in section-05. Combined with section-05's additions (Fluent Bit 64 MB, Loki 256 MB, Grafana 256 MB), the full monitoring + logging stack adds ~960 MB. The host needs at least 16 GB RAM for the complete Docker Compose stack (~11 GB total).

## Key Files

| File | Action | Description |
|------|--------|-------------|
| `docker-compose.yml` | Modify | Add `cadvisor` and `prometheus` services, add `prometheus-data` volume |
| `config/prometheus/prometheus.yml` | Create | Scrape config for cAdvisor + 3 graph services |
| `config/grafana/provisioning/datasources/datasources.yaml` | Modify | Add Prometheus datasource (Loki already added by section-05) |
| `config/grafana/provisioning/alerting/alerts.yaml` | Modify | Add PromQL alert rules for memory, OOM, restarts (LogQL rules from section-05 already present) |
| `config/grafana/provisioning/dashboards/dashboards.yaml` | Create | Dashboard auto-provisioning config |
| `config/grafana/provisioning/dashboards/quantstack.json` | Create | Pre-built operational dashboard |

## Dependencies

- **section-05-log-aggregation** (required): Provides the Grafana service in `docker-compose.yml`, the `config/grafana/provisioning/` directory structure, the Loki datasource, and LogQL alert rules. This section extends those artifacts -- it does not create Grafana from scratch.

## Risks and Mitigations

**cAdvisor on macOS Docker Desktop:** cAdvisor is primarily Linux-oriented. On macOS, Docker runs inside a Linux VM and some metrics (particularly CPU) may not report accurately. Memory metrics work. Mitigation: test on the Linux deployment target; document macOS limitations as a comment in `docker-compose.yml`.

**Prometheus disk usage:** With 15-day retention and 15-second scrape interval across 4 jobs (~20 targets including cAdvisor container metrics), expect roughly 500 MB - 1 GB of TSDB storage. The `prometheus-data` volume should not be included in cleanup scripts. Same as Loki: `docker compose down -v` destroys metric history.

**Alert noise during development:** Container restarts are frequent during development (code changes, docker-compose restarts). The "Container Restart" alert will fire constantly in dev. Mitigation: either silence the alert in dev via a Grafana silence rule, or add a `for: 5m` duration so it only fires on unexpected repeated restarts. The current spec uses `for: 0m` (immediate) which is correct for production but noisy in dev. The implementer should decide based on workflow.

**Grafana provisioning ordering:** Grafana provisioning loads datasources before alert rules. If the Prometheus datasource UID in alert rules doesn't match the provisioned datasource, alerts will fail silently. Mitigation: use the datasource `name` field for lookups rather than hardcoded UIDs, or ensure the UID is set explicitly in the datasource provisioning YAML.
