# Section 05: Log Aggregation + Alerting

## Goal

Deploy a centralized log aggregation stack -- Fluent Bit, Loki, and Grafana -- into the existing Docker Compose environment. This gives QuantStack a single pane of glass for log search, LogQL-based alert rules, and Discord/email notification channels. The existing supervisor-graph Discord alerting remains as a parallel alerting path (resilient if Grafana goes down).

## Dependencies

- **No upstream dependencies.** This section can be implemented in Batch 1 alongside sections 01, 02, 03, and 09.
- **Blocks section-06 (Monitoring Stack)** -- Grafana and its provisioning directory structure created here are reused by section-06 when it adds Prometheus + cAdvisor.
- **Blocks section-07 (Health Check Granularity)** -- health metrics are displayed in the Grafana dashboard and alerted on via the alerting pipeline created here.

## Background

**Current state:** Loguru writes to stderr (colorized) and `~/.quantstack/logs/quantstack.jsonl` (structured JSON, 50MB rotation, 30-day retention). Docker's json-file driver rotates at 50MB x 5 files per container. There is no centralized log shipping. Logs are lost if the host fails or a container is removed.

**Existing alerting:** Discord webhook client (`src/quantstack/tools/discord/client.py`) and daily digest (`src/quantstack/coordination/daily_digest.py`) already post to Discord. These must remain operational. Grafana alerting supplements them with log-based and metric-based alerts.

**No changes to application logging code.** Fluent Bit reads the Docker json-file driver output (stdout/stderr from containers). The JSONL file continues as a local backup.

## Tests First

All tests are integration tests requiring running Docker Compose services. Place them in `tests/integration/test_log_aggregation.py` and mark with `@pytest.mark.integration`.

```python
# tests/integration/test_log_aggregation.py

import pytest
import subprocess
import requests


@pytest.mark.integration
class TestFluentBitConfig:
    """Verify Fluent Bit config parses without errors."""

    def test_fluent_bit_dry_run(self):
        """Run fluent-bit --dry-run against the config file.
        Asserts exit code 0 and no parse errors in output.
        """


@pytest.mark.integration
class TestLokiAcceptsPush:
    """Verify Loki is running and accepts log pushes."""

    def test_loki_push_endpoint(self):
        """POST a test log entry to /loki/api/v1/push.
        Asserts HTTP 204 (accepted).
        """


@pytest.mark.integration
class TestGrafanaProvisioning:
    """Verify Grafana auto-provisions datasources and alert rules on startup."""

    def test_datasources_provisioned(self):
        """GET /api/datasources returns both 'loki' and 'prometheus' datasources.
        (Prometheus datasource is provisioned here but only used after section-06.)
        """

    def test_alert_rules_provisioned(self):
        """GET /api/v1/provisioning/alert-rules returns the expected LogQL alert rules:
        - CRITICAL log detection
        - Error spike detection
        - Kill switch active
        - Container restart
        """

    def test_discord_contact_point_configured(self):
        """GET /api/v1/provisioning/contact-points includes a Discord contact point."""
```

These tests validate that infrastructure configuration files are correct and that services start in a working state. They are not unit tests -- they require `docker compose up` to be running.

## Implementation

### 2.1 Fluent Bit Collector Sidecar

**New file: `config/fluent-bit/fluent-bit.conf`**

The Fluent Bit configuration must:

- Use the `tail` input plugin to read Docker container JSON logs from `/var/lib/docker/containers/*/*.log`.
- Enable `Docker_Mode On` to handle multi-line log messages (Python tracebacks).
- Use `Tag_Regex` to extract the container name into the log tag.
- Use the `loki` output plugin (Fluent Bit has native Loki support) to push to `http://loki:3100/loki/api/v1/push`.
- Set labels on each log line: `job` (container name), `level` (parsed from structured JSON if present), `graph` (trading/research/supervisor -- derived from container name).

The `[INPUT]` section reads all container logs. The `[OUTPUT]` section forwards to Loki with the label set. A `[FILTER]` section can use the `parser` filter to extract the `level` field from Loguru's JSON output.

**New service in `docker-compose.yml`:**

```yaml
fluent-bit:
  image: fluent/fluent-bit:3.1
  volumes:
    - /var/lib/docker/containers:/var/lib/docker/containers:ro
    - ./config/fluent-bit/fluent-bit.conf:/fluent-bit/etc/fluent-bit.conf:ro
  depends_on:
    - loki
  mem_limit: 64m
  restart: unless-stopped
```

**macOS vs Linux note:** On macOS with Docker Desktop, `/var/lib/docker/containers` is inside the Docker VM and is accessible from within containers. On native Linux, it maps directly. The volume mount works on both but metric fidelity may differ. Document this difference in a comment in the compose file.

### 2.2 Loki Log Backend

**New file: `config/loki/loki-config.yaml`**

Loki runs in single-binary mode (sufficient for a single host). The configuration must include:

- `auth_enabled: false` (single-tenant, local deployment)
- `schema_config` using `boltdb-shipper` for the index and `filesystem` for chunk storage
- `limits_config` with `retention_period: 720h` (30 days)
- `compactor` enabled with `retention_enabled: true` to enforce the retention period
- `server.http_listen_port: 3100`

**New service in `docker-compose.yml`:**

```yaml
loki:
  image: grafana/loki:3.1.0
  volumes:
    - ./config/loki/loki-config.yaml:/etc/loki/loki-config.yaml:ro
    - loki-data:/loki
  command: -config.file=/etc/loki/loki-config.yaml
  ports:
    - "3100:3100"
  mem_limit: 256m
  healthcheck:
    test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:3100/ready"]
    interval: 15s
    timeout: 5s
    retries: 3
  restart: unless-stopped
```

The `loki-data` named volume stores all log data. This volume is critical -- see section 2.5.

### 2.3 Grafana Dashboard and Alerting

**New service in `docker-compose.yml`:**

```yaml
grafana:
  image: grafana/grafana:11.1.0
  volumes:
    - ./config/grafana/provisioning:/etc/grafana/provisioning:ro
    - grafana-data:/var/lib/grafana
  environment:
    - GF_AUTH_ANONYMOUS_ENABLED=true
    - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
    - GF_ALERTING_ENABLED=true
  ports:
    - "3000:3000"
  depends_on:
    - loki
  mem_limit: 256m
  restart: unless-stopped
```

Anonymous auth is enabled because this is a local, single-host deployment not exposed to the internet. The Admin role allows dashboard editing without login.

**New directory tree: `config/grafana/provisioning/`**

```
config/grafana/provisioning/
  datasources/
    datasources.yaml
  alerting/
    alerts.yaml
  dashboards/
    dashboards.yaml
    quantstack.json
```

**`config/grafana/provisioning/datasources/datasources.yaml`** -- provisions two datasources:

- **Loki** at `http://loki:3100` (used immediately by this section)
- **Prometheus** at `http://prometheus:9090` (used after section-06 adds the Prometheus service; Grafana tolerates an unreachable datasource gracefully)

**`config/grafana/provisioning/alerting/alerts.yaml`** -- LogQL alert rules:

| Alert Name | LogQL Expression | For Duration | Severity |
|---|---|---|---|
| CRITICAL log detected | `rate({job=~".*-graph"} \|= "CRITICAL" [5m]) > 0` | 0m | Critical |
| Error spike | `rate({job=~".*-graph"} \|= "ERROR" [5m]) > 5` | 0m | Warning |
| Kill switch active | `quantstack_kill_switch_active == 1` (PromQL, fires after section-06) | 0m | Critical |
| Container restart | `changes(container_start_time_seconds[5m]) > 0` (PromQL, fires after section-06) | 0m | Warning |

The PromQL rules (kill switch active, container restart) are provisioned now but will only fire once Prometheus is scraping targets (section-06). Grafana handles rules against unreachable datasources by marking them as "no data" rather than erroring.

**Contact points:**

- **Discord webhook** -- primary, immediate delivery. Uses the `DISCORD_WEBHOOK_URL` environment variable (same one used by the existing supervisor alerts). Configure via the Grafana contact point provisioning YAML with the webhook type.
- **Email** -- escalation channel (4h+). Mark as a TODO in the provisioning file with a comment: "Configure Google mail bot SMTP credentials to enable email contact point." Do not block on this.

**`config/grafana/provisioning/dashboards/dashboards.yaml`** -- tells Grafana to auto-load dashboards from `/etc/grafana/provisioning/dashboards/`.

**`config/grafana/provisioning/dashboards/quantstack.json`** -- a pre-built Grafana dashboard JSON. At minimum, include panels for:

- Log volume by level (Loki query, bar chart)
- Recent CRITICAL/ERROR logs (Loki query, log panel)
- Placeholder panels for container metrics (will populate after section-06)

The full dashboard with alert panels, container metrics, and trading metrics is completed in section-06. This section provides the foundation panels that work with Loki alone.

### 2.4 Loki Volume Protection

The `loki-data` named volume must be declared in the `volumes:` section of `docker-compose.yml`:

```yaml
volumes:
  loki-data:
  grafana-data:
```

**Critical safeguard:** `docker compose down -v` destroys named volumes including log history. The `stop.sh` script must use `docker compose down` (without `-v`). Add a comment at the top of `stop.sh` warning about this:

```bash
# WARNING: Do NOT use 'docker compose down -v' -- this destroys log history (loki-data)
# and Grafana state (grafana-data). Use 'docker compose down' to preserve volumes.
```

If `stop.sh` currently uses `docker compose down -v`, change it to `docker compose down`.

Similarly, document in `start.sh` that `docker compose down -v` is destructive to observability data.

### 2.5 Parallel Alerting Architecture

This is a design constraint, not a code change. The existing supervisor graph sends Discord alerts for kill switch events, daily digests, and critical operational conditions via direct webhook calls. This path must remain untouched.

Grafana adds a second alerting path for:
- Log pattern detection (CRITICAL messages, error spikes)
- Metric threshold breaches (after section-06)
- Infrastructure events (container restarts, OOM -- after section-06)

If Grafana goes down, supervisor alerts still fire. If the supervisor graph is unhealthy, Grafana alerts still fire (it reads logs/metrics independently). This gives defense-in-depth for alerting.

## New Files Summary

| File | Purpose |
|------|---------|
| `config/fluent-bit/fluent-bit.conf` | Fluent Bit input/output configuration |
| `config/loki/loki-config.yaml` | Loki single-binary configuration |
| `config/grafana/provisioning/datasources/datasources.yaml` | Loki + Prometheus datasource definitions |
| `config/grafana/provisioning/alerting/alerts.yaml` | LogQL and PromQL alert rules |
| `config/grafana/provisioning/dashboards/dashboards.yaml` | Dashboard auto-discovery config |
| `config/grafana/provisioning/dashboards/quantstack.json` | Pre-built system dashboard |
| `tests/integration/test_log_aggregation.py` | Integration tests for the log stack |

## Modified Files Summary

| File | Change |
|------|--------|
| `docker-compose.yml` | Add `fluent-bit`, `loki`, `grafana` services + `loki-data`, `grafana-data` volumes |
| `stop.sh` | Ensure uses `docker compose down` (not `-v`), add warning comment |
| `start.sh` | Add comment warning about `docker compose down -v` destroying observability data |

## Memory Budget

| Service | Memory Limit |
|---------|-------------|
| fluent-bit | 64 MB |
| loki | 256 MB |
| grafana | 256 MB |
| **Total new** | **576 MB** |

This brings the total Docker Compose stack from ~10.4 GB to ~11.0 GB. Ensure the host has at least 16 GB RAM.

## Risks and Mitigations

1. **Fluent Bit log path differs between macOS and Linux.** `/var/lib/docker/containers` is inside Docker Desktop's VM on macOS but maps directly on Linux. The volume mount works in both cases because the mount is from the Docker engine's perspective, not the host OS. Document this in a compose file comment.

2. **Loki disk usage.** With 30-day retention and 3 active graphs logging at moderate volume, expect 1-5 GB of log data. The compactor enforces retention. Monitor via `loki-data` volume size.

3. **Grafana alert rules referencing unavailable datasources.** The PromQL rules (kill switch, container restart) reference Prometheus, which does not exist until section-06. Grafana marks these as "no data" -- they do not error or spam notifications. They will auto-activate once Prometheus is added.

4. **Discord webhook rate limits.** If Grafana fires many alerts simultaneously, Discord may throttle. Grafana's built-in alert grouping and repeat interval (default 4h) mitigates this. Configure `group_wait: 30s` and `group_interval: 5m` in the notification policy.

## Verification Checklist

After implementation, verify:

- [ ] `docker compose up -d` starts fluent-bit, loki, and grafana without errors
- [ ] `curl http://localhost:3100/ready` returns "ready" (Loki health)
- [ ] Grafana UI at `http://localhost:3000` loads without login
- [ ] Grafana datasources page shows Loki (connected) and Prometheus (unreachable -- expected)
- [ ] Grafana Alerting > Alert rules page shows all 4 provisioned rules
- [ ] Grafana Alerting > Contact points page shows Discord contact point
- [ ] Grafana Explore page can query Loki logs: `{job=~".*-graph"}`
- [ ] `stop.sh` does NOT use `docker compose down -v`
- [ ] `pytest tests/integration/test_log_aggregation.py -m integration` passes (with Docker running)
