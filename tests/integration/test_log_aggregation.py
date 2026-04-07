"""Integration tests for the log aggregation stack (Fluent Bit + Loki + Grafana).

Requires running Docker Compose services.
Run with: pytest -m integration tests/integration/test_log_aggregation.py
"""

import subprocess

import pytest
import requests

pytestmark = pytest.mark.integration


class TestFluentBitConfig:
    """Verify Fluent Bit config parses without errors."""

    def test_fluent_bit_dry_run(self):
        """Run fluent-bit --dry-run against the config file."""
        result = subprocess.run(
            [
                "docker", "compose", "exec", "fluent-bit",
                "/fluent-bit/bin/fluent-bit",
                "--dry-run",
                "-c", "/fluent-bit/etc/fluent-bit.conf",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Fluent Bit dry-run failed: {result.stderr}"


class TestLokiAcceptsPush:
    """Verify Loki is running and accepts log pushes."""

    def test_loki_ready(self):
        """Loki /ready endpoint returns 'ready'."""
        resp = requests.get("http://localhost:3101/ready", timeout=5)
        assert resp.status_code == 200

    def test_loki_push_endpoint(self):
        """POST a test log entry to /loki/api/v1/push. Asserts HTTP 204."""
        import json
        import time

        payload = {
            "streams": [
                {
                    "stream": {"job": "test", "level": "info"},
                    "values": [
                        [str(int(time.time() * 1e9)), "integration test log entry"]
                    ],
                }
            ]
        }
        resp = requests.post(
            "http://localhost:3101/loki/api/v1/push",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        assert resp.status_code == 204


class TestGrafanaProvisioning:
    """Verify Grafana auto-provisions datasources and alert rules on startup."""

    def test_datasources_provisioned(self):
        """GET /api/datasources returns Loki datasource."""
        resp = requests.get("http://localhost:3000/api/datasources", timeout=5)
        assert resp.status_code == 200
        names = {ds["name"] for ds in resp.json()}
        assert "Loki" in names

    def test_alert_rules_provisioned(self):
        """GET /api/v1/provisioning/alert-rules returns expected alert rules."""
        resp = requests.get(
            "http://localhost:3000/api/v1/provisioning/alert-rules", timeout=5
        )
        assert resp.status_code == 200
        titles = {rule["title"] for rule in resp.json()}
        assert "CRITICAL log detected" in titles
        assert "Error spike" in titles

    def test_discord_contact_point_configured(self):
        """GET /api/v1/provisioning/contact-points includes Discord."""
        resp = requests.get(
            "http://localhost:3000/api/v1/provisioning/contact-points", timeout=5
        )
        assert resp.status_code == 200
        types = {cp["type"] for cp in resp.json()}
        assert "discord" in types
