"""Tests for Docker resource limits, log rotation, and Langfuse retention."""

import re
import yaml
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_mem(value: str) -> int:
    """Parse Docker memory string to bytes."""
    value = value.strip().lower()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(b|k|kb|m|mb|g|gb)?$", value)
    if not m:
        raise ValueError(f"Cannot parse memory value: {value}")
    num = float(m.group(1))
    unit = (m.group(2) or "b").rstrip("b") or "b"
    multipliers = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}
    return int(num * multipliers[unit])


@pytest.fixture
def compose_config():
    path = PROJECT_ROOT / "docker-compose.yml"
    return yaml.safe_load(path.read_text())


EXPECTED_LIMITS = {
    "postgres": "512m",
    "langfuse-db": "256m",
    "langfuse": "512m",
    "ollama": "4g",
    "trading-graph": "1g",
    "research-graph": "1g",
    "supervisor-graph": "512m",
}


class TestDockerResourceLimits:
    def test_all_services_have_memory_limits(self, compose_config):
        services = compose_config.get("services", {})
        for name, svc in services.items():
            # Skip legacy/optional profile services
            if svc.get("profiles"):
                continue
            assert "mem_limit" in svc, f"Service '{name}' missing mem_limit"

    def test_total_memory_under_10gb(self, compose_config):
        services = compose_config.get("services", {})
        total = 0
        for name, svc in services.items():
            if svc.get("profiles"):
                continue
            mem = svc.get("mem_limit")
            if mem:
                total += _parse_mem(str(mem))
        ten_gb = 10 * 1024**3
        assert total <= ten_gb, f"Total memory {total / 1024**3:.1f}GB exceeds 10GB"

    def test_expected_per_service_limits(self, compose_config):
        services = compose_config.get("services", {})
        for svc_name, expected in EXPECTED_LIMITS.items():
            svc = services.get(svc_name, {})
            actual = str(svc.get("mem_limit", ""))
            assert actual == expected, (
                f"Service '{svc_name}': expected mem_limit={expected}, got {actual}"
            )


class TestLogRotation:
    def test_all_services_have_logging_config(self, compose_config):
        services = compose_config.get("services", {})
        for name, svc in services.items():
            if svc.get("profiles"):
                continue
            logging_cfg = svc.get("logging", {})
            assert logging_cfg.get("driver") == "json-file", (
                f"Service '{name}' missing json-file logging driver"
            )
            opts = logging_cfg.get("options", {})
            assert "max-size" in opts, f"Service '{name}' missing log max-size"
            assert "max-file" in opts, f"Service '{name}' missing log max-file"

    def test_log_max_size_is_50m(self, compose_config):
        services = compose_config.get("services", {})
        for name, svc in services.items():
            if svc.get("profiles"):
                continue
            opts = svc.get("logging", {}).get("options", {})
            assert opts.get("max-size") == "50m", (
                f"Service '{name}': expected log max-size=50m, got {opts.get('max-size')}"
            )

    def test_log_max_file_is_5(self, compose_config):
        services = compose_config.get("services", {})
        for name, svc in services.items():
            if svc.get("profiles"):
                continue
            opts = svc.get("logging", {}).get("options", {})
            assert opts.get("max-file") == "5", (
                f"Service '{name}': expected log max-file=5, got {opts.get('max-file')}"
            )
