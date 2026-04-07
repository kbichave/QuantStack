"""Tests for non-root container hardening (Section 05).

These tests require Docker and are tagged as integration tests.
They validate that the Dockerfile USER directive, volume ownership,
and init: true settings are effective.
"""

import subprocess

import pytest

pytestmark = pytest.mark.integration


def _docker_run(*args: str, image: str = "quantstack") -> subprocess.CompletedProcess:
    """Run a command inside the quantstack Docker image."""
    return subprocess.run(
        ["docker", "run", "--rm", image, *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_container_runs_as_non_root():
    """Container process runs as 'quantstack' user, not root."""
    result = _docker_run("whoami")
    assert result.returncode == 0
    assert result.stdout.strip() == "quantstack"


def test_application_can_write_logs():
    """The quantstack user can write to /app/logs inside the container."""
    result = _docker_run("touch", "/app/logs/test.log")
    assert result.returncode == 0, f"Cannot write to /app/logs: {result.stderr}"


def test_kill_switch_sentinel_writable():
    """The quantstack user can create the kill switch sentinel file."""
    result = _docker_run("touch", "/data/quantstack/KILL_SWITCH_ACTIVE")
    assert result.returncode == 0, f"Cannot write sentinel: {result.stderr}"


def test_init_prevents_zombie_processes():
    """With init: true, PID 1 is tini/docker-init, not the application.

    Note: --init flag simulates docker-compose init: true.
    """
    result = subprocess.run(
        ["docker", "run", "--rm", "--init", "quantstack",
         "cat", "/proc/1/cmdline"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    # PID 1 should be docker-init or tini, not python
    cmdline = result.stdout.replace("\x00", " ").strip()
    assert "init" in cmdline.lower() or "tini" in cmdline.lower(), (
        f"PID 1 is not an init process: {cmdline}"
    )
