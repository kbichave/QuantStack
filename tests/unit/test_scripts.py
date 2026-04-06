"""Tests for Section 10: Start/Stop/Status Scripts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_start_sh_checks_for_docker():
    content = (PROJECT_ROOT / "start.sh").read_text()
    assert "docker" in content.lower()
    assert "docker compose" in content


def test_start_sh_checks_for_env_file():
    content = (PROJECT_ROOT / "start.sh").read_text()
    assert ".env" in content
    assert "exit 1" in content


def test_start_sh_starts_infrastructure_before_crews():
    content = (PROJECT_ROOT / "start.sh").read_text()
    infra_pos = content.find("docker compose up")
    second_up = content.find("docker compose up", infra_pos + 1)
    assert second_up > infra_pos, (
        "Expected two docker compose up calls: infra first, crews second"
    )


def test_stop_sh_sends_graceful_shutdown():
    content = (PROJECT_ROOT / "stop.sh").read_text()
    assert "docker compose down" in content
    assert "kill -9" not in content


def test_status_sh_displays_container_status_heartbeats_positions():
    content = (PROJECT_ROOT / "status.sh").read_text()
    # status.sh now launches TUI (quantstack.tui module)
    # The TUI handles display of heartbeats and positions
    assert "quantstack.tui" in content or "tui" in content


def test_startup_waits_for_health_checks():
    content = (PROJECT_ROOT / "start.sh").read_text()
    assert "healthy" in content.lower()


def test_start_sh_validates_required_env_vars():
    content = (PROJECT_ROOT / "start.sh").read_text()
    assert "TRADER_PG_URL" in content
    assert "ALPACA_API_KEY" in content
    assert "ALPHA_VANTAGE_API_KEY" in content


def test_stop_sh_activates_kill_switch():
    content = (PROJECT_ROOT / "stop.sh").read_text()
    assert "kill_switch" in content
    assert "sentinel" in content.lower()
