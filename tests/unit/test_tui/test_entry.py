"""Unit tests for TUI entry points."""
import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestEntryPoints:
    def test_dashboard_script_imports_from_tui(self):
        """scripts/dashboard.py imports from quantstack.tui, not quantstack.dashboard."""
        script = PROJECT_ROOT / "scripts" / "dashboard.py"
        tree = ast.parse(script.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        assert any("quantstack.tui" in imp for imp in imports), (
            "dashboard.py should import from quantstack.tui"
        )
        assert not any("quantstack.dashboard" in imp for imp in imports), (
            "dashboard.py should not import from quantstack.dashboard"
        )

    def test_main_module_is_importable(self):
        """quantstack.tui.__main__ can be imported without error."""
        import importlib
        mod = importlib.import_module("quantstack.tui.__main__")
        assert hasattr(mod, "main")

    def test_tui_package_exports_app(self):
        """quantstack.tui exports QuantStackApp."""
        from quantstack.tui import QuantStackApp
        assert QuantStackApp is not None


class TestDockerHealth:
    def test_fetch_docker_health_returns_list(self):
        """fetch_docker_health returns a list of ServiceHealth (no DB needed)."""
        from quantstack.tui.queries.system import fetch_docker_health
        result = fetch_docker_health()
        assert isinstance(result, list)
        assert len(result) >= 0

    def test_probe_port_handles_closed(self):
        """_probe_port returns False for a port nothing listens on."""
        from quantstack.tui.queries.system import _probe_port
        assert _probe_port("localhost", 19999, timeout=0.1) is False

    def test_service_health_dataclass(self):
        """ServiceHealth is a proper dataclass."""
        from quantstack.tui.queries.system import ServiceHealth
        sh = ServiceHealth(name="test", status="running", port=5432)
        assert sh.name == "test"
        assert sh.status == "running"
