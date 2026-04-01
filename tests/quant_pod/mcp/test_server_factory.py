# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for server_factory.py — domain-scoped MCP server creation.

Validates:
  - Tool modules are imported and TOOLS collected per domain
  - Domain filtering: tools only register on servers where @domain() matches
  - Cross-cutting tools (analysis.py) appear in all target servers
  - Import failures are isolated — one bad module doesn't crash the server
  - Each Domain enum value maps to at least one module
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

from quantstack.mcp.domains import Domain
from quantstack.mcp.server_factory import (
    _DOMAIN_MODULES,
    _count_tools,
    create_server,
)
from quantstack.mcp.tools._registry import TOOL_DOMAINS
from quantstack.mcp.tools._tool_def import ToolDefinition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_names(server) -> set[str]:
    """Extract registered tool names from a FastMCP server."""
    try:
        return set(server._tool_manager._tools.keys())
    except AttributeError:
        return set()


# ---------------------------------------------------------------------------
# _DOMAIN_MODULES coverage
# ---------------------------------------------------------------------------


class TestDomainModuleMapping:
    """Verify the _DOMAIN_MODULES mapping is complete and consistent."""

    def test_every_domain_has_modules(self):
        """Each Domain enum value should map to at least one module."""
        for d in Domain:
            assert d in _DOMAIN_MODULES, f"Domain {d.name} missing from _DOMAIN_MODULES"
            assert len(_DOMAIN_MODULES[d]) > 0, f"Domain {d.name} has empty module list"

    def test_analysis_is_cross_cutting(self):
        """analysis.py should be listed in every domain (it provides get_system_status, get_regime)."""
        for d in Domain:
            modules = _DOMAIN_MODULES[d]
            assert "analysis" in modules, (
                f"Domain {d.name} missing 'analysis' — "
                f"cross-cutting tools (get_system_status, get_regime) won't be available"
            )

    def test_no_unknown_modules(self):
        """Every module name in _DOMAIN_MODULES should be importable."""
        import importlib.util as _importlib_util

        all_modules = set()
        for modules in _DOMAIN_MODULES.values():
            all_modules.update(modules)

        for mod_name in all_modules:
            fqn = f"quantstack.mcp.tools.{mod_name}"
            try:
                importlib.import_module(fqn)
            except ImportError:
                # qc_fundamentals may fail if API key not set — that's fine,
                # the module itself exists
                try:
                    spec = _importlib_util.find_spec(fqn)
                    assert spec is not None, f"Module {fqn} not found on sys.path"
                except (ModuleNotFoundError, ValueError):
                    pytest.fail(f"Module {fqn} in _DOMAIN_MODULES does not exist")


# ---------------------------------------------------------------------------
# create_server — tool registration
# ---------------------------------------------------------------------------


class TestCreateServer:
    """Test create_server registers the right tools per domain."""

    def test_creates_fastmcp_instance(self):
        server = create_server("test-ml", Domain.ML, "test")
        assert server.name == "test-ml"

    def test_ml_server_has_ml_tools(self):
        """ML server should have tools from ml.py and analysis.py."""
        server = create_server("test-ml", Domain.ML, "ML server")
        names = _tool_names(server)
        # ml.py exports train_ml_model, get_ml_model_status, predict_ml_signal (at minimum)
        assert "train_ml_model" in names
        # analysis.py is cross-cutting
        assert "get_regime" in names or "get_system_status" in names

    def test_execution_server_has_trade_tools(self):
        server = create_server("test-exec", Domain.EXECUTION, "Execution server")
        names = _tool_names(server)
        assert "execute_trade" in names
        assert "close_position" in names
        assert "get_fills" in names

    def test_research_server_has_strategy_tools(self):
        server = create_server("test-research", Domain.RESEARCH, "Research server")
        names = _tool_names(server)
        assert "register_strategy" in names
        assert "run_backtest" in names
        assert "list_strategies" in names

    def test_risk_server_has_risk_tools(self):
        server = create_server("test-risk", Domain.RISK, "Risk server")
        names = _tool_names(server)
        # qc_risk.py should provide at least compute_var
        assert "compute_var" in names

    def test_signals_server_has_signal_tools(self):
        server = create_server("test-signals", Domain.SIGNALS, "Signals server")
        names = _tool_names(server)
        assert "get_signal_brief" in names

    def test_tool_count_is_positive(self):
        """Every domain server should register at least 1 tool."""
        for d in Domain:
            server = create_server(f"test-{d.name.lower()}", d, "test")
            count = _count_tools(server)
            assert count > 0, f"Domain {d.name} server registered 0 tools"


# ---------------------------------------------------------------------------
# Domain filtering
# ---------------------------------------------------------------------------


class TestDomainFiltering:
    """Verify that tools are filtered by domain, not just blindly registered."""

    def test_ml_server_does_not_have_execution_tools(self):
        """ML server should NOT have execute_trade (that's EXECUTION domain)."""
        server = create_server("test-ml", Domain.ML, "ML only")
        names = _tool_names(server)
        assert "execute_trade" not in names
        assert "close_position" not in names

    def test_execution_server_does_not_have_backtest_tools(self):
        """Execution server should NOT have backtesting tools."""
        server = create_server("test-exec", Domain.EXECUTION, "Exec only")
        names = _tool_names(server)
        assert "run_backtest" not in names
        assert "register_strategy" not in names

    def test_risk_server_does_not_have_ml_tools(self):
        server = create_server("test-risk", Domain.RISK, "Risk only")
        names = _tool_names(server)
        assert "train_ml_model" not in names

    def test_cross_cutting_tools_appear_in_multiple_servers(self):
        """get_regime (tagged SIGNALS | INTEL | RESEARCH | ...) should appear in many servers."""
        servers_with_regime = []
        for d in Domain:
            server = create_server(f"test-{d.name}", d, "test")
            names = _tool_names(server)
            if "get_regime" in names:
                servers_with_regime.append(d.name)

        # get_regime is in analysis.py which is listed in every domain
        assert len(servers_with_regime) >= 5, (
            f"get_regime only appeared in {servers_with_regime} — expected 5+ servers"
        )


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    """Verify that import failures in one module don't break the server."""

    def test_broken_module_does_not_crash_server(self):
        """If a module import fails, other tools still register."""
        # Patch importlib.import_module to fail on 'ml' but succeed on others
        real_import = importlib.import_module

        def _selective_fail(fqn, *args, **kwargs):
            if fqn == "quantstack.mcp.tools.ml":
                raise ImportError("simulated failure")
            return real_import(fqn, *args, **kwargs)

        with patch("quantstack.mcp.server_factory.importlib.import_module", side_effect=_selective_fail):
            server = create_server("test-ml", Domain.ML, "ML server")

        names = _tool_names(server)
        # ml.py tools should be absent
        assert "train_ml_model" not in names
        # but analysis.py tools should still be present
        assert "get_regime" in names or "get_system_status" in names

    def test_module_without_tools_export_is_harmless(self):
        """A module that lacks TOOLS attribute should not cause errors."""
        real_import = importlib.import_module

        def _mock_module(fqn, *args, **kwargs):
            if fqn == "quantstack.mcp.tools.ml":
                mock_mod = MagicMock()
                del mock_mod.TOOLS  # simulate missing TOOLS
                return mock_mod
            return real_import(fqn, *args, **kwargs)

        with patch("quantstack.mcp.server_factory.importlib.import_module", side_effect=_mock_module):
            server = create_server("test-ml", Domain.ML, "ML server")

        # Should not crash; analysis tools should still register
        names = _tool_names(server)
        assert "get_regime" in names or "get_system_status" in names


# ---------------------------------------------------------------------------
# TOOL_DOMAINS consistency
# ---------------------------------------------------------------------------


class TestToolDomainsConsistency:
    """Verify TOOL_DOMAINS (from @domain() decorators) is populated correctly."""

    def test_tool_domains_is_populated(self):
        """After importing tool modules, TOOL_DOMAINS should have entries."""
        # Force import of a few key modules
        importlib.import_module("quantstack.mcp.tools.analysis")
        importlib.import_module("quantstack.mcp.tools.execution")
        importlib.import_module("quantstack.mcp.tools.strategy")

        assert len(TOOL_DOMAINS) > 0
        assert "get_regime" in TOOL_DOMAINS
        assert "execute_trade" in TOOL_DOMAINS
        assert "register_strategy" in TOOL_DOMAINS

    def test_domain_tags_are_valid_flags(self):
        """Every value in TOOL_DOMAINS should be a valid Domain flag."""
        for name, domain_flag in TOOL_DOMAINS.items():
            assert isinstance(domain_flag, Domain), (
                f"TOOL_DOMAINS[{name!r}] = {domain_flag!r} is not a Domain flag"
            )

    def test_cross_cutting_tools_have_multiple_domains(self):
        """analysis.py tools should be tagged with multiple domains."""
        importlib.import_module("quantstack.mcp.tools.analysis")
        regime_domain = TOOL_DOMAINS.get("get_regime")
        if regime_domain is not None:
            # get_regime should have at least 2 domain bits set
            bits = bin(regime_domain.value).count("1")
            assert bits >= 2, (
                f"get_regime has {bits} domain bit(s) — expected cross-cutting (2+)"
            )
