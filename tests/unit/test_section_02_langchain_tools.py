"""Tests for Section 2: LangChain tools are bridge-free and import correctly."""

import importlib


def test_no_bridge_imports_in_langchain_tools():
    """No langchain tool module should import from mcp_bridge."""
    import pathlib
    langchain_dir = pathlib.Path("src/quantstack/tools/langchain")
    for py_file in langchain_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        text = py_file.read_text()
        assert "mcp_bridge" not in text, f"{py_file.name} still imports mcp_bridge"
        assert "get_bridge" not in text, f"{py_file.name} still uses get_bridge"


def test_risk_tools_importable():
    mod = importlib.import_module("quantstack.tools.langchain.risk_tools")
    assert hasattr(mod, "compute_risk_metrics")
    assert hasattr(mod, "compute_position_size")


def test_ml_tools_importable():
    mod = importlib.import_module("quantstack.tools.langchain.ml_tools")
    assert hasattr(mod, "train_model")
    assert hasattr(mod, "compute_features")


def test_options_tools_importable():
    mod = importlib.import_module("quantstack.tools.langchain.options_tools")
    assert hasattr(mod, "fetch_options_chain")
    assert hasattr(mod, "compute_greeks")


def test_execution_tools_importable():
    mod = importlib.import_module("quantstack.tools.langchain.execution_tools")
    assert hasattr(mod, "execute_order")


def test_data_tools_importable():
    mod = importlib.import_module("quantstack.tools.langchain.data_tools")
    assert hasattr(mod, "fetch_market_data")
    assert hasattr(mod, "fetch_fundamentals")
    assert hasattr(mod, "fetch_earnings_data")


def test_signal_tools_importable():
    mod = importlib.import_module("quantstack.tools.langchain.signal_tools")
    assert hasattr(mod, "signal_brief")
    assert hasattr(mod, "multi_signal_brief")


def test_strategy_tools_importable():
    mod = importlib.import_module("quantstack.tools.langchain.strategy_tools")
    assert hasattr(mod, "fetch_strategy_registry")


def test_portfolio_tools_importable():
    mod = importlib.import_module("quantstack.tools.langchain.portfolio_tools")
    assert hasattr(mod, "fetch_portfolio")


def test_learning_tools_importable():
    mod = importlib.import_module("quantstack.tools.langchain.learning_tools")
    assert hasattr(mod, "search_knowledge_base")


def test_backtest_tools_importable():
    mod = importlib.import_module("quantstack.tools.langchain.backtest_tools")
    assert hasattr(mod, "run_backtest")


def test_intelligence_tools_importable():
    mod = importlib.import_module("quantstack.tools.langchain.intelligence_tools")
    assert hasattr(mod, "web_search")


def test_execution_tools_has_helpers():
    """execute_order uses private helpers that must exist in module."""
    mod = importlib.import_module("quantstack.tools.langchain.execution_tools")
    assert callable(mod._calc_quantity_from_size)


def test_calc_quantity_from_size():
    from quantstack.tools.langchain.execution_tools import _calc_quantity_from_size
    qty = _calc_quantity_from_size("quarter", 100000.0, 150.0)
    # quarter = 2.5% of equity / price = 2500 / 150 = 16
    assert qty == 16

    qty_full = _calc_quantity_from_size("full", 100000.0, 150.0)
    # full = 10% of equity / price = 10000 / 150 = 66
    assert qty_full == 66

    # Zero price should return 0
    assert _calc_quantity_from_size("full", 100000.0, 0.0) == 0
