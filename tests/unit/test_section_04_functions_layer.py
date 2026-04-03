"""Tests for Section 4: Functions layer after bridge removal."""

import ast
import importlib
import inspect
from pathlib import Path

import pytest

FUNCTIONS_DIR = Path("src/quantstack/tools/functions")

FUNCTION_MODULES = [
    "quantstack.tools.functions.data_functions",
    "quantstack.tools.functions.execution_functions",
    "quantstack.tools.functions.risk_functions",
    "quantstack.tools.functions.system_functions",
]


@pytest.mark.parametrize("module_path", FUNCTION_MODULES)
def test_no_bridge_imports(module_path):
    """No functions/ module should import get_bridge or MCPBridge."""
    source_file = FUNCTIONS_DIR / (module_path.split(".")[-1] + ".py")
    source = source_file.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_str = ast.dump(node)
            assert "mcp_bridge" not in import_str, (
                f"{module_path} still imports from mcp_bridge"
            )


@pytest.mark.parametrize("module_path", FUNCTION_MODULES)
def test_modules_importable(module_path):
    """All function modules import successfully."""
    importlib.import_module(module_path)


def test_data_functions_has_get_regime():
    mod = importlib.import_module("quantstack.tools.functions.data_functions")
    assert callable(mod.get_regime)
    assert inspect.iscoroutinefunction(mod.get_regime)


def test_data_functions_has_get_portfolio_state():
    mod = importlib.import_module("quantstack.tools.functions.data_functions")
    assert callable(mod.get_portfolio_state)
    assert inspect.iscoroutinefunction(mod.get_portfolio_state)


def test_execution_functions_has_submit_order():
    mod = importlib.import_module("quantstack.tools.functions.execution_functions")
    assert callable(mod.submit_order)
    assert inspect.iscoroutinefunction(mod.submit_order)


def test_execution_functions_has_close_position():
    mod = importlib.import_module("quantstack.tools.functions.execution_functions")
    assert callable(mod.close_position)
    assert inspect.iscoroutinefunction(mod.close_position)


def test_risk_functions_has_validate_risk_gate():
    mod = importlib.import_module("quantstack.tools.functions.risk_functions")
    assert callable(mod.validate_risk_gate)
    assert inspect.iscoroutinefunction(mod.validate_risk_gate)


def test_system_functions_has_all_functions():
    mod = importlib.import_module("quantstack.tools.functions.system_functions")
    for fn_name in ("check_system_status", "check_heartbeat", "record_heartbeat"):
        assert callable(getattr(mod, fn_name))
        assert inspect.iscoroutinefunction(getattr(mod, fn_name))


@pytest.mark.parametrize("module_path", FUNCTION_MODULES)
def test_all_functions_return_dict(module_path):
    """Every public async function returns dict, not str."""
    mod = importlib.import_module(module_path)
    for name, obj in inspect.getmembers(mod, inspect.iscoroutinefunction):
        if name.startswith("_"):
            continue
        hints = obj.__annotations__
        assert "return" in hints, f"{module_path}.{name} missing return annotation"
        ret = hints["return"]
        assert "dict" in str(ret).lower(), (
            f"{module_path}.{name} returns {ret}, expected dict"
        )
