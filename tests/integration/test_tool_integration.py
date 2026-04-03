"""Integration tests for refactored tool layer.

Tests validate return types, JSON structure, and error handling
for the tool layer after MCP removal.
"""

import inspect
import json

import pytest


class TestReturnTypeInvariants:
    """All langchain tools return str (JSON), all functions return dict."""

    def test_all_langchain_tools_return_str(self):
        from quantstack.tools.registry import TOOL_REGISTRY

        for name, tool_obj in TOOL_REGISTRY.items():
            fn = tool_obj.coroutine or tool_obj.func
            sig = inspect.signature(fn)
            ret = sig.return_annotation
            assert ret is str or ret == "str", (
                f"Tool '{name}' return annotation is {ret}, expected str"
            )

    def test_functions_layer_returns_dict(self):
        from quantstack.tools.functions.data_functions import get_regime, get_portfolio_state
        from quantstack.tools.functions.execution_functions import submit_order
        from quantstack.tools.functions.risk_functions import validate_risk_gate
        from quantstack.tools.functions.system_functions import (
            check_system_status as sys_status_fn,
            check_heartbeat as heartbeat_fn,
        )

        for fn in [get_regime, get_portfolio_state, submit_order,
                    validate_risk_gate, sys_status_fn, heartbeat_fn]:
            sig = inspect.signature(fn)
            ret = sig.return_annotation
            assert "dict" in str(ret).lower(), (
                f"Function {fn.__name__} return annotation is {ret}, expected dict"
            )


class TestStubToolsReturnValidJSON:
    """Tools marked as pending migration return parseable JSON error objects."""

    @pytest.fixture
    def stub_tools(self):
        from quantstack.tools.registry import TOOL_REGISTRY
        stubs = {}
        for name, tool_obj in TOOL_REGISTRY.items():
            fn = tool_obj.coroutine or tool_obj.func
            source = inspect.getsource(fn)
            if "Tool pending migration" in source:
                stubs[name] = tool_obj
        return stubs

    async def test_stub_tools_return_valid_json(self, stub_tools):
        for name, tool_obj in stub_tools.items():
            # Call with minimal args — stub ignores them
            try:
                fn = tool_obj.coroutine
                sig = inspect.signature(fn)
                # Build minimal kwargs
                kwargs = {}
                for pname, param in sig.parameters.items():
                    if param.default is inspect.Parameter.empty:
                        if param.annotation == str or param.annotation is str:
                            kwargs[pname] = "test"
                        elif param.annotation == float or param.annotation is float:
                            kwargs[pname] = 1.0
                        elif param.annotation == int or param.annotation is int:
                            kwargs[pname] = 1
                        elif "list" in str(param.annotation).lower():
                            kwargs[pname] = []
                        elif "dict" in str(param.annotation).lower():
                            kwargs[pname] = {}
                        else:
                            kwargs[pname] = "test"

                result = await fn(**kwargs)
                parsed = json.loads(result)
                assert isinstance(parsed, dict), f"Tool '{name}' returned non-dict JSON"
                assert "error" in parsed, f"Stub tool '{name}' missing 'error' key"
            except Exception as e:
                pytest.fail(f"Stub tool '{name}' raised {type(e).__name__}: {e}")
