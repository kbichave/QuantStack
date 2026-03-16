# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for MCP Bridge tool tests."""

from unittest.mock import MagicMock

import pytest
from quant_pod.tools.mcp_bridge import MCPBridge


@pytest.fixture
def mock_bridge():
    """Create a mock MCP bridge."""
    bridge = MagicMock(spec=MCPBridge)
    bridge._quantcore_available = True
    bridge._etrade_available = True
    return bridge


@pytest.fixture
def mock_quantcore_response():
    """Mock successful QuantCore response."""

    async def _mock_call(tool_name, **kwargs):
        return {"status": "success", "tool": tool_name, "params": kwargs}

    return _mock_call


@pytest.fixture
def mock_quantcore_error():
    """Mock QuantCore unavailable response."""

    async def _mock_call(tool_name, **kwargs):
        return {"error": "QuantCore MCP not available"}

    return _mock_call
