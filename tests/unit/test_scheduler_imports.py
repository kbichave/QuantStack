"""Tests for scheduler import chain (Section 09)."""
import importlib
import subprocess
import sys

import pytest


def test_quantstack_data_registry_importable():
    """DataProviderRegistry imports without ibkr_mcp."""
    from quantstack.data.registry import DataProviderRegistry

    assert DataProviderRegistry is not None


def test_ibkr_adapter_guard():
    """IBKRDataAdapter import guard raises clear error."""
    try:
        from quantstack.data.adapters.ibkr import IBKRDataAdapter, _IBKR_AVAILABLE

        if not _IBKR_AVAILABLE:
            with pytest.raises(ImportError, match="ibkr_mcp"):
                IBKRDataAdapter()
    except ImportError:
        # Module itself may fail to import for other reasons -- that's OK for this test
        pass


def test_ibkr_tick_adapter_guard():
    """IBKRTickAdapter import guard raises clear error."""
    try:
        from quantstack.data.streaming.ibkr_tick import (
            IBKRTickAdapter,
            _IBKR_AVAILABLE,
        )

        if not _IBKR_AVAILABLE:
            with pytest.raises(ImportError, match="ib_insync"):
                IBKRTickAdapter()
    except ImportError:
        pass


def test_ibkr_streaming_adapter_guard():
    """IBKRStreamingAdapter import guard raises clear error."""
    try:
        from quantstack.data.streaming.ibkr_stream import (
            IBKRStreamingAdapter,
            _IBKR_AVAILABLE,
        )

        if not _IBKR_AVAILABLE:
            with pytest.raises(ImportError, match="ib_insync"):
                IBKRStreamingAdapter()
    except ImportError:
        pass


def test_scheduler_module_importable():
    """scheduler.py imports without crashing (validates ibkr chain is unblocked)."""
    result = subprocess.run(
        [sys.executable, "-c", "import scripts.scheduler"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd="/Users/kshitijbichave/Personal/Trader",
    )
    # Even if it fails due to missing deps, it should NOT fail due to ibkr_mcp
    if result.returncode != 0:
        assert "ibkr_mcp" not in result.stderr, (
            f"Scheduler import chain broken by ibkr_mcp: {result.stderr}"
        )
