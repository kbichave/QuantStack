"""Shared fixtures for TUI widget tests."""
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _patch_static_update():
    """Patch Static.update() so widget tests work without a running Textual app.

    Static.update() calls app.console which requires an active app context.
    In unit tests we only verify that update_view() processes data without
    errors — the actual rendering is Textual's responsibility.
    """
    with patch("textual.widgets._static.Static.update"):
        yield
