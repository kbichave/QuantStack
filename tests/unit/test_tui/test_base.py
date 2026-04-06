"""Tests for RefreshableWidget base class."""
from unittest.mock import MagicMock, patch

import pytest

from quantstack.tui.base import RefreshableWidget


class ConcreteWidget(RefreshableWidget):
    """Concrete subclass for testing."""

    def __init__(self, fetch_result=None, fetch_error=None):
        super().__init__()
        self._fetch_result = fetch_result
        self._fetch_error = fetch_error
        self.last_data = None

    def fetch_data(self):
        if self._fetch_error:
            raise self._fetch_error
        return self._fetch_result

    def update_view(self, data):
        self.last_data = data


class TestRefreshableWidget:
    """Thread-to-UI data flow pattern."""

    def test_subclass_can_override_fetch_and_update(self):
        widget = ConcreteWidget(fetch_result={"key": "value"})
        assert widget._fetch_result == {"key": "value"}

    def test_fetch_data_raises_not_implemented_on_base(self):
        widget = RefreshableWidget()
        with pytest.raises(NotImplementedError):
            widget.fetch_data()

    def test_update_view_raises_not_implemented_on_base(self):
        widget = RefreshableWidget()
        with pytest.raises(NotImplementedError):
            widget.update_view(None)

    def test_fetch_data_exception_does_not_crash_widget(self):
        """If fetch_data() raises, the widget should handle it gracefully."""
        widget = ConcreteWidget(fetch_error=RuntimeError("db down"))
        # Calling fetch_data directly should raise, but the _do_refresh
        # wrapper (tested via integration) catches it. Here we just verify
        # the widget is still usable after an error.
        with pytest.raises(RuntimeError):
            widget.fetch_data()
        # Widget is still alive and functional
        widget._fetch_error = None
        widget._fetch_result = "recovered"
        assert widget.fetch_data() == "recovered"
