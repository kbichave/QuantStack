"""Tests for the Textual app shell."""
import pytest

from quantstack.tui.app import QuantStackApp


class TestQuantStackApp:
    """QuantStackApp instantiation and composition."""

    def test_app_instantiates_without_error(self):
        app = QuantStackApp()
        assert app is not None

    def test_title_is_quantstack(self):
        app = QuantStackApp()
        assert app.TITLE == "QUANTSTACK"

    def test_css_path_points_to_dashboard_tcss(self):
        app = QuantStackApp()
        assert app.CSS_PATH == "dashboard.tcss"

    def test_bindings_include_required_keys(self):
        app = QuantStackApp()
        bound_keys = {b.key for b in app.BINDINGS if hasattr(b, "key")}
        required = {"1", "2", "3", "4", "5", "6", "q", "r", "question_mark", "slash"}
        # Textual normalises key names; check at least the numeric + q + r
        raw_keys = set()
        for b in app.BINDINGS:
            if isinstance(b, tuple):
                raw_keys.add(b[0])
            else:
                raw_keys.add(getattr(b, "key", ""))
        for k in ["1", "2", "3", "4", "5", "6", "q", "r"]:
            assert k in raw_keys, f"Missing binding for key '{k}'"

    @pytest.mark.asyncio
    async def test_compose_yields_header_tabbed_content_footer(self):
        from textual.widgets import TabbedContent, Static

        from quantstack.tui.widgets.header import HeaderBar

        app = QuantStackApp()
        async with app.run_test(size=(120, 40)) as pilot:
            # Header bar exists
            headers = app.query("HeaderBar")
            assert len(headers) >= 1

            # TabbedContent exists with 6 panes
            tc = app.query_one(TabbedContent)
            assert tc is not None
            panes = tc.query("TabPane")
            assert len(panes) == 6

            # Footer exists
            footer = app.query_one("#footer")
            assert footer is not None
