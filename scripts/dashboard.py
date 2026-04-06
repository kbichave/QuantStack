#!/usr/bin/env python3
"""Launch the QuantStack TUI dashboard.

This replaces the original Rich Live dashboard. Run directly or via:
    python -m quantstack.tui
"""
from quantstack.tui import QuantStackApp

if __name__ == "__main__":
    QuantStackApp().run()
