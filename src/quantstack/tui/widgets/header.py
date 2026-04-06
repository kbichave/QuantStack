from __future__ import annotations

from datetime import datetime

from textual.reactive import reactive
from textual.widgets import Static


class HeaderBar(Static):
    """Single-line status bar docked to the top of the app.

    Renders: QUANTSTACK  HH:MM:SS | MODE | Kill: ok | Regime: name (conf%) | AV: used/limit | Universe: N
    """

    kill_status: reactive[str] = reactive("ok")
    regime_text: reactive[str] = reactive("unknown (?%)")
    av_count: reactive[str] = reactive("0/25000")
    trading_mode: reactive[str] = reactive("PAPER")
    universe_size: reactive[int] = reactive(0)

    def render(self) -> str:
        now = datetime.now().strftime("%H:%M:%S")
        return (
            f"QUANTSTACK  {now}"
            f"  | {self.trading_mode}"
            f"  | Kill: {self.kill_status}"
            f"  | Regime: {self.regime_text}"
            f"  | AV: {self.av_count}"
            f"  | Universe: {self.universe_size}"
        )
