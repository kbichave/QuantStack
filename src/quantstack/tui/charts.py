"""Unicode chart rendering utilities — pure functions returning Rich renderables."""
from __future__ import annotations

import math
from datetime import date
from typing import Sequence

from rich.table import Table
from rich.text import Text

BLOCKS = "▁▂▃▄▅▆▇█"
FILLED = "█"
EMPTY = "░"


def sparkline(data: Sequence[float], width: int = 0, color: str = "green") -> Text:
    """Render a numeric series as Unicode block characters."""
    if not data:
        return Text("")
    values = [v for v in data if math.isfinite(v)]
    if not values:
        return Text("")
    if width > 0 and len(values) > width:
        bucket_size = len(values) / width
        resampled = []
        for i in range(width):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            resampled.append(sum(values[start:end]) / max(1, end - start))
        values = resampled
    lo, hi = min(values), max(values)
    if hi == lo:
        chars = BLOCKS[3] * len(values)
    else:
        chars = "".join(
            BLOCKS[min(7, int((v - lo) / (hi - lo) * 7 + 0.5))]
            for v in values
        )
    return Text(chars, style=color)


def horizontal_bar(
    value: float, max_value: float, width: int = 20, color: str = "cyan",
) -> Text:
    """Render a single horizontal bar with percentage label."""
    if max_value <= 0 or not math.isfinite(value):
        pct = 0.0
    else:
        pct = max(0.0, min(1.0, value / max_value))
    filled = round(pct * width)
    empty = width - filled
    pct_label = f" {int(pct * 100)}%"
    result = Text()
    result.append(FILLED * filled, style=color)
    result.append(EMPTY * empty, style="dim")
    result.append(pct_label)
    return result


def progress_bar(current: float, total: float, width: int = 20) -> Text:
    """Render a progress bar with adaptive color based on completion."""
    if total <= 0 or not math.isfinite(current):
        pct = 0.0
    else:
        pct = max(0.0, min(1.0, current / total))
    if pct > 0.7:
        color = "green"
    elif pct > 0.4:
        color = "yellow"
    else:
        color = "red"
    filled = round(pct * width)
    empty = width - filled
    pct_label = f" {int(pct * 100)}%"
    result = Text()
    result.append(FILLED * filled, style=color)
    result.append(EMPTY * empty, style="dim")
    result.append(pct_label)
    return result


def daily_heatmap(daily_values: Sequence[float], dates: Sequence[date]) -> Table:
    """Render a Mon-Fri grid of daily P&L values."""
    table = Table(show_edge=False, show_header=True, box=None)
    for day in ("Mon", "Tue", "Wed", "Thu", "Fri"):
        table.add_column(day, justify="right", width=8)
    if not daily_values or not dates:
        return table
    max_abs = max((abs(v) for v in daily_values if math.isfinite(v)), default=1.0) or 1.0
    by_week: dict[int, dict[int, float]] = {}
    for d, v in zip(dates, daily_values):
        iso = d.isocalendar()
        week_key = iso[0] * 100 + iso[1]
        by_week.setdefault(week_key, {})[iso[2]] = v
    for week_key in sorted(by_week):
        week = by_week[week_key]
        cells = []
        for dow in range(1, 6):
            if dow in week:
                val = week[dow]
                color = "green" if val >= 0 else "red"
                bold = "bold " if abs(val) > max_abs * 0.5 else ""
                cells.append(Text(f"{val:+.1f}", style=f"{bold}{color}"))
            else:
                cells.append(Text(""))
        table.add_row(*cells)
    return table


def equity_curve(values: Sequence[float], width: int = 60, height: int = 5) -> str:
    """Render a multi-line ASCII chart."""
    if not values:
        return ""
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return ""
    if width > 0 and len(clean) > width:
        bucket_size = len(clean) / width
        resampled = []
        for i in range(width):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            resampled.append(sum(clean[start:end]) / max(1, end - start))
        clean = resampled
    lo, hi = min(clean), max(clean)
    label_w = max(len(f"{lo:.0f}"), len(f"{hi:.0f}")) + 1
    if hi == lo:
        mid = height // 2
        lines = []
        for row in range(height - 1, -1, -1):
            label = f"{hi:.0f}".rjust(label_w) if row == mid else " " * label_w
            line = "─" * len(clean) if row == mid else " " * len(clean)
            lines.append(f"{label}│{line}")
        return "\n".join(lines)
    lines = []
    for row in range(height - 1, -1, -1):
        if row == height - 1:
            label = f"{hi:.0f}".rjust(label_w)
        elif row == 0:
            label = f"{lo:.0f}".rjust(label_w)
        else:
            label = " " * label_w
        chars = []
        for v in clean:
            mapped = int((v - lo) / (hi - lo) * (height - 1) + 0.5)
            chars.append("•" if mapped == row else " ")
        lines.append(f"{label}│{''.join(chars)}")
    return "\n".join(lines)
