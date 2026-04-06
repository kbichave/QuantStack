# Section 3: Unicode Chart Renderers

## Overview

Build reusable chart rendering functions in `src/quantstack/tui/charts.py` that return Rich `Text` or `Table` objects for embedding in Textual widgets. These renderers are pure functions with no database or I/O dependencies -- they transform numeric data into terminal-renderable objects.

This module is consumed by every tab widget in sections 4-10 (Portfolio, Strategies, Data & Signals, etc.) wherever visual data representation is needed: sparklines for equity curves, horizontal bars for P&L attribution, progress bars for forward-testing gates, heatmaps for daily P&L, and multi-line ASCII charts for the equity curve widget.

## Dependencies

- **Section 1 (Package Scaffolding):** The `src/quantstack/tui/` package must exist. `charts.py` is created inside it.
- **No other section dependencies.** This module is self-contained -- it only depends on the `rich` library (already a transitive dependency of `textual`).

## File to Create

`src/quantstack/tui/charts.py`

## Tests (Write First)

Create `tests/unit/test_tui/test_charts.py`. Ensure `tests/unit/test_tui/__init__.py` exists.

```python
# tests/unit/test_tui/test_charts.py

from rich.text import Text
from rich.table import Table

from quantstack.tui.charts import (
    sparkline,
    horizontal_bar,
    progress_bar,
    daily_heatmap,
    equity_curve,
)


# --- sparkline ---

# Test: sparkline([1,2,3,4,5,6,7,8]) returns 8 block characters ascending
# Verify the returned Text contains exactly 8 characters and they are
# monotonically non-decreasing from the Unicode block set (▁▂▃▄▅▆▇█).

# Test: sparkline([]) returns empty string
# Verify sparkline returns a Text object whose plain text is "".

# Test: sparkline with all same values returns all same block char
# e.g. sparkline([5, 5, 5, 5]) should have 4 identical characters.

# Test: sparkline resamples when data exceeds width
# e.g. sparkline(list(range(100)), width=10) returns exactly 10 characters.

# Test: sparkline returns Rich Text with specified color
# Verify the style on the returned Text matches the color argument.


# --- horizontal_bar ---

# Test: horizontal_bar(75, 100, 20) renders ~15 filled + 5 empty + "75%"
# Count filled chars (█) and empty chars (░). Verify percentage label.

# Test: horizontal_bar(0, 100, 20) renders all empty
# All bar characters should be ░, label should be "0%".

# Test: horizontal_bar handles max_value=0 without division error
# Should not raise ZeroDivisionError. Return an empty/zero bar gracefully.


# --- progress_bar ---

# Test: progress_bar(80, 100) renders green
# Verify the style includes green (>70% threshold).

# Test: progress_bar(50, 100) renders yellow
# Verify the style includes yellow (>40%, <=70% threshold).

# Test: progress_bar(20, 100) renders red
# Verify the style includes red (<=40% threshold).


# --- daily_heatmap ---

# Test: daily_heatmap renders Mon-Fri grid
# Verify the returned Table has 5 columns (Mon through Fri).

# Test: daily_heatmap colors positive green, negative red
# Inspect cell styles for sign-based coloring.

# Test: daily_heatmap handles empty input
# Should return a Table (or fallback) without raising.


# --- equity_curve ---

# Test: equity_curve renders multi-line output with Y-axis labels
# Verify the output string contains multiple lines and includes
# the min and max values as Y-axis labels.

# Test: equity_curve handles single data point
# Should not crash; render a single-point "chart" gracefully.

# Test: equity_curve handles flat data (all same value)
# Should render without division-by-zero errors on the Y-axis range.
```

## Implementation Details

### Function Signatures and Behavior

All functions are module-level, stateless, and pure. They accept numeric data and rendering parameters, and return Rich renderables.

**`sparkline(data: Sequence[float], width: int = 0, color: str = "green") -> Text`**

- Renders a numeric series as Unicode block characters using the 8-level block set: `▁▂▃▄▅▆▇█`.
- Normalizes the input range to indices 0-7 by mapping `(value - min) / (max - min) * 7`, rounding to the nearest integer.
- If all values are equal (max == min), use the middle block character (`▄`) for every position.
- If `width` is specified and `len(data) > width`, resample by dividing data into `width` equal-sized buckets and averaging each bucket.
- If `width` is 0 or unset, use `len(data)` as the width (no resampling).
- If `data` is empty, return `Text("")`.
- Return a `rich.text.Text` object with the full sparkline string styled with `color`.

**`horizontal_bar(value: float, max_value: float, width: int = 20, color: str = "cyan") -> Text`**

- Renders a single horizontal bar: `████████░░░░ 75%`.
- Uses `█` (U+2588) for the filled portion and `░` (U+2591) for the empty portion.
- Filled character count: `round(value / max_value * width)` if `max_value > 0`, else 0.
- Guard against `max_value == 0`: return all-empty bar with `0%` label instead of raising `ZeroDivisionError`.
- Clamp the ratio to [0.0, 1.0] so values exceeding `max_value` don't overflow the bar width.
- Append a space and the percentage label (e.g., `" 75%"`).
- Return a `rich.text.Text` with the filled portion styled with `color` and the empty portion unstyled (or dim).

**`progress_bar(current: float, total: float, width: int = 20) -> Text`**

- Same bar rendering as `horizontal_bar` but with adaptive color based on completion percentage:
  - `> 70%`: green
  - `> 40%`: yellow
  - `<= 40%`: red
- Guard against `total == 0` the same way as `horizontal_bar`.
- Return a `rich.text.Text` with the bar styled in the determined color.

**`daily_heatmap(daily_values: Sequence[float], dates: Sequence[date]) -> Table`**

- Renders a Mon-Fri grid of daily P&L values as a `rich.table.Table`.
- `daily_values` and `dates` are parallel sequences of the same length, representing trading-day P&L and their calendar dates.
- Build rows by week: each row represents one calendar week. Columns are Mon through Fri. Cells contain the P&L value for that day, or empty if no trading occurred (weekend/holiday).
- Cell styling: green text for positive values, red text for negative values. Intensity is proportional to magnitude relative to the max absolute value in the dataset (use Rich style `bold` for values exceeding 50% of the max).
- Table configuration: no borders (`show_edge=False, show_header=True, box=None`), column headers are `Mon Tue Wed Thu Fri`.
- If `daily_values` is empty, return an empty `Table` with column headers only.

**`equity_curve(values: Sequence[float], width: int = 60, height: int = 5) -> str`**

- Renders a multi-line ASCII chart using box-drawing characters (`---`, `|`, and position markers).
- Returns a plain `str` (not `Text`) since it's multi-line and will be wrapped in a `Static` widget.
- Y-axis: left-aligned labels showing min and max values of the series. The Y-axis spans `height` rows.
- X-axis: implicit (no explicit labels required, but if dates are available they can be added by the caller).
- Plotting: for each column position (up to `width`), resample data if needed, then map the value to a row index (0 = bottom = min, height-1 = top = max). Place a marker character at that row/column.
- If `values` has a single data point, render a single horizontal line at that value.
- If all values are the same (flat), render a horizontal line at the midpoint row with the value as the Y-axis label.
- Guard against empty `values`: return an empty string.

### Character Sets Reference

For implementation, use these exact Unicode code points:

- Block elements for sparkline: `▁` (U+2581), `▂` (U+2582), `▃` (U+2583), `▄` (U+2584), `▅` (U+2585), `▆` (U+2586), `▇` (U+2587), `█` (U+2588)
- Filled bar: `█` (U+2588)
- Empty bar: `░` (U+2591)
- Box drawing for equity_curve: `─` (U+2500), `│` (U+2502) and standard ASCII as needed

### Error Handling

These are pure rendering functions. They should not raise exceptions for any valid numeric input. Defend against:

- Empty sequences (return empty renderable)
- Zero ranges (max == min) causing division by zero
- Negative values in bars (clamp to zero)
- `max_value == 0` in bar functions (return zero-width bar)
- Non-finite values (NaN, inf) -- filter them out or treat as zero before rendering

### Where These Renderers Are Used

For context (do not implement these consumers -- they belong to other sections):

| Renderer | Consumer Widget | Section |
|----------|----------------|---------|
| `sparkline` | EquityCurveWidget, DataHealthCompact | 4, 5 |
| `horizontal_bar` | PnlBySymbolWidget, DataHealthMatrixWidget | 5, 7 |
| `progress_bar` | PipelineKanbanWidget (forward test progress) | 6 |
| `daily_heatmap` | DailyHeatmapWidget | 5 |
| `equity_curve` | EquityCurveWidget (multi-line mode) | 5 |

## Implementation Checklist

1. Create `tests/unit/test_tui/__init__.py` (if it does not exist)
2. Create `tests/unit/test_tui/test_charts.py` with all test stubs above
3. Run tests -- all should fail (no implementation yet)
4. Create `src/quantstack/tui/charts.py` with the five functions
5. Implement `sparkline` -- run its tests until green
6. Implement `horizontal_bar` -- run its tests until green
7. Implement `progress_bar` -- run its tests until green
8. Implement `daily_heatmap` -- run its tests until green
9. Implement `equity_curve` -- run its tests until green
10. Run full test suite: `uv run pytest tests/unit/test_tui/test_charts.py -v`
11. Verify no import side effects: `uv run python -c "from quantstack.tui.charts import sparkline, horizontal_bar, progress_bar, daily_heatmap, equity_curve"`
