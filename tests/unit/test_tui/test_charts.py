"""Tests for Unicode chart renderers."""
from datetime import date

from rich.table import Table
from rich.text import Text

from quantstack.tui.charts import (
    daily_heatmap,
    equity_curve,
    horizontal_bar,
    progress_bar,
    sparkline,
)


class TestSparkline:

    def test_ascending_series(self):
        result = sparkline([1, 2, 3, 4, 5, 6, 7, 8])
        assert isinstance(result, Text)
        assert len(result.plain) == 8
        chars = list(result.plain)
        assert chars == sorted(chars)

    def test_empty_data(self):
        result = sparkline([])
        assert result.plain == ""

    def test_all_same_values(self):
        result = sparkline([5, 5, 5, 5])
        assert len(set(result.plain)) == 1
        assert len(result.plain) == 4

    def test_resamples_when_exceeds_width(self):
        result = sparkline(list(range(100)), width=10)
        assert len(result.plain) == 10

    def test_color_applied(self):
        result = sparkline([1, 2, 3], color="red")
        assert result.style == "red"


class TestHorizontalBar:

    def test_renders_filled_and_empty(self):
        result = horizontal_bar(75, 100, 20)
        assert isinstance(result, Text)
        assert "75%" in result.plain

    def test_zero_value(self):
        result = horizontal_bar(0, 100, 20)
        assert "0%" in result.plain

    def test_max_value_zero(self):
        result = horizontal_bar(50, 0, 20)
        assert "0%" in result.plain


class TestProgressBar:

    def test_green_above_70(self):
        result = progress_bar(80, 100)
        spans = result._spans
        assert any("green" in str(s.style) for s in spans)

    def test_yellow_between_40_and_70(self):
        result = progress_bar(50, 100)
        spans = result._spans
        assert any("yellow" in str(s.style) for s in spans)

    def test_red_below_40(self):
        result = progress_bar(20, 100)
        spans = result._spans
        assert any("red" in str(s.style) for s in spans)


class TestDailyHeatmap:

    def test_renders_table_with_5_columns(self):
        result = daily_heatmap([], [])
        assert isinstance(result, Table)
        assert len(result.columns) == 5

    def test_positive_negative_coloring(self):
        dates = [date(2026, 3, 30), date(2026, 3, 31)]  # Mon, Tue
        values = [100.0, -50.0]
        result = daily_heatmap(values, dates)
        assert result.row_count == 1

    def test_empty_input(self):
        result = daily_heatmap([], [])
        assert isinstance(result, Table)
        assert result.row_count == 0


class TestEquityCurve:

    def test_multiline_output(self):
        result = equity_curve([100, 105, 103, 110, 108], height=5)
        lines = result.split("\n")
        assert len(lines) == 5

    def test_single_data_point(self):
        result = equity_curve([100])
        assert result != ""
        assert "100" in result

    def test_flat_data(self):
        result = equity_curve([50, 50, 50, 50])
        assert result != ""
        assert "50" in result

    def test_empty_data(self):
        assert equity_curve([]) == ""
