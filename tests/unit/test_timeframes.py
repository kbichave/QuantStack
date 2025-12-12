# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.config.timeframes module."""

import pytest

from quantcore.config.timeframes import (
    Timeframe,
    TimeframeParams,
    TIMEFRAME_HIERARCHY,
    TIMEFRAME_PARAMS,
    get_higher_timeframes,
    get_lower_timeframes,
    get_next_higher_timeframe,
    get_next_lower_timeframe,
)


class TestTimeframeEnum:
    """Test Timeframe enum."""

    def test_timeframe_values(self):
        """Test enum values."""
        assert Timeframe.H1.value == "1H"
        assert Timeframe.H4.value == "4H"
        assert Timeframe.D1.value == "1D"
        assert Timeframe.W1.value == "1W"

    def test_timeframe_count(self):
        """Test number of timeframes."""
        assert len(Timeframe) == 4

    def test_timeframe_from_value(self):
        """Test creating timeframe from value."""
        assert Timeframe("1H") == Timeframe.H1
        assert Timeframe("4H") == Timeframe.H4
        assert Timeframe("1D") == Timeframe.D1
        assert Timeframe("1W") == Timeframe.W1


class TestTimeframeHierarchy:
    """Test timeframe hierarchy."""

    def test_hierarchy_order(self):
        """Test hierarchy is ordered from highest to lowest."""
        assert TIMEFRAME_HIERARCHY == [
            Timeframe.W1,
            Timeframe.D1,
            Timeframe.H4,
            Timeframe.H1,
        ]

    def test_hierarchy_length(self):
        """Test hierarchy contains all timeframes."""
        assert len(TIMEFRAME_HIERARCHY) == len(Timeframe)


class TestTimeframeParams:
    """Test TimeframeParams dataclass."""

    def test_all_timeframes_have_params(self):
        """Test all timeframes have parameters defined."""
        for tf in Timeframe:
            assert tf in TIMEFRAME_PARAMS

    def test_params_are_frozen(self):
        """Test params are immutable."""
        params = TIMEFRAME_PARAMS[Timeframe.D1]
        with pytest.raises(AttributeError):
            params.ema_fast = 100  # type: ignore

    def test_weekly_params(self):
        """Test weekly timeframe parameters."""
        params = TIMEFRAME_PARAMS[Timeframe.W1]

        assert params.ema_fast == 10
        assert params.ema_slow == 20
        assert params.rsi_period == 14
        assert params.atr_period == 14
        assert params.bb_std == 2.0
        assert params.swing_lookback == 3
        assert params.resample_rule == "W-FRI"

    def test_daily_params(self):
        """Test daily timeframe parameters."""
        params = TIMEFRAME_PARAMS[Timeframe.D1]

        assert params.ema_fast == 20
        assert params.ema_slow == 50
        assert params.rsi_period == 14
        assert params.resample_rule == "D"

    def test_hourly_params(self):
        """Test hourly timeframe parameters."""
        params = TIMEFRAME_PARAMS[Timeframe.H1]

        assert params.ema_fast == 20
        assert params.ema_slow == 50
        assert params.resample_rule == "1h"

    def test_four_hour_params(self):
        """Test 4-hour timeframe parameters."""
        params = TIMEFRAME_PARAMS[Timeframe.H4]

        assert params.resample_rule == "4h"

    def test_all_params_have_resample_rule(self):
        """Test all params have valid resample rules."""
        for tf, params in TIMEFRAME_PARAMS.items():
            assert params.resample_rule is not None
            assert len(params.resample_rule) > 0


class TestGetHigherTimeframes:
    """Test get_higher_timeframes function."""

    def test_from_hourly(self):
        """Test getting higher timeframes from H1."""
        higher = get_higher_timeframes(Timeframe.H1)
        assert higher == [Timeframe.W1, Timeframe.D1, Timeframe.H4]

    def test_from_four_hour(self):
        """Test getting higher timeframes from H4."""
        higher = get_higher_timeframes(Timeframe.H4)
        assert higher == [Timeframe.W1, Timeframe.D1]

    def test_from_daily(self):
        """Test getting higher timeframes from D1."""
        higher = get_higher_timeframes(Timeframe.D1)
        assert higher == [Timeframe.W1]

    def test_from_weekly(self):
        """Test getting higher timeframes from W1 (none higher)."""
        higher = get_higher_timeframes(Timeframe.W1)
        assert higher == []


class TestGetLowerTimeframes:
    """Test get_lower_timeframes function."""

    def test_from_weekly(self):
        """Test getting lower timeframes from W1."""
        lower = get_lower_timeframes(Timeframe.W1)
        assert lower == [Timeframe.D1, Timeframe.H4, Timeframe.H1]

    def test_from_daily(self):
        """Test getting lower timeframes from D1."""
        lower = get_lower_timeframes(Timeframe.D1)
        assert lower == [Timeframe.H4, Timeframe.H1]

    def test_from_four_hour(self):
        """Test getting lower timeframes from H4."""
        lower = get_lower_timeframes(Timeframe.H4)
        assert lower == [Timeframe.H1]

    def test_from_hourly(self):
        """Test getting lower timeframes from H1 (none lower)."""
        lower = get_lower_timeframes(Timeframe.H1)
        assert lower == []


class TestGetNextHigherTimeframe:
    """Test get_next_higher_timeframe function."""

    def test_from_hourly(self):
        """Test next higher from H1."""
        assert get_next_higher_timeframe(Timeframe.H1) == Timeframe.H4

    def test_from_four_hour(self):
        """Test next higher from H4."""
        assert get_next_higher_timeframe(Timeframe.H4) == Timeframe.D1

    def test_from_daily(self):
        """Test next higher from D1."""
        assert get_next_higher_timeframe(Timeframe.D1) == Timeframe.W1

    def test_from_weekly(self):
        """Test next higher from W1 (returns None)."""
        assert get_next_higher_timeframe(Timeframe.W1) is None


class TestGetNextLowerTimeframe:
    """Test get_next_lower_timeframe function."""

    def test_from_weekly(self):
        """Test next lower from W1."""
        assert get_next_lower_timeframe(Timeframe.W1) == Timeframe.D1

    def test_from_daily(self):
        """Test next lower from D1."""
        assert get_next_lower_timeframe(Timeframe.D1) == Timeframe.H4

    def test_from_four_hour(self):
        """Test next lower from H4."""
        assert get_next_lower_timeframe(Timeframe.H4) == Timeframe.H1

    def test_from_hourly(self):
        """Test next lower from H1 (returns None)."""
        assert get_next_lower_timeframe(Timeframe.H1) is None


class TestTimeframeParamsAttributes:
    """Test all attributes of TimeframeParams."""

    @pytest.fixture
    def daily_params(self) -> TimeframeParams:
        """Get daily params fixture."""
        return TIMEFRAME_PARAMS[Timeframe.D1]

    def test_ema_attributes(self, daily_params):
        """Test EMA period attributes."""
        assert daily_params.ema_fast > 0
        assert daily_params.ema_slow > 0
        assert daily_params.ema_fast < daily_params.ema_slow

    def test_momentum_attributes(self, daily_params):
        """Test momentum attributes."""
        assert daily_params.rsi_period > 0
        assert daily_params.stoch_k_period > 0
        assert daily_params.stoch_d_period > 0
        assert daily_params.macd_fast > 0
        assert daily_params.macd_slow > 0
        assert daily_params.macd_signal > 0
        assert daily_params.roc_period > 0

    def test_volatility_attributes(self, daily_params):
        """Test volatility attributes."""
        assert daily_params.atr_period > 0
        assert daily_params.bb_period > 0
        assert daily_params.bb_std > 0
        assert daily_params.realized_vol_period > 0

    def test_volume_attributes(self, daily_params):
        """Test volume attributes."""
        assert daily_params.volume_ma_period > 0
        assert daily_params.obv_period > 0

    def test_market_structure_attributes(self, daily_params):
        """Test market structure attributes."""
        assert daily_params.swing_lookback > 0
        assert daily_params.trend_exhaustion_bars > 0

    def test_zscore_attributes(self, daily_params):
        """Test z-score attributes."""
        assert daily_params.zscore_period > 0
        assert daily_params.zscore_entry_threshold > 0
        assert daily_params.zscore_exit_threshold >= 0
        assert daily_params.zscore_entry_threshold > daily_params.zscore_exit_threshold

    def test_trade_attributes(self, daily_params):
        """Test trade parameter attributes."""
        assert daily_params.tp_atr_multiple > 0
        assert daily_params.sl_atr_multiple > 0
        assert daily_params.max_hold_bars > 0
