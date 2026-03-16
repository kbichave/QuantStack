"""Tests for shared.serializers."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from shared.serializers import serialize_for_json


class TestSerializeForJson:
    def test_none(self):
        assert serialize_for_json(None) is None

    def test_primitives_passthrough(self):
        assert serialize_for_json(42) == 42
        assert serialize_for_json(3.14) == 3.14
        assert serialize_for_json("hello") == "hello"
        assert serialize_for_json(True) is True

    def test_datetime(self):
        dt = datetime(2024, 6, 15, 12, 30, 0)
        assert serialize_for_json(dt) == "2024-06-15T12:30:00"

    def test_dataclass(self):
        @dataclass
        class Point:
            x: float
            y: float

        result = serialize_for_json(Point(1.0, 2.0))
        assert result == {"x": 1.0, "y": 2.0}

    def test_dataclass_with_datetime(self):
        @dataclass
        class Event:
            name: str
            ts: datetime

        result = serialize_for_json(Event("test", datetime(2024, 1, 1)))
        assert result["name"] == "test"
        assert result["ts"] == "2024-01-01T00:00:00"

    def test_numpy_integer(self):
        assert serialize_for_json(np.int64(42)) == 42.0
        assert isinstance(serialize_for_json(np.int64(42)), float)

    def test_numpy_float(self):
        assert serialize_for_json(np.float64(3.14)) == pytest.approx(3.14)

    def test_numpy_bool(self):
        assert serialize_for_json(np.bool_(True)) is True
        assert serialize_for_json(np.bool_(False)) is False

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = serialize_for_json(arr)
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_pandas_series(self):
        s = pd.Series([10, 20, 30], index=["a", "b", "c"])
        result = serialize_for_json(s)
        assert result == {"a": 10, "b": 20, "c": 30}

    def test_pandas_dataframe(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = serialize_for_json(df)
        assert result == [{"x": 1, "y": 3}, {"x": 2, "y": 4}]

    def test_pandas_dataframe_truncation(self):
        df = pd.DataFrame({"x": range(1000)})
        result = serialize_for_json(df, max_rows=5)
        assert len(result) == 5

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2024-06-15")
        result = serialize_for_json(ts)
        assert "2024-06-15" in result

    def test_nested_dict(self):
        data = {"a": np.int64(1), "b": datetime(2024, 1, 1), "c": "plain"}
        result = serialize_for_json(data)
        assert result == {"a": 1.0, "b": "2024-01-01T00:00:00", "c": "plain"}

    def test_nested_list(self):
        data = [np.float64(1.1), datetime(2024, 1, 1), "plain"]
        result = serialize_for_json(data)
        assert result[0] == pytest.approx(1.1)
        assert result[1] == "2024-01-01T00:00:00"
        assert result[2] == "plain"

    def test_deeply_nested(self):
        data = {"outer": [{"inner": np.int64(42)}]}
        result = serialize_for_json(data)
        assert result["outer"][0]["inner"] == 42.0


import pytest  # noqa: E402 — needed for approx
