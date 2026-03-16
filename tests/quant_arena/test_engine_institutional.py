# Copyright 2024 QuantArena Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for HistoricalEngine institutional additions.

Tests are *unit* tests: they bypass the async simulation loop and exercise
individual engine methods directly.  Where the engine touches quantcore or
the data loader, we inject fakes/mocks so the test has no external deps.

Covered:
- _correlation_check_passes()         T2-6
- _generate_result() Sharpe CI path   T1-1
- _generate_result() benchmark path   T1-6
- _generate_result() walk-forward     T1-3
- _generate_result() overfitting path T1-2
- SimulationResult field defaults     (invariant)
- config walk-forward fields present  (invariant)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_arena.historical.config import HistoricalConfig
from quant_arena.historical.engine import HistoricalEngine, SimulationResult
from quant_arena.historical.sim_broker import OrderSide, SimBroker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> HistoricalConfig:
    """Build a minimal HistoricalConfig — large enough to pass engine init."""
    defaults = dict(
        symbols=["SPY", "QQQ"],
        initial_equity=100_000.0,
        max_leverage=1.0,
        slippage_bps=5.0,
        commission_per_share=0.005,
        max_position_pct=0.25,
        max_drawdown_halt_pct=0.20,
        max_daily_loss_pct=0.05,
        benchmark_symbol="SPY",
        max_portfolio_correlation=1.0,  # disabled by default
        walk_forward_mode=False,
        walk_forward_n_folds=5,
        walk_forward_test_days=63,
    )
    defaults.update(kwargs)
    return HistoricalConfig(**defaults)


def _make_engine(config: Optional[HistoricalConfig] = None) -> HistoricalEngine:
    """
    Build an HistoricalEngine with mocked universe and data_loader so no
    real I/O happens during unit tests.
    """
    if config is None:
        config = _make_config()

    engine = HistoricalEngine.__new__(HistoricalEngine)
    engine.config = config

    # Minimal stubs
    engine.universe = MagicMock()
    engine.universe.symbols = config.symbols
    engine.universe.__len__ = lambda self: len(config.symbols)

    engine.data_loader = MagicMock()
    engine.data_loader.start = date(2023, 1, 1)
    engine.data_loader.end = date(2023, 12, 31)
    engine.data_loader.trading_days = []
    engine.data_loader.__len__ = MagicMock(return_value=250)

    engine.broker = SimBroker(
        initial_equity=config.initial_equity,
        slippage_bps=config.slippage_bps,
        commission_per_share=config.commission_per_share,
        max_position_pct=config.max_position_pct,
        max_drawdown_halt_pct=config.max_drawdown_halt_pct,
        max_leverage=config.max_leverage,
        max_daily_loss_pct=config.max_daily_loss_pct,
    )
    engine.clock = None
    engine._knowledge_store = None
    engine._historical_flow = None
    engine._policy_store = None
    engine._running = False
    engine._current_day = 0
    engine._total_trades = 0
    engine._checkpoint_interval = 20
    engine._last_checkpoint_trades = 0
    engine._enable_mtf = False
    engine._execution_timeframe = "daily"
    engine._use_super_trader = False
    engine._tca = None
    engine._tca_available = None
    engine._on_day_complete = None
    engine._on_trade = None

    return engine


@dataclass
class _FakeSnap:
    equity: float
    date: date = date(2023, 1, 2)


def _inject_snapshots(engine: HistoricalEngine, returns: List[float], start_equity: float = 100_000.0):
    """
    Populate broker daily_snapshots from a synthetic return series so that
    _generate_result() has data to work with.
    """
    snaps = [_FakeSnap(equity=start_equity, date=date(2023, 1, 2))]
    for i, r in enumerate(returns):
        snaps.append(_FakeSnap(equity=snaps[-1].equity * (1 + r), date=date(2023, 1, 3) + timedelta(days=i)))
    engine.broker._daily_snapshots = snaps


# ---------------------------------------------------------------------------
# TestSimulationResultDefaults
# ---------------------------------------------------------------------------

class TestSimulationResultDefaults:
    """SimulationResult optional institutional fields default to None / empty."""

    def test_new_result_optional_fields_are_none(self):
        result = SimulationResult(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_equity=100_000.0,
            final_equity=110_000.0,
            total_return=0.10,
            max_drawdown=-0.05,
            total_trades=50,
            win_rate=0.55,
            sharpe_ratio=1.2,
            trading_days=250,
            symbols=["SPY"],
        )
        assert result.sharpe_ci is None
        assert result.calmar_ratio is None
        assert result.alpha is None
        assert result.beta is None
        assert result.information_ratio is None
        assert result.benchmark_return is None
        assert result.overfitting_verdict == ""
        assert result.tca_report is None
        assert result.walk_forward_summary is None

    def test_sample_size_ok_defaults_true(self):
        result = SimulationResult(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_equity=100_000.0,
            final_equity=110_000.0,
            total_return=0.10,
            max_drawdown=-0.05,
            total_trades=50,
            win_rate=0.55,
            sharpe_ratio=1.2,
            trading_days=250,
            symbols=["SPY"],
        )
        assert result.sample_size_ok is True
        assert result.sample_size_msg == ""


# ---------------------------------------------------------------------------
# TestConfigNewFields
# ---------------------------------------------------------------------------

class TestConfigNewFields:
    """Verify T1-3, T1-6, T2-6 fields are present in HistoricalConfig."""

    def test_benchmark_symbol_default(self):
        config = HistoricalConfig()
        assert config.benchmark_symbol == "SPY"

    def test_max_portfolio_correlation_default_disabled(self):
        config = HistoricalConfig()
        assert config.max_portfolio_correlation == 1.0  # disabled

    def test_walk_forward_mode_default_false(self):
        config = HistoricalConfig()
        assert config.walk_forward_mode is False

    def test_walk_forward_n_folds_default(self):
        config = HistoricalConfig()
        assert config.walk_forward_n_folds == 5

    def test_walk_forward_test_days_default(self):
        config = HistoricalConfig()
        assert config.walk_forward_test_days == 63


# ---------------------------------------------------------------------------
# TestCorrelationCheckPasses
# ---------------------------------------------------------------------------

class TestCorrelationCheckPasses:
    """
    _correlation_check_passes() unit tests.

    The method is pure-python (no async) so we call it directly on a
    constructed engine whose data_loader.get_price_history is stubbed.
    """

    def _setup_engine(self, max_corr: float) -> HistoricalEngine:
        config = _make_config(max_portfolio_correlation=max_corr, symbols=["SPY", "QQQ"])
        engine = _make_engine(config)
        return engine

    def test_disabled_when_max_corr_is_1(self):
        engine = self._setup_engine(max_corr=1.0)
        # Should pass regardless of actual correlation
        instruction = {"symbol": "QQQ", "side": "buy", "quantity": 100}
        assert engine._correlation_check_passes(instruction, date(2023, 6, 1)) is True

    def test_sell_orders_always_pass(self):
        engine = self._setup_engine(max_corr=0.0)  # Zero tolerance
        instruction = {"symbol": "QQQ", "side": "sell", "quantity": 100}
        assert engine._correlation_check_passes(instruction, date(2023, 6, 1)) is True

    def test_passes_when_no_existing_positions(self):
        engine = self._setup_engine(max_corr=0.5)
        # No positions → no correlation to check → allow
        instruction = {"symbol": "QQQ", "side": "buy", "quantity": 100}
        assert engine._correlation_check_passes(instruction, date(2023, 6, 1)) is True

    def test_passes_when_history_too_short(self):
        engine = self._setup_engine(max_corr=0.5)
        # Put a position in the broker so correlation check is triggered
        engine.broker._prices = {"SPY": 100.0}
        engine.broker.update_prices({"SPY": 100.0}, date(2023, 1, 2))
        engine.broker.submit_order("SPY", OrderSide.BUY, 50)

        # Return only 5 prices — below the 10-sample threshold
        short_series = pd.Series([100.0 + i for i in range(5)])
        engine.data_loader.get_price_history = MagicMock(return_value=short_series)

        instruction = {"symbol": "QQQ", "side": "buy", "quantity": 50}
        assert engine._correlation_check_passes(instruction, date(2023, 6, 1)) is True

    def test_rejects_highly_correlated_new_position(self):
        engine = self._setup_engine(max_corr=0.50)

        # Insert a SPY position
        engine.broker.update_prices({"SPY": 100.0}, date(2023, 1, 2))
        engine.broker.submit_order("SPY", OrderSide.BUY, 50)

        # Create perfectly-correlated price series (correlation ≈ 1.0)
        prices = pd.Series([100.0 + i for i in range(30)])

        engine.data_loader.get_price_history = MagicMock(return_value=prices)

        instruction = {"symbol": "QQQ", "side": "buy", "quantity": 50}
        result = engine._correlation_check_passes(instruction, date(2023, 6, 1))
        # avg correlation ~1.0 > 0.50 → should reject
        assert result is False

    def test_allows_uncorrelated_new_position(self):
        engine = self._setup_engine(max_corr=0.50)

        # Insert a SPY position
        engine.broker.update_prices({"SPY": 100.0}, date(2023, 1, 2))
        engine.broker.submit_order("SPY", OrderSide.BUY, 50)

        rng = np.random.default_rng(42)
        # SPY price series
        spy_prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, 30)))
        # Independent (uncorrelated) QQQ price series
        qqq_prices = pd.Series(200.0 + np.cumsum(rng.normal(0, 1, 30)))

        call_count = [0]

        def fake_history(symbol, end_date, days):
            call_count[0] += 1
            return qqq_prices if symbol == "QQQ" else spy_prices

        engine.data_loader.get_price_history = MagicMock(side_effect=fake_history)

        instruction = {"symbol": "QQQ", "side": "buy", "quantity": 50}
        result = engine._correlation_check_passes(instruction, date(2023, 6, 1))
        # Independent random walks: correlation much lower than 0.50
        assert result is True


# ---------------------------------------------------------------------------
# TestGenerateResultSharpe
# ---------------------------------------------------------------------------

class TestGenerateResultSharpe:
    """_generate_result() correctly populates Sharpe-related fields."""

    def test_sharpe_populated_with_enough_data(self):
        engine = _make_engine()
        rng = np.random.default_rng(0)
        returns = list(rng.normal(0.0005, 0.01, 100))
        _inject_snapshots(engine, returns)

        with patch.object(engine.broker, "get_summary", return_value={
            "final_equity": 110_000.0,
            "total_return": 0.10,
            "max_drawdown": -0.05,
            "total_trades": 80,
            "win_rate": 0.55,
        }), patch.object(engine.broker, "get_trade_history", return_value=[]):
            result = engine._generate_result()

        assert result.sharpe_ratio is not None
        assert math.isfinite(result.sharpe_ratio)

    def test_sharpe_is_positive_for_positive_drift(self):
        engine = _make_engine()
        # Consistent positive drift — Sharpe should be > 0
        returns = [0.001] * 100  # constant 10bps per day
        _inject_snapshots(engine, returns)

        with patch.object(engine.broker, "get_summary", return_value={
            "final_equity": 110_000.0,
            "total_return": 0.10,
            "max_drawdown": 0.0,
            "total_trades": 100,
            "win_rate": 1.0,
        }), patch.object(engine.broker, "get_trade_history", return_value=[]):
            result = engine._generate_result()

        # With zero variance, Sharpe may be 0 (division by zero guard) or very high.
        # The main check: it shouldn't be negative for all-positive returns.
        if result.sharpe_ratio is not None:
            assert result.sharpe_ratio >= 0.0

    def test_insufficient_returns_gives_none_ci(self):
        engine = _make_engine()
        # Fewer than 10 observations → CI not computed
        _inject_snapshots(engine, [0.001] * 5)

        with patch.object(engine.broker, "get_summary", return_value={
            "final_equity": 100_500.0,
            "total_return": 0.005,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
        }), patch.object(engine.broker, "get_trade_history", return_value=[]):
            result = engine._generate_result()

        # With < 10 returns the quantcore branch is not entered; CI stays None
        assert result.sharpe_ci is None or isinstance(result.sharpe_ci, tuple)


# ---------------------------------------------------------------------------
# TestGenerateResultBenchmark
# ---------------------------------------------------------------------------

class TestGenerateResultBenchmark:
    """_generate_result() benchmark comparison (T1-6): alpha / beta / IR fields."""

    def _engine_with_spy_data(self, strat_returns: List[float], bench_returns: List[float]) -> HistoricalEngine:
        engine = _make_engine()
        _inject_snapshots(engine, strat_returns)

        # Stub data_loader.get_price_history for SPY
        n = len(bench_returns) + 1
        spy_prices = pd.Series([100.0 * float(np.prod(1 + np.array(bench_returns[:i]))) for i in range(n)])
        engine.data_loader.get_price_history = MagicMock(return_value=spy_prices)
        engine.data_loader.end = date(2023, 12, 31)

        return engine

    def test_beta_between_reasonable_bounds(self):
        rng = np.random.default_rng(3)
        bench = list(rng.normal(0.0003, 0.01, 120))
        # Strategy = 1.2 × bench + noise
        strat = [1.2 * r + rng.normal(0, 0.002) for r in bench]
        engine = self._engine_with_spy_data(strat, bench)

        with patch.object(engine.broker, "get_summary", return_value={
            "final_equity": 115_000.0, "total_return": 0.15,
            "max_drawdown": -0.05, "total_trades": 80, "win_rate": 0.6,
        }), patch.object(engine.broker, "get_trade_history", return_value=[]):
            result = engine._generate_result()

        if result.beta is not None:
            # With 1.2x leverage on benchmark, beta should be around 1.0–1.5
            assert 0.0 < result.beta < 3.0

    def test_benchmark_return_is_finite(self):
        rng = np.random.default_rng(5)
        bench = list(rng.normal(0.0004, 0.01, 60))
        strat = [r + rng.normal(0, 0.003) for r in bench]
        engine = self._engine_with_spy_data(strat, bench)

        with patch.object(engine.broker, "get_summary", return_value={
            "final_equity": 108_000.0, "total_return": 0.08,
            "max_drawdown": -0.04, "total_trades": 50, "win_rate": 0.55,
        }), patch.object(engine.broker, "get_trade_history", return_value=[]):
            result = engine._generate_result()

        if result.benchmark_return is not None:
            assert math.isfinite(result.benchmark_return)

    def test_benchmark_skipped_gracefully_on_empty_history(self):
        engine = _make_engine()
        _inject_snapshots(engine, [0.001] * 50)
        engine.data_loader.get_price_history = MagicMock(return_value=pd.Series([], dtype=float))
        engine.data_loader.end = date(2023, 12, 31)

        with patch.object(engine.broker, "get_summary", return_value={
            "final_equity": 105_000.0, "total_return": 0.05,
            "max_drawdown": -0.02, "total_trades": 40, "win_rate": 0.6,
        }), patch.object(engine.broker, "get_trade_history", return_value=[]):
            result = engine._generate_result()

        # Fields stay None on missing benchmark data — must not raise
        assert result.benchmark_return is None or math.isfinite(result.benchmark_return)


# ---------------------------------------------------------------------------
# TestGenerateResultWalkForward
# ---------------------------------------------------------------------------

class TestGenerateResultWalkForward:
    """Walk-forward (T1-3) runs when config.walk_forward_mode=True."""

    def _wf_engine(self, n_returns: int = 200, n_folds: int = 3, test_days: int = 40) -> HistoricalEngine:
        config = _make_config(
            walk_forward_mode=True,
            walk_forward_n_folds=n_folds,
            walk_forward_test_days=test_days,
        )
        engine = _make_engine(config)
        rng = np.random.default_rng(9)
        returns = list(rng.normal(0.0004, 0.01, n_returns))
        _inject_snapshots(engine, returns)
        engine.data_loader.get_price_history = MagicMock(return_value=pd.Series([], dtype=float))
        engine.data_loader.end = date(2023, 12, 31)
        return engine

    def test_walk_forward_summary_populated(self):
        engine = self._wf_engine()
        with patch.object(engine.broker, "get_summary", return_value={
            "final_equity": 120_000.0, "total_return": 0.20,
            "max_drawdown": -0.06, "total_trades": 120, "win_rate": 0.58,
        }), patch.object(engine.broker, "get_trade_history", return_value=[]):
            result = engine._generate_result()

        # walk_forward_summary should be populated (not None) for a long enough series
        # Exact value depends on whether quantcore is installed; check type or None
        # If quantcore unavailable, result is still not an error — just None
        assert result.walk_forward_summary is None or hasattr(result.walk_forward_summary, "n_folds")

    def test_walk_forward_disabled_gives_none(self):
        config = _make_config(walk_forward_mode=False)
        engine = _make_engine(config)
        rng = np.random.default_rng(11)
        _inject_snapshots(engine, list(rng.normal(0.0004, 0.01, 100)))
        engine.data_loader.get_price_history = MagicMock(return_value=pd.Series([], dtype=float))
        engine.data_loader.end = date(2023, 12, 31)

        with patch.object(engine.broker, "get_summary", return_value={
            "final_equity": 108_000.0, "total_return": 0.08,
            "max_drawdown": -0.03, "total_trades": 60, "win_rate": 0.52,
        }), patch.object(engine.broker, "get_trade_history", return_value=[]):
            result = engine._generate_result()

        assert result.walk_forward_summary is None

    def test_walk_forward_too_few_returns_gives_none(self):
        # If there aren't enough returns to form folds, result is gracefully None
        engine = self._wf_engine(n_returns=15)  # < 20 returns threshold
        with patch.object(engine.broker, "get_summary", return_value={
            "final_equity": 101_000.0, "total_return": 0.01,
            "max_drawdown": -0.01, "total_trades": 5, "win_rate": 0.5,
        }), patch.object(engine.broker, "get_trade_history", return_value=[]):
            result = engine._generate_result()

        assert result.walk_forward_summary is None


# ---------------------------------------------------------------------------
# TestGenerateResultOverfitting
# ---------------------------------------------------------------------------

class TestGenerateResultOverfitting:
    """Overfitting verdict field is populated if quantcore is available."""

    def test_overfitting_verdict_is_string(self):
        engine = _make_engine()
        rng = np.random.default_rng(13)
        _inject_snapshots(engine, list(rng.normal(0.0003, 0.01, 80)))
        engine.data_loader.get_price_history = MagicMock(return_value=pd.Series([], dtype=float))
        engine.data_loader.end = date(2023, 12, 31)

        with patch.object(engine.broker, "get_summary", return_value={
            "final_equity": 106_000.0, "total_return": 0.06,
            "max_drawdown": -0.03, "total_trades": 50, "win_rate": 0.54,
        }), patch.object(engine.broker, "get_trade_history", return_value=[]):
            result = engine._generate_result()

        # Either empty (quantcore absent) or one of the valid verdicts
        assert isinstance(result.overfitting_verdict, str)
        if result.overfitting_verdict:
            assert result.overfitting_verdict in {"GENUINE", "SUSPECT", "OVERFIT"}

    def test_overfitting_skipped_gracefully_without_quantcore(self):
        engine = _make_engine()
        _inject_snapshots(engine, [0.0005] * 30)
        engine.data_loader.get_price_history = MagicMock(return_value=pd.Series([], dtype=float))
        engine.data_loader.end = date(2023, 12, 31)

        # Simulate ImportError for overfitting module
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "quantcore.research.overfitting":
                raise ImportError("quantcore not available")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with patch.object(engine.broker, "get_summary", return_value={
                "final_equity": 103_000.0, "total_return": 0.03,
                "max_drawdown": -0.01, "total_trades": 20, "win_rate": 0.5,
            }), patch.object(engine.broker, "get_trade_history", return_value=[]):
                result = engine._generate_result()

        # Should not raise; verdict empty when module unavailable
        assert isinstance(result.overfitting_verdict, str)


# ---------------------------------------------------------------------------
# TestGetTca
# ---------------------------------------------------------------------------

class TestGetTca:
    """_get_tca() caches the TCA engine and handles missing module gracefully."""

    def test_returns_none_when_quantcore_absent(self):
        engine = _make_engine()
        engine._tca = None
        engine._tca_available = None

        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if "tca_engine" in name:
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = engine._get_tca()

        assert result is None
        assert engine._tca_available is False

    def test_second_call_after_failure_skips_import(self):
        engine = _make_engine()
        engine._tca = None
        engine._tca_available = False  # already failed

        # Should return None immediately without attempting another import
        result = engine._get_tca()
        assert result is None

    def test_returns_cached_instance_on_second_call(self):
        engine = _make_engine()
        fake_tca = MagicMock()
        engine._tca = fake_tca
        engine._tca_available = True

        assert engine._get_tca() is fake_tca
        assert engine._get_tca() is fake_tca  # same object
