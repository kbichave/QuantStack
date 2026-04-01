# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for finrl_tools.py — FinRL MCP tools for RL model lifecycle.

Tests cover:
  - finrl_create_environment (validation, config caching)
  - finrl_train_model (env lookup, training flow)
  - finrl_train_ensemble
  - finrl_evaluate_model
  - finrl_predict
  - finrl_list_models
  - finrl_compare_models
  - finrl_get_model_status
  - finrl_promote_model
  - finrl_screen_stocks
  - finrl_screen_options
  - _pick_best helper

All heavy dependencies (FinRLTrainer, ModelRegistry, environments) are mocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.quant_pod.mcp.conftest import _fn


# ---------------------------------------------------------------------------
# finrl_create_environment
# ---------------------------------------------------------------------------


class TestFinrlCreateEnvironment:

    @pytest.mark.asyncio
    async def test_invalid_env_type(self):
        from quantstack.mcp.tools.finrl_tools import finrl_create_environment

        result = await _fn(finrl_create_environment)(env_type="invalid_type")
        assert result["success"] is False
        assert "Invalid env_type" in result["error"]

    @pytest.mark.asyncio
    async def test_stock_trading_requires_symbols(self):
        from quantstack.mcp.tools.finrl_tools import finrl_create_environment

        result = await _fn(finrl_create_environment)(
            env_type="stock_trading",
            start_date="2023-01-01",
            end_date="2024-01-01",
        )
        assert result["success"] is False
        assert "symbols" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_stock_trading_requires_dates(self):
        from quantstack.mcp.tools.finrl_tools import finrl_create_environment

        result = await _fn(finrl_create_environment)(
            env_type="stock_trading",
            symbols=["SPY"],
        )
        assert result["success"] is False
        assert "start_date" in result["error"] or "end_date" in result["error"]

    @pytest.mark.asyncio
    async def test_execution_env_no_requirements(self):
        """execution env type has no symbol/date requirements."""
        from quantstack.mcp.tools.finrl_tools import finrl_create_environment

        result = await _fn(finrl_create_environment)(env_type="execution")
        assert result["success"] is True
        assert result["env_type"] == "execution"
        assert result["env_id"].startswith("env_execution_")

    @pytest.mark.asyncio
    async def test_sizing_env(self):
        from quantstack.mcp.tools.finrl_tools import finrl_create_environment

        result = await _fn(finrl_create_environment)(
            env_type="sizing",
            initial_capital=50_000,
        )
        assert result["success"] is True
        assert result["config"]["initial_capital"] == 50_000

    @pytest.mark.asyncio
    async def test_env_cached(self):
        """Created env config is stored in _env_configs cache."""
        from quantstack.mcp.tools.finrl_tools import finrl_create_environment, _env_configs

        result = await _fn(finrl_create_environment)(env_type="alpha_selection")
        assert result["success"] is True
        env_id = result["env_id"]
        assert env_id in _env_configs

        # Cleanup
        _env_configs.pop(env_id, None)


# ---------------------------------------------------------------------------
# finrl_train_model
# ---------------------------------------------------------------------------


class TestFinrlTrainModel:

    @pytest.mark.asyncio
    async def test_env_not_found(self):
        from quantstack.mcp.tools.finrl_tools import finrl_train_model

        result = await _fn(finrl_train_model)(env_id="nonexistent_env")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.finrl_tools import finrl_train_model, _env_configs

        env_id = "env_test_train"
        _env_configs[env_id] = {
            "env_id": env_id,
            "env_type": "execution",
            "symbols": [],
            "custom_params": {},
        }

        @dataclass
        class MockTrainResult:
            model_id: str = "model_abc123"
            algorithm: str = "ppo"
            checkpoint_path: str = "/tmp/models/model_abc123"
            total_timesteps: int = 10000
            training_time_s: float = 12.5
            metrics: dict = None

            def __post_init__(self):
                if self.metrics is None:
                    self.metrics = {"loss": 0.01}

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MockTrainResult()

        with (
            patch("quantstack.mcp.tools.finrl_tools._build_env", return_value=MagicMock()),
            # FinRLTrainer is imported inside the function body — patch at source
            patch("quantstack.finrl.trainer.FinRLTrainer", return_value=mock_trainer),
            patch("quantstack.mcp.tools.finrl_tools.live_db_or_error", return_value=(None, {"success": False})),
        ):
            result = await _fn(finrl_train_model)(env_id=env_id, algorithm="ppo")

        assert result["success"] is True
        assert result["model_id"] == "model_abc123"
        assert result["status"] == "shadow"

        _env_configs.pop(env_id, None)


# ---------------------------------------------------------------------------
# finrl_train_ensemble
# ---------------------------------------------------------------------------


class TestFinrlTrainEnsemble:

    @pytest.mark.asyncio
    async def test_env_not_found(self):
        from quantstack.mcp.tools.finrl_tools import finrl_train_ensemble

        result = await _fn(finrl_train_ensemble)(env_id="bad_env")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.finrl_tools import finrl_train_ensemble, _env_configs

        env_id = "env_test_ensemble"
        _env_configs[env_id] = {
            "env_id": env_id,
            "env_type": "execution",
            "symbols": [],
            "custom_params": {},
        }

        @dataclass
        class MockResult:
            model_id: str = "ensemble_winner"
            algorithm: str = "a2c"
            checkpoint_path: str = "/tmp/models/ensemble"
            training_time_s: float = 30.0
            metrics: dict = None

            def __post_init__(self):
                if self.metrics is None:
                    self.metrics = {"ensemble_results": {"ppo": 1.0, "a2c": 1.5}}

        mock_trainer = MagicMock()
        mock_trainer.train_ensemble.return_value = MockResult()

        with (
            patch("quantstack.mcp.tools.finrl_tools._build_env", return_value=MagicMock()),
            patch("quantstack.finrl.trainer.FinRLTrainer", return_value=mock_trainer),
        ):
            result = await _fn(finrl_train_ensemble)(env_id=env_id)

        assert result["success"] is True
        assert result["winner_algorithm"] == "a2c"

        _env_configs.pop(env_id, None)


# ---------------------------------------------------------------------------
# finrl_evaluate_model
# ---------------------------------------------------------------------------


class TestFinrlEvaluateModel:

    @pytest.mark.asyncio
    async def test_no_ctx(self):
        from quantstack.mcp.tools.finrl_tools import finrl_evaluate_model

        err = {"success": False, "error": "not initialized"}
        with patch("quantstack.mcp.tools.finrl_tools.live_db_or_error", return_value=(None, err)):
            result = await _fn(finrl_evaluate_model)(model_id="any")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_model_not_found(self):
        from quantstack.mcp.tools.finrl_tools import finrl_evaluate_model

        mock_ctx = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with (
            patch("quantstack.mcp.tools.finrl_tools.live_db_or_error", return_value=(mock_ctx, None)),
            patch("quantstack.finrl.model_registry.ModelRegistry", return_value=mock_registry),
        ):
            result = await _fn(finrl_evaluate_model)(model_id="missing")

        assert result["success"] is False
        assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# finrl_predict
# ---------------------------------------------------------------------------


class TestFinrlPredict:

    @pytest.mark.asyncio
    async def test_no_ctx(self):
        from quantstack.mcp.tools.finrl_tools import finrl_predict

        err = {"success": False, "error": "not initialized"}
        with patch("quantstack.mcp.tools.finrl_tools.live_db_or_error", return_value=(None, err)):
            result = await _fn(finrl_predict)(model_id="any")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_model_not_found(self):
        from quantstack.mcp.tools.finrl_tools import finrl_predict

        mock_ctx = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with (
            patch("quantstack.mcp.tools.finrl_tools.live_db_or_error", return_value=(mock_ctx, None)),
            patch("quantstack.finrl.model_registry.ModelRegistry", return_value=mock_registry),
        ):
            result = await _fn(finrl_predict)(model_id="missing")

        assert result["success"] is False
        assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# finrl_list_models
# ---------------------------------------------------------------------------


class TestFinrlListModels:

    @pytest.mark.asyncio
    async def test_no_ctx(self):
        from quantstack.mcp.tools.finrl_tools import finrl_list_models

        err = {"success": False, "error": "not initialized"}
        with patch("quantstack.mcp.tools.finrl_tools.live_db_or_error", return_value=(None, err)):
            result = await _fn(finrl_list_models)()

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_happy_path(self):
        from quantstack.mcp.tools.finrl_tools import finrl_list_models

        mock_ctx = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_models.return_value = [
            {"model_id": "m1", "status": "shadow", "algorithm": "ppo"},
            {"model_id": "m2", "status": "live", "algorithm": "a2c"},
        ]

        with (
            patch("quantstack.mcp.tools.finrl_tools.live_db_or_error", return_value=(mock_ctx, None)),
            patch("quantstack.finrl.model_registry.ModelRegistry", return_value=mock_registry),
        ):
            result = await _fn(finrl_list_models)()

        assert result["success"] is True
        assert result["count"] == 2


# ---------------------------------------------------------------------------
# finrl_compare_models
# ---------------------------------------------------------------------------


class TestFinrlCompareModels:

    @pytest.mark.asyncio
    async def test_delegates_to_evaluate(self):
        """compare_models calls finrl_evaluate_model for each model."""
        from quantstack.mcp.tools.finrl_tools import finrl_compare_models

        async def mock_eval(model_id, **kwargs):
            return {
                "success": True,
                "model_id": model_id,
                "metrics": {"sharpe_ratio": 1.5 if model_id == "m1" else 0.8},
            }

        with patch("quantstack.mcp.tools.finrl_tools.finrl_evaluate_model", side_effect=mock_eval):
            result = await _fn(finrl_compare_models)(model_ids=["m1", "m2"])

        assert result["success"] is True
        assert result["recommendation"]["best_model_id"] == "m1"
        assert result["recommendation"]["best_sharpe"] == 1.5


# ---------------------------------------------------------------------------
# finrl_get_model_status
# ---------------------------------------------------------------------------


class TestFinrlGetModelStatus:

    @pytest.mark.asyncio
    async def test_no_ctx(self):
        from quantstack.mcp.tools.finrl_tools import finrl_get_model_status

        err = {"success": False, "error": "not initialized"}
        with patch("quantstack.mcp.tools.finrl_tools.live_db_or_error", return_value=(None, err)):
            result = await _fn(finrl_get_model_status)(model_id="any")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_model_not_found(self):
        from quantstack.mcp.tools.finrl_tools import finrl_get_model_status

        mock_ctx = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with (
            patch("quantstack.mcp.tools.finrl_tools.live_db_or_error", return_value=(mock_ctx, None)),
            patch("quantstack.finrl.model_registry.ModelRegistry", return_value=mock_registry),
        ):
            result = await _fn(finrl_get_model_status)(model_id="missing")

        assert result["success"] is False


# ---------------------------------------------------------------------------
# finrl_promote_model
# ---------------------------------------------------------------------------


class TestFinrlPromoteModel:

    @pytest.mark.asyncio
    async def test_no_ctx(self):
        from quantstack.mcp.tools.finrl_tools import finrl_promote_model

        err = {"success": False, "error": "not initialized"}
        with patch("quantstack.mcp.tools.finrl_tools.live_db_or_error", return_value=(None, err)):
            result = await _fn(finrl_promote_model)(model_id="any", evidence="test")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_model_not_found(self):
        from quantstack.mcp.tools.finrl_tools import finrl_promote_model

        mock_ctx = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with (
            patch("quantstack.mcp.tools.finrl_tools.live_db_or_error", return_value=(mock_ctx, None)),
            patch("quantstack.finrl.model_registry.ModelRegistry", return_value=mock_registry),
        ):
            result = await _fn(finrl_promote_model)(model_id="missing", evidence="test")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_not_shadow_rejected(self):
        """Promoting a non-shadow model returns error."""
        from quantstack.mcp.tools.finrl_tools import finrl_promote_model

        mock_ctx = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = {
            "model_id": "m1",
            "status": "live",
            "algorithm": "ppo",
        }

        with (
            patch("quantstack.mcp.tools.finrl_tools.live_db_or_error", return_value=(mock_ctx, None)),
            patch("quantstack.finrl.model_registry.ModelRegistry", return_value=mock_registry),
        ):
            result = await _fn(finrl_promote_model)(model_id="m1", evidence="test")

        assert result["success"] is False
        assert "shadow" in result["error"].lower()


# ---------------------------------------------------------------------------
# finrl_screen_stocks
# ---------------------------------------------------------------------------


class TestFinrlScreenStocks:

    @pytest.mark.asyncio
    async def test_no_data(self):
        """Empty DataFrame returns error."""
        from quantstack.mcp.tools.finrl_tools import finrl_screen_stocks

        import pandas as pd

        mock_adapter = MagicMock()
        mock_adapter.fetch_and_format.return_value = pd.DataFrame()

        with patch("quantstack.finrl.data_adapter.FinRLDataAdapter", return_value=mock_adapter):
            result = await _fn(finrl_screen_stocks)(
                symbols=["SPY"], start_date="2023-01-01", end_date="2024-01-01",
            )

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_error(self):
        """When adapter fetch raises, returns error."""
        from quantstack.mcp.tools.finrl_tools import finrl_screen_stocks

        mock_adapter = MagicMock()
        mock_adapter.fetch_and_format.side_effect = RuntimeError("data unavailable")

        with patch("quantstack.finrl.data_adapter.FinRLDataAdapter", return_value=mock_adapter):
            result = await _fn(finrl_screen_stocks)(
                symbols=["SPY"], start_date="2023-01-01", end_date="2024-01-01",
            )

        assert result["success"] is False


# ---------------------------------------------------------------------------
# finrl_screen_options
# ---------------------------------------------------------------------------


class TestFinrlScreenOptions:

    @pytest.mark.asyncio
    async def test_no_data(self):
        from quantstack.mcp.tools.finrl_tools import finrl_screen_options

        import pandas as pd

        mock_adapter = MagicMock()
        mock_adapter.fetch_and_format.return_value = pd.DataFrame()

        with patch("quantstack.finrl.data_adapter.FinRLDataAdapter", return_value=mock_adapter):
            result = await _fn(finrl_screen_options)(
                symbols=["SPY"], start_date="2023-01-01", end_date="2024-01-01",
            )

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_error(self):
        """When adapter fetch raises, returns error."""
        from quantstack.mcp.tools.finrl_tools import finrl_screen_options

        mock_adapter = MagicMock()
        mock_adapter.fetch_and_format.side_effect = RuntimeError("data unavailable")

        with patch("quantstack.finrl.data_adapter.FinRLDataAdapter", return_value=mock_adapter):
            result = await _fn(finrl_screen_options)(
                symbols=["SPY"], start_date="2023-01-01", end_date="2024-01-01",
            )

        assert result["success"] is False


# ---------------------------------------------------------------------------
# _pick_best helper
# ---------------------------------------------------------------------------


class TestPickBest:

    def test_picks_highest_sharpe(self):
        from quantstack.mcp.tools.finrl_tools import _pick_best

        results = {
            "m1": {"sharpe_ratio": 1.2, "max_drawdown": -0.15},
            "m2": {"sharpe_ratio": 2.1, "max_drawdown": -0.10},
            "m3": {"error": "evaluation failed"},
        }
        best = _pick_best(results)
        assert best["best_model_id"] == "m2"
        assert best["best_sharpe"] == 2.1

    def test_all_errors(self):
        from quantstack.mcp.tools.finrl_tools import _pick_best

        results = {
            "m1": {"error": "failed"},
            "m2": {"error": "also failed"},
        }
        best = _pick_best(results)
        assert best["best_model_id"] is None
        assert best["best_sharpe"] is None

    def test_empty(self):
        from quantstack.mcp.tools.finrl_tools import _pick_best

        best = _pick_best({})
        assert best["best_model_id"] is None
