# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ML pipeline MCP tools — train, predict, model registry,
drift detection, feature store, model QA.

Heavy ML libraries (lightgbm, xgboost, catboost, joblib, optuna, shap)
are mocked to avoid real training. Tests focus on control flow, error
handling, and data pipeline correctness.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from tests.quant_pod.mcp.conftest import _fn, synthetic_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_pg_data_store(ohlcv_df=None):
    """Return a mock PgDataStore that returns the given OHLCV DataFrame."""
    store = MagicMock()
    store.load_ohlcv.return_value = ohlcv_df
    store.close.return_value = None
    return store


def _patch_pg_data_store(ohlcv_df=None):
    """Patch PgDataStore() constructor to return a mock with given data."""
    store = _mock_pg_data_store(ohlcv_df)
    return patch(
        "quantstack.mcp.tools.ml.PgDataStore",
        return_value=store,
    )


def _patch_live_db_ok():
    """Patch live_db_or_error() to return (mock_ctx, None)."""
    return patch(
        "quantstack.mcp.tools.ml.live_db_or_error",
        return_value=(MagicMock(), None),
    )


def _patch_live_db_error(msg="DB unavailable"):
    """Patch live_db_or_error() to return (None, error_dict)."""
    return patch(
        "quantstack.mcp.tools.ml.live_db_or_error",
        return_value=(None, {"success": False, "error": msg}),
    )


def _make_model_metadata(symbol="SPY", **overrides):
    """Build a model metadata dict suitable for writing to JSON."""
    meta = {
        "symbol": symbol,
        "model_type": "lightgbm",
        "feature_names": ["rsi_14", "macd_signal", "sma_20"],
        "feature_tiers": ["technical"],
        "features_total": 3,
        "features_after_filter": 3,
        "features_dropped": [],
        "accuracy": 0.62,
        "auc": 0.67,
        "cv_scores": {"accuracy": [0.60, 0.61, 0.63], "auc": [0.65, 0.67, 0.69]},
        "label_method": "event",
        "lookback_days": 756,
        "training_samples": 500,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "apply_causal_filter": False,
        "feature_whitelist": None,
    }
    meta.update(overrides)
    return meta


# ---------------------------------------------------------------------------
# get_ml_model_status
# ---------------------------------------------------------------------------


class TestGetMlModelStatus:
    @pytest.mark.asyncio
    async def test_no_models_directory(self, tmp_path):
        """When models dir doesn't exist, return empty list."""
        from quantstack.mcp.tools.ml import get_ml_model_status

        nonexistent = tmp_path / "nonexistent_models"
        with patch("quantstack.mcp.tools.ml._MODELS_DIR", nonexistent):
            result = await _fn(get_ml_model_status)(symbol=None)
        assert result["success"] is True
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_specific_symbol_not_found(self, tmp_path):
        """When requesting status for a specific symbol that has no model."""
        from quantstack.mcp.tools.ml import get_ml_model_status

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir):
            result = await _fn(get_ml_model_status)(symbol="NOMODEL")
        assert result["success"] is True
        assert result["total"] == 0
        assert "No model found" in result.get("message", "")

    @pytest.mark.asyncio
    async def test_model_found_and_stale(self, tmp_path):
        """Model with old trained_at should be flagged as needs_retrain."""
        from quantstack.mcp.tools.ml import get_ml_model_status

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Write metadata with old trained_at (60 days ago)
        meta = _make_model_metadata(
            symbol="AAPL",
            trained_at="2024-01-01T00:00:00+00:00",
        )
        (models_dir / "AAPL_latest.json").write_text(json.dumps(meta))
        # Also write a dummy .joblib so model_exists is True
        (models_dir / "AAPL_latest.joblib").write_bytes(b"dummy")

        with patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir):
            result = await _fn(get_ml_model_status)(symbol="AAPL")
        assert result["success"] is True
        assert result["total"] == 1
        model_info = result["models"][0]
        assert model_info["needs_retrain"] is True
        assert model_info["model_exists"] is True
        assert model_info["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_scan_all_models(self, tmp_path):
        """When symbol=None, scan all metadata files."""
        from quantstack.mcp.tools.ml import get_ml_model_status

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        for sym in ("AAPL", "MSFT"):
            meta = _make_model_metadata(symbol=sym)
            (models_dir / f"{sym}_latest.json").write_text(json.dumps(meta))
            (models_dir / f"{sym}_latest.joblib").write_bytes(b"dummy")

        with patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir):
            result = await _fn(get_ml_model_status)(symbol=None)
        assert result["success"] is True
        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_corrupt_metadata_skipped(self, tmp_path):
        """Corrupt JSON metadata should be silently skipped."""
        from quantstack.mcp.tools.ml import get_ml_model_status

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "BAD_latest.json").write_text("not valid json {{{")

        with patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir):
            result = await _fn(get_ml_model_status)(symbol=None)
        assert result["success"] is True
        assert result["total"] == 0


# ---------------------------------------------------------------------------
# train_ml_model — tests via _train_sync
# ---------------------------------------------------------------------------


class TestTrainMlModel:
    @pytest.mark.asyncio
    async def test_insufficient_data(self):
        """Return error when OHLCV has fewer than 100 bars."""
        from quantstack.mcp.tools.ml import train_ml_model

        short_df = synthetic_ohlcv("SPY", n_days=50)
        with _patch_pg_data_store(short_df):
            result = await _fn(train_ml_model)(symbol="SPY", save=False)
        assert result["success"] is False
        assert "Insufficient" in result["error"]

    @pytest.mark.asyncio
    async def test_none_ohlcv(self):
        """Return error when store returns None."""
        from quantstack.mcp.tools.ml import train_ml_model

        with _patch_pg_data_store(None):
            result = await _fn(train_ml_model)(symbol="SPY", save=False)
        assert result["success"] is False
        assert "Insufficient" in result["error"]

    @pytest.mark.asyncio
    async def test_label_generation_failure(self):
        """When labeler fails, the pipeline should handle gracefully."""
        from quantstack.mcp.tools.ml import train_ml_model

        df = synthetic_ohlcv("SPY", n_days=200)
        with (
            _patch_pg_data_store(df),
            patch(
                "quantstack.mcp.tools.ml.TechnicalIndicators"
            ) as mock_ti_cls,
            patch("quantstack.mcp.tools.ml._generate_labels") as mock_labels,
        ):
            mock_ti = MagicMock()
            # Return a DataFrame with numeric feature columns
            features_df = df.copy()
            features_df["rsi_14"] = 50.0
            features_df["macd_signal"] = 0.1
            mock_ti.compute.return_value = features_df
            mock_ti_cls.return_value = mock_ti
            # _generate_labels returns df without label_long column
            mock_labels.return_value = features_df

            result = await _fn(train_ml_model)(
                symbol="SPY",
                save=False,
                feature_tiers=["technical"],
            )
        assert result["success"] is False
        assert "label" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_happy_path_with_mocked_trainer(self, tmp_path):
        """Full pipeline with mocked ModelTrainer and save."""
        from quantstack.mcp.tools.ml import train_ml_model

        df = synthetic_ohlcv("SPY", n_days=200)

        features_df = df.copy()
        features_df["rsi_14"] = 50.0
        features_df["macd_signal"] = 0.1
        features_df["label_long"] = np.random.choice([0, 1], size=len(df))

        mock_train_result = MagicMock()
        mock_train_result.model = MagicMock()
        mock_train_result.metrics = {"accuracy": 0.62, "auc": 0.67}
        mock_train_result.cv_scores = [0.60, 0.62, 0.64]
        mock_train_result.feature_importance = {"rsi_14": 0.5, "macd_signal": 0.3}

        models_dir = tmp_path / "models"

        with (
            _patch_pg_data_store(df),
            patch("quantstack.mcp.tools.ml.TechnicalIndicators") as mock_ti_cls,
            patch("quantstack.mcp.tools.ml._generate_labels") as mock_labels,
            patch("quantstack.mcp.tools.ml.FeatureTiers") as mock_ft_cls,
            patch("quantstack.mcp.tools.ml.ModelTrainer") as mock_trainer_cls,
            patch("quantstack.mcp.tools.ml.joblib") as mock_joblib,
            patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir),
        ):
            mock_ti = MagicMock()
            mock_ti.compute.return_value = features_df
            mock_ti_cls.return_value = mock_ti

            mock_labels.return_value = features_df

            mock_ft = MagicMock()
            mock_ft.any_active.return_value = False
            mock_ft_cls.return_value = mock_ft

            mock_trainer = MagicMock()
            mock_trainer.train.return_value = mock_train_result
            mock_trainer_cls.return_value = mock_trainer

            result = await _fn(train_ml_model)(
                symbol="SPY",
                save=True,
                feature_tiers=["technical"],
            )

        assert result["success"] is True
        assert result["symbol"] == "SPY"
        assert "test_accuracy" in result
        assert "test_auc" in result
        assert result["model_path"] is not None

    @pytest.mark.asyncio
    async def test_feature_whitelist_no_match(self):
        """When all whitelisted features are missing, return error."""
        from quantstack.mcp.tools.ml import train_ml_model

        df = synthetic_ohlcv("SPY", n_days=200)

        features_df = df.copy()
        features_df["rsi_14"] = 50.0
        features_df["label_long"] = np.random.choice([0, 1], size=len(df))

        with (
            _patch_pg_data_store(df),
            patch("quantstack.mcp.tools.ml.TechnicalIndicators") as mock_ti_cls,
            patch("quantstack.mcp.tools.ml._generate_labels") as mock_labels,
            patch("quantstack.mcp.tools.ml.FeatureTiers") as mock_ft_cls,
        ):
            mock_ti = MagicMock()
            mock_ti.compute.return_value = features_df
            mock_ti_cls.return_value = mock_ti

            mock_labels.return_value = features_df

            mock_ft = MagicMock()
            mock_ft.any_active.return_value = False
            mock_ft_cls.return_value = mock_ft

            result = await _fn(train_ml_model)(
                symbol="SPY",
                save=False,
                feature_tiers=["technical"],
                feature_whitelist=["nonexistent_feature_abc"],
            )

        assert result["success"] is False
        assert "whitelist" in result["error"].lower()


# ---------------------------------------------------------------------------
# predict_ml_signal
# ---------------------------------------------------------------------------


class TestPredictMlSignal:
    @pytest.mark.asyncio
    async def test_no_model_file(self, tmp_path):
        """Return error when model .joblib doesn't exist."""
        from quantstack.mcp.tools.ml import predict_ml_signal

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir):
            result = await _fn(predict_ml_signal)(symbol="NOMODEL")
        assert result["success"] is False
        assert "No trained model" in result["error"]

    @pytest.mark.asyncio
    async def test_insufficient_data_for_prediction(self, tmp_path):
        """Return error when OHLCV is too short for prediction."""
        from quantstack.mcp.tools.ml import predict_ml_signal

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Write model and metadata
        meta = _make_model_metadata()
        (models_dir / "SPY_latest.json").write_text(json.dumps(meta))

        short_df = synthetic_ohlcv("SPY", n_days=30)

        with (
            patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir),
            patch("quantstack.mcp.tools.ml.joblib") as mock_joblib,
            _patch_pg_data_store(short_df),
        ):
            mock_joblib.load.return_value = MagicMock()
            # Write dummy joblib file so path.exists() returns True
            (models_dir / "SPY_latest.joblib").write_bytes(b"dummy")

            result = await _fn(predict_ml_signal)(symbol="SPY")
        assert result["success"] is False
        assert "Insufficient" in result["error"]

    @pytest.mark.asyncio
    async def test_happy_path_prediction(self, tmp_path):
        """Full prediction pipeline with mocked model."""
        from quantstack.mcp.tools.ml import predict_ml_signal

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        feature_names = ["rsi_14", "macd_signal", "sma_20"]
        meta = _make_model_metadata(feature_names=feature_names)
        (models_dir / "SPY_latest.json").write_text(json.dumps(meta))
        (models_dir / "SPY_latest.joblib").write_bytes(b"dummy")

        df = synthetic_ohlcv("SPY", n_days=252)

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])

        features_df = df.copy()
        for col in feature_names:
            features_df[col] = np.random.randn(len(df))

        with (
            patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir),
            patch("quantstack.mcp.tools.ml.joblib") as mock_joblib,
            _patch_pg_data_store(df),
            patch("quantstack.mcp.tools.ml.TechnicalIndicators") as mock_ti_cls,
            patch("quantstack.mcp.tools.ml.FeatureTiers") as mock_ft_cls,
        ):
            mock_joblib.load.return_value = mock_model
            mock_ti = MagicMock()
            mock_ti.compute.return_value = features_df
            mock_ti_cls.return_value = mock_ti
            mock_ft = MagicMock()
            mock_ft.any_active.return_value = False
            mock_ft_cls.return_value = mock_ft

            result = await _fn(predict_ml_signal)(symbol="SPY")

        assert result["success"] is True
        assert result["symbol"] == "SPY"
        assert result["direction"] == "bullish"  # 0.7 > 0.55
        assert 0.0 <= result["probability"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["top_features"]) > 0


# ---------------------------------------------------------------------------
# register_model
# ---------------------------------------------------------------------------


class TestRegisterModel:
    @pytest.mark.asyncio
    async def test_db_unavailable(self):
        """Return error when live_db_or_error fails."""
        from quantstack.mcp.tools.ml import register_model

        with _patch_live_db_error():
            result = await _fn(register_model)(
                symbol="SPY",
                model_path="/tmp/fake.joblib",
            )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_happy_path_first_registration(self, tmp_path):
        """First model registration should auto-promote as champion."""
        from quantstack.mcp.tools.ml import register_model

        models_dir = tmp_path / "models"
        # Create a source model file
        src_model = tmp_path / "src_model.joblib"
        src_model.write_bytes(b"model_data")

        mock_conn = MagicMock()
        # execute() returns self (like PgConnection.execute)
        mock_conn.execute.return_value = mock_conn
        # First fetchone: MAX(model_version) = 0 (no prior versions)
        # Second fetchone: champion query returns None (no existing champion)
        mock_conn.fetchone.side_effect = [(0,), None]

        with (
            _patch_live_db_ok(),
            patch("quantstack.mcp.tools.ml.pg_conn") as mock_pg,
            patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir),
            patch("quantstack.mcp.tools.ml._REGISTRY_TABLE_CREATED", True),
        ):
            mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_pg.return_value.__exit__ = MagicMock(return_value=False)

            result = await _fn(register_model)(
                symbol="SPY",
                model_path=str(src_model),
                metadata={"auc": 0.68, "model_type": "lightgbm"},
            )

        assert result["success"] is True
        assert result["version"] == 1
        assert result["promoted"] is True


# ---------------------------------------------------------------------------
# get_model_history
# ---------------------------------------------------------------------------


class TestGetModelHistory:
    @pytest.mark.asyncio
    async def test_db_unavailable(self):
        """Return error when live_db_or_error fails."""
        from quantstack.mcp.tools.ml import get_model_history

        with _patch_live_db_error():
            result = await _fn(get_model_history)(symbol="SPY")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_empty_history(self):
        """Return empty versions when no models registered."""
        from quantstack.mcp.tools.ml import get_model_history

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_conn
        mock_conn.fetchall.return_value = []

        with (
            _patch_live_db_ok(),
            patch("quantstack.mcp.tools.ml.pg_conn") as mock_pg,
            patch("quantstack.mcp.tools.ml._REGISTRY_TABLE_CREATED", True),
        ):
            mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_pg.return_value.__exit__ = MagicMock(return_value=False)

            result = await _fn(get_model_history)(symbol="SPY")

        assert result["success"] is True
        assert result["total"] == 0
        assert result["versions"] == []


# ---------------------------------------------------------------------------
# rollback_model
# ---------------------------------------------------------------------------


class TestRollbackModel:
    @pytest.mark.asyncio
    async def test_db_unavailable(self):
        """Return error when live_db_or_error fails."""
        from quantstack.mcp.tools.ml import rollback_model

        with _patch_live_db_error():
            result = await _fn(rollback_model)(symbol="SPY", version=1)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_version_not_found(self):
        """Return error when target version doesn't exist."""
        from quantstack.mcp.tools.ml import rollback_model

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_conn
        mock_conn.fetchone.return_value = None

        with (
            _patch_live_db_ok(),
            patch("quantstack.mcp.tools.ml.pg_conn") as mock_pg,
            patch("quantstack.mcp.tools.ml._REGISTRY_TABLE_CREATED", True),
        ):
            mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_pg.return_value.__exit__ = MagicMock(return_value=False)

            result = await _fn(rollback_model)(symbol="SPY", version=99)

        assert result["success"] is False
        assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    @pytest.mark.asyncio
    async def test_db_unavailable(self):
        """Return error when live_db_or_error fails."""
        from quantstack.mcp.tools.ml import compare_models

        with _patch_live_db_error():
            result = await _fn(compare_models)(
                symbol="SPY", version_a=1, version_b=2
            )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_version_a_not_found(self):
        """Return error when version_a doesn't exist in registry."""
        from quantstack.mcp.tools.ml import compare_models

        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_conn
        # Both fetches return None
        mock_conn.fetchone.return_value = None

        with (
            _patch_live_db_ok(),
            patch("quantstack.mcp.tools.ml.pg_conn") as mock_pg,
            patch("quantstack.mcp.tools.ml._REGISTRY_TABLE_CREATED", True),
        ):
            mock_pg.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_pg.return_value.__exit__ = MagicMock(return_value=False)

            result = await _fn(compare_models)(
                symbol="SPY", version_a=1, version_b=2
            )

        assert result["success"] is False
        assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# check_concept_drift
# ---------------------------------------------------------------------------


class TestCheckConceptDrift:
    @pytest.mark.asyncio
    async def test_no_model_metadata(self, tmp_path):
        """Return error when model metadata file doesn't exist."""
        from quantstack.mcp.tools.ml import check_concept_drift

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir):
            result = await _fn(check_concept_drift)(symbol="NOMODEL")
        assert result["success"] is False
        assert "metadata" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_no_feature_names_in_metadata(self, tmp_path):
        """Return error when metadata has empty feature_names."""
        from quantstack.mcp.tools.ml import check_concept_drift

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        meta = _make_model_metadata(feature_names=[])
        (models_dir / "SPY_latest.json").write_text(json.dumps(meta))

        with patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir):
            result = await _fn(check_concept_drift)(symbol="SPY")
        assert result["success"] is False
        assert "feature names" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_insufficient_data_for_drift(self, tmp_path):
        """Return error when OHLCV is too short for drift check."""
        from quantstack.mcp.tools.ml import check_concept_drift

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        meta = _make_model_metadata()
        (models_dir / "SPY_latest.json").write_text(json.dumps(meta))

        short_df = synthetic_ohlcv("SPY", n_days=50)
        with (
            patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir),
            _patch_pg_data_store(short_df),
        ):
            result = await _fn(check_concept_drift)(
                symbol="SPY", window_days=30
            )
        assert result["success"] is False
        assert "Insufficient" in result["error"]


# ---------------------------------------------------------------------------
# review_model_quality — bug detection
# ---------------------------------------------------------------------------


class TestReviewModelQuality:
    @pytest.mark.asyncio
    async def test_model_not_found(self, tmp_path):
        """Return error when model file doesn't exist."""
        from quantstack.mcp.tools.ml import review_model_quality

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir):
            result = await _fn(review_model_quality)(symbol="NOMODEL")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_metadata_not_found(self, tmp_path):
        """Return error when metadata JSON doesn't exist but model does."""
        from quantstack.mcp.tools.ml import review_model_quality

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "SPY_latest.joblib").write_bytes(b"dummy")

        with patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir):
            result = await _fn(review_model_quality)(symbol="SPY")
        assert result["success"] is False
        assert "Metadata not found" in result["error"]

    @pytest.mark.asyncio
    async def test_low_auc_rejected(self, tmp_path):
        """Model with AUC < 0.52 should be rejected."""
        from quantstack.mcp.tools.ml import review_model_quality

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        meta = _make_model_metadata(
            auc=0.51,
            accuracy=0.51,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            cv_scores={"accuracy": [0.50, 0.51, 0.52], "auc": [0.50, 0.51, 0.52]},
        )
        (models_dir / "SPY_latest.json").write_text(json.dumps(meta))
        (models_dir / "SPY_latest.joblib").write_bytes(b"dummy")

        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        with (
            patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir),
            patch("quantstack.mcp.tools.ml.joblib") as mock_joblib,
            _patch_pg_data_store(None),  # skip calibration data
        ):
            mock_joblib.load.return_value = mock_model
            result = await _fn(review_model_quality)(symbol="SPY")

        assert result["success"] is True
        assert result["verdict"] == "reject"

    @pytest.mark.asyncio
    async def test_good_model_accepted(self, tmp_path):
        """Model with all quality checks passing should be accepted."""
        from quantstack.mcp.tools.ml import review_model_quality

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        meta = _make_model_metadata(
            auc=0.72,
            accuracy=0.68,
            feature_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
            cv_scores={"accuracy": [0.66, 0.68, 0.70], "auc": [0.70, 0.72, 0.74]},
        )
        (models_dir / "SPY_latest.json").write_text(json.dumps(meta))
        (models_dir / "SPY_latest.joblib").write_bytes(b"dummy")

        mock_model = MagicMock()
        # Feature importances with no single dominant feature
        mock_model.feature_importances_ = np.array(
            [0.12, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.09, 0.09, 0.09]
        )

        with (
            patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir),
            patch("quantstack.mcp.tools.ml.joblib") as mock_joblib,
            _patch_pg_data_store(None),  # skip calibration data
        ):
            mock_joblib.load.return_value = mock_model
            result = await _fn(review_model_quality)(symbol="SPY")

        assert result["success"] is True
        assert result["verdict"] == "accept"
        assert result["score"] > 70

    @pytest.mark.asyncio
    async def test_feature_concentration_flag(self, tmp_path):
        """Model where one feature > 30% importance should flag concentration."""
        from quantstack.mcp.tools.ml import review_model_quality

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        meta = _make_model_metadata(
            auc=0.70,
            feature_names=["dominant_feature", "f2", "f3", "f4", "f5"],
            cv_scores={"accuracy": [0.66, 0.68, 0.70], "auc": [0.68, 0.70, 0.72]},
        )
        (models_dir / "SPY_latest.json").write_text(json.dumps(meta))
        (models_dir / "SPY_latest.joblib").write_bytes(b"dummy")

        mock_model = MagicMock()
        # One dominant feature: 80% importance
        mock_model.feature_importances_ = np.array([0.80, 0.05, 0.05, 0.05, 0.05])

        with (
            patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir),
            patch("quantstack.mcp.tools.ml.joblib") as mock_joblib,
            _patch_pg_data_store(None),
        ):
            mock_joblib.load.return_value = mock_model
            result = await _fn(review_model_quality)(symbol="SPY")

        assert result["success"] is True
        concentration_check = next(
            c for c in result["checks"] if c["name"] == "feature_concentration"
        )
        assert concentration_check["passed"] is False
        assert "dominant_feature" in result["recommended_changes"]["features_to_drop"]

    @pytest.mark.asyncio
    async def test_class_balance_check_with_proba_not_in_scope(self, tmp_path):
        """BUG TEST: class_balance check uses `'proba' in dir()` which is
        unreliable for checking local variable existence.

        When calibration data is unavailable (PgDataStore returns None),
        `proba` is never defined in the local scope of
        _review_model_quality_sync. The `dir()` check on line 1809 should
        use `locals()` instead. This test verifies the class_balance check
        still produces a sensible result even when proba is undefined.
        """
        from quantstack.mcp.tools.ml import review_model_quality

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # All CV accuracies are different, so the first class_balance heuristic passes.
        # The second check (proba-based) depends on the buggy `dir()` call.
        meta = _make_model_metadata(
            auc=0.65,
            accuracy=0.62,
            feature_names=["f1", "f2", "f3", "f4", "f5", "f6"],
            cv_scores={
                "accuracy": [0.60, 0.62, 0.64],
                "auc": [0.63, 0.65, 0.67],
            },
        )
        (models_dir / "SPY_latest.json").write_text(json.dumps(meta))
        (models_dir / "SPY_latest.joblib").write_bytes(b"dummy")

        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array(
            [0.20, 0.18, 0.17, 0.16, 0.15, 0.14]
        )

        with (
            patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir),
            patch("quantstack.mcp.tools.ml.joblib") as mock_joblib,
            _patch_pg_data_store(None),  # No calibration data → proba never defined
        ):
            mock_joblib.load.return_value = mock_model
            result = await _fn(review_model_quality)(symbol="SPY")

        assert result["success"] is True
        # The class_balance check should still be present
        class_balance_check = next(
            c for c in result["checks"] if c["name"] == "class_balance"
        )
        assert isinstance(class_balance_check["passed"], bool)


# ---------------------------------------------------------------------------
# compute_and_store_features
# ---------------------------------------------------------------------------


class TestComputeAndStoreFeatures:
    @pytest.mark.asyncio
    async def test_insufficient_data(self):
        """Return error when OHLCV has < 50 bars."""
        from quantstack.mcp.tools.ml import compute_and_store_features

        short_df = synthetic_ohlcv("SPY", n_days=30)
        with _patch_pg_data_store(short_df):
            result = await _fn(compute_and_store_features)(symbol="SPY")
        assert result["success"] is False
        assert "Insufficient" in result["error"]

    @pytest.mark.asyncio
    async def test_none_data(self):
        """Return error when store returns None."""
        from quantstack.mcp.tools.ml import compute_and_store_features

        with _patch_pg_data_store(None):
            result = await _fn(compute_and_store_features)(symbol="SPY")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_db_unavailable_for_storage(self):
        """Return error when DB is unavailable for feature storage."""
        from quantstack.mcp.tools.ml import compute_and_store_features

        df = synthetic_ohlcv("SPY", n_days=100)
        features_df = df.copy()
        features_df["rsi_14"] = 50.0

        with (
            _patch_pg_data_store(df),
            patch("quantstack.mcp.tools.ml.TechnicalIndicators") as mock_ti_cls,
            patch("quantstack.mcp.tools.ml.FeatureTiers") as mock_ft_cls,
            _patch_live_db_error(),
        ):
            mock_ti = MagicMock()
            mock_ti.compute.return_value = features_df
            mock_ti_cls.return_value = mock_ti
            mock_ft = MagicMock()
            mock_ft.any_active.return_value = False
            mock_ft_cls.return_value = mock_ft

            result = await _fn(compute_and_store_features)(
                symbol="SPY",
                tiers=["technical"],
            )
        assert result["success"] is False
        assert "DB unavailable" in result["error"]


# ---------------------------------------------------------------------------
# get_feature_lineage
# ---------------------------------------------------------------------------


class TestGetFeatureLineage:
    @pytest.mark.asyncio
    async def test_no_data_found(self, tmp_path):
        """Return error when no feature data exists and DB is down."""
        from quantstack.mcp.tools.ml import get_feature_lineage

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        with (
            _patch_live_db_error(),
            patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir),
        ):
            result = await _fn(get_feature_lineage)(symbol="NOMODEL")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_fallback_to_metadata_file(self, tmp_path):
        """When DB is unavailable but metadata file exists, use it."""
        from quantstack.mcp.tools.ml import get_feature_lineage

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        meta = _make_model_metadata(feature_names=["rsi_14", "macd_signal"])
        (models_dir / "SPY_latest.json").write_text(json.dumps(meta))

        with (
            _patch_live_db_error(),
            patch("quantstack.mcp.tools.ml._MODELS_DIR", models_dir),
        ):
            result = await _fn(get_feature_lineage)(symbol="SPY")
        assert result["success"] is True
        assert "rsi_14" in result["feature_names"]
        assert result["source"] == "metadata_file"

    @pytest.mark.asyncio
    async def test_model_version_lookup_db_unavailable(self):
        """When requesting specific model_version and DB is down."""
        from quantstack.mcp.tools.ml import get_feature_lineage

        with _patch_live_db_error():
            result = await _fn(get_feature_lineage)(
                symbol="SPY", model_version=1
            )
        assert result["success"] is False
        assert "DB unavailable" in result["error"]
