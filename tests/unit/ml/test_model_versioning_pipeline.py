"""Tests for model versioning in training pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantstack.ml.training_service import _cleanup_old_versions


class TestVersionedFilenames:
    def test_cleanup_keeps_5_versions(self, tmp_path):
        """_cleanup_old_versions should keep only 5 most recent files."""
        import time

        # Create 7 model files with staggered mtimes
        for i in range(7):
            model_file = tmp_path / f"SPY_lightgbm_v{i+1}_0.0500.joblib"
            model_file.write_text(f"model_{i}")
            json_file = tmp_path / f"SPY_lightgbm_v{i+1}_0.0500.json"
            json_file.write_text(f'{{"version": {i+1}}}')
            # Ensure different mtimes
            time.sleep(0.01)

        with patch("quantstack.ml.training_service._MODELS_DIR", tmp_path):
            _cleanup_old_versions("SPY", "lightgbm", keep=5)

        remaining = list(tmp_path.glob("SPY_lightgbm_v*.joblib"))
        assert len(remaining) == 5

        # Oldest 2 should be deleted
        assert not (tmp_path / "SPY_lightgbm_v1_0.0500.joblib").exists()
        assert not (tmp_path / "SPY_lightgbm_v2_0.0500.joblib").exists()

        # Newest 5 should remain
        assert (tmp_path / "SPY_lightgbm_v7_0.0500.joblib").exists()

    def test_cleanup_with_fewer_than_keep(self, tmp_path):
        """With fewer files than keep limit, nothing should be deleted."""
        for i in range(3):
            model_file = tmp_path / f"SPY_lightgbm_v{i+1}_0.0500.joblib"
            model_file.write_text(f"model_{i}")

        with patch("quantstack.ml.training_service._MODELS_DIR", tmp_path):
            _cleanup_old_versions("SPY", "lightgbm", keep=5)

        remaining = list(tmp_path.glob("SPY_lightgbm_v*.joblib"))
        assert len(remaining) == 3

    def test_cleanup_also_removes_json(self, tmp_path):
        """JSON metadata files should also be cleaned up."""
        import time

        for i in range(7):
            (tmp_path / f"SPY_lightgbm_v{i+1}_0.0500.joblib").write_text("model")
            (tmp_path / f"SPY_lightgbm_v{i+1}_0.0500.json").write_text("{}")
            time.sleep(0.01)

        with patch("quantstack.ml.training_service._MODELS_DIR", tmp_path):
            _cleanup_old_versions("SPY", "lightgbm", keep=5)

        remaining_json = list(tmp_path.glob("SPY_lightgbm_v*.json"))
        assert len(remaining_json) == 5


class TestFeatureImportanceInResult:
    def test_consensus_features_in_training_config(self):
        """TrainingConfig should have validate_features flag."""
        from quantstack.ml.trainer import TrainingConfig

        config = TrainingConfig(validate_features=True)
        assert config.validate_features is True

    def test_training_result_has_consensus_fields(self):
        """TrainingResult should have consensus_features and importance_by_method."""
        from quantstack.ml.trainer import TrainingResult

        result = TrainingResult(
            model=None,
            feature_names=["f1", "f2"],
            feature_importance={"f1": 0.5, "f2": 0.5},
            metrics={"auc": 0.8},
            cv_scores=[0.7, 0.8],
            consensus_features=["f1"],
            importance_by_method={"mdi": {"f1": 0.6, "f2": 0.4}},
        )
        assert result.consensus_features == ["f1"]
        assert "mdi" in result.importance_by_method
