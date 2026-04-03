"""ML models module for trade classification."""

from quantstack.ml.ensemble import HierarchicalEnsemble
from quantstack.ml.explainer import SHAPExplainer
from quantstack.ml.feature_importance import FeatureImportanceProtocol
from quantstack.ml.fractional_diff import batch_find_min_d, frac_diff, find_min_d
from quantstack.ml.labeling import MetaLabeler, label_series, triple_barrier_label
from quantstack.ml.predictor import Predictor
from quantstack.ml.trainer import ModelTrainer, TrainingConfig

__all__ = [
    "ModelTrainer",
    "TrainingConfig",
    "Predictor",
    "HierarchicalEnsemble",
    "SHAPExplainer",
    "FeatureImportanceProtocol",
    "frac_diff",
    "find_min_d",
    "batch_find_min_d",
    "triple_barrier_label",
    "label_series",
    "MetaLabeler",
]
