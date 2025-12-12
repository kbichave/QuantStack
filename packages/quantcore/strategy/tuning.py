"""
Hyperparameter tuning for ML and RL strategies.

CRITICAL: All tuning uses VALIDATION data only. Test data is NEVER touched.

Provides:
- ML hyperparameter grid search
- RL hyperparameter configuration
- Cross-validation utilities
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class MLHyperparameters:
    """ML model hyperparameters."""

    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    subsample: float = 1.0


@dataclass
class RLHyperparameters:
    """RL agent hyperparameters."""

    # Network architecture
    hidden_dim: int = 128
    num_layers: int = 2

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    gamma: float = 0.99

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Experience replay
    replay_buffer_size: int = 10000
    target_update_freq: int = 100

    # Regularization
    dropout: float = 0.2
    grad_clip: float = 1.0


# Default hyperparameter grids for tuning
ML_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
}

RL_PARAM_GRID = {
    "hidden_dim": [64, 128, 256],
    "learning_rate": [1e-4, 1e-3, 5e-3],
    "gamma": [0.95, 0.99],
    "batch_size": [32, 64, 128],
}


@dataclass
class DataSplit:
    """Container for train/val/test data splits."""

    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    def __post_init__(self):
        """Verify no data leakage."""
        assert (
            self.train_end <= self.val_start
        ), f"Data leakage: train_end ({self.train_end}) > val_start ({self.val_start})"
        assert (
            self.val_end <= self.test_start
        ), f"Data leakage: val_end ({self.val_end}) > test_start ({self.test_start})"

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def val_size(self) -> int:
        return self.val_end - self.val_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


def create_temporal_split(
    n_samples: int,
    train_pct: float = 0.6,
    val_pct: float = 0.2,
) -> DataSplit:
    """
    Create temporal train/val/test split.

    CRITICAL: Preserves temporal order. No shuffling.

    Args:
        n_samples: Total number of samples
        train_pct: Percentage for training (default 60%)
        val_pct: Percentage for validation (default 20%)

    Returns:
        DataSplit with indices
    """
    train_end = int(n_samples * train_pct)
    val_end = int(n_samples * (train_pct + val_pct))

    split = DataSplit(
        train_start=0,
        train_end=train_end,
        val_start=train_end,
        val_end=val_end,
        test_start=val_end,
        test_end=n_samples,
    )

    logger.info(
        f"Created split: train={split.train_size}, val={split.val_size}, test={split.test_size}"
    )

    return split


def split_dataframe(
    df: pd.DataFrame,
    split: DataSplit,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame according to DataSplit.

    Args:
        df: DataFrame to split
        split: DataSplit indices

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = df.iloc[split.train_start : split.train_end]
    val_df = df.iloc[split.val_start : split.val_end]
    test_df = df.iloc[split.test_start : split.test_end]

    return train_df, val_df, test_df


class WalkForwardValidator:
    """
    Walk-forward validation for time series.

    Provides proper temporal cross-validation without lookahead bias.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: int = 252,
        test_size: int = 21,
        gap: int = 0,
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_splits: Number of folds
            train_size: Training window size
            test_size: Test window size per fold
            gap: Gap between train and test (to avoid lookahead)
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for walk-forward validation.

        Args:
            n_samples: Total samples

        Yields:
            Tuples of (train_indices, test_indices)
        """
        splits = []

        # Calculate step size to get n_splits
        total_test_size = self.test_size * self.n_splits
        available = n_samples - self.train_size - self.gap

        if available < total_test_size:
            # Adjust test size
            step = max(1, available // self.n_splits)
        else:
            step = self.test_size

        for i in range(self.n_splits):
            test_start = self.train_size + self.gap + i * step
            test_end = min(test_start + step, n_samples)

            if test_end > n_samples:
                break

            # Train on everything before gap
            train_end = test_start - self.gap
            train_start = max(0, train_end - self.train_size)

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits


def log_split_info(
    df: pd.DataFrame,
    split: DataSplit,
    name: str = "Data",
) -> None:
    """
    Log information about data split for auditing.

    Args:
        df: DataFrame being split
        split: DataSplit indices
        name: Name for logging
    """
    logger.info("=" * 60)
    logger.info(f"{name} SPLIT SUMMARY (No Data Leakage Verification)")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(df)}")
    logger.info("")
    logger.info(
        f"TRAIN: {split.train_size} samples ({split.train_size/len(df)*100:.1f}%)"
    )
    logger.info(f"  Indices: [{split.train_start}, {split.train_end})")
    if hasattr(df.index[0], "strftime"):
        logger.info(
            f"  Dates: {df.index[split.train_start]} to {df.index[split.train_end-1]}"
        )
    logger.info("")
    logger.info(
        f"VALIDATION: {split.val_size} samples ({split.val_size/len(df)*100:.1f}%)"
    )
    logger.info(f"  Indices: [{split.val_start}, {split.val_end})")
    if hasattr(df.index[0], "strftime"):
        logger.info(
            f"  Dates: {df.index[split.val_start]} to {df.index[split.val_end-1]}"
        )
    logger.info("")
    logger.info(
        f"TEST (HOLDOUT): {split.test_size} samples ({split.test_size/len(df)*100:.1f}%)"
    )
    logger.info(f"  Indices: [{split.test_start}, {split.test_end})")
    if hasattr(df.index[0], "strftime"):
        logger.info(f"  Dates: {df.index[split.test_start]} to {df.index[-1]}")
    logger.info("")
    logger.info("CRITICAL: Test data must NEVER be used for training or tuning!")
    logger.info("=" * 60)
