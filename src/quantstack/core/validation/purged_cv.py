"""
Purged cross-validation for time-series data.

Implements Lopez de Prado's purged k-fold and combinatorial purged CV
to prevent data leakage from overlapping labels.
"""

from collections.abc import Generator
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class CVSplit:
    """A single cross-validation split."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class PurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation for time-series.

    Key features:
    1. Purging: Remove training samples whose labels overlap with test period
    2. Embargo: Add gap between train and test to prevent leakage

    Based on: Advances in Financial Machine Learning (Lopez de Prado)
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.0,
    ):
        """
        Initialize purged k-fold CV.

        Args:
            n_splits: Number of folds
            embargo_pct: Percentage of data to embargo after train
            purge_pct: Additional purge percentage
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        label_end_times: pd.Series | None = None,
        regime_labels: pd.Series | None = None,
    ) -> Generator[CVSplit, None, None]:
        """
        Generate train/test indices for each fold.

        Args:
            X: Feature DataFrame with DatetimeIndex
            y: Target series (optional)
            label_end_times: End time for each label (for purging)
            regime_labels: Optional regime label per sample (e.g. "trending_up",
                "ranging"). When provided, each fold contains proportional
                representation of all regimes.

        Yields:
            CVSplit for each fold
        """
        if regime_labels is not None:
            yield from self._split_regime_stratified(
                X, label_end_times, regime_labels
            )
        else:
            yield from self._split_standard(X, label_end_times)

    def _split_standard(
        self,
        X: pd.DataFrame,
        label_end_times: pd.Series | None = None,
    ) -> Generator[CVSplit, None, None]:
        """Standard purged k-fold (no regime stratification)."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate embargo size
        embargo_size = int(n_samples * self.embargo_pct)

        # Split into k folds
        fold_size = n_samples // self.n_splits

        for fold_idx in range(self.n_splits):
            # Test indices for this fold
            test_start_idx = fold_idx * fold_size
            test_end_idx = (
                (fold_idx + 1) * fold_size
                if fold_idx < self.n_splits - 1
                else n_samples
            )
            test_indices = indices[test_start_idx:test_end_idx]

            # Train indices: everything except test + embargo
            train_mask = np.ones(n_samples, dtype=bool)

            # Exclude test period
            train_mask[test_start_idx:test_end_idx] = False

            # Apply embargo after training (before test)
            if test_start_idx > 0:
                embargo_start = max(0, test_start_idx - embargo_size)
                train_mask[embargo_start:test_start_idx] = False

            # Apply purge if label_end_times provided
            if label_end_times is not None:
                test_start_time = X.index[test_start_idx]

                for i in range(test_start_idx):
                    if train_mask[i]:
                        label_end = label_end_times.iloc[i]
                        if pd.notna(label_end) and label_end >= test_start_time:
                            train_mask[i] = False

            train_indices = indices[train_mask]

            yield CVSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=(
                    X.index[train_indices[0]] if len(train_indices) > 0 else None
                ),
                train_end=(
                    X.index[train_indices[-1]] if len(train_indices) > 0 else None
                ),
                test_start=X.index[test_indices[0]],
                test_end=X.index[test_indices[-1]],
            )

    def _split_regime_stratified(
        self,
        X: pd.DataFrame,
        label_end_times: pd.Series | None,
        regime_labels: pd.Series,
    ) -> Generator[CVSplit, None, None]:
        """Regime-stratified purged k-fold.

        Groups samples by regime, divides each group into n_splits sub-groups
        chronologically, then builds each fold's test set as the union of the
        k-th sub-group from each regime. This ensures proportional regime
        representation in every fold.

        Regimes with fewer samples than n_splits are assigned entirely to
        the train set in every fold (with a warning).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        embargo_size = int(n_samples * self.embargo_pct)

        # Group sample indices by regime (preserving chronological order)
        unique_regimes = regime_labels.unique()
        regime_groups: dict[str, np.ndarray] = {}
        small_regime_indices: list[int] = []

        for regime in unique_regimes:
            mask = regime_labels.values == regime
            group_idx = indices[mask]
            if len(group_idx) < self.n_splits:
                logger.warning(
                    f"[PurgedKFoldCV] Regime '{regime}' has only {len(group_idx)} "
                    f"samples (< {self.n_splits} splits), assigning all to train"
                )
                small_regime_indices.extend(group_idx.tolist())
            else:
                regime_groups[regime] = group_idx

        # Divide each regime group into n_splits sub-groups chronologically
        regime_subgroups: dict[str, list[np.ndarray]] = {}
        for regime, group_idx in regime_groups.items():
            splits = np.array_split(group_idx, self.n_splits)
            regime_subgroups[regime] = splits

        for fold_idx in range(self.n_splits):
            # Test set: union of fold_idx-th sub-group from each regime
            test_parts = [
                regime_subgroups[r][fold_idx] for r in regime_subgroups
            ]
            test_indices = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=int)

            if len(test_indices) == 0:
                continue

            # Train set: everything not in test, plus small regime indices
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_indices] = False

            # Apply embargo around each contiguous test block
            test_min, test_max = int(test_indices.min()), int(test_indices.max())
            embargo_start = max(0, test_min - embargo_size)
            embargo_end = min(n_samples, test_max + embargo_size + 1)

            # Only embargo the gap between train and test, not within test
            for idx in range(embargo_start, embargo_end):
                if idx not in test_indices:
                    train_mask[idx] = False

            # Apply purge if label_end_times provided
            if label_end_times is not None:
                test_start_time = X.index[test_indices[0]]
                for i in range(int(test_indices[0])):
                    if train_mask[i]:
                        label_end = label_end_times.iloc[i]
                        if pd.notna(label_end) and label_end >= test_start_time:
                            train_mask[i] = False

            train_indices = indices[train_mask]

            yield CVSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=(
                    X.index[train_indices[0]] if len(train_indices) > 0 else None
                ),
                train_end=(
                    X.index[train_indices[-1]] if len(train_indices) > 0 else None
                ),
                test_start=X.index[test_indices[0]],
                test_end=X.index[test_indices[-1]],
            )

    def get_n_splits(self) -> int:
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation.

    Tests all combinations of train/test groups for more robust
    out-of-sample estimation.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize combinatorial CV.

        Args:
            n_groups: Number of groups to divide data into
            n_test_groups: Number of groups to use for testing
            embargo_pct: Embargo percentage
        """
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> Generator[CVSplit, None, None]:
        """
        Generate combinatorial splits.

        Args:
            X: Feature DataFrame
            y: Target series

        Yields:
            CVSplit for each combination
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        group_size = n_samples // self.n_groups
        embargo_size = int(n_samples * self.embargo_pct)

        # Create groups
        groups = []
        for i in range(self.n_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_groups - 1 else n_samples
            groups.append(indices[start:end])

        # Generate all combinations of test groups
        for test_group_indices in combinations(
            range(self.n_groups), self.n_test_groups
        ):
            # Test indices
            test_indices = np.concatenate([groups[i] for i in test_group_indices])

            # Train indices (with embargo)
            train_mask = np.ones(n_samples, dtype=bool)

            for test_group_idx in test_group_indices:
                group_start = groups[test_group_idx][0]
                group_end = groups[test_group_idx][-1]

                # Exclude test group
                train_mask[group_start : group_end + 1] = False

                # Embargo around test group
                embargo_start = max(0, group_start - embargo_size)
                embargo_end = min(n_samples, group_end + embargo_size + 1)
                train_mask[embargo_start:embargo_end] = False

            train_indices = indices[train_mask]

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield CVSplit(
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_start=X.index[train_indices[0]],
                    train_end=X.index[train_indices[-1]],
                    test_start=X.index[test_indices[0]],
                    test_end=X.index[test_indices[-1]],
                )


class WalkForwardValidator:
    """
    Walk-forward validation with expanding or sliding window.

    More realistic simulation of live trading where model
    is retrained periodically on new data.
    """

    def __init__(
        self,
        train_period_days: int = 252 * 2,  # 2 years
        test_period_days: int = 63,  # 1 quarter
        step_days: int = 21,  # Monthly steps
        min_train_samples: int = 500,
        expanding_window: bool = True,
        embargo_days: int = 5,
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_period_days: Initial training period
            test_period_days: Test period length
            step_days: Days to step forward each iteration
            min_train_samples: Minimum training samples required
            expanding_window: Use expanding (True) or sliding (False) window
            embargo_days: Gap between train and test
        """
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_days = step_days
        self.min_train_samples = min_train_samples
        self.expanding_window = expanding_window
        self.embargo_days = embargo_days

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> Generator[CVSplit, None, None]:
        """
        Generate walk-forward splits.

        Args:
            X: Feature DataFrame with DatetimeIndex
            y: Target series

        Yields:
            CVSplit for each walk-forward period
        """
        dates = pd.Series(X.index)
        n_samples = len(X)

        # Estimate samples per day
        total_days = (dates.iloc[-1] - dates.iloc[0]).days
        samples_per_day = n_samples / total_days if total_days > 0 else 1

        # Convert days to samples
        train_samples = int(self.train_period_days * samples_per_day)
        test_samples = int(self.test_period_days * samples_per_day)
        step_samples = int(self.step_days * samples_per_day)
        embargo_samples = int(self.embargo_days * samples_per_day)

        # Initial position
        train_start = 0
        train_end = train_samples

        while train_end + embargo_samples + test_samples <= n_samples:
            # Test period
            test_start = train_end + embargo_samples
            test_end = min(test_start + test_samples, n_samples)

            if train_end - train_start < self.min_train_samples:
                train_start = max(0, train_end - self.min_train_samples)

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield CVSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=X.index[train_indices[0]],
                train_end=X.index[train_indices[-1]],
                test_start=X.index[test_indices[0]],
                test_end=X.index[test_indices[-1]],
            )

            # Move forward
            if self.expanding_window:
                train_end += step_samples
            else:
                train_start += step_samples
                train_end += step_samples

    def get_summary(self, X: pd.DataFrame) -> dict:
        """Get summary of walk-forward splits."""
        splits = list(self.split(X))

        return {
            "n_splits": len(splits),
            "first_test_start": splits[0].test_start if splits else None,
            "last_test_end": splits[-1].test_end if splits else None,
            "avg_train_size": (
                np.mean([len(s.train_indices) for s in splits]) if splits else 0
            ),
            "avg_test_size": (
                np.mean([len(s.test_indices) for s in splits]) if splits else 0
            ),
        }
