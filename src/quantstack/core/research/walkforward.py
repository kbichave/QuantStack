"""
Walk-Forward Validation Framework.

Provides rigorous time-series cross-validation:
- Expanding window (anchored) validation
- Rolling window validation
- Combinatorial purged cross-validation
- Out-of-sample performance tracking
"""

from collections.abc import Callable, Generator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class WalkForwardFold:
    """Single fold in walk-forward validation."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_size: int
    test_size: int


@dataclass
class WalkForwardResult:
    """Result from walk-forward validation."""

    folds: list[WalkForwardFold]
    train_metrics: list[dict]
    test_metrics: list[dict]
    aggregate_train: dict
    aggregate_test: dict
    overfit_ratio: float  # train/test performance ratio


class WalkForwardValidator:
    """
    Walk-forward validation for trading strategies.

    Respects temporal ordering and prevents lookahead bias.
    Supports both expanding and rolling windows.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 252,  # ~1 year
        min_train_size: int = 504,  # ~2 years
        gap: int = 0,  # Gap between train and test (embargo)
        expanding: bool = True,  # Expanding vs rolling
    ):
        """
        Initialize validator.

        Args:
            n_splits: Number of walk-forward folds
            test_size: Size of each test period (in bars)
            min_train_size: Minimum training set size
            gap: Gap between train and test to prevent leakage
            expanding: If True, training window expands; if False, rolls
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.gap = gap
        self.expanding = expanding

    def split(
        self,
        data: pd.DataFrame,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for walk-forward validation.

        Args:
            data: DataFrame with DatetimeIndex

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n = len(data)

        # Calculate split points
        total_test = self.n_splits * self.test_size

        if n < self.min_train_size + total_test:
            raise ValueError(
                f"Insufficient data: need {self.min_train_size + total_test}, have {n}"
            )

        # First test starts after minimum training period
        first_test_start = self.min_train_size + self.gap

        for i in range(self.n_splits):
            test_start = first_test_start + i * self.test_size
            test_end = min(test_start + self.test_size, n)

            if self.expanding:
                train_start = 0
            else:
                # Rolling window: fixed training size
                train_start = max(0, test_start - self.gap - self.min_train_size)

            train_end = test_start - self.gap

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

    def get_folds(self, data: pd.DataFrame) -> list[WalkForwardFold]:
        """Get detailed fold information."""
        folds = []

        for i, (train_idx, test_idx) in enumerate(self.split(data)):
            fold = WalkForwardFold(
                fold_id=i,
                train_start=data.index[train_idx[0]],
                train_end=data.index[train_idx[-1]],
                test_start=data.index[test_idx[0]],
                test_end=data.index[test_idx[-1]],
                train_size=len(train_idx),
                test_size=len(test_idx),
            )
            folds.append(fold)

        return folds

    def validate(
        self,
        data: pd.DataFrame,
        model_fn: Callable,
        evaluate_fn: Callable,
        features: list[str],
        target: str,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            data: DataFrame with features and target
            model_fn: Function(X_train, y_train) -> trained_model
            evaluate_fn: Function(model, X, y) -> metrics_dict
            features: List of feature column names
            target: Target column name

        Returns:
            WalkForwardResult with all metrics
        """
        folds = self.get_folds(data)
        train_metrics = []
        test_metrics = []

        X = data[features].values
        y = data[target].values

        for i, (train_idx, test_idx) in enumerate(self.split(data)):
            logger.info(f"Walk-forward fold {i + 1}/{self.n_splits}")

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Train model
            model = model_fn(X_train, y_train)

            # Evaluate on train and test
            train_met = evaluate_fn(model, X_train, y_train)
            test_met = evaluate_fn(model, X_test, y_test)

            train_metrics.append(train_met)
            test_metrics.append(test_met)

            logger.info(f"  Train: {train_met}")
            logger.info(f"  Test:  {test_met}")

        # Aggregate metrics
        aggregate_train = self._aggregate_metrics(train_metrics)
        aggregate_test = self._aggregate_metrics(test_metrics)

        # Calculate overfit ratio
        # Compare key metric (e.g., Sharpe) between train and test
        train_sharpe = aggregate_train.get("sharpe_mean", 1.0)
        test_sharpe = aggregate_test.get("sharpe_mean", 1.0)
        overfit_ratio = train_sharpe / test_sharpe if test_sharpe != 0 else np.inf

        return WalkForwardResult(
            folds=folds,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            aggregate_train=aggregate_train,
            aggregate_test=aggregate_test,
            overfit_ratio=overfit_ratio,
        )

    def _aggregate_metrics(self, metrics_list: list[dict]) -> dict:
        """Aggregate metrics across folds."""
        if not metrics_list:
            return {}

        aggregated = {}
        keys = metrics_list[0].keys()

        for key in keys:
            values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
                aggregated[f"{key}_min"] = np.min(values)
                aggregated[f"{key}_max"] = np.max(values)

        return aggregated


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    From Marcos Lopez de Prado's "Advances in Financial Machine Learning".
    Prevents leakage in overlapping samples.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize CPCV.

        Args:
            n_splits: Number of groups to split data into
            n_test_splits: Number of groups to use as test set
            embargo_pct: Percentage of data to embargo after each test group
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        data: pd.DataFrame,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test splits with purging and embargo.

        Yields:
            Tuple of (train_indices, test_indices)
        """
        from itertools import combinations

        n = len(data)
        group_size = n // self.n_splits
        embargo_size = int(n * self.embargo_pct)

        # Create group boundaries
        groups = []
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n
            groups.append((start, end))

        # Generate all combinations of test groups
        for test_groups in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = []
            purge_idx = set()

            for g in test_groups:
                start, end = groups[g]
                test_idx.extend(range(start, end))

                # Add embargo after test group
                purge_start = end
                purge_end = min(end + embargo_size, n)
                purge_idx.update(range(purge_start, purge_end))

                # Purge before test group (for overlapping labels)
                purge_start = max(0, start - embargo_size)
                purge_end = start
                purge_idx.update(range(purge_start, purge_end))

            # Train indices = everything except test and purged
            all_test = set(test_idx)
            train_idx = [
                i for i in range(n) if i not in all_test and i not in purge_idx
            ]

            yield np.array(train_idx), np.array(sorted(test_idx))


def backtest_walk_forward(
    data: pd.DataFrame,
    signal_col: str,
    returns_col: str,
    n_splits: int = 5,
    test_size: int = 252,
) -> dict:
    """
    Simple walk-forward backtest of a signal.

    Args:
        data: DataFrame with signal and returns
        signal_col: Column name for trading signal
        returns_col: Column name for returns
        n_splits: Number of walk-forward folds
        test_size: Size of each test period

    Returns:
        Dictionary with backtest results
    """
    validator = WalkForwardValidator(
        n_splits=n_splits,
        test_size=test_size,
        expanding=True,
    )

    results = {
        "fold_returns": [],
        "fold_sharpes": [],
        "fold_dates": [],
    }

    for _fold_id, (_train_idx, test_idx) in enumerate(validator.split(data)):
        # Get test period
        test_data = data.iloc[test_idx]

        # Simple signal-based returns
        signal = test_data[signal_col]
        returns = test_data[returns_col]

        # Strategy returns: signal * next-period return
        strat_returns = signal.shift(1) * returns
        strat_returns = strat_returns.dropna()

        # Metrics
        total_return = (1 + strat_returns).prod() - 1
        sharpe = (
            strat_returns.mean() / strat_returns.std() * np.sqrt(252)
            if strat_returns.std() > 0
            else 0
        )

        results["fold_returns"].append(total_return)
        results["fold_sharpes"].append(sharpe)
        results["fold_dates"].append((test_data.index[0], test_data.index[-1]))

    # Aggregate
    results["mean_return"] = np.mean(results["fold_returns"])
    results["mean_sharpe"] = np.mean(results["fold_sharpes"])
    results["std_sharpe"] = np.std(results["fold_sharpes"])
    results["min_sharpe"] = np.min(results["fold_sharpes"])
    results["max_sharpe"] = np.max(results["fold_sharpes"])

    return results


class CPCVEvaluator:
    """
    Combinatorial Purged CV evaluator.

    Wraps CombinatorialPurgedCV with full performance measurement across splits,
    then feeds the result into DSR and PBO for overfitting detection.

    Typical usage::

        evaluator = CPCVEvaluator(n_splits=6, n_test_splits=2)
        report = evaluator.evaluate(returns_matrix)
        print(report.summary)
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        embargo_pct: float = 0.01,
        n_trials: int = 1,
        significance_level: float = 0.95,
    ):
        """
        Args:
            n_splits: CPCV groups.
            n_test_splits: Groups held out per split.
            embargo_pct: Embargo window fraction.
            n_trials: Number of strategy variants tried before selecting this one.
                      Used for DSR multiple-testing correction.
            significance_level: DSR confidence threshold.
        """
        self.cpcv = CombinatorialPurgedCV(n_splits, n_test_splits, embargo_pct)
        self.n_trials = n_trials
        self.significance_level = significance_level

    def evaluate(
        self,
        returns: pd.DataFrame,
        signal_col: str = "signal",
        returns_col: str = "returns",
    ):
        """
        Evaluate a single signal across CPCV splits and return DSR + PBO.

        Args:
            returns: DataFrame with signal and returns columns, DatetimeIndex.
            signal_col: Column name of the trading signal.
            returns_col: Column name of asset returns.

        Returns:
            OverfittingReport from quantstack.core.research.overfitting.
        """
        from quantstack.core.research.overfitting import run_overfitting_analysis

        # Collect OOS returns across all splits
        oos_return_series = []
        for _, test_idx in self.cpcv.split(returns):
            test = returns.iloc[test_idx]
            strat_ret = test[signal_col].shift(1) * test[returns_col]
            oos_return_series.append(strat_ret.dropna())

        if not oos_return_series:
            logger.warning("CPCVEvaluator: no valid splits produced")
            from quantstack.core.research.overfitting import (
                DSRResult,
                OverfittingReport,
            )

            dummy_dsr = DSRResult(
                observed_sharpe=0.0,
                benchmark_sharpe=0.0,
                dsr=0.0,
                is_genuine=False,
                n_trials=self.n_trials,
                skewness=0.0,
                kurtosis=0.0,
            )
            return OverfittingReport(
                dsr_result=dummy_dsr,
                pbo_result=None,
                verdict="OVERFIT",
                summary="CPCVEvaluator: insufficient data",
            )

        combined_oos = pd.concat(oos_return_series).sort_index()

        # Build a (T × 1) returns matrix for PBO (single strategy)
        r_matrix = combined_oos.values.reshape(-1, 1)

        return run_overfitting_analysis(
            strategy_returns=combined_oos,
            n_trials=self.n_trials,
            all_strategy_returns=r_matrix if len(combined_oos) >= 20 else None,
            n_cpcv_splits=self.cpcv.n_splits,
            significance_level=self.significance_level,
        )

    def evaluate_multiple(
        self,
        returns: pd.DataFrame,
        strategy_signals: dict,
        returns_col: str = "returns",
    ):
        """
        Evaluate multiple strategy signals simultaneously, computing PBO
        across all variants.

        Args:
            returns: DataFrame with returns column and DatetimeIndex.
            strategy_signals: Dict of {strategy_name: signal_series}.
            returns_col: Column name of asset returns.

        Returns:
            Dict of {strategy_name: OverfittingReport} plus a combined PBO.
        """
        import numpy as np

        from quantstack.core.research.overfitting import (
            probability_of_backtest_overfitting,
            run_overfitting_analysis,
        )

        n = len(strategy_signals)
        all_names = list(strategy_signals.keys())

        # Align all signals to the returns index
        aligned = {}
        for name, sig in strategy_signals.items():
            common = sig.index.intersection(returns.index)
            strat_ret = sig.reindex(common).shift(1) * returns[returns_col].reindex(
                common
            )
            aligned[name] = strat_ret.dropna()

        # Build returns matrix (T × N) across common time axis
        common_dates = aligned[all_names[0]].index
        for s in aligned.values():
            common_dates = common_dates.intersection(s.index)

        if len(common_dates) < 20:
            logger.warning(
                "CPCVEvaluator.evaluate_multiple: insufficient overlapping data"
            )
            return {}

        r_matrix = np.column_stack(
            [aligned[nm].reindex(common_dates).values for nm in all_names]
        )

        # PBO across all strategies
        pbo_result = probability_of_backtest_overfitting(
            r_matrix, n_splits=self.cpcv.n_splits
        )

        # Per-strategy DSR
        results = {}
        for name in all_names:
            report = run_overfitting_analysis(
                strategy_returns=aligned[name].reindex(common_dates),
                n_trials=n,  # All variants count as trials
                all_strategy_returns=r_matrix,
                n_cpcv_splits=self.cpcv.n_splits,
                significance_level=self.significance_level,
            )
            results[name] = report

        results["_pbo_combined"] = pbo_result
        return results


def generate_walk_forward_report(result: WalkForwardResult) -> str:
    """Generate text report from walk-forward validation."""
    report = """
Walk-Forward Validation Report
==============================

Fold Details:
"""
    for fold in result.folds:
        report += f"""
Fold {fold.fold_id + 1}:
  Train: {fold.train_start.date()} to {fold.train_end.date()} ({fold.train_size} bars)
  Test:  {fold.test_start.date()} to {fold.test_end.date()} ({fold.test_size} bars)
"""

    report += """
Aggregate Results:
------------------
Train Performance:
"""
    for key, val in result.aggregate_train.items():
        report += f"  {key}: {val:.4f}\n"

    report += """
Test Performance:
"""
    for key, val in result.aggregate_test.items():
        report += f"  {key}: {val:.4f}\n"

    report += f"""
Overfit Analysis:
  Train/Test ratio: {result.overfit_ratio:.2f}
  {"WARNING: Possible overfitting!" if result.overfit_ratio > 2.0 else "Acceptable train/test consistency"}
"""

    return report
