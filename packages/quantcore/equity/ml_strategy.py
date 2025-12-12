"""
ML strategy for equity direction prediction.

Uses GradientBoosting classifier with feature selection.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold

from quantcore.equity.backtester import backtest_signals, BacktestResult
from quantcore.equity.reports import TickerStrategyResult, StrategyResult


def run_ml_strategy(
    symbol_data: Dict[str, Any],
    initial_equity: float = 100000,
    calculate_data_split: callable = None,
) -> StrategyResult:
    """
    Train and run ML strategy for direction prediction.

    Args:
        symbol_data: Dict mapping symbol -> SymbolData (with .ohlcv and .features)
        initial_equity: Initial equity for backtesting
        calculate_data_split: Function to calculate train/val/test split

    Returns:
        StrategyResult with per-ticker breakdown
    """
    logger.info("\n" + "=" * 60)
    logger.info("ML STRATEGY (GradientBoosting)")
    logger.info("=" * 60)

    per_ticker = {}
    total_pnl = 0
    total_trades = 0
    total_wins = 0
    train_metrics = {}
    max_dd = 0

    for symbol, data in symbol_data.items():
        if data.features is None or data.features.empty:
            continue

        logger.info(f"\n[{symbol}] Training ML model...")

        if calculate_data_split:
            split = calculate_data_split(len(data.features))
        else:
            # Default 60/20/20 split
            n = len(data.features)
            train_end = int(n * 0.6)
            val_end = int(n * 0.8)

            class Split:
                train_start = 0
                train_end = train_end
                val_start = train_end
                val_end = val_end
                test_start = val_end
                test_end = n

            split = Split()

        train_features = data.features.iloc[split.train_start : split.val_end].copy()

        if len(train_features) < 100:
            logger.warning(f"  Insufficient training data: {len(train_features)}")
            continue

        try:
            # Feature selection
            exclude_cols = ["open", "high", "low", "close", "volume", "label"]
            feature_cols = [c for c in train_features.columns if c not in exclude_cols]

            X_train_full = train_features[feature_cols].values
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(X_train_full)
            selected_features = [
                f for f, s in zip(feature_cols, selector.get_support()) if s
            ]

            if len(selected_features) > 100:
                correlations = (
                    train_features[selected_features]
                    .corrwith(train_features["label"])
                    .abs()
                )
                selected_features = correlations.nlargest(100).index.tolist()

            logger.info(f"  Selected {len(selected_features)} features")

            X = train_features[selected_features].values
            y = train_features["label"].values

            val_split = int(len(X) * 0.8)
            X_train, X_val = X[:val_split], X[val_split:]
            y_train, y_val = y[:val_split], y[val_split:]

            model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                min_samples_split=20,
                min_samples_leaf=10,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
            model.fit(X_train, y_train)

            train_acc = model.score(X_train, y_train)
            val_acc = model.score(X_val, y_val)

            logger.info(f"  Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

            train_metrics[symbol] = {"train_acc": train_acc, "val_acc": val_acc}

            # Generate signals on test data
            test_features = data.features.iloc[split.test_start : split.test_end]
            test_prices = data.ohlcv.iloc[split.test_start : split.test_end]

            X_test = test_features[selected_features].values
            predictions = model.predict(X_test)

            signals = pd.Series(
                [1 if p == 1 else -1 for p in predictions], index=test_features.index
            )

            result = backtest_signals(
                signals=signals,
                prices=test_prices,
                shares_per_trade=100,
                initial_equity=initial_equity,
            )

            per_ticker[symbol] = TickerStrategyResult(
                ticker=symbol,
                strategy="ML (GBM)",
                pnl=result.total_pnl,
                num_trades=result.num_trades,
                win_rate=result.win_rate,
                sharpe=result.sharpe_ratio,
            )

            total_pnl += result.total_pnl
            total_trades += result.num_trades
            total_wins += int(result.win_rate * result.num_trades)
            max_dd = max(max_dd, result.max_drawdown)

            logger.info(
                f"  Test PnL=${result.total_pnl:,.0f}, Trades={result.num_trades}"
            )

        except Exception as e:
            logger.error(f"  Error: {e}")

    if per_ticker:
        best_ticker = max(per_ticker.items(), key=lambda x: x[1].pnl)[0]
        worst_ticker = min(per_ticker.items(), key=lambda x: x[1].pnl)[0]
    else:
        best_ticker = ""
        worst_ticker = ""

    total_return = total_pnl / initial_equity
    win_rate = total_wins / total_trades if total_trades > 0 else 0

    return StrategyResult(
        strategy_name="ML (GBM)",
        strategy_type="ml",
        total_pnl=total_pnl,
        total_return=total_return,
        sharpe_ratio=0,
        max_drawdown=max_dd,
        win_rate=win_rate,
        num_trades=total_trades,
        avg_trade_pnl=total_pnl / total_trades if total_trades > 0 else 0,
        per_ticker=per_ticker,
        best_ticker=best_ticker,
        worst_ticker=worst_ticker,
        train_metrics=train_metrics,
    )
