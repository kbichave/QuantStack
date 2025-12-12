"""
Main entry point for the Multi-Timeframe Mean Reversion Trading Platform.

This module provides the high-level orchestration for:
1. Data fetching and preprocessing
2. Feature computation across timeframes
3. Model training and prediction
4. Signal generation with hierarchical filtering
5. Backtesting and performance analysis
"""

from typing import Dict, Optional, List
from datetime import datetime
import pandas as pd
from loguru import logger

from quantcore.config.settings import get_settings, Settings
from quantcore.config.timeframes import Timeframe, TIMEFRAME_HIERARCHY
from quantcore.data.fetcher import AlphaVantageClient
from quantcore.data.storage import DataStore
from quantcore.data.resampler import OHLCVResampler
from quantcore.data.preprocessor import DataPreprocessor
from quantcore.features.factory import MultiTimeframeFeatureFactory
from quantcore.labeling.event_labeler import MultiTimeframeLabelBuilder
from quantcore.hierarchy.cascade import SignalCascade
from quantcore.models.trainer import ModelTrainer, TrainingConfig
from quantcore.models.ensemble import HierarchicalEnsemble
from quantcore.strategy.signals import SignalGenerator
from quantcore.backtesting.engine import BacktestEngine, BacktestConfig
from quantcore.backtesting.reports import PerformanceReport
from quantcore.risk.controls import RiskController


class TradingPlatform:
    """
    Main orchestrator for the trading platform.

    Provides high-level methods for:
    - Data pipeline management
    - Model training
    - Signal generation
    - Backtesting
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the trading platform.

        Args:
            settings: Optional settings override
        """
        self.settings = settings or get_settings()

        # Initialize components
        self.data_client = AlphaVantageClient()
        self.data_store = DataStore()
        self.resampler = OHLCVResampler()
        self.preprocessor = DataPreprocessor()
        self.feature_factory = MultiTimeframeFeatureFactory()
        self.label_builder = MultiTimeframeLabelBuilder()
        self.cascade = SignalCascade()
        self.ensemble = HierarchicalEnsemble()
        self.risk_controller = RiskController()

        logger.info("Trading platform initialized")

    def fetch_and_store_data(
        self,
        symbols: Optional[List[str]] = None,
    ) -> None:
        """
        Fetch data for all symbols and store in database.

        Args:
            symbols: List of symbols (uses settings if not provided)
        """
        symbols = symbols or self.settings.symbols

        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")

            try:
                # Fetch hourly data
                df_1h = self.data_client.fetch_all_intraday_history(symbol)

                if df_1h.empty:
                    logger.warning(f"No data fetched for {symbol}")
                    continue

                # Preprocess
                df_1h = self.preprocessor.preprocess(df_1h, Timeframe.H1)

                # Resample to all timeframes
                mtf_data = self.resampler.resample_all_timeframes(df_1h)

                # Store each timeframe
                for tf, df in mtf_data.items():
                    self.data_store.save_ohlcv(df, symbol, tf)

                logger.info(f"Data stored for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

    def load_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[Timeframe, pd.DataFrame]:
        """
        Load multi-timeframe data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Dictionary of DataFrames per timeframe
        """
        return self.data_store.load_multi_timeframe(symbol, start_date, end_date)

    def compute_features(
        self,
        data: Dict[Timeframe, pd.DataFrame],
        benchmark_data: Optional[Dict[Timeframe, pd.DataFrame]] = None,
    ) -> Dict[Timeframe, pd.DataFrame]:
        """
        Compute features for all timeframes.

        Args:
            data: OHLCV data per timeframe
            benchmark_data: Benchmark data for RRG

        Returns:
            Feature DataFrames per timeframe
        """
        return self.feature_factory.compute_all_timeframes(data, benchmark_data)

    def train_models(
        self,
        training_data: Dict[Timeframe, pd.DataFrame],
        model_dir: str = "models",
    ) -> None:
        """
        Train ML models for each timeframe.

        Args:
            training_data: Labeled feature data per TF
            model_dir: Directory to save models
        """
        trainer = ModelTrainer(TrainingConfig())

        for tf in [Timeframe.H1, Timeframe.H4]:
            if tf not in training_data:
                continue

            df = training_data[tf]

            # Train long model
            if "label_long" in df.columns:
                feature_cols = self.feature_factory.get_feature_names_for_ml(tf)
                feature_cols = [c for c in feature_cols if c in df.columns]

                X = df[feature_cols]
                y = df["label_long"]

                logger.info(f"Training {tf.value} LONG model")
                result = trainer.train(X, y, feature_cols)
                self.ensemble.add_model(tf, "long", result)

                trainer.save_model(result, f"{model_dir}/{tf.value}_long.pkl")

            # Train short model
            if "label_short" in df.columns:
                feature_cols = self.feature_factory.get_feature_names_for_ml(tf)
                feature_cols = [c for c in feature_cols if c in df.columns]

                X = df[feature_cols]
                y = df["label_short"]

                logger.info(f"Training {tf.value} SHORT model")
                result = trainer.train(X, y, feature_cols)
                self.ensemble.add_model(tf, "short", result)

                trainer.save_model(result, f"{model_dir}/{tf.value}_short.pkl")

    def generate_signals(
        self,
        symbol: str,
        data: Dict[Timeframe, pd.DataFrame],
    ) -> List:
        """
        Generate trading signals.

        Args:
            symbol: Stock symbol
            data: Feature data per timeframe

        Returns:
            List of generated signals
        """
        generator = SignalGenerator(
            timeframe=Timeframe.H1,
            cascade=self.cascade,
        )

        h1_data = data.get(Timeframe.H1, pd.DataFrame())
        return generator.generate(symbol, h1_data, data)

    def run_backtest(
        self,
        symbol: str,
        data: Dict[Timeframe, pd.DataFrame],
        config: Optional[BacktestConfig] = None,
    ) -> PerformanceReport:
        """
        Run backtest on historical data.

        Args:
            symbol: Stock symbol
            data: Feature data per timeframe
            config: Backtest configuration

        Returns:
            PerformanceReport
        """
        config = config or BacktestConfig()
        engine = BacktestEngine(config)

        # Generate historical signals
        generator = SignalGenerator(timeframe=Timeframe.H1, cascade=self.cascade)
        h1_data = data.get(Timeframe.H1, pd.DataFrame())

        signal_df = generator.scan_historical(symbol, h1_data, data)

        # Run backtest
        result = engine.run(signal_df, h1_data)

        return PerformanceReport(result)

    def run_full_pipeline(
        self,
        symbol: str,
        train_end_date: Optional[datetime] = None,
    ) -> Dict:
        """
        Run the complete pipeline: data → features → labels → train → backtest.

        Args:
            symbol: Stock symbol
            train_end_date: End of training period

        Returns:
            Dictionary with results
        """
        logger.info(f"Running full pipeline for {symbol}")

        # 1. Load data
        logger.info("Loading data...")
        data = self.load_data(symbol)

        if not data or all(df.empty for df in data.values()):
            raise ValueError(f"No data available for {symbol}")

        # 2. Load benchmark
        benchmark_data = self.load_data(self.settings.benchmark_symbol)

        # 3. Compute features
        logger.info("Computing features...")
        features = self.compute_features(data, benchmark_data)

        # 4. Generate labels
        logger.info("Generating labels...")
        labeled_data = self.label_builder.label_all_timeframes(features)

        # 5. Split train/test
        if train_end_date is None:
            train_end_date = datetime.strptime(self.settings.train_end_date, "%Y-%m-%d")

        train_data = {}
        test_data = {}

        for tf, df in labeled_data.items():
            if df.empty:
                continue
            train_data[tf] = df[df.index <= train_end_date]
            test_data[tf] = df[df.index > train_end_date]

        # 6. Train models
        logger.info("Training models...")
        self.train_models(train_data)

        # 7. Run backtest on test data
        logger.info("Running backtest...")
        report = self.run_backtest(symbol, test_data)

        # 8. Print results
        report.print_summary()

        return {
            "symbol": symbol,
            "train_samples": {tf.value: len(df) for tf, df in train_data.items()},
            "test_samples": {tf.value: len(df) for tf, df in test_data.items()},
            "model_summary": self.ensemble.get_model_summary(),
            "backtest_summary": report.summary(),
        }

    def close(self) -> None:
        """Clean up resources."""
        self.data_store.close()
        logger.info("Platform closed")


def main():
    """Main entry point."""
    import sys

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/trader.log", rotation="10 MB", level="DEBUG")

    # Initialize platform
    platform = TradingPlatform()

    try:
        # Example: Run full pipeline for SPY
        results = platform.run_full_pipeline("SPY")

        logger.info("Pipeline completed successfully")
        logger.info(f"Results: {results}")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
    finally:
        platform.close()


if __name__ == "__main__":
    main()
