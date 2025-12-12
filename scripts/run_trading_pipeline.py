#!/usr/bin/env python3
"""
WTI Trading System - Full End-to-End Test Suite.

This is the main entry point that orchestrates:
1. Data fetching (WTI, Brent, ETFs, economic indicators, news/sentiment)
2. Feature engineering (273+ features)
3. HMM regime detection training
4. RL agent training (Execution, Sizing, Alpha Selection, Spread)
5. Backtesting with profit/loss reporting
6. News sentiment analysis

Usage:
    python -m scripts.test_wti_pipeline --api-key YOUR_API_KEY

    Or set environment variable in .env file:
    ALPHA_VANTAGE_API_KEY=YOUR_API_KEY

    Then simply run:
    uv run python scripts/run_trading_pipeline.py
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file (if exists)
# This allows the script to automatically pick up API keys without manual export
load_dotenv()

# Ensure trader package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
from quantcore.data.fetcher import AlphaVantageClient
from quantcore.data.storage import DataStore
from quantcore.config.timeframes import Timeframe

# New modular imports
from quantcore.utils.formatting import (
    print_header,
    print_section,
    print_success,
    print_error,
    print_info,
)
from quantcore.backtesting import run_backtest, run_strategy_comparison
from quantcore.analysis import (
    tune_hyperparameters,
    run_monte_carlo_simulation,
    generate_report,
)
from quantcore.visualization import generate_strategy_plots

# Validation imports
from quantcore.validation import (
    LeakageDetector,
    FeatureShiftTest,
    PurgedKFoldCV,
    WalkForwardValidator,
    validate_data_integrity,
)
from quantcore.validation.input_validation import DataFrameValidator


# ============================================================================
# WTI-Specific Database Paths
# ============================================================================

WTI_DATA_DIR = Path("data/wti")


def get_wti_db_paths() -> Dict[str, str]:
    """Get database paths for WTI pipeline data."""
    WTI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "ohlcv": str(WTI_DATA_DIR / "wti.ohlcv.db"),
        "news": str(WTI_DATA_DIR / "wti.news.db"),
        "features": str(WTI_DATA_DIR / "wti.features.db"),
        "models": str(WTI_DATA_DIR / "wti.models.db"),
    }


# ============================================================================
# Data Fetching (with caching)
# ============================================================================


def fetch_all_data(
    client: AlphaVantageClient,
    ohlcv_store: DataStore,
    news_store: DataStore,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch all required WTI data with caching.
    Data is saved to DuckDB and reused on subsequent runs unless force_refresh=True.
    """
    all_data = {}

    print_section("Phase 1: Loading/Fetching Commodity Data")

    # Check if we have cached data
    if not force_refresh:
        print_info("Checking for cached data in DuckDB...")
        cached_count = 0

        for symbol in ["WTI", "BRENT", "NATURAL_GAS"]:
            try:
                df = ohlcv_store.load_ohlcv(symbol, Timeframe.D1)
                if df is not None and len(df) > 100:
                    all_data[f"{symbol}_daily"] = df
                    cached_count += 1
                    print_success(f"  Cached {symbol}: {len(df)} bars")
            except:
                pass

        for symbol in ["USO", "XLE", "UUP", "SPY"]:
            try:
                df = ohlcv_store.load_ohlcv(symbol, Timeframe.D1)
                if df is not None and len(df) > 100:
                    all_data[f"{symbol}_daily"] = df
                    cached_count += 1
                    print_success(f"  Cached {symbol}: {len(df)} bars")
            except:
                pass

        if cached_count >= 6:
            print_success(f"Found {cached_count} cached datasets. Skipping API fetch.")
            print_info("  Use --force-refresh to fetch new data from AlphaVantage")
            return all_data
        else:
            print_info(
                f"Only {cached_count} datasets cached. Fetching remaining from API..."
            )

    # Commodities
    for commodity in ["WTI", "BRENT", "NATURAL_GAS"]:
        print_info(f"Fetching {commodity}...")
        try:
            df = client.fetch_commodity(commodity, interval="daily")
            if not df.empty:
                all_data[f"{commodity}_daily"] = df
                ohlcv_store.save_ohlcv(df, commodity, Timeframe.D1)
                print_success(f"{commodity}: {len(df)} bars")
            time.sleep(0.5)
        except Exception as e:
            print_error(f"{commodity}: {e}")

    print_section("Phase 2: Fetching ETF Proxy Data (Full History)")

    etfs = {"USO": "WTI ETF", "XLE": "Energy ETF", "UUP": "USD ETF", "SPY": "S&P 500"}
    for symbol, desc in etfs.items():
        print_info(f"Fetching {symbol} ({desc}) - full history...")
        try:
            df = client.fetch_daily(symbol, outputsize="full")
            if not df.empty:
                all_data[f"{symbol}_daily"] = df
                ohlcv_store.save_ohlcv(df, symbol, Timeframe.D1)
                years = len(df) / 252
                print_success(f"{symbol}: {len(df)} bars (~{years:.1f} years)")
            time.sleep(1)
        except Exception as e:
            print_error(f"{symbol}: {e}")

    print_section("Phase 3: Fetching Economic Indicators")

    print_info("Fetching Federal Funds Rate...")
    try:
        fed = client.fetch_economic_indicator("FEDERAL_FUNDS_RATE", "daily")
        if not fed.empty:
            all_data["FED_FUNDS"] = fed
            print_success(f"Fed Funds: {len(fed)} points")
    except Exception as e:
        print_error(f"Fed Funds: {e}")

    print_info("Fetching 10Y Treasury...")
    try:
        ust = client.fetch_economic_indicator(
            "TREASURY_YIELD", "daily", maturity="10year"
        )
        if not ust.empty:
            all_data["UST10Y"] = ust
            print_success(f"10Y Treasury: {len(ust)} points")
    except Exception as e:
        print_error(f"10Y Treasury: {e}")

    print_section("Phase 4: Fetching News & Sentiment")

    # Try to load cached historical news sentiment first
    print_info("Checking for cached historical news sentiment...")
    cached_news = None
    try:
        cached_news = news_store.load_news_sentiment(
            start_date=datetime(2020, 1, 1),
            end_date=datetime.now(),
            tickers=["CRUDE", "XLE", "USO"],
        )
        if cached_news is not None and not cached_news.empty and len(cached_news) > 100:
            print_success(f"Loaded {len(cached_news)} cached news articles")
            all_data["NEWS_SENTIMENT"] = cached_news
        else:
            cached_news = None
    except Exception as e:
        logger.debug(f"No cached news: {e}")

    # Fetch new historical news if not cached (or if force_refresh)
    if cached_news is None or force_refresh:
        print_info("Fetching historical news sentiment (this may take a while)...")
        try:
            # Fetch historical news going back 3 years
            from datetime import timedelta

            start_date = datetime.now() - timedelta(days=365 * 3)

            news = client.fetch_historical_news_sentiment(
                tickers="CRUDE,XLE,USO",
                topics="energy_transportation",
                start_date=start_date,
                end_date=datetime.now(),
                batch_months=3,
            )

            if not news.empty:
                # Save to database for future caching
                rows_saved = news_store.save_news_sentiment(news)
                all_data["NEWS_SENTIMENT"] = news
                avg_sentiment = news["overall_sentiment_score"].astype(float).mean()
                print_success(
                    f"Fetched {len(news)} news articles, saved {rows_saved} to cache"
                )
                print_info(
                    f"  Date range: {news.index.min().date()} to {news.index.max().date()}"
                )
                print_info(f"  Avg sentiment: {avg_sentiment:.3f}")
            else:
                print_info("No historical news data returned")
        except Exception as e:
            print_error(f"Historical news fetch: {e}")
            # Fall back to single recent fetch
            print_info("Falling back to recent news only...")
            try:
                news = client.fetch_news_sentiment(
                    tickers="CRUDE,XLE,USO", topics="energy_transportation", limit=100
                )
                if not news.empty:
                    all_data["NEWS_SENTIMENT"] = news
                    avg_sentiment = news["overall_sentiment_score"].astype(float).mean()
                    print_success(
                        f"News articles: {len(news)}, Avg sentiment: {avg_sentiment:.3f}"
                    )
            except Exception as e2:
                print_error(f"Recent news fetch also failed: {e2}")

    # Validate all OHLCV data before returning
    print_section("Validating Data Quality")
    ohlcv_keys = [
        "WTI_daily",
        "BRENT_daily",
        "NATURAL_GAS_daily",
        "USO_daily",
        "XLE_daily",
        "UUP_daily",
        "SPY_daily",
    ]
    validation_passed = 0
    validation_failed = 0

    for key in ohlcv_keys:
        if key in all_data:
            result = DataFrameValidator.validate_ohlcv(all_data[key], name=key)
            if result.is_valid:
                validation_passed += 1
            else:
                validation_failed += 1
                print_error(f"{key}: {'; '.join(result.errors[:2])}")
            # Log warnings
            for warning in result.warnings[:2]:
                logger.warning(f"{key}: {warning}")

    print_success(f"Validation: {validation_passed} passed, {validation_failed} failed")

    if validation_failed > 0:
        print_error("Some data failed validation. Results may be unreliable.")

    return all_data


def calculate_spreads(all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate WTI-Brent spread and signals."""
    print_section("Phase 5: Calculating Spreads")

    if "WTI_daily" not in all_data or "BRENT_daily" not in all_data:
        print_error("Missing WTI or Brent data")
        return pd.DataFrame()

    wti = all_data["WTI_daily"]
    brent = all_data["BRENT_daily"]

    common_idx = wti.index.intersection(brent.index)

    spread_df = pd.DataFrame(index=common_idx)
    spread_df["wti"] = wti.loc[common_idx, "close"]
    spread_df["brent"] = brent.loc[common_idx, "close"]
    spread_df["spread"] = spread_df["wti"] - spread_df["brent"]
    spread_df["spread_pct"] = spread_df["spread"] / spread_df["brent"] * 100

    # Z-score
    spread_df["spread_ma60"] = spread_df["spread"].rolling(60).mean()
    spread_df["spread_std60"] = spread_df["spread"].rolling(60).std()
    spread_df["spread_zscore"] = (spread_df["spread"] - spread_df["spread_ma60"]) / (
        spread_df["spread_std60"] + 1e-8
    )

    # Signals
    spread_df["signal"] = "NEUTRAL"
    spread_df.loc[spread_df["spread_zscore"] < -2, "signal"] = "LONG"
    spread_df.loc[spread_df["spread_zscore"] > 2, "signal"] = "SHORT"

    print_success(f"Spread calculated: {len(spread_df)} data points")
    print_info(f"  Latest spread: ${spread_df['spread'].iloc[-1]:.2f}")
    print_info(f"  Z-score: {spread_df['spread_zscore'].iloc[-1]:.2f}")
    print_info(f"  Signal: {spread_df['signal'].iloc[-1]}")

    return spread_df


def compute_features(
    all_data: Dict[str, pd.DataFrame],
    spread_df: pd.DataFrame,
    features_store: Optional[DataStore] = None,
) -> pd.DataFrame:
    """Compute all features including Gann, mean reversion, market structure, and sentiment."""
    print_section("Phase 6: Feature Engineering")

    try:
        from quantcore.features.commodity_factory import CommodityFeatureFactory

        if "USO_daily" in all_data:
            uso = all_data["USO_daily"].copy()
        elif "WTI_daily" in all_data:
            uso = all_data["WTI_daily"].copy()
        else:
            print_error("No price data for features")
            return pd.DataFrame()

        # Get news sentiment data if available
        news_sentiment = all_data.get("NEWS_SENTIMENT", None)
        if news_sentiment is not None and not news_sentiment.empty:
            print_info(
                f"  Using {len(news_sentiment)} news articles for sentiment features"
            )
        else:
            print_info(
                "  No news sentiment data available (neutral features will be used)"
            )

        factory = CommodityFeatureFactory(
            include_spread_features=True,
            include_curve_features=True,
            include_seasonality_features=True,
            include_event_features=True,
            include_microstructure_features=True,
            include_cross_asset_features=False,
            # NEW: Enable advanced features
            include_gann_features=True,
            include_mean_reversion_features=True,
            include_market_structure_features=True,
            include_sentiment_features=True,
            mr_lookback=20,
            mr_zscore_threshold=2.0,
        )

        featured = factory.compute_features(
            data={Timeframe.D1: uso},
            spread_data=spread_df,
            news_sentiment_data=news_sentiment,
        )

        if Timeframe.D1 in featured:
            df = featured[Timeframe.D1]

            # Log feature groups
            feature_groups = factory.get_feature_groups()
            print_success(
                f"Generated {len(df.columns)} features across {len(feature_groups)} groups:"
            )
            for group, features in feature_groups.items():
                present = [f for f in features if f in df.columns]
                if present:
                    print_info(f"  {group}: {len(present)} features")

            # Save features to database if store provided
            if features_store is not None:
                try:
                    features_store.save_ohlcv(df, "WTI_FEATURED", Timeframe.D1)
                    print_success(f"Saved {len(df)} feature rows to features database")
                except Exception as e:
                    print_error(f"Failed to save features: {e}")

            return df

    except Exception as e:
        print_error(f"Feature engineering failed: {e}")
        import traceback

        traceback.print_exc()

    return pd.DataFrame()


def run_validation_checks(
    featured_df: pd.DataFrame,
    spread_df: pd.DataFrame,
    train_end_date: str = "2021-01-01",
) -> Dict:
    """
    Run comprehensive validation checks on features and data.

    This includes:
    1. Data integrity validation
    2. Feature leakage detection (distribution drift)
    3. Purged cross-validation setup verification
    4. Walk-forward validation structure

    Args:
        featured_df: DataFrame with all computed features
        spread_df: DataFrame with spread data and signals
        train_end_date: End of training period

    Returns:
        Dictionary with validation results
    """
    print_section("Phase 3.5: Validation & Leakage Checks")

    results = {
        "data_integrity": {"passed": True, "details": []},
        "leakage_checks": {"passed": True, "details": []},
        "cv_structure": {"passed": True, "details": []},
        "walk_forward": {"passed": True, "details": []},
    }

    # 1. Data Integrity Validation
    print_info("Running data integrity checks...")
    try:
        if not featured_df.empty:
            # Check for lookahead bias in index
            if isinstance(featured_df.index, pd.DatetimeIndex):
                # Verify data is sorted
                is_sorted = featured_df.index.is_monotonic_increasing
                results["data_integrity"]["details"].append(
                    f"Index sorted: {'Yes' if is_sorted else 'WARNING: No'}"
                )

                # Check for future dates
                today = pd.Timestamp.now().normalize()
                future_data = (featured_df.index > today).sum()
                if future_data > 0:
                    results["data_integrity"]["passed"] = False
                    results["data_integrity"]["details"].append(
                        f"CRITICAL: {future_data} rows have future dates"
                    )
                else:
                    results["data_integrity"]["details"].append(
                        "No future dates detected"
                    )

                # Check for gaps
                if len(featured_df) > 1:
                    gaps = featured_df.index.to_series().diff().dropna()
                    max_gap = gaps.max()
                    results["data_integrity"]["details"].append(
                        f"Max data gap: {max_gap}"
                    )

            # Check for NaN patterns
            nan_pct = featured_df.isna().mean().mean() * 100
            results["data_integrity"]["details"].append(
                f"Overall NaN rate: {nan_pct:.2f}%"
            )

            if nan_pct > 50:
                results["data_integrity"]["passed"] = False
                results["data_integrity"]["details"].append(
                    "WARNING: High NaN rate (>50%)"
                )

        if results["data_integrity"]["passed"]:
            print_success("Data integrity: PASSED")
        else:
            print_error("Data integrity: FAILED")

        for detail in results["data_integrity"]["details"]:
            print_info(f"  {detail}")

    except Exception as e:
        print_error(f"Data integrity check failed: {e}")
        results["data_integrity"]["passed"] = False

    # 2. Feature Leakage Detection (Distribution Drift)
    print()
    print_info("Running feature leakage detection...")
    try:
        if not featured_df.empty and len(featured_df) > 100:
            # Split into train/test
            train_df = featured_df[featured_df.index < train_end_date]
            test_df = featured_df[featured_df.index >= train_end_date]

            if len(train_df) > 50 and len(test_df) > 50:
                from scipy import stats

                # Check distribution drift for numeric features
                numeric_cols = featured_df.select_dtypes(include=[np.number]).columns[
                    :20
                ]
                drift_warnings = []

                for col in numeric_cols:
                    train_vals = train_df[col].dropna()
                    test_vals = test_df[col].dropna()

                    if len(train_vals) > 10 and len(test_vals) > 10:
                        ks_stat, p_value = stats.ks_2samp(train_vals, test_vals)

                        if ks_stat > 0.3 and p_value < 0.01:
                            drift_warnings.append(f"{col}: KS={ks_stat:.3f}")

                if drift_warnings:
                    results["leakage_checks"]["details"].append(
                        f"Distribution drift in {len(drift_warnings)} features"
                    )
                    for warning in drift_warnings[:5]:
                        results["leakage_checks"]["details"].append(f"  - {warning}")
                else:
                    results["leakage_checks"]["details"].append(
                        "No significant distribution drift detected"
                    )

                print_success(f"Leakage checks: {len(drift_warnings)} warnings")
            else:
                print_info("  Insufficient data for leakage detection")

    except Exception as e:
        print_error(f"Leakage detection failed: {e}")
        results["leakage_checks"]["passed"] = False

    # 3. Purged CV Structure
    print()
    print_info("Validating purged cross-validation structure...")
    try:
        if not featured_df.empty and len(featured_df) > 252:
            purged_cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.02)

            splits_info = []
            for i, split in enumerate(purged_cv.split(featured_df)):
                splits_info.append(
                    {
                        "fold": i + 1,
                        "train_size": len(split.train_indices),
                        "test_size": len(split.test_indices),
                        "train_start": split.train_start,
                        "test_end": split.test_end,
                    }
                )

            if splits_info:
                results["cv_structure"]["details"].append(
                    f"Purged CV: {len(splits_info)} folds generated"
                )
                results["cv_structure"]["details"].append(
                    f"  Fold 1 train size: {splits_info[0]['train_size']}"
                )
                results["cv_structure"]["details"].append(
                    f"  Fold 1 test size: {splits_info[0]['test_size']}"
                )
                results["cv_structure"]["details"].append(
                    f"  Embargo: 2% = {int(len(featured_df) * 0.02)} bars"
                )
                print_success(f"Purged CV structure: {len(splits_info)} folds ready")

    except Exception as e:
        print_error(f"CV structure validation failed: {e}")
        results["cv_structure"]["passed"] = False

    # 4. Walk-Forward Validation Structure
    print()
    print_info("Validating walk-forward structure...")
    try:
        if not featured_df.empty and len(featured_df) > 504:  # At least 2 years
            wf_validator = WalkForwardValidator(
                n_splits=5,
                test_size=252,  # 1 year
                min_train_size=504,  # 2 years
                gap=5,  # 5 day embargo
                expanding=True,
            )

            splits_count = 0
            for split in wf_validator.split(featured_df):
                splits_count += 1

            results["walk_forward"]["details"].append(
                f"Walk-forward: {splits_count} expanding windows"
            )
            results["walk_forward"]["details"].append(
                f"  Test window: 252 bars (1 year)"
            )
            results["walk_forward"]["details"].append(
                f"  Min train: 504 bars (2 years)"
            )
            results["walk_forward"]["details"].append(f"  Embargo gap: 5 bars")
            print_success(f"Walk-forward structure: {splits_count} windows ready")

    except Exception as e:
        print_error(f"Walk-forward validation failed: {e}")
        results["walk_forward"]["passed"] = False

    # Summary
    print()
    all_passed = all(r["passed"] for r in results.values())
    if all_passed:
        print_success("All validation checks PASSED")
    else:
        failed = [k for k, v in results.items() if not v["passed"]]
        print_error(f"Validation checks FAILED: {', '.join(failed)}")

    return results


def train_regime_models(
    all_data: Dict[str, pd.DataFrame], train_end_date: str = "2020-12-31"
) -> Dict[str, object]:
    """Train all regime detection models: HMM, Changepoint, and TFT."""
    print_section("Phase 7: Regime Detection Training (HMM + Changepoint + TFT)")

    models = {}

    df = all_data.get("WTI_daily", all_data.get("USO_daily"))

    if df is None or len(df) < 252:
        print_error(f"Insufficient data for regime training")
        return models

    # CRITICAL: Only use data BEFORE train_end_date (no lookahead!)
    df = df[df.index <= train_end_date]

    if len(df) < 252:
        print_error(f"Insufficient pre-{train_end_date} data for training")
        return models

    df = df.tail(252 * 5)
    print_info(f"⚠️  Training on data BEFORE {train_end_date} only (no lookahead)")
    print_info(
        f"Using {len(df)} bars for regime training ({df.index[0].date()} to {df.index[-1].date()})"
    )

    # HMM
    print()
    print_info("Training HMM (4-state Gaussian)...")
    try:
        from quantcore.hierarchy.regime.hmm_model import HMMRegimeModel, HMM_AVAILABLE

        if not HMM_AVAILABLE:
            print_error("hmmlearn not installed. Install with: pip install hmmlearn")
        else:
            hmm = HMMRegimeModel(n_states=4, lookback=252)
            hmm.fit(df)
            result = hmm.predict(df)
            models["hmm"] = hmm
            print_success(f"HMM trained - Current state: {result.state.name}")

    except Exception as e:
        print_error(f"HMM training failed: {e}")

    # Changepoint
    print()
    print_info("Training Bayesian Changepoint Detector...")
    try:
        from quantcore.hierarchy.regime.changepoint import BayesianChangepointDetector

        changepoint = BayesianChangepointDetector(hazard_rate=1 / 250)
        result = changepoint.detect(df, feature="returns")
        models["changepoint"] = changepoint
        print_success(
            f"Changepoint detector trained - Change prob: {result.regime_change_probability:.2%}"
        )

    except Exception as e:
        print_error(f"Changepoint training failed: {e}")

    # TFT
    print()
    print_info("Training TFT Regime Model...")
    try:
        from quantcore.hierarchy.regime.tft_regime import (
            TFTRegimeModel,
            TORCH_AVAILABLE,
        )

        if not TORCH_AVAILABLE:
            print_error("PyTorch not installed. Install with: pip install torch")
        else:
            tft = TFTRegimeModel(lookback=60, hidden_size=32, n_heads=2, epochs=50)
            print_info("  Training for 50 epochs...")
            tft.fit(df)
            result = tft.predict(df)
            models["tft"] = tft
            print_success(
                f"TFT trained - Predicted regime: {result.predicted_regime.name}"
            )

    except Exception as e:
        print_error(f"TFT training failed: {e}")

    # Combined Detector
    print()
    print_info("Training Combined Commodity Regime Detector...")
    try:
        from quantcore.hierarchy.regime.commodity_regime import CommodityRegimeDetector

        detector = CommodityRegimeDetector(
            use_hmm="hmm" in models,
            use_changepoint="changepoint" in models,
            use_tft="tft" in models,
        )

        features = df.copy()
        features["returns"] = features["close"].pct_change()
        features["volatility"] = features["returns"].rolling(20).std()
        features = features.dropna()

        detector.fit(features)
        result = detector.detect(features)
        models["combined"] = detector
        print_success(f"Combined detector trained - Regime: {result.primary_regime}")

    except Exception as e:
        print_error(f"Combined detector training failed: {e}")

    return models


def train_rl_agents(
    all_data: Dict[str, pd.DataFrame],
    spread_df: pd.DataFrame,
    train_end_date: str = "2020-12-31",
) -> Tuple[Dict[str, object], Dict]:
    """Train RL agents on pre-train_end_date data only."""
    print_section("Phase 8: RL Agent Training")

    agents = {}
    rl_metrics = {}

    df = all_data.get("USO_daily", all_data.get("WTI_daily"))
    if df is None or len(df) < 100:
        print_error("Insufficient data for RL training")
        return agents, rl_metrics

    # CRITICAL: Only use data BEFORE train_end_date (no lookahead!)
    df = df[df.index <= train_end_date]
    spread_df_train = (
        spread_df[spread_df.index <= train_end_date]
        if not spread_df.empty
        else spread_df
    )

    print_info(f"⚠️  Training RL on data BEFORE {train_end_date} only (no lookahead)")
    print_info(f"Training data: {len(df)} bars")

    try:
        from quantcore.rl.execution import ExecutionRLAgent, ExecutionEnvironment
        from quantcore.rl.sizing import SizingRLAgent, SizingEnvironment
        from quantcore.rl.spread import SpreadArbitrageAgent, SpreadArbitrageEnvironment
        from quantcore.rl.training import RLTrainer, TrainingConfig

        # Use more training timesteps for proper convergence
        config = TrainingConfig(
            total_timesteps=10000,  # Increased from 1000 for better training
            batch_size=64,
            learning_starts=200,
            eval_freq=1000,
            save_freq=2000,
            save_path="models/rl_test",
        )

        # Execution RL
        print_info("Training Execution RL Agent...")
        try:
            exec_env = ExecutionEnvironment(data=df.tail(500))
            exec_agent = ExecutionRLAgent(
                state_dim=exec_env.get_state_dim(), action_dim=exec_env.get_action_dim()
            )
            exec_trainer = RLTrainer(exec_agent, exec_env, config)
            metrics = exec_trainer.train()
            agents["execution"] = exec_agent
            final_reward = metrics.episode_rewards[-1] if metrics.episode_rewards else 0
            rl_metrics["execution"] = {
                "final_reward": final_reward,
                "episodes": len(metrics.episode_rewards),
                "trained": True,
            }
            print_success(f"Execution RL: trained, reward: {final_reward:.2f}")
        except Exception as e:
            print_error(f"Execution RL failed: {e}")
            rl_metrics["execution"] = {"trained": False, "error": str(e)}

        # Sizing RL
        print_info("Training Sizing RL Agent...")
        try:
            # Pass actual price data for more realistic training
            sizing_env = SizingEnvironment(
                initial_equity=100000,
                data=df.tail(500),  # Use actual price data
            )
            sizing_agent = SizingRLAgent(
                state_dim=sizing_env.get_state_dim(),
                action_dim=sizing_env.get_action_dim(),
            )
            sizing_trainer = RLTrainer(sizing_agent, sizing_env, config)
            metrics = sizing_trainer.train()
            agents["sizing"] = sizing_agent
            final_reward = metrics.episode_rewards[-1] if metrics.episode_rewards else 0
            rl_metrics["sizing"] = {
                "final_reward": final_reward,
                "episodes": len(metrics.episode_rewards),
                "trained": True,
            }
            print_success(f"Sizing RL: trained, reward: {final_reward:.2f}")
        except Exception as e:
            print_error(f"Sizing RL failed: {e}")
            rl_metrics["sizing"] = {"trained": False, "error": str(e)}

        # Spread RL
        if not spread_df_train.empty and len(spread_df_train) > 500:
            print_info("Training Spread Arbitrage RL Agent...")
            try:
                spread_data_for_rl = (
                    spread_df_train.tail(2000).copy().reset_index(drop=True)
                )
                if "spread" not in spread_data_for_rl.columns:
                    spread_data_for_rl["spread"] = spread_data_for_rl.get(
                        "wti_brent_spread", 0
                    )

                spread_env = SpreadArbitrageEnvironment(
                    spread_data=spread_data_for_rl,
                    zscore_lookback=30,
                    max_holding_bars=30,
                )
                spread_agent = SpreadArbitrageAgent(
                    state_dim=spread_env.get_state_dim(),
                    action_dim=spread_env.get_action_dim(),
                )
                spread_trainer = RLTrainer(spread_agent, spread_env, config)
                metrics = spread_trainer.train()
                agents["spread"] = spread_agent
                final_reward = (
                    metrics.episode_rewards[-1] if metrics.episode_rewards else 0
                )
                rl_metrics["spread"] = {
                    "final_reward": final_reward,
                    "episodes": len(metrics.episode_rewards),
                    "trained": True,
                }
                print_success(f"Spread RL: trained, reward: {final_reward:.2f}")
            except Exception as e:
                print_error(f"Spread RL failed: {e}")
                rl_metrics["spread"] = {"trained": False, "error": str(e)}

    except ImportError as e:
        print_error(f"RL modules not available: {e}")
    except Exception as e:
        print_error(f"RL training failed: {e}")

    return agents, rl_metrics


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="WTI Trading System - Full Test Suite")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ALPHA_VANTAGE_API_KEY"),
        help="AlphaVantage API key",
    )
    parser.add_argument("--rate-limit", type=int, default=75, help="API rate limit")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--skip-training", action="store_true", help="Skip RL training")
    parser.add_argument(
        "--force-refresh", action="store_true", help="Force refresh data from API"
    )

    args = parser.parse_args()

    logger.remove()
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="WARNING")

    if not args.api_key:
        print("Error: API key required. Use --api-key or set ALPHA_VANTAGE_API_KEY")
        sys.exit(1)

    print_header("WTI TRADING SYSTEM - FULL TEST SUITE")
    print(f"  API Key: {args.api_key[:8]}...")
    print(f"  Initial Capital: ${args.capital:,.2f}")
    print(f"  Rate Limit: {args.rate_limit} calls/min")

    # Initialize with WTI-specific database paths
    db_paths = get_wti_db_paths()
    print(f"  OHLCV DB: {db_paths['ohlcv']}")
    print(f"  News DB: {db_paths['news']}")
    print(f"  Features DB: {db_paths['features']}")

    client = AlphaVantageClient(api_key=args.api_key)
    client.rate_limit = args.rate_limit

    # Create separate stores for different data types
    ohlcv_store = DataStore(db_path=db_paths["ohlcv"])
    news_store = DataStore(db_path=db_paths["news"])
    features_store = DataStore(db_path=db_paths["features"])

    # Define temporal splits (CRITICAL: No lookahead bias)
    # Train: data < TRAIN_END (for ML model fitting)
    # Validation: TRAIN_END <= data < VAL_END (for hyperparameter selection)
    # Test: data >= VAL_END (for final unbiased evaluation)
    TRAIN_END_DATE = "2018-01-01"
    VAL_END_DATE = "2021-01-01"

    # 1. Fetch Data
    all_data = fetch_all_data(
        client, ohlcv_store, news_store, force_refresh=args.force_refresh
    )

    # 2. Calculate Spreads
    spread_df = calculate_spreads(all_data)

    # 3. Compute Features
    featured_df = compute_features(all_data, spread_df, features_store=features_store)

    # 3.5. Validation & Leakage Checks
    validation_results = run_validation_checks(
        featured_df, spread_df, train_end_date=VAL_END_DATE
    )

    # 4. Train Regime Models (on all data before test period)
    # Note: Models are trained on data < VAL_END_DATE (before test period starts)
    regime_models = train_regime_models(all_data, train_end_date=VAL_END_DATE)

    # 5. Train RL Agents (on all data before test period)
    rl_agents = {}
    rl_metrics = {}
    if not args.skip_training:
        rl_agents, rl_metrics = train_rl_agents(
            all_data, spread_df, train_end_date=VAL_END_DATE
        )
    else:
        print_section("Phase 8: RL Agent Training (SKIPPED)")

    # 6. Hyperparameter Tuning (uses VALIDATION set for param selection)
    best_params, tuning_results = tune_hyperparameters(
        spread_df, args.capital, train_end=TRAIN_END_DATE, val_end=VAL_END_DATE
    )

    # 7. Monte Carlo Robustness Test (on TRUE holdout - 2023+)
    print_section("Phase 9.5: Monte Carlo Robustness Test")
    mc_results = None
    if best_params:
        print_info("Running 500 Monte Carlo simulations on HOLDOUT data (2023+)...")
        mc_results = run_monte_carlo_simulation(
            spread_df,
            best_params,
            args.capital,
            n_simulations=500,
            holdout_start="2023-01-01",  # True holdout for robustness
        )
        if "error" not in mc_results:
            print_success(f"Monte Carlo Results (on holdout):")
            print(
                f"  Mean Return: {mc_results['mean_return']:.1f}% ± {mc_results['std_return']:.1f}%"
            )
            print(f"  5th Percentile: {mc_results['percentile_5']:.1f}%")
            print(f"  Probability of Profit: {mc_results['prob_positive']:.1f}%")

    # 8. Run Final Backtest (on TEST data only - no lookahead)
    # Extract test results from tuning if available, otherwise run separate backtest
    if tuning_results and "test_results" in tuning_results:
        # Use pre-computed test results from hyperparameter tuning
        backtest_results = tuning_results["test_results"]
        backtest_results["initial_capital"] = args.capital
        backtest_results["final_capital"] = args.capital * (
            1 + backtest_results["total_return_pct"] / 100
        )
        backtest_results["total_return"] = (
            backtest_results["final_capital"] - args.capital
        )
        print_section("Final Backtest Results (from Test Set Evaluation)")
        print_info(f"Using pre-computed test results from hyperparameter tuning")
        print_info(
            f"Test Period: {tuning_results['split_info']['test_start']} to {tuning_results['split_info']['test_end']}"
        )
        print_info(
            f"Return: {backtest_results['total_return_pct']:.1f}%, Sharpe: {backtest_results['sharpe_ratio']:.2f}"
        )
    else:
        # Fallback: Run backtest on test data only
        test_data = spread_df[spread_df.index >= VAL_END_DATE].copy()
        backtest_results = run_backtest(
            all_data,
            test_data,
            args.capital,
            params=best_params if best_params else None,
        )

    # 9. Strategy Comparison Suite
    strategy_results = run_strategy_comparison(
        all_data, spread_df, regime_models, args.capital, rl_agents
    )

    # 10. Generate Report
    report = generate_report(
        all_data,
        spread_df,
        backtest_results,
        regime_models,
        rl_agents,
        best_params,
        strategy_results,
        rl_metrics,
        mc_results,
        tuning_results,
        validation_results,
    )

    # Save report
    report_path = Path("reports/WTI-Brent-spread")
    report_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"wti_test_report_{timestamp}"
    report_file = report_path / f"{report_name}.txt"
    report_file.write_text(report)
    print(f"\n📄 Report saved to: {report_file}")

    # 11. Generate Plots
    print_section("Phase 12: Generating Plots")
    plots_dir = report_path / report_name
    plots_dir.mkdir(exist_ok=True)
    plots_generated = generate_strategy_plots(
        spread_df, strategy_results, plots_dir, timestamp, regime_models, rl_agents
    )
    if plots_generated:
        print_success(f"Generated {len(plots_generated)} plots/visualizations")
        for plot in plots_generated:
            print_info(f"  → {plot}")

    # Exit status
    if backtest_results["total_return"] > 0:
        print(
            f"\n🎉 SUCCESS: Strategy generated ${backtest_results['total_return']:,.2f} profit!"
        )
        sys.exit(0)
    else:
        print(f"\n⚠️ Strategy lost ${abs(backtest_results['total_return']):,.2f}")
        sys.exit(0)


if __name__ == "__main__":
    main()
