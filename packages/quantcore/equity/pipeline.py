"""
Main pipeline orchestration for equity trading.

Provides:
- Data fetching and feature building
- Rule-based strategy execution
- ML strategy execution
- Report generation
"""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from loguru import logger

from quantcore.config.timeframes import Timeframe
from quantcore.data.fetcher import AlphaVantageClient
from quantcore.data.storage import DataStore
from quantcore.data.resampler import TimeframeResampler
from quantcore.data.universe import UniverseManager
from quantcore.features.factory import MultiTimeframeFeatureFactory

from quantcore.equity.strategies import (
    EquityStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
    RRGStrategy,
    CompositeStrategy,
)
from quantcore.equity.backtester import backtest_signals
from quantcore.equity.reports import (
    TickerStrategyResult,
    StrategyResult,
    PipelineReport,
    generate_text_report,
)
from quantcore.equity.ml_strategy import run_ml_strategy
from quantcore.equity.tuning import tune_all_tickers, TunedParams

# Try to import RL strategy
try:
    from quantcore.equity.rl_strategy import run_rl_strategy, RL_AVAILABLE
except ImportError:
    RL_AVAILABLE = False
    run_rl_strategy = None


@dataclass
class DataSplit:
    """Temporal data split indices."""

    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def val_size(self) -> int:
        return self.val_end - self.val_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


@dataclass
class SymbolData:
    """Complete data for a symbol."""

    symbol: str
    ohlcv: pd.DataFrame
    features: pd.DataFrame


def calculate_data_split(
    n_samples: int,
    train_pct: float = 0.6,
    val_pct: float = 0.2,
) -> DataSplit:
    """Calculate strict temporal split indices."""
    train_end = int(n_samples * train_pct)
    val_end = int(n_samples * (train_pct + val_pct))

    return DataSplit(
        train_start=0,
        train_end=train_end,
        val_start=train_end,
        val_end=val_end,
        test_start=val_end,
        test_end=n_samples,
    )


def get_universe_symbols() -> List[str]:
    """Get all symbols from the universe."""
    universe = UniverseManager()
    return universe.symbols


def fetch_equity_data(
    symbols: List[str],
    fetcher: AlphaVantageClient,
    data_store: DataStore,
    skip_fetch: bool = False,
    force_fetch: bool = False,
    soft_fetch: bool = False,
    start_year: int = 2015,
    end_year: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical equity data for all symbols.

    Args:
        symbols: List of symbols to fetch
        fetcher: AlphaVantage API client
        data_store: Data storage
        skip_fetch: Skip fetching entirely, use cached only
        force_fetch: Force re-fetch all data even if cached
        soft_fetch: Only fetch missing data (check date coverage)
        start_year: Start year for data range
        end_year: End year for data range (default: current year)
    """
    # Default end_year to current year
    if end_year is None:
        end_year = datetime.now().year

    logger.info("=" * 60)
    logger.info("PHASE 1: FETCHING EQUITY DATA")
    logger.info("=" * 60)
    logger.info(f"Symbols to fetch: {len(symbols)}")
    logger.info(
        f"Date range: {start_year}-{end_year} (through {datetime.now().strftime('%Y-%m-%d')})"
    )
    logger.info(
        f"Mode: {'force-fetch' if force_fetch else 'soft-fetch' if soft_fetch else 'skip-fetch' if skip_fetch else 'normal'}"
    )

    data = {}

    # Target date range
    target_start = datetime(start_year, 1, 1)
    target_end = datetime.now()

    for i, symbol in enumerate(symbols):
        logger.info(f"\n[{i+1}/{len(symbols)}] [{symbol}] Fetching equity data...")

        try:
            cached = data_store.load_ohlcv(symbol=symbol, timeframe=Timeframe.H1)

            # Skip fetch mode - use cached only
            if skip_fetch:
                if not cached.empty:
                    data[symbol] = cached
                    logger.info(f"  [CACHE] Using {len(cached)} cached bars")
                else:
                    logger.warning(f"  [SKIP] No cached data available")
                continue

            # Force fetch mode - always fetch fresh
            if force_fetch:
                logger.info(
                    f"  [FORCE-FETCH] Fetching full intraday history ({start_year}-{end_year})..."
                )
                df = fetcher.fetch_all_intraday_history(
                    symbol=symbol,
                    interval="60min",
                    start_year=start_year,
                    end_year=end_year,
                )

                if df is not None and not df.empty:
                    data_store.save_ohlcv(df, symbol=symbol, timeframe=Timeframe.H1)
                    data[symbol] = df
                    days = (df.index.max() - df.index.min()).days
                    logger.info(f"  [SUCCESS] {len(df)} bars, {days} days")
                else:
                    if not cached.empty:
                        data[symbol] = cached
                        logger.warning(f"  [FALLBACK] Using {len(cached)} cached bars")
                    else:
                        logger.warning(f"  [EMPTY] No data returned")

                time.sleep(1)
                continue

            # Soft fetch mode - check coverage and fetch only missing
            if soft_fetch:
                if not cached.empty:
                    cached_start = cached.index.min()
                    cached_end = cached.index.max()

                    # Check if we have sufficient coverage
                    has_start_coverage = cached_start <= target_start + pd.Timedelta(
                        days=30
                    )
                    has_end_coverage = cached_end >= target_end - pd.Timedelta(days=7)
                    min_bars = (
                        (end_year - start_year) * 252 * 7
                    )  # ~7 bars per trading day
                    has_enough_bars = (
                        len(cached) >= min_bars * 0.5
                    )  # At least 50% coverage

                    if has_start_coverage and has_end_coverage and has_enough_bars:
                        days = (cached_end - cached_start).days
                        logger.info(
                            f"  [CACHE OK] {len(cached)} bars, {days} days ({cached_start.strftime('%Y-%m-%d')} to {cached_end.strftime('%Y-%m-%d')})"
                        )
                        data[symbol] = cached
                        continue
                    else:
                        # Identify missing ranges
                        missing_start = None
                        missing_end = None

                        if not has_start_coverage:
                            missing_start = start_year
                            missing_end_year = cached_start.year
                            logger.info(
                                f"  [SOFT-FETCH] Missing early data: {start_year} to {missing_end_year}"
                            )

                        if not has_end_coverage:
                            missing_start_year = cached_end.year
                            missing_end = end_year
                            logger.info(
                                f"  [SOFT-FETCH] Missing recent data: {missing_start_year} to {end_year}"
                            )

                        # Fetch missing ranges
                        fetch_start = (
                            missing_start if missing_start else cached_end.year
                        )
                        fetch_end = missing_end if missing_end else end_year

                        logger.info(
                            f"  [SOFT-FETCH] Fetching {fetch_start}-{fetch_end}..."
                        )
                        new_df = fetcher.fetch_all_intraday_history(
                            symbol=symbol,
                            interval="60min",
                            start_year=fetch_start,
                            end_year=fetch_end,
                        )

                        if new_df is not None and not new_df.empty:
                            # Merge with existing cached data
                            combined = pd.concat([cached, new_df])
                            combined = combined[~combined.index.duplicated(keep="last")]
                            combined = combined.sort_index()

                            data_store.save_ohlcv(
                                combined, symbol=symbol, timeframe=Timeframe.H1
                            )
                            data[symbol] = combined
                            days = (combined.index.max() - combined.index.min()).days
                            logger.info(
                                f"  [SUCCESS] {len(combined)} bars, {days} days (merged)"
                            )
                        else:
                            data[symbol] = cached
                            logger.warning(
                                f"  [PARTIAL] Using existing {len(cached)} cached bars"
                            )

                        time.sleep(1)
                        continue
                else:
                    # No cached data - fetch full range
                    logger.info(
                        f"  [SOFT-FETCH] No cache, fetching full range ({start_year}-{end_year})..."
                    )
                    df = fetcher.fetch_all_intraday_history(
                        symbol=symbol,
                        interval="60min",
                        start_year=start_year,
                        end_year=end_year,
                    )

                    if df is not None and not df.empty:
                        data_store.save_ohlcv(df, symbol=symbol, timeframe=Timeframe.H1)
                        data[symbol] = df
                        days = (df.index.max() - df.index.min()).days
                        logger.info(f"  [SUCCESS] {len(df)} bars, {days} days")
                    else:
                        logger.warning(f"  [EMPTY] No data returned")

                    time.sleep(1)
                    continue

            # Normal mode (neither force nor soft) - use cache if good enough
            if not cached.empty and len(cached) >= 500:
                days = (cached.index.max() - cached.index.min()).days
                logger.info(f"  [CACHE] {len(cached)} bars, {days} days")
                data[symbol] = cached
                continue

            # Fetch if cache insufficient
            logger.info(
                f"  [FETCH] Fetching full intraday history ({start_year}-{end_year})..."
            )
            df = fetcher.fetch_all_intraday_history(
                symbol=symbol,
                interval="60min",
                start_year=start_year,
                end_year=end_year,
            )

            if df is not None and not df.empty:
                data_store.save_ohlcv(df, symbol=symbol, timeframe=Timeframe.H1)
                data[symbol] = df
                days = (df.index.max() - df.index.min()).days
                logger.info(f"  [SUCCESS] {len(df)} bars, {days} days")
            else:
                if not cached.empty:
                    data[symbol] = cached
                    logger.warning(f"  [FALLBACK] Using {len(cached)} cached bars")
                else:
                    logger.warning(f"  [EMPTY] No data returned")

            time.sleep(1)

        except Exception as e:
            logger.error(f"  [ERROR] {e}")
            # Try cached on error
            cached = data_store.load_ohlcv(symbol=symbol, timeframe=Timeframe.H1)
            if not cached.empty:
                data[symbol] = cached
                logger.info(f"  [FALLBACK] Using cached data after error")

    logger.info(f"\nFetched data for {len(data)} symbols")
    return data


def fetch_news_sentiment_for_symbols(
    symbols: List[str],
    fetcher: AlphaVantageClient,
    data_store: DataStore,
    force_fetch: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch news sentiment for equity symbols with caching.

    Args:
        symbols: List of symbols to fetch news for
        fetcher: AlphaVantage API client
        data_store: Data storage for caching
        force_fetch: Force re-fetch even if cached

    Returns:
        Dict mapping symbol -> news sentiment DataFrame
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1B: FETCHING NEWS SENTIMENT")
    logger.info("=" * 60)

    news_data = {}

    for i, symbol in enumerate(symbols[:5]):  # Limit to 5 symbols to respect API limits
        logger.info(
            f"\n[{i+1}/{min(5, len(symbols))}] [{symbol}] Fetching news sentiment..."
        )

        try:
            # Try cached first
            if not force_fetch:
                cached_news = data_store.load_news_sentiment(
                    start_date=datetime(2020, 1, 1),
                    end_date=datetime.now(),
                    tickers=[symbol],
                )
                if (
                    cached_news is not None
                    and not cached_news.empty
                    and len(cached_news) > 10
                ):
                    news_data[symbol] = cached_news
                    logger.info(f"  [CACHE] {len(cached_news)} cached articles")
                    continue

            # Fetch from API
            from datetime import timedelta

            start_date = datetime.now() - timedelta(days=365 * 2)

            news = fetcher.fetch_historical_news_sentiment(
                tickers=symbol,
                topics="earnings,financial_markets",
                start_date=start_date,
                end_date=datetime.now(),
                batch_months=6,
            )

            if not news.empty:
                # Save to cache
                rows_saved = data_store.save_news_sentiment(news)
                news_data[symbol] = news
                logger.info(f"  [SUCCESS] {len(news)} articles, saved {rows_saved}")
            else:
                logger.info(f"  [EMPTY] No news found")

            time.sleep(1)  # Rate limit

        except Exception as e:
            logger.error(f"  [ERROR] {e}")

    logger.info(f"\nFetched news for {len(news_data)} symbols")
    return news_data


def build_features(
    ohlcv_data: Dict[str, pd.DataFrame],
    benchmark_symbol: str = "SPY",
    news_sentiment_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """Build comprehensive features using existing trader modules."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: BUILDING FEATURES (All TA + RRG + MR + Gann + Sentiment)")
    logger.info("=" * 60)

    features = {}

    factory = MultiTimeframeFeatureFactory(
        include_waves=True,
        include_rrg=True,
        include_technical_indicators=True,
        enable_moving_averages=True,
        enable_oscillators=True,
        enable_volatility_indicators=True,
        enable_volume_indicators=True,
        enable_hilbert=False,
        include_trendlines=True,
        include_candlestick_patterns=True,
        include_quant_trend=True,
        include_quant_pattern=True,
        # NEW: Enable Gann, MR, and sentiment features
        include_gann_features=True,
        include_mean_reversion=True,
        include_sentiment_features=True,
    )

    resampler = TimeframeResampler()
    benchmark_df = ohlcv_data.get(benchmark_symbol)

    if benchmark_df is None:
        logger.warning(
            f"Benchmark {benchmark_symbol} not available, RRG features will be NaN"
        )

    for symbol, df in ohlcv_data.items():
        logger.info(f"\n[{symbol}] Building features...")

        try:
            tf_data = resampler.build_multi_timeframe_dataset(df)

            # Get news sentiment for this symbol if available
            symbol_news = None
            if news_sentiment_data and symbol in news_sentiment_data:
                symbol_news = news_sentiment_data[symbol]
                logger.info(
                    f"  Using {len(symbol_news)} news articles for sentiment features"
                )

            tf_features = factory.compute_all_timeframes(
                tf_data,
                lag_features=True,
                benchmark_data=(
                    {Timeframe.H1: benchmark_df} if benchmark_df is not None else None
                ),
                news_sentiment_data=symbol_news,
            )

            if tf_features:
                aligned = resampler.align_all_timeframes(
                    tf_features, base_tf=Timeframe.H1
                )

                base_df = df.copy().reindex(aligned.index)
                for col in ["open", "high", "low", "close", "volume"]:
                    if col not in aligned.columns and col in base_df.columns:
                        aligned[col] = base_df[col]

                aligned = aligned.ffill().bfill()

                if "close" in aligned.columns:
                    aligned["label"] = (
                        aligned["close"].shift(-1) > aligned["close"]
                    ).astype(int)

                # Ensure zscore_price exists
                if "zscore_price" not in aligned.columns:
                    if "1H_zscore_price" in aligned.columns:
                        aligned["zscore_price"] = aligned["1H_zscore_price"]
                    else:
                        close = aligned["close"]
                        aligned["zscore_price"] = (
                            close - close.rolling(20).mean()
                        ) / close.rolling(20).std()

                # Ensure ATR exists
                if "atr" not in aligned.columns:
                    if "1H_atr" in aligned.columns:
                        aligned["atr"] = aligned["1H_atr"]
                    else:
                        high = aligned["high"]
                        low = aligned["low"]
                        close = aligned["close"]
                        tr = pd.concat(
                            [
                                high - low,
                                (high - close.shift(1)).abs(),
                                (low - close.shift(1)).abs(),
                            ],
                            axis=1,
                        ).max(axis=1)
                        aligned["atr"] = tr.rolling(14).mean()

                numeric_aligned = aligned.select_dtypes(include=[np.number])
                numeric_aligned = numeric_aligned.replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(0)

                features[symbol] = numeric_aligned
                logger.info(
                    f"  [SUCCESS] {numeric_aligned.shape[0]} bars, {numeric_aligned.shape[1]} features"
                )

        except Exception as e:
            logger.error(f"  [ERROR] {e}")
            import traceback

            traceback.print_exc()

    return features


def run_rule_based_strategies(
    symbol_data: Dict[str, SymbolData],
    initial_equity: float = 100000,
    tuned_params: Optional[Dict[str, TunedParams]] = None,
) -> Dict[str, StrategyResult]:
    """
    Run all rule-based strategies with per-ticker tuned parameters.

    Args:
        symbol_data: Dict mapping symbol -> SymbolData
        initial_equity: Initial equity for backtesting
        tuned_params: Optional dict of TunedParams per ticker
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: RULE-BASED STRATEGIES (with per-ticker params)")
    logger.info("=" * 60)

    # Define which strategies to run
    strategy_names = ["MeanReversion", "Momentum", "TrendFollowing", "RRG", "Composite"]

    results = {}

    for strategy_name in strategy_names:
        logger.info(f"\n[{strategy_name}] Running backtest...")

        per_ticker = {}
        total_pnl = 0
        total_trades = 0
        total_wins = 0
        max_dd = 0

        for symbol, data in symbol_data.items():
            if data.features is None or data.features.empty:
                continue

            split = calculate_data_split(len(data.features))

            test_features = data.features.iloc[split.test_start : split.test_end]
            test_prices = data.ohlcv.iloc[split.test_start : split.test_end]

            if len(test_prices) < 20:
                continue

            # Use tuned params if available
            if tuned_params and symbol in tuned_params:
                tp = tuned_params[symbol]
                if strategy_name == "MeanReversion":
                    strategy = MeanReversionStrategy(**tp.mean_reversion)
                elif strategy_name == "Momentum":
                    strategy = MomentumStrategy(**tp.momentum)
                elif strategy_name == "TrendFollowing":
                    strategy = TrendFollowingStrategy()
                elif strategy_name == "RRG":
                    strategy = RRGStrategy()
                elif strategy_name == "Composite":
                    strategy = CompositeStrategy(
                        [
                            MeanReversionStrategy(**tp.mean_reversion),
                            MomentumStrategy(**tp.momentum),
                            TrendFollowingStrategy(),
                        ]
                    )
                else:
                    strategy = MeanReversionStrategy()
            else:
                # Default params
                if strategy_name == "MeanReversion":
                    strategy = MeanReversionStrategy(
                        zscore_threshold=2.0, reversion_delta=0.2
                    )
                elif strategy_name == "Momentum":
                    strategy = MomentumStrategy(rsi_oversold=30, rsi_overbought=70)
                elif strategy_name == "TrendFollowing":
                    strategy = TrendFollowingStrategy()
                elif strategy_name == "RRG":
                    strategy = RRGStrategy()
                elif strategy_name == "Composite":
                    strategy = CompositeStrategy(
                        [
                            MeanReversionStrategy(),
                            MomentumStrategy(),
                            TrendFollowingStrategy(),
                        ]
                    )
                else:
                    strategy = MeanReversionStrategy()

            signals = strategy.generate_signals(test_features)

            result = backtest_signals(
                signals=signals,
                prices=test_prices,
                shares_per_trade=100,
                initial_equity=initial_equity,
            )

            per_ticker[symbol] = TickerStrategyResult(
                ticker=symbol,
                strategy=strategy_name,
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
                f"  {symbol}: PnL=${result.total_pnl:,.0f}, Trades={result.num_trades}"
            )

        if per_ticker:
            best_ticker = max(per_ticker.items(), key=lambda x: x[1].pnl)[0]
            worst_ticker = min(per_ticker.items(), key=lambda x: x[1].pnl)[0]
        else:
            best_ticker = ""
            worst_ticker = ""

        total_return = total_pnl / initial_equity
        win_rate = total_wins / total_trades if total_trades > 0 else 0

        results[strategy_name] = StrategyResult(
            strategy_name=strategy_name,
            strategy_type="rule-based",
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
        )

        logger.info(
            f"  TOTAL: PnL=${total_pnl:,.0f}, Return={total_return:.2%}, Trades={total_trades}"
        )

    return results


def run_pipeline(
    symbols: Optional[List[str]] = None,
    skip_fetch: bool = False,
    force_fetch: bool = False,
    soft_fetch: bool = False,
    start_year: int = 2015,
    end_year: Optional[int] = None,
    initial_equity: float = 100000,
    tune_hyperparams: bool = True,
    run_rl: bool = True,
    fetch_news: bool = True,
) -> PipelineReport:
    """
    Run the complete equity signal pipeline.

    Args:
        symbols: List of symbols to process (default: all from universe)
        skip_fetch: Skip data fetching, use cached data only
        force_fetch: Force re-fetch ALL data even if cached
        soft_fetch: Smart fetch - only fetch missing data for date range
        start_year: Start year for data (default: 2015)
        end_year: End year for data (default: current year)
        initial_equity: Initial equity for backtesting
        tune_hyperparams: Whether to tune hyperparameters per ticker
        run_rl: Whether to run RL strategy
        fetch_news: Whether to fetch news sentiment for symbols

    Returns:
        PipelineReport with all results
    """
    if symbols is None:
        symbols = get_universe_symbols()

    # Default end_year to current year
    if end_year is None:
        end_year = datetime.now().year

    # Determine fetch mode
    fetch_mode = (
        "skip"
        if skip_fetch
        else "force" if force_fetch else "soft" if soft_fetch else "normal"
    )

    logger.info("=" * 80)
    logger.info("EQUITY SIGNAL PIPELINE - FULL FEATURE SET")
    logger.info("=" * 80)
    logger.info(f"Symbols: {len(symbols)} tickers")
    logger.info(f"Fetch Mode: {fetch_mode}")
    logger.info(f"Date Range: {start_year}-{end_year}")
    logger.info(f"Tune Hyperparams: {tune_hyperparams}")
    logger.info(f"Run RL: {run_rl}")
    logger.info(f"Fetch News: {fetch_news}")
    logger.info(f"Trade Size: 100 shares, $0 commission")

    fetcher = AlphaVantageClient()
    data_store = DataStore()

    # Phase 1: Fetch equity data
    ohlcv_data = fetch_equity_data(
        symbols,
        fetcher,
        data_store,
        skip_fetch=skip_fetch,
        force_fetch=force_fetch,
        soft_fetch=soft_fetch,
        start_year=start_year,
        end_year=end_year,
    )

    if not ohlcv_data:
        raise ValueError("No equity data available")

    # Phase 1B: Fetch news sentiment (optional)
    news_sentiment_data = None
    if fetch_news and not skip_fetch:
        news_sentiment_data = fetch_news_sentiment_for_symbols(
            list(ohlcv_data.keys()),
            fetcher,
            data_store,
            force_fetch=force_fetch,
        )

    # Phase 2: Build features (including sentiment if available)
    features = build_features(
        ohlcv_data, benchmark_symbol="SPY", news_sentiment_data=news_sentiment_data
    )

    # Assemble symbol data
    symbol_data = {}
    for symbol in symbols:
        if symbol in ohlcv_data and symbol in features:
            symbol_data[symbol] = SymbolData(
                symbol=symbol,
                ohlcv=ohlcv_data[symbol],
                features=features[symbol],
            )

    # Phase 3: Hyperparameter tuning (on validation data)
    tuned_params = None
    if tune_hyperparams:
        tuned_params = tune_all_tickers(
            symbol_data, calculate_data_split, initial_equity
        )

    # Phase 4: Rule-based strategies (with tuned params)
    rule_results = run_rule_based_strategies(symbol_data, initial_equity, tuned_params)

    # Phase 5: ML strategy
    ml_result = run_ml_strategy(symbol_data, initial_equity, calculate_data_split)

    # Combine results
    all_results = {**rule_results}
    if ml_result:
        all_results[ml_result.strategy_name] = ml_result

    # Phase 6: RL strategy
    if run_rl and RL_AVAILABLE and run_rl_strategy:
        rl_result = run_rl_strategy(
            symbol_data,
            initial_equity,
            total_timesteps=10000,
            calculate_data_split=calculate_data_split,
        )
        if rl_result:
            all_results[rl_result.strategy_name] = rl_result
    elif run_rl and not RL_AVAILABLE:
        logger.warning("RL not available - install gymnasium and stable-baselines3")

    # Build ticker breakdown
    ticker_breakdown = {}
    for symbol in symbol_data.keys():
        ticker_breakdown[symbol] = {}
        for strategy_name, result in all_results.items():
            if symbol in result.per_ticker:
                ticker_breakdown[symbol][strategy_name] = result.per_ticker[symbol].pnl

    # Find best strategy
    best_strategy = max(all_results.items(), key=lambda x: x[1].total_pnl)[0]

    # Build report with dates
    sample_symbol = list(symbol_data.keys())[0] if symbol_data else None
    split_info = {}
    if sample_symbol:
        data = symbol_data[sample_symbol]
        split = calculate_data_split(len(data.features))

        # Get actual dates
        idx = data.ohlcv.index
        train_start_date = (
            idx[split.train_start].strftime("%Y-%m-%d")
            if hasattr(idx[split.train_start], "strftime")
            else str(idx[split.train_start])
        )
        train_end_date = (
            idx[min(split.train_end - 1, len(idx) - 1)].strftime("%Y-%m-%d")
            if hasattr(idx[min(split.train_end - 1, len(idx) - 1)], "strftime")
            else str(idx[min(split.train_end - 1, len(idx) - 1)])
        )
        val_start_date = (
            idx[split.val_start].strftime("%Y-%m-%d")
            if hasattr(idx[split.val_start], "strftime")
            else str(idx[split.val_start])
        )
        val_end_date = (
            idx[min(split.val_end - 1, len(idx) - 1)].strftime("%Y-%m-%d")
            if hasattr(idx[min(split.val_end - 1, len(idx) - 1)], "strftime")
            else str(idx[min(split.val_end - 1, len(idx) - 1)])
        )
        test_start_date = (
            idx[split.test_start].strftime("%Y-%m-%d")
            if hasattr(idx[split.test_start], "strftime")
            else str(idx[split.test_start])
        )
        test_end_date = (
            idx[-1].strftime("%Y-%m-%d")
            if hasattr(idx[-1], "strftime")
            else str(idx[-1])
        )

        split_info = {
            "Train": f"{split.train_size} bars (60%) | {train_start_date} to {train_end_date}",
            "Validation": f"{split.val_size} bars (20%) | {val_start_date} to {val_end_date}",
            "Test": f"{split.test_size} bars (20%) | {test_start_date} to {test_end_date} - HOLDOUT",
        }

    # Build data summary with date ranges
    data_summary = {}
    for s, d in symbol_data.items():
        start_date = (
            d.ohlcv.index[0].strftime("%Y-%m-%d")
            if hasattr(d.ohlcv.index[0], "strftime")
            else str(d.ohlcv.index[0])
        )
        end_date = (
            d.ohlcv.index[-1].strftime("%Y-%m-%d")
            if hasattr(d.ohlcv.index[-1], "strftime")
            else str(d.ohlcv.index[-1])
        )
        data_summary[s] = f"{len(d.ohlcv)} hourly bars ({start_date} to {end_date})"

    report = PipelineReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        symbols_count=len(symbol_data),
        data_summary=data_summary,
        split_info=split_info,
        strategy_results=all_results,
        ticker_breakdown=ticker_breakdown,
        best_strategy=best_strategy,
        recommendations=[
            "Consider ensemble of top-performing strategies",
            "Use paper trading before live deployment",
            "Monitor strategy performance by sector",
            (
                "Hyperparameters tuned per-ticker on validation data"
                if tune_hyperparams
                else "Using default hyperparameters"
            ),
        ],
    )

    return report
