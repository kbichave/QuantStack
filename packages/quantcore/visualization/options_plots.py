"""
Options Trading Visualization.

Comprehensive plotting for options trading performance:
- Strategy comparison (Rule-based vs ML vs RL)
- Per-ticker performance
- Greeks exposure over time
- Equity curves
- Animated trade GIFs
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from loguru import logger

# Import generalized trade animation module
from quantcore.visualization.trade_animation import (
    generate_trade_animation_gif,
    generate_options_trade_gif,
)


# Color scheme for strategies
STRATEGY_COLORS = {
    "rule_based": "#2ecc71",  # Green
    "ml": "#3498db",  # Blue
    "rl": "#9b59b6",  # Purple
}

# Color scheme for tickers (categorical)
TICKER_CMAP = plt.cm.tab20


def plot_strategy_comparison(
    results_by_strategy: Dict[str, Dict[str, Any]],
    title: str = "Strategy Comparison",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compare performance across Rule-based, ML, and RL strategies.

    Args:
        results_by_strategy: Dict of strategy_name -> backtest results
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    strategies = list(results_by_strategy.keys())

    # Panel 1: Equity curves
    ax1 = axes[0, 0]
    for strategy, results in results_by_strategy.items():
        if "equity_curve" in results:
            equity = results["equity_curve"]
            color = STRATEGY_COLORS.get(strategy.lower(), "#333333")
            ax1.plot(equity, label=strategy, color=color, linewidth=1.5)

    ax1.set_title("Equity Curves", fontsize=11)
    ax1.set_xlabel("Bar")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Returns distribution
    ax2 = axes[0, 1]
    returns_data = []
    labels = []
    colors = []

    for strategy, results in results_by_strategy.items():
        if "equity_curve" in results:
            equity = np.array(results["equity_curve"])
            returns = np.diff(equity) / equity[:-1]
            returns_data.append(returns * 100)  # Convert to percent
            labels.append(strategy)
            colors.append(STRATEGY_COLORS.get(strategy.lower(), "#333333"))

    if returns_data:
        bp = ax2.boxplot(returns_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax2.set_title("Returns Distribution", fontsize=11)
    ax2.set_ylabel("Daily Return (%)")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Performance metrics bar chart
    ax3 = axes[1, 0]
    metrics = ["sharpe_ratio", "total_return", "win_rate"]
    x = np.arange(len(metrics))
    width = 0.25

    for i, (strategy, results) in enumerate(results_by_strategy.items()):
        values = [
            results.get("sharpe_ratio", 0),
            results.get("total_return", 0) * 100,  # Convert to %
            results.get("win_rate", 0) * 100,  # Convert to %
        ]
        color = STRATEGY_COLORS.get(strategy.lower(), "#333333")
        ax3.bar(x + i * width, values, width, label=strategy, color=color, alpha=0.8)

    ax3.set_title("Performance Metrics", fontsize=11)
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(["Sharpe", "Return (%)", "Win Rate (%)"])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: Drawdown comparison
    ax4 = axes[1, 1]
    for strategy, results in results_by_strategy.items():
        if "drawdown_curve" in results:
            dd = np.array(results["drawdown_curve"]) * 100  # Convert to %
            color = STRATEGY_COLORS.get(strategy.lower(), "#333333")
            ax4.fill_between(
                range(len(dd)), 0, -dd, alpha=0.3, color=color, label=strategy
            )
            ax4.plot(-dd, color=color, linewidth=1)

    ax4.set_title("Drawdown", fontsize=11)
    ax4.set_xlabel("Bar")
    ax4.set_ylabel("Drawdown (%)")
    ax4.legend(loc="lower left")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved strategy comparison plot to {output_path}")

    return fig


def plot_per_ticker_performance(
    results_by_ticker: Dict[str, Dict[str, Any]],
    strategy_name: str = "Strategy",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot performance metrics for each ticker.

    Args:
        results_by_ticker: Dict of ticker -> backtest results
        strategy_name: Name of the strategy
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    tickers = list(results_by_ticker.keys())
    n_tickers = len(tickers)

    if n_tickers == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"{strategy_name} - Per-Ticker Performance", fontsize=14, fontweight="bold"
    )

    # Extract metrics
    returns = [results_by_ticker[t].get("total_return", 0) * 100 for t in tickers]
    sharpes = [results_by_ticker[t].get("sharpe_ratio", 0) for t in tickers]
    win_rates = [results_by_ticker[t].get("win_rate", 0) * 100 for t in tickers]
    num_trades = [results_by_ticker[t].get("num_trades", 0) for t in tickers]

    # Color by return (green for positive, red for negative)
    colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in returns]

    # Panel 1: Total Return by ticker
    ax1 = axes[0, 0]
    bars = ax1.barh(tickers, returns, color=colors, alpha=0.8)
    ax1.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
    ax1.set_title("Total Return (%)", fontsize=11)
    ax1.set_xlabel("Return (%)")
    ax1.grid(True, alpha=0.3, axis="x")

    # Panel 2: Sharpe Ratio by ticker
    ax2 = axes[0, 1]
    sharpe_colors = ["#2ecc71" if s >= 0 else "#e74c3c" for s in sharpes]
    ax2.barh(tickers, sharpes, color=sharpe_colors, alpha=0.8)
    ax2.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
    ax2.axvline(x=1, color="green", linestyle="--", alpha=0.3, label="Sharpe=1")
    ax2.set_title("Sharpe Ratio", fontsize=11)
    ax2.set_xlabel("Sharpe")
    ax2.grid(True, alpha=0.3, axis="x")

    # Panel 3: Win Rate by ticker
    ax3 = axes[1, 0]
    wr_colors = ["#2ecc71" if w >= 50 else "#e74c3c" for w in win_rates]
    ax3.barh(tickers, win_rates, color=wr_colors, alpha=0.8)
    ax3.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax3.set_title("Win Rate (%)", fontsize=11)
    ax3.set_xlabel("Win Rate (%)")
    ax3.set_xlim(0, 100)
    ax3.grid(True, alpha=0.3, axis="x")

    # Panel 4: Number of trades by ticker
    ax4 = axes[1, 1]
    ax4.barh(tickers, num_trades, color="#3498db", alpha=0.8)
    ax4.set_title("Number of Trades", fontsize=11)
    ax4.set_xlabel("Trades")
    ax4.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved per-ticker plot to {output_path}")

    return fig


def plot_greeks_exposure(
    greeks_history: pd.DataFrame,
    title: str = "Portfolio Greeks Over Time",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot Greeks exposure over time.

    Args:
        greeks_history: DataFrame with columns: timestamp, delta, gamma, theta, vega
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Extract data
    if "timestamp" in greeks_history.columns:
        x = greeks_history["timestamp"]
    else:
        x = range(len(greeks_history))

    # Delta
    ax1 = axes[0, 0]
    if "delta" in greeks_history.columns:
        delta = greeks_history["delta"]
        ax1.fill_between(x, 0, delta, where=(delta >= 0), alpha=0.3, color="green")
        ax1.fill_between(x, 0, delta, where=(delta < 0), alpha=0.3, color="red")
        ax1.plot(x, delta, color="black", linewidth=1)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Delta Exposure", fontsize=11)
    ax1.set_ylabel("Delta")
    ax1.grid(True, alpha=0.3)

    # Gamma
    ax2 = axes[0, 1]
    if "gamma" in greeks_history.columns:
        gamma = greeks_history["gamma"]
        ax2.fill_between(x, 0, gamma, alpha=0.3, color="blue")
        ax2.plot(x, gamma, color="blue", linewidth=1)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("Gamma Exposure", fontsize=11)
    ax2.set_ylabel("Gamma")
    ax2.grid(True, alpha=0.3)

    # Theta
    ax3 = axes[1, 0]
    if "theta" in greeks_history.columns:
        theta = greeks_history["theta"]
        ax3.fill_between(x, 0, theta, where=(theta >= 0), alpha=0.3, color="green")
        ax3.fill_between(x, 0, theta, where=(theta < 0), alpha=0.3, color="red")
        ax3.plot(x, theta, color="black", linewidth=1)
    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_title("Theta (Daily Decay)", fontsize=11)
    ax3.set_ylabel("Theta ($/day)")
    ax3.grid(True, alpha=0.3)

    # Vega
    ax4 = axes[1, 1]
    if "vega" in greeks_history.columns:
        vega = greeks_history["vega"]
        ax4.fill_between(x, 0, vega, alpha=0.3, color="purple")
        ax4.plot(x, vega, color="purple", linewidth=1)
    ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_title("Vega Exposure", fontsize=11)
    ax4.set_ylabel("Vega")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved Greeks plot to {output_path}")

    return fig


def plot_combined_equity_curves(
    results_by_strategy_and_ticker: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot combined equity curves for all strategies and tickers.

    Args:
        results_by_strategy_and_ticker: Nested dict: strategy -> ticker -> results
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    strategies = list(results_by_strategy_and_ticker.keys())

    fig, axes = plt.subplots(1, len(strategies), figsize=(6 * len(strategies), 6))
    fig.suptitle("Equity Curves by Strategy", fontsize=14, fontweight="bold")

    if len(strategies) == 1:
        axes = [axes]

    for i, strategy in enumerate(strategies):
        ax = axes[i]
        ticker_results = results_by_strategy_and_ticker[strategy]

        for j, (ticker, results) in enumerate(ticker_results.items()):
            if "equity_curve" in results:
                equity = results["equity_curve"]
                # Normalize to start at 100
                if equity[0] != 0:
                    normalized = [e / equity[0] * 100 for e in equity]
                else:
                    normalized = equity
                color = TICKER_CMAP(j % 20)
                ax.plot(normalized, label=ticker, color=color, linewidth=0.8, alpha=0.7)

        ax.set_title(f"{strategy}", fontsize=11)
        ax.set_xlabel("Bar")
        ax.set_ylabel("Normalized Equity")
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved combined equity curves to {output_path}")

    return fig


def plot_best_strategy_per_ticker(
    results_by_strategy_and_ticker: Dict[str, Dict[str, Dict[str, Any]]],
    metric: str = "sharpe_ratio",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Show best strategy for each ticker.

    Args:
        results_by_strategy_and_ticker: Nested dict: strategy -> ticker -> results
        metric: Metric to compare ('sharpe_ratio', 'total_return', 'win_rate')
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Get all tickers
    all_tickers = set()
    for strategy_results in results_by_strategy_and_ticker.values():
        all_tickers.update(strategy_results.keys())
    tickers = sorted(all_tickers)
    strategies = list(results_by_strategy_and_ticker.keys())

    # Build comparison matrix
    data = []
    best_strategies = []

    for ticker in tickers:
        ticker_metrics = {}
        for strategy in strategies:
            if ticker in results_by_strategy_and_ticker.get(strategy, {}):
                value = results_by_strategy_and_ticker[strategy][ticker].get(metric, 0)
                ticker_metrics[strategy] = value
            else:
                ticker_metrics[strategy] = 0

        data.append(ticker_metrics)

        # Find best strategy
        if ticker_metrics:
            best = max(ticker_metrics, key=ticker_metrics.get)
            best_strategies.append(best)
        else:
            best_strategies.append("None")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Panel 1: Heatmap of metrics
    ax1 = axes[0]
    matrix = np.array([[d.get(s, 0) for s in strategies] for d in data])

    im = ax1.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategies, rotation=45, ha="right")
    ax1.set_yticks(range(len(tickers)))
    ax1.set_yticklabels(tickers)
    ax1.set_title(f"{metric} by Strategy and Ticker", fontsize=11)

    # Add colorbar
    plt.colorbar(im, ax=ax1, label=metric)

    # Panel 2: Best strategy count
    ax2 = axes[1]
    strategy_counts = {}
    for s in best_strategies:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    colors = [STRATEGY_COLORS.get(s.lower(), "#333333") for s in strategy_counts.keys()]
    bars = ax2.bar(
        strategy_counts.keys(), strategy_counts.values(), color=colors, alpha=0.8
    )

    ax2.set_title("Best Strategy Count", fontsize=11)
    ax2.set_ylabel("Number of Tickers")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add count labels on bars
    for bar, count in zip(bars, strategy_counts.values()):
        height = bar.get_height()
        ax2.annotate(
            f"{count}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved best strategy plot to {output_path}")

    return fig


def generate_all_strategy_plots(
    results_by_strategy: Dict[str, Dict[str, Any]],
    results_by_ticker: Dict[str, Dict[str, Dict[str, Any]]],
    greeks_history: Optional[pd.DataFrame] = None,
    output_dir: Path = Path("reports/options"),
    report_timestamp: Optional[str] = None,
) -> List[Path]:
    """
    Generate all strategy plots.

    Args:
        results_by_strategy: Dict of strategy_name -> aggregated results
        results_by_ticker: Dict of strategy_name -> ticker -> results
        greeks_history: Optional Greeks history DataFrame
        output_dir: Output directory
        report_timestamp: Timestamp for file naming

    Returns:
        List of generated file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if report_timestamp is None:
        report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    generated_files = []

    try:
        # Strategy comparison
        fig = plot_strategy_comparison(
            results_by_strategy,
            title="Options Strategy Comparison",
            output_path=output_dir / f"strategy_comparison_{report_timestamp}.png",
        )
        generated_files.append(
            output_dir / f"strategy_comparison_{report_timestamp}.png"
        )
        plt.close(fig)

        # Per-ticker performance for each strategy
        for strategy, ticker_results in results_by_ticker.items():
            fig = plot_per_ticker_performance(
                ticker_results,
                strategy_name=strategy,
                output_path=output_dir
                / f"{strategy}_per_ticker_{report_timestamp}.png",
            )
            generated_files.append(
                output_dir / f"{strategy}_per_ticker_{report_timestamp}.png"
            )
            plt.close(fig)

        # Combined equity curves
        if results_by_ticker:
            fig = plot_combined_equity_curves(
                results_by_ticker,
                output_path=output_dir / f"combined_equity_{report_timestamp}.png",
            )
            generated_files.append(
                output_dir / f"combined_equity_{report_timestamp}.png"
            )
            plt.close(fig)

        # Best strategy per ticker
        if results_by_ticker:
            fig = plot_best_strategy_per_ticker(
                results_by_ticker,
                metric="sharpe_ratio",
                output_path=output_dir / f"best_strategy_{report_timestamp}.png",
            )
            generated_files.append(output_dir / f"best_strategy_{report_timestamp}.png")
            plt.close(fig)

        # Greeks exposure
        if greeks_history is not None and not greeks_history.empty:
            fig = plot_greeks_exposure(
                greeks_history,
                output_path=output_dir / f"greeks_exposure_{report_timestamp}.png",
            )
            generated_files.append(
                output_dir / f"greeks_exposure_{report_timestamp}.png"
            )
            plt.close(fig)

    except Exception as e:
        logger.error(f"Error generating plots: {e}")

    return generated_files


def generate_options_backtest_animation(
    data: pd.DataFrame,
    trades: List[Dict],
    symbol: str,
    strategy_name: str,
    output_dir: Path = Path("reports/options"),
    report_timestamp: Optional[str] = None,
    initial_capital: float = 100000,
    **kwargs,
) -> Optional[str]:
    """
    Generate animated GIF for options backtest results.

    Shows:
    - Underlying equity price with candlesticks
    - LONG/SHORT entry markers
    - Take-profit/stop-loss exit markers with P&L labels
    - Running cumulative P&L (updates during position)

    Args:
        data: OHLCV DataFrame for underlying equity
        trades: List of trade dicts with entry_idx, exit_idx, direction, pnl
        symbol: Underlying symbol
        strategy_name: Strategy name for title
        output_dir: Output directory
        report_timestamp: Timestamp for filename
        initial_capital: Starting capital
        **kwargs: Additional args passed to generate_options_trade_gif

    Returns:
        Path to generated GIF, or None if generation failed
    """
    if report_timestamp is None:
        report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return generate_options_trade_gif(
        data=data,
        trades=trades,
        output_dir=output_dir,
        symbol=symbol,
        strategy_name=strategy_name,
        report_timestamp=report_timestamp,
        initial_capital=initial_capital,
        **kwargs,
    )
