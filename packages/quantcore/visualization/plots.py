"""
Plotting functions for WTI trading system.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from loguru import logger

from quantcore.backtesting.engine import run_backtest_with_params

# Try to import PIL for GIF generation
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def plot_strategy_signals(
    data: pd.DataFrame,
    strategy_name: str,
    trades: List[Dict],
    equity_curve: List[float],
    output_dir: Path,
    report_timestamp: str,
) -> Optional[str]:
    """
    Plot strategy signals on price chart.
    """
    try:
        fig, axes = plt.subplots(
            3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1, 1]}
        )
        fig.suptitle(
            f"{strategy_name} - Trading Signals (Test Period)",
            fontsize=14,
            fontweight="bold",
        )

        plt.style.use("seaborn-v0_8-darkgrid")

        dates = data.index if hasattr(data.index, "date") else range(len(data))
        spread = (
            data["spread"].values if "spread" in data.columns else data["wti"].values
        )
        zscore = (
            data["spread_zscore"].values
            if "spread_zscore" in data.columns
            else np.zeros(len(data))
        )

        # Panel 1: Price/Spread with Signals
        ax1 = axes[0]
        ax1.plot(dates, spread, "b-", linewidth=1, alpha=0.8, label="WTI-Brent Spread")
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        if trades:
            for trade in trades:
                if "entry_idx" in trade and "exit_idx" in trade:
                    entry_idx = trade["entry_idx"]
                    exit_idx = trade["exit_idx"]
                    direction = trade.get("direction", "LONG")

                    if 0 <= entry_idx < len(dates) and 0 <= exit_idx < len(dates):
                        entry_date = dates[entry_idx]
                        exit_date = dates[exit_idx]
                        entry_price = spread[entry_idx]
                        exit_price = spread[exit_idx]

                        if "LONG" in direction:
                            ax1.scatter(
                                entry_date,
                                entry_price,
                                marker="^",
                                color="green",
                                s=100,
                                zorder=5,
                            )
                            ax1.scatter(
                                exit_date,
                                exit_price,
                                marker="v",
                                color="red",
                                s=100,
                                zorder=5,
                            )
                        else:
                            ax1.scatter(
                                entry_date,
                                entry_price,
                                marker="v",
                                color="red",
                                s=100,
                                zorder=5,
                            )
                            ax1.scatter(
                                exit_date,
                                exit_price,
                                marker="^",
                                color="green",
                                s=100,
                                zorder=5,
                            )

        ax1.set_ylabel("Spread ($)", fontsize=10)
        ax1.legend(loc="upper left")
        ax1.set_title("WTI-Brent Spread with Entry/Exit Signals", fontsize=11)

        # Panel 2: Z-Score
        ax2 = axes[1]
        ax2.fill_between(
            dates,
            zscore,
            0,
            where=(zscore > 0),
            alpha=0.3,
            color="red",
            label="Overbought",
        )
        ax2.fill_between(
            dates,
            zscore,
            0,
            where=(zscore < 0),
            alpha=0.3,
            color="green",
            label="Oversold",
        )
        ax2.plot(dates, zscore, "k-", linewidth=1)
        ax2.axhline(y=2, color="red", linestyle="--", alpha=0.7, label="Entry Short")
        ax2.axhline(y=-2, color="green", linestyle="--", alpha=0.7, label="Entry Long")
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax2.set_ylabel("Z-Score", fontsize=10)
        ax2.set_ylim(-4, 4)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.set_title("Spread Z-Score", fontsize=11)

        # Panel 3: Equity Curve
        ax3 = axes[2]
        if len(equity_curve) > 0:
            eq_dates = (
                dates[: len(equity_curve)]
                if len(equity_curve) <= len(dates)
                else range(len(equity_curve))
            )
            ax3.plot(eq_dates, equity_curve, "purple", linewidth=2, label="Equity")
            ax3.fill_between(
                eq_dates, equity_curve[0], equity_curve, alpha=0.3, color="purple"
            )
        ax3.set_ylabel("Equity ($)", fontsize=10)
        ax3.set_xlabel("Date", fontsize=10)
        ax3.legend(loc="upper left")
        ax3.set_title("Equity Curve", fontsize=11)

        # Format x-axis
        for ax in axes:
            if hasattr(dates[0], "date"):
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        safe_name = (
            strategy_name.replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        filename = output_dir / f"strategy_{safe_name}_{report_timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return str(filename)

    except Exception as e:
        logger.warning(f"Failed to plot {strategy_name}: {e}")
        plt.close("all")
        return None


def _draw_candlestick(
    ax,
    idx: int,
    open_price: float,
    high: float,
    low: float,
    close: float,
    width: float = 0.6,
    up_color: str = "#26a69a",
    down_color: str = "#ef5350",
):
    """Draw a single candlestick on the given axis."""
    if close >= open_price:
        color = up_color
        body_bottom = open_price
        body_height = close - open_price
    else:
        color = down_color
        body_bottom = close
        body_height = open_price - close

    # Draw wick (high-low line)
    ax.plot([idx, idx], [low, high], color=color, linewidth=1, zorder=1)

    # Draw body
    if body_height > 0:
        rect = Rectangle(
            (idx - width / 2, body_bottom),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=1,
            zorder=2,
        )
        ax.add_patch(rect)
    else:
        # Doji - just a horizontal line
        ax.plot(
            [idx - width / 2, idx + width / 2],
            [close, close],
            color=color,
            linewidth=2,
            zorder=2,
        )


def generate_candlestick_gif(
    data: pd.DataFrame,
    trades: List[Dict],
    output_dir: Path,
    report_timestamp: str,
    strategy_name: str = "Spread",
    frame_duration: int = 300,  # Slower - 300ms per frame
    initial_capital: float = 100000,
    use_full_data: bool = True,  # Use full test period
    show_zscore: bool = False,  # Hide Z-score panel by default
) -> Optional[str]:
    """
    Generate TradingView-style animated GIF showing backtest with proper candlesticks,
    trade markers, P&L labels, and running equity curve.

    Args:
        show_zscore: If True, shows Z-score panel. Default False for cleaner view.
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, skipping GIF generation")
        return None

    try:
        # Use full data for complete test period coverage
        if use_full_data:
            plot_data = data.copy()
        else:
            plot_data = data.tail(300).copy()

        if len(plot_data) < 60:
            return None

        frames = []
        # Adjust window and step based on data size for smooth animation
        if len(plot_data) > 500:
            window_size = 80  # Larger window for long datasets
            step = 5  # Faster scrolling for long data
        else:
            window_size = 60
            step = 2

        # Check if we have OHLC data for proper candlesticks
        has_ohlc = all(
            col in plot_data.columns for col in ["open", "high", "low", "close"]
        )

        # Extract price data
        if "spread" in plot_data.columns:
            prices = plot_data["spread"].values
            price_label = "WTI-Brent Spread ($)"
        elif "wti" in plot_data.columns:
            prices = plot_data["wti"].values
            price_label = "WTI Price ($)"
        else:
            prices = (
                plot_data["close"].values
                if "close" in plot_data.columns
                else np.zeros(len(plot_data))
            )
            price_label = "Price ($)"

        # Generate synthetic OHLC from spread if not available
        if not has_ohlc:
            # Create synthetic OHLC based on spread movement
            opens = np.roll(prices, 1)
            opens[0] = prices[0]
            volatility = np.abs(np.diff(prices, prepend=prices[0])) * 0.5 + 0.05
            highs = np.maximum(prices, opens) + volatility
            lows = np.minimum(prices, opens) - volatility
            closes = prices
        else:
            opens = plot_data["open"].values
            highs = plot_data["high"].values
            lows = plot_data["low"].values
            closes = plot_data["close"].values

        zscore = (
            plot_data["spread_zscore"].values
            if "spread_zscore" in plot_data.columns
            else np.zeros(len(plot_data))
        )
        dates = plot_data.index

        # Map trades to indices in plot_data
        trade_entries = {}
        trade_exits = {}
        for trade in trades:
            if "entry_idx" in trade:
                local_entry = trade["entry_idx"] - (len(data) - len(plot_data))
                if 0 <= local_entry < len(plot_data):
                    trade_entries[local_entry] = trade
            if "exit_idx" in trade:
                local_exit = trade["exit_idx"] - (len(data) - len(plot_data))
                if 0 <= local_exit < len(plot_data):
                    trade_exits[local_exit] = trade

        # TradingView dark theme colors
        bg_color = "#131722"
        panel_color = "#1e222d"
        grid_color = "#2a2e39"
        grid_color_strong = "#434651"
        text_color = "#d1d4dc"
        text_dim = "#787b86"
        green_color = "#26a69a"
        red_color = "#ef5350"
        blue_color = "#2962ff"
        purple_color = "#ab47bc"
        orange_color = "#ff9800"

        for start_idx in range(0, len(plot_data) - window_size, step):
            end_idx = start_idx + window_size

            # Create figure with dark theme - layout depends on whether we show Z-score
            if show_zscore:
                fig = plt.figure(figsize=(14, 9), facecolor=bg_color)
                gs = fig.add_gridspec(
                    4, 1, height_ratios=[3.5, 1, 1.2, 0.6], hspace=0.05
                )
                ax_price = fig.add_subplot(gs[0])
                ax_zscore = fig.add_subplot(gs[1])
                ax_equity = fig.add_subplot(gs[2])
                ax_stats = fig.add_subplot(gs[3])
                main_axes = [ax_price, ax_zscore, ax_equity]
            else:
                # Cleaner 3-panel layout without Z-score
                fig = plt.figure(figsize=(14, 8), facecolor=bg_color)
                gs = fig.add_gridspec(3, 1, height_ratios=[4, 1.5, 0.6], hspace=0.05)
                ax_price = fig.add_subplot(gs[0])
                ax_zscore = None  # No Z-score panel
                ax_equity = fig.add_subplot(gs[1])
                ax_stats = fig.add_subplot(gs[2])
                main_axes = [ax_price, ax_equity]

            # Style all main axes
            for ax in main_axes:
                ax.set_facecolor(bg_color)
                ax.tick_params(colors=text_color, labelsize=9)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color(grid_color_strong)
                ax.spines["left"].set_color(grid_color_strong)
                # Enhanced grid
                ax.grid(
                    True,
                    which="major",
                    color=grid_color_strong,
                    alpha=0.6,
                    linestyle="-",
                    linewidth=0.8,
                )
                ax.grid(
                    True,
                    which="minor",
                    color=grid_color,
                    alpha=0.3,
                    linestyle="-",
                    linewidth=0.5,
                )
                ax.minorticks_on()

            ax_stats.set_facecolor(panel_color)
            ax_stats.axis("off")

            window_opens = opens[start_idx:end_idx]
            window_highs = highs[start_idx:end_idx]
            window_lows = lows[start_idx:end_idx]
            window_closes = closes[start_idx:end_idx]
            window_prices = prices[start_idx:end_idx]
            window_zscore = zscore[start_idx:end_idx]
            window_dates = dates[start_idx:end_idx]

            # === PRICE PANEL with proper candlesticks ===
            for i in range(len(window_prices)):
                _draw_candlestick(
                    ax_price,
                    i,
                    window_opens[i],
                    window_highs[i],
                    window_lows[i],
                    window_closes[i],
                    width=0.7,
                    up_color=green_color,
                    down_color=red_color,
                )

            # Draw moving average line for context
            if len(window_prices) >= 20:
                ma20 = pd.Series(window_prices).rolling(20).mean().values
                ax_price.plot(
                    range(len(ma20)),
                    ma20,
                    color=orange_color,
                    linewidth=1.5,
                    alpha=0.7,
                    label="MA20",
                    linestyle="--",
                )

            # Draw entry/exit markers with P&L labels
            for i in range(len(window_prices)):
                global_idx = start_idx + i
                price_y = window_prices[i]
                y_range = max(window_highs) - min(window_lows)

                # Entry marker
                if global_idx in trade_entries:
                    trade = trade_entries[global_idx]
                    direction = trade.get("direction", "LONG")
                    if "LONG" in direction:
                        # Green triangle pointing up
                        ax_price.scatter(
                            i,
                            window_lows[i] - y_range * 0.08,
                            marker="^",
                            s=200,
                            color=green_color,
                            edgecolors="white",
                            linewidths=1.5,
                            zorder=10,
                        )
                        ax_price.text(
                            i,
                            window_lows[i] - y_range * 0.15,
                            "LONG",
                            ha="center",
                            va="top",
                            fontsize=9,
                            color=green_color,
                            fontweight="bold",
                        )
                    else:
                        # Red triangle pointing down
                        ax_price.scatter(
                            i,
                            window_highs[i] + y_range * 0.08,
                            marker="v",
                            s=200,
                            color=red_color,
                            edgecolors="white",
                            linewidths=1.5,
                            zorder=10,
                        )
                        ax_price.text(
                            i,
                            window_highs[i] + y_range * 0.15,
                            "SHORT",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            color=red_color,
                            fontweight="bold",
                        )

                # Exit marker with P&L
                if global_idx in trade_exits:
                    trade = trade_exits[global_idx]
                    pnl = trade.get("pnl", 0)
                    pnl_color = green_color if pnl > 0 else red_color
                    pnl_text = f"+${pnl:,.0f}" if pnl > 0 else f"${pnl:,.0f}"

                    # Exit X marker
                    ax_price.scatter(
                        i,
                        price_y,
                        marker="X",
                        s=150,
                        color=pnl_color,
                        edgecolors="white",
                        linewidths=1.5,
                        zorder=10,
                    )
                    ax_price.annotate(
                        pnl_text,
                        xy=(i, price_y),
                        xytext=(i + 2, price_y),
                        fontsize=10,
                        color=pnl_color,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor=bg_color,
                            edgecolor=pnl_color,
                            alpha=0.9,
                        ),
                        arrowprops=dict(arrowstyle="-", color=pnl_color, alpha=0.5),
                    )

            # Price axis formatting
            y_range = max(window_highs) - min(window_lows)
            ax_price.set_ylabel(
                price_label, fontsize=10, color=text_color, fontweight="bold"
            )
            ax_price.set_xlim(-2, window_size + 5)
            ax_price.set_ylim(
                min(window_lows) - y_range * 0.2, max(window_highs) + y_range * 0.2
            )
            ax_price.set_xticklabels([])

            # Current price indicator on right
            current_price = window_closes[-1]
            price_color = (
                green_color if current_price >= window_opens[-1] else red_color
            )
            ax_price.axhline(
                y=current_price,
                color=price_color,
                linestyle="--",
                alpha=0.5,
                linewidth=1,
            )
            ax_price.text(
                window_size + 1,
                current_price,
                f"${current_price:.2f}",
                ha="left",
                va="center",
                fontsize=10,
                color=price_color,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=price_color,
                    edgecolor="none",
                    alpha=0.2,
                ),
            )

            # Title with date and strategy
            if hasattr(window_dates[-1], "strftime"):
                date_title = window_dates[-1].strftime("%b %d, %Y")
            else:
                date_title = f"Bar {end_idx}"
            ax_price.set_title(
                f"{strategy_name}  |  {date_title}",
                fontsize=13,
                color=text_color,
                fontweight="bold",
                loc="left",
                pad=12,
            )

            # === Z-SCORE PANEL (optional) ===
            current_z = window_zscore[-1]  # Always compute for stats
            z_color = (
                green_color
                if current_z < -2
                else (red_color if current_z > 2 else text_color)
            )

            if show_zscore and ax_zscore is not None:
                # Draw Z-score as area chart
                ax_zscore.fill_between(
                    range(len(window_zscore)),
                    0,
                    window_zscore,
                    where=[z >= 0 for z in window_zscore],
                    color=red_color,
                    alpha=0.4,
                )
                ax_zscore.fill_between(
                    range(len(window_zscore)),
                    0,
                    window_zscore,
                    where=[z < 0 for z in window_zscore],
                    color=green_color,
                    alpha=0.4,
                )
                ax_zscore.plot(
                    range(len(window_zscore)),
                    window_zscore,
                    color=blue_color,
                    linewidth=2,
                )

                # Entry threshold lines
                ax_zscore.axhline(
                    y=2,
                    color=red_color,
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                    label="Sell Zone",
                )
                ax_zscore.axhline(
                    y=-2,
                    color=green_color,
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                    label="Buy Zone",
                )
                ax_zscore.axhline(
                    y=0, color=text_dim, linestyle="-", alpha=0.5, linewidth=1
                )

                # Zone shading
                ax_zscore.axhspan(2, 5, color=red_color, alpha=0.1)
                ax_zscore.axhspan(-5, -2, color=green_color, alpha=0.1)

                ax_zscore.set_ylabel(
                    "Z-Score", fontsize=10, color=text_color, fontweight="bold"
                )
                ax_zscore.set_xlim(-2, window_size + 5)
                ax_zscore.set_ylim(-4.5, 4.5)
                ax_zscore.set_xticklabels([])

                # Current Z-score indicator
                ax_zscore.text(
                    window_size + 1,
                    current_z,
                    f"{current_z:.2f}",
                    ha="left",
                    va="center",
                    fontsize=10,
                    color=z_color,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=z_color,
                        edgecolor="none",
                        alpha=0.2,
                    ),
                )

                # Zone labels
                ax_zscore.text(
                    -1,
                    2.5,
                    "OVERBOUGHT",
                    fontsize=8,
                    color=red_color,
                    alpha=0.7,
                    va="center",
                )
                ax_zscore.text(
                    -1,
                    -2.5,
                    "OVERSOLD",
                    fontsize=8,
                    color=green_color,
                    alpha=0.7,
                    va="center",
                )

            # === EQUITY CURVE PANEL ===
            # First compute cumulative equity UP TO the start of the window
            cumulative_eq = initial_capital
            for idx in range(0, start_idx):
                if idx in trade_exits:
                    cumulative_eq += trade_exits[idx].get("pnl", 0)

            # Now compute equity for the current window
            equity_slice = [cumulative_eq]
            running_eq = cumulative_eq
            for idx in range(start_idx, end_idx):
                if idx in trade_exits:
                    running_eq += trade_exits[idx].get("pnl", 0)
                equity_slice.append(running_eq)

            x_eq = range(len(equity_slice))

            # Draw equity curve with gradient fill
            ax_equity.fill_between(
                x_eq,
                initial_capital,
                equity_slice,
                where=[e >= initial_capital for e in equity_slice],
                color=green_color,
                alpha=0.3,
            )
            ax_equity.fill_between(
                x_eq,
                initial_capital,
                equity_slice,
                where=[e < initial_capital for e in equity_slice],
                color=red_color,
                alpha=0.3,
            )
            ax_equity.plot(x_eq, equity_slice, color=purple_color, linewidth=2.5)

            # Baseline
            ax_equity.axhline(
                y=initial_capital,
                color=text_dim,
                linestyle="--",
                alpha=0.5,
                linewidth=1,
            )
            ax_equity.text(
                -1,
                initial_capital,
                f"${initial_capital/1000:.0f}K",
                fontsize=8,
                color=text_dim,
                va="center",
                ha="right",
            )

            ax_equity.set_ylabel(
                "Equity ($)", fontsize=10, color=text_color, fontweight="bold"
            )
            ax_equity.set_xlim(-2, window_size + 5)

            # Add date labels on X-axis
            if hasattr(window_dates[0], "strftime"):
                # Show ~5 evenly spaced date labels
                n_labels = 5
                label_indices = np.linspace(
                    0, len(window_dates) - 1, n_labels, dtype=int
                )
                date_labels = [
                    window_dates[idx].strftime("%b %y") for idx in label_indices
                ]
                ax_equity.set_xticks(label_indices)
                ax_equity.set_xticklabels(date_labels, fontsize=8, color=text_dim)
            else:
                ax_equity.set_xlabel("Bars", fontsize=9, color=text_dim)

            # Current equity indicator
            eq_color = green_color if running_eq >= initial_capital else red_color
            ax_equity.text(
                window_size + 1,
                running_eq,
                f"${running_eq:,.0f}",
                ha="left",
                va="center",
                fontsize=10,
                color=eq_color,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=eq_color,
                    edgecolor="none",
                    alpha=0.2,
                ),
            )

            # === STATS PANEL ===
            total_pnl = running_eq - initial_capital
            pnl_color = green_color if total_pnl >= 0 else red_color
            pnl_text = f"+${total_pnl:,.0f}" if total_pnl >= 0 else f"${total_pnl:,.0f}"
            return_pct = (total_pnl / initial_capital) * 100

            # Count trades up to current point
            trades_so_far = sum(1 for idx in trade_exits if idx < end_idx)
            wins_so_far = sum(
                1
                for idx in trade_exits
                if idx < end_idx and trade_exits[idx].get("pnl", 0) > 0
            )
            win_rate = (wins_so_far / trades_so_far * 100) if trades_so_far > 0 else 0

            # Create stats display
            stats_items = [
                (f"P&L: {pnl_text}", pnl_color),
                (f"Return: {return_pct:+.1f}%", pnl_color),
                (f"Trades: {trades_so_far}", text_color),
                (
                    f"Win Rate: {win_rate:.0f}%",
                    green_color if win_rate >= 50 else red_color,
                ),
                (f"Z-Score: {current_z:.2f}", z_color),
            ]

            x_positions = np.linspace(0.1, 0.9, len(stats_items))
            for (text, color), x_pos in zip(stats_items, x_positions):
                ax_stats.text(
                    x_pos,
                    0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=color,
                    fontweight="bold",
                    transform=ax_stats.transAxes,
                )

            # Adjust spacing
            plt.subplots_adjust(
                left=0.07, right=0.93, top=0.94, bottom=0.04, hspace=0.12
            )

            buf = io.BytesIO()
            plt.savefig(
                buf, format="png", dpi=100, facecolor=bg_color, edgecolor="none"
            )
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()
            plt.close(fig)

        if not frames:
            return None

        safe_name = strategy_name.replace(" ", "_").replace("/", "_")
        gif_path = output_dir / f"animation_{safe_name}_{report_timestamp}.gif"

        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
        )

        return str(gif_path)

    except Exception as e:
        logger.warning(f"Failed to generate GIF: {e}")
        import traceback

        traceback.print_exc()
        plt.close("all")
        return None


def plot_all_strategies_comparison(
    strategy_results: pd.DataFrame,
    output_dir: Path,
    report_timestamp: str,
) -> Optional[str]:
    """
    Plot comparison bar chart of all strategies.
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Strategy Comparison - Out-of-Sample Test Results",
            fontsize=14,
            fontweight="bold",
        )

        type_colors = {
            "Rule-Based": "#3498db",
            "ML-Based": "#e74c3c",
            "RL-Based": "#9b59b6",
            "Benchmark": "#95a5a6",
        }

        colors = [type_colors.get(t, "#333333") for t in strategy_results["type"]]
        strategies = strategy_results["strategy"].tolist()

        ax1 = axes[0, 0]
        ax1.barh(strategies, strategy_results["total_return_pct"], color=colors)
        ax1.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax1.set_xlabel("Return (%)")
        ax1.set_title("Total Return")

        ax2 = axes[0, 1]
        ax2.barh(strategies, strategy_results["sharpe_ratio"], color=colors)
        ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax2.set_xlabel("Sharpe Ratio")
        ax2.set_title("Sharpe Ratio")

        ax3 = axes[1, 0]
        ax3.barh(strategies, strategy_results["max_drawdown"], color=colors)
        ax3.set_xlabel("Max Drawdown (%)")
        ax3.set_title("Maximum Drawdown")

        ax4 = axes[1, 1]
        ax4.barh(strategies, strategy_results["win_rate"], color=colors)
        ax4.axvline(x=50, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
        ax4.set_xlabel("Win Rate (%)")
        ax4.set_title("Win Rate")

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=c, label=t)
            for t, c in type_colors.items()
        ]
        fig.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 0.99)
        )

        plt.tight_layout()

        filename = output_dir / f"strategy_comparison_{report_timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return str(filename)

    except Exception as e:
        logger.warning(f"Failed to plot comparison: {e}")
        plt.close("all")
        return None


def generate_strategy_plots(
    spread_df: pd.DataFrame,
    strategy_results: pd.DataFrame,
    output_dir: Path,
    report_timestamp: str,
    regime_models: Dict[str, object] = None,
    rl_agents: Dict[str, object] = None,
) -> List[str]:
    """
    Generate all strategy plots and GIF animations for winning strategies.
    Creates one GIF per winning strategy category: Rule-Based, ML-Based, RL-Based.
    Uses the FULL test period from 2021-01-01 onward.
    """
    from quantcore.backtesting.strategies import (
        backtest_changepoint_strategy,
        backtest_rl_spread_strategy,
    )

    generated_files = []

    if spread_df.empty:
        return generated_files

    # Use proper test period: 2021-01-01 onward (matching the hyperparameter tuning)
    test_data = spread_df[spread_df.index >= "2021-01-01"].copy()

    if len(test_data) < 100:
        logger.warning("Not enough test data for plotting")
        return generated_files

    logger.info(
        f"Generating plots for test period: {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')} ({len(test_data)} bars)"
    )

    # Strategy Comparison Chart
    if strategy_results is not None and not strategy_results.empty:
        comp_plot = plot_all_strategies_comparison(
            strategy_results, output_dir, report_timestamp
        )
        if comp_plot:
            generated_files.append(comp_plot)

    # ============================================================
    # RULE-BASED WINNER: Spread Mean Reversion
    # ============================================================
    logger.info("Generating plots for Rule-Based winner: Spread Mean Reversion...")
    result = run_backtest_with_params(
        test_data, 100000, 2.0, 0.0, 1000, 0.05, None, return_trades_list=True
    )
    trades = result.get("trades_list", [])
    equity = result.get("equity_curve", [])

    plot_path = plot_strategy_signals(
        test_data,
        "Rule_Spread_Mean_Reversion",
        trades,
        equity,
        output_dir,
        report_timestamp,
    )
    if plot_path:
        generated_files.append(plot_path)

    logger.info("Generating animation for Rule-Based: Spread Mean Reversion...")
    gif_path = generate_candlestick_gif(
        test_data,
        trades,
        output_dir,
        report_timestamp,
        strategy_name="Rule_Spread_Mean_Reversion",
        frame_duration=300,
        initial_capital=100000,
        use_full_data=True,
    )
    if gif_path:
        generated_files.append(gif_path)

    # ============================================================
    # ML-BASED WINNER: Changepoint Strategy
    # ============================================================
    if regime_models and "changepoint" in regime_models:
        logger.info("Generating plots for ML-Based winner: Changepoint Strategy...")
        try:
            cp_result = backtest_changepoint_strategy(
                test_data, regime_models["changepoint"], 100000
            )
            # Run again to get trades list (need to modify or re-run with tracking)
            # For now, create trades from the strategy manually
            capital = 100000
            position = 0
            entry_price = 0
            entry_idx = 0
            trades_ml = []
            equity_ml = [capital]

            POSITION_SIZE = 1000
            SPREAD_COST = 0.05

            for i in range(60, len(test_data)):
                row = test_data.iloc[i]
                zscore = row.get("spread_zscore", 0)
                spread = row.get("spread", 0)

                # Detect high change probability
                recent_data = test_data.iloc[max(0, i - 60) : i + 1]
                try:
                    cp_result_local = regime_models["changepoint"].detect(recent_data)
                    high_change_prob = cp_result_local.change_probability > 0.5
                except:
                    high_change_prob = False

                effective_size = (
                    POSITION_SIZE // 2 if high_change_prob else POSITION_SIZE
                )

                if position == 0:
                    if zscore < -2:
                        position = 1
                        entry_price = spread
                        entry_idx = i
                        entry_size = effective_size
                        capital -= SPREAD_COST * entry_size
                    elif zscore > 2:
                        position = -1
                        entry_price = spread
                        entry_idx = i
                        entry_size = effective_size
                        capital -= SPREAD_COST * entry_size
                elif position == 1:
                    if zscore > 0 or high_change_prob:
                        pnl = (
                            spread - entry_price
                        ) * entry_size - SPREAD_COST * entry_size
                        capital += pnl
                        trades_ml.append(
                            {
                                "pnl": pnl,
                                "entry_idx": entry_idx,
                                "exit_idx": i,
                                "direction": "LONG",
                            }
                        )
                        position = 0
                elif position == -1:
                    if zscore < 0 or high_change_prob:
                        pnl = (
                            entry_price - spread
                        ) * entry_size - SPREAD_COST * entry_size
                        capital += pnl
                        trades_ml.append(
                            {
                                "pnl": pnl,
                                "entry_idx": entry_idx,
                                "exit_idx": i,
                                "direction": "SHORT",
                            }
                        )
                        position = 0

                mtm = capital + (
                    position * (spread - entry_price) * entry_size
                    if position != 0
                    else 0
                )
                equity_ml.append(mtm)

            plot_path = plot_strategy_signals(
                test_data,
                "ML_Changepoint",
                trades_ml,
                equity_ml,
                output_dir,
                report_timestamp,
            )
            if plot_path:
                generated_files.append(plot_path)

            logger.info("Generating animation for ML-Based: Changepoint...")
            gif_path = generate_candlestick_gif(
                test_data,
                trades_ml,
                output_dir,
                report_timestamp,
                strategy_name="ML_Changepoint",
                frame_duration=300,
                initial_capital=100000,
                use_full_data=True,
            )
            if gif_path:
                generated_files.append(gif_path)

        except Exception as e:
            logger.warning(f"Failed to generate ML Changepoint plots: {e}")

    # ============================================================
    # RL-BASED WINNER: RL Spread Agent
    # ============================================================
    if rl_agents and "spread" in rl_agents:
        logger.info("Generating plots for RL-Based winner: RL Spread Agent...")
        try:
            from quantcore.rl.base import State as RLState

            # Run RL spread strategy with trade tracking
            capital = 100000
            position = 0
            position_size_dir = 0.0  # Track position size direction for state
            entry_price = 0
            entry_idx = 0
            unrealized_pnl = 0.0
            bars_held = 0
            trades_rl = []
            equity_rl = [capital]

            POSITION_SIZE = 1000
            SPREAD_COST = 0.05
            spread_agent = rl_agents["spread"]

            # Use the agent's thresholds
            zscore_entry = getattr(spread_agent, "zscore_entry_threshold", 1.5)
            zscore_exit = getattr(spread_agent, "zscore_exit_threshold", 0.5)

            for i in range(60, len(test_data)):
                row = test_data.iloc[i]
                zscore = row.get("spread_zscore", 0)
                spread = row.get("spread", 0)

                # Calculate momentum
                if i >= 5:
                    mom_5 = (
                        test_data.iloc[i]["spread"] - test_data.iloc[i - 5]["spread"]
                    )
                else:
                    mom_5 = 0
                if i >= 20:
                    mom_20 = (
                        test_data.iloc[i]["spread"] - test_data.iloc[i - 20]["spread"]
                    )
                else:
                    mom_20 = 0

                # Build proper 12-feature state for the spread agent
                state_features = np.array(
                    [
                        zscore / 3,  # Normalized zscore
                        mom_5 * 100,  # 5-bar momentum
                        mom_20 * 100,  # 20-bar momentum
                        0.5,  # Percentile rank placeholder
                        position,  # Position direction
                        position_size_dir,  # Position size
                        unrealized_pnl / 1000,  # Normalized unrealized PnL
                        min(bars_held / 50, 1.0),  # Bars held normalized
                        0.3,  # Volatility regime placeholder
                        0.9,  # Correlation placeholder
                        0.0,  # USD regime placeholder
                        0.0,  # Curve shape placeholder
                    ],
                    dtype=np.float32,
                )

                state = RLState(features=state_features)

                # Set agent to eval mode and use trained Q-network
                spread_agent.training = False
                action_obj = spread_agent.select_action(state, explore=False)
                action = action_obj.value

                # Actions: 0=Close, 1=Small Long, 2=Full Long, 3=Small Short, 4=Full Short
                if action in [1, 2] and position == 0:  # Long signal
                    position = 1
                    position_size_dir = 1.0 if action == 2 else 0.25
                    entry_price = spread
                    entry_idx = i
                    bars_held = 0
                    capital -= SPREAD_COST * POSITION_SIZE
                elif action in [3, 4] and position == 0:  # Short signal
                    position = -1
                    position_size_dir = 1.0 if action == 4 else 0.25
                    entry_price = spread
                    entry_idx = i
                    bars_held = 0
                    capital -= SPREAD_COST * POSITION_SIZE
                elif action == 0 and position != 0:  # Close
                    if position == 1:
                        pnl = (
                            spread - entry_price
                        ) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
                    else:
                        pnl = (
                            entry_price - spread
                        ) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
                    capital += pnl
                    trades_rl.append(
                        {
                            "pnl": pnl,
                            "entry_idx": entry_idx,
                            "exit_idx": i,
                            "direction": "LONG" if position == 1 else "SHORT",
                        }
                    )
                    position = 0
                    position_size_dir = 0.0
                    unrealized_pnl = 0.0
                    bars_held = 0

                # Update unrealized PnL and bars held
                if position != 0:
                    unrealized_pnl = position * (spread - entry_price) * POSITION_SIZE
                    bars_held += 1

                mtm = capital + unrealized_pnl
                equity_rl.append(mtm)

            logger.info(f"RL Spread Agent generated {len(trades_rl)} trades")

            plot_path = plot_strategy_signals(
                test_data,
                "RL_Spread_Agent",
                trades_rl,
                equity_rl,
                output_dir,
                report_timestamp,
            )
            if plot_path:
                generated_files.append(plot_path)

            logger.info("Generating animation for RL-Based: RL Spread Agent...")
            gif_path = generate_candlestick_gif(
                test_data,
                trades_rl,
                output_dir,
                report_timestamp,
                strategy_name="RL_Spread_Agent",
                frame_duration=300,
                initial_capital=100000,
                use_full_data=True,
            )
            if gif_path:
                generated_files.append(gif_path)

        except Exception as e:
            logger.warning(f"Failed to generate RL Spread Agent plots: {e}")
            import traceback

            traceback.print_exc()

    # ============================================================
    # RL-BASED: RL-Enhanced (Execution + Sizing Agents)
    # ============================================================
    if rl_agents and ("execution" in rl_agents or "sizing" in rl_agents):
        logger.info("Generating plots for RL-Enhanced (Execution + Sizing)...")
        try:
            capital = 100000
            position = 0
            entry_price = 0
            entry_idx = 0
            entry_position_size = 0
            trades_enhanced = []
            equity_enhanced = [capital]

            BASE_POSITION_SIZE = 1000
            SPREAD_COST = 0.05

            sizing_agent = rl_agents.get("sizing")
            exec_agent = rl_agents.get("execution")

            for i in range(60, len(test_data)):
                row = test_data.iloc[i]
                zscore = row.get("spread_zscore", 0)
                spread = row.get("spread", 0)

                # Base signal from Z-score (mean reversion)
                if zscore < -2:
                    base_signal = 1  # Long
                elif zscore > 2:
                    base_signal = -1  # Short
                elif abs(zscore) < 0.5:
                    base_signal = 0  # Close
                else:
                    base_signal = position  # Hold current position

                # Get position sizing from RL agent (if available)
                size_scale = 1.0
                if sizing_agent and base_signal != 0 and position == 0:
                    try:
                        # Build state for sizing agent (10 features expected)
                        state_arr = np.array(
                            [
                                abs(base_signal),  # Signal confidence
                                float(base_signal),  # Signal direction
                                0.02,  # Volatility
                                0.0,  # Current drawdown
                                0.0,  # Risk budget used
                                0.0,  # Recent Sharpe
                                0.0,  # Current position
                                0.0,  # Time since last trade
                                0.0,  # Regime indicator
                                0.5,  # Rolling win rate
                            ],
                            dtype=np.float32,
                        )
                        # Truncate to agent's state dim
                        state_arr = state_arr[: sizing_agent.state_dim]
                        from quantcore.rl.base import State as RLState

                        state = RLState(features=state_arr)
                        action_obj = sizing_agent.select_action(state, explore=False)
                        # Action is typically a continuous value 0-1
                        if hasattr(action_obj, "value"):
                            action_val = action_obj.value
                            if isinstance(action_val, np.ndarray):
                                action_val = float(action_val[0])
                            size_scale = (
                                0.5 + float(action_val) * 0.5
                            )  # Scale 0.5x to 1.5x
                    except Exception as sizing_err:
                        size_scale = 1.0

                current_position_size = int(BASE_POSITION_SIZE * size_scale)

                if base_signal == 1 and position == 0:
                    position = 1
                    entry_price = spread
                    entry_idx = i
                    entry_position_size = current_position_size
                    capital -= SPREAD_COST * entry_position_size
                elif base_signal == -1 and position == 0:
                    position = -1
                    entry_price = spread
                    entry_idx = i
                    entry_position_size = current_position_size
                    capital -= SPREAD_COST * entry_position_size
                elif base_signal == 0 and position != 0:
                    if position == 1:
                        pnl = (
                            spread - entry_price
                        ) * entry_position_size - SPREAD_COST * entry_position_size
                    else:
                        pnl = (
                            entry_price - spread
                        ) * entry_position_size - SPREAD_COST * entry_position_size
                    capital += pnl
                    trades_enhanced.append(
                        {
                            "pnl": pnl,
                            "entry_idx": entry_idx,
                            "exit_idx": i,
                            "direction": "LONG" if position == 1 else "SHORT",
                        }
                    )
                    position = 0
                    entry_position_size = 0

                mtm = capital + (
                    position * (spread - entry_price) * entry_position_size
                    if position != 0
                    else 0
                )
                equity_enhanced.append(mtm)

            logger.info(f"RL-Enhanced generated {len(trades_enhanced)} trades")

            plot_path = plot_strategy_signals(
                test_data,
                "RL_Enhanced",
                trades_enhanced,
                equity_enhanced,
                output_dir,
                report_timestamp,
            )
            if plot_path:
                generated_files.append(plot_path)

            logger.info("Generating animation for RL-Enhanced...")
            gif_path = generate_candlestick_gif(
                test_data,
                trades_enhanced,
                output_dir,
                report_timestamp,
                strategy_name="RL_Enhanced",
                frame_duration=300,
                initial_capital=100000,
                use_full_data=True,
            )
            if gif_path:
                generated_files.append(gif_path)

        except Exception as e:
            logger.warning(f"Failed to generate RL-Enhanced plots: {e}")
            import traceback

            traceback.print_exc()

    return generated_files
