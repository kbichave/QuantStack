"""
Generalized Trade Animation GIF Generator.

Creates TradingView-style animated GIFs showing:
- Candlestick price action
- Trade entry/exit markers with direction (LONG/SHORT)
- P&L labels on exits
- Running cumulative PnL panel (updates during position, not just post-trade)

Usage:
    from quantcore.visualization.trade_animation import generate_trade_animation_gif

    gif_path = generate_trade_animation_gif(
        data=ohlcv_df,
        trades=trades_list,
        output_dir=Path("reports/"),
        strategy_name="Options_Momentum",
    )
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from loguru import logger

# Try to import PIL for GIF generation
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available, GIF generation disabled")


# TradingView dark theme colors
THEME = {
    "bg_color": "#131722",
    "panel_color": "#1e222d",
    "grid_color": "#2a2e39",
    "grid_color_strong": "#434651",
    "text_color": "#d1d4dc",
    "text_dim": "#787b86",
    "green_color": "#26a69a",
    "red_color": "#ef5350",
    "blue_color": "#2962ff",
    "purple_color": "#ab47bc",
    "orange_color": "#ff9800",
    "yellow_color": "#ffeb3b",
}


def _draw_candlestick(
    ax,
    idx: int,
    open_price: float,
    high: float,
    low: float,
    close: float,
    width: float = 0.6,
    up_color: str = None,
    down_color: str = None,
):
    """Draw a single candlestick on the given axis."""
    up_color = up_color or THEME["green_color"]
    down_color = down_color or THEME["red_color"]

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


def generate_trade_animation_gif(
    data: pd.DataFrame,
    trades: List[Dict],
    output_dir: Path,
    report_timestamp: str = None,
    strategy_name: str = "Strategy",
    frame_duration: int = 300,
    initial_capital: float = 100000,
    window_size: int = 60,
    step: int = 2,
    price_column: str = "close",
    price_label: str = "Price ($)",
    show_zscore: bool = False,
    zscore_column: str = "zscore",
) -> Optional[str]:
    """
    Generate TradingView-style animated GIF showing backtest with candlesticks,
    trade markers, P&L labels, and running equity curve.

    Args:
        data: DataFrame with OHLCV data (open, high, low, close, volume)
               Can also include a spread or zscore column
        trades: List of trade dicts with keys:
                - entry_idx: Bar index of entry
                - exit_idx: Bar index of exit
                - direction: "LONG" or "SHORT"
                - pnl: Profit/loss of the trade
        output_dir: Directory to save the GIF
        report_timestamp: Timestamp for filename (generated if None)
        strategy_name: Name shown in title
        frame_duration: Milliseconds per frame (higher = slower)
        initial_capital: Starting capital for equity curve
        window_size: Number of bars visible in each frame
        step: Number of bars to advance per frame
        price_column: Column to use for price display (default "close")
        price_label: Y-axis label for price panel
        show_zscore: Whether to show Z-score panel
        zscore_column: Column name for Z-score data

    Returns:
        Path to generated GIF, or None if generation failed
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, skipping GIF generation")
        return None

    if data.empty or len(data) < window_size:
        logger.warning(f"Insufficient data for GIF: {len(data)} bars < {window_size}")
        return None

    try:
        # Generate timestamp if not provided
        if report_timestamp is None:
            from datetime import datetime

            report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = []
        plot_data = data.copy()

        # Adjust window and step based on data size
        if len(plot_data) > 500:
            window_size = max(window_size, 80)
            step = max(step, 5)

        # Check if we have OHLC data
        has_ohlc = all(
            col in plot_data.columns for col in ["open", "high", "low", "close"]
        )

        # Extract price data
        if price_column in plot_data.columns:
            prices = plot_data[price_column].values
        elif "close" in plot_data.columns:
            prices = plot_data["close"].values
        else:
            logger.error("No price data found in DataFrame")
            return None

        # Generate OHLC from prices if not available
        if not has_ohlc:
            opens = np.roll(prices, 1)
            opens[0] = prices[0]
            volatility = np.abs(np.diff(prices, prepend=prices[0])) * 0.5 + 0.01
            highs = np.maximum(prices, opens) + volatility
            lows = np.minimum(prices, opens) - volatility
            closes = prices
        else:
            opens = plot_data["open"].values
            highs = plot_data["high"].values
            lows = plot_data["low"].values
            closes = plot_data["close"].values

        # Get Z-score data if needed
        if show_zscore and zscore_column in plot_data.columns:
            zscore = plot_data[zscore_column].values
        else:
            zscore = np.zeros(len(plot_data))
            show_zscore = False

        dates = plot_data.index

        # Map trades to indices
        trade_entries = {}
        trade_exits = {}
        for trade in trades:
            if "entry_idx" in trade:
                entry_idx = trade["entry_idx"]
                if 0 <= entry_idx < len(plot_data):
                    trade_entries[entry_idx] = trade
            if "exit_idx" in trade:
                exit_idx = trade["exit_idx"]
                if 0 <= exit_idx < len(plot_data):
                    trade_exits[exit_idx] = trade

        # Generate frames
        for start_idx in range(0, len(plot_data) - window_size, step):
            end_idx = start_idx + window_size

            # Create figure
            if show_zscore:
                fig = plt.figure(figsize=(14, 9), facecolor=THEME["bg_color"])
                gs = fig.add_gridspec(
                    4, 1, height_ratios=[3.5, 1, 1.2, 0.6], hspace=0.05
                )
                ax_price = fig.add_subplot(gs[0])
                ax_zscore = fig.add_subplot(gs[1])
                ax_equity = fig.add_subplot(gs[2])
                ax_stats = fig.add_subplot(gs[3])
                main_axes = [ax_price, ax_zscore, ax_equity]
            else:
                fig = plt.figure(figsize=(14, 8), facecolor=THEME["bg_color"])
                gs = fig.add_gridspec(3, 1, height_ratios=[4, 1.5, 0.6], hspace=0.05)
                ax_price = fig.add_subplot(gs[0])
                ax_zscore = None
                ax_equity = fig.add_subplot(gs[1])
                ax_stats = fig.add_subplot(gs[2])
                main_axes = [ax_price, ax_equity]

            # Style axes
            for ax in main_axes:
                ax.set_facecolor(THEME["bg_color"])
                ax.tick_params(colors=THEME["text_color"], labelsize=9)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color(THEME["grid_color_strong"])
                ax.spines["left"].set_color(THEME["grid_color_strong"])
                ax.grid(
                    True,
                    which="major",
                    color=THEME["grid_color_strong"],
                    alpha=0.6,
                    linestyle="-",
                    linewidth=0.8,
                )
                ax.grid(
                    True,
                    which="minor",
                    color=THEME["grid_color"],
                    alpha=0.3,
                    linestyle="-",
                    linewidth=0.5,
                )
                ax.minorticks_on()

            ax_stats.set_facecolor(THEME["panel_color"])
            ax_stats.axis("off")

            # Window data
            window_opens = opens[start_idx:end_idx]
            window_highs = highs[start_idx:end_idx]
            window_lows = lows[start_idx:end_idx]
            window_closes = closes[start_idx:end_idx]
            window_prices = prices[start_idx:end_idx]
            window_zscore = zscore[start_idx:end_idx]
            window_dates = dates[start_idx:end_idx]

            # === PRICE PANEL ===
            for i in range(len(window_prices)):
                _draw_candlestick(
                    ax_price,
                    i,
                    window_opens[i],
                    window_highs[i],
                    window_lows[i],
                    window_closes[i],
                    width=0.7,
                )

            # Moving average
            if len(window_prices) >= 20:
                ma20 = pd.Series(window_prices).rolling(20).mean().values
                ax_price.plot(
                    range(len(ma20)),
                    ma20,
                    color=THEME["orange_color"],
                    linewidth=1.5,
                    alpha=0.7,
                    linestyle="--",
                )

            # Draw entry/exit markers
            y_range = max(window_highs) - min(window_lows)
            for i in range(len(window_prices)):
                global_idx = start_idx + i

                # Entry marker
                if global_idx in trade_entries:
                    trade = trade_entries[global_idx]
                    direction = trade.get("direction", "LONG")

                    if "LONG" in str(direction).upper():
                        ax_price.scatter(
                            i,
                            window_lows[i] - y_range * 0.08,
                            marker="^",
                            s=200,
                            color=THEME["green_color"],
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
                            color=THEME["green_color"],
                            fontweight="bold",
                        )
                    else:
                        ax_price.scatter(
                            i,
                            window_highs[i] + y_range * 0.08,
                            marker="v",
                            s=200,
                            color=THEME["red_color"],
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
                            color=THEME["red_color"],
                            fontweight="bold",
                        )

                # Exit marker with P&L
                if global_idx in trade_exits:
                    trade = trade_exits[global_idx]
                    pnl = trade.get("pnl", 0)
                    pnl_color = THEME["green_color"] if pnl > 0 else THEME["red_color"]
                    pnl_text = f"+${pnl:,.0f}" if pnl > 0 else f"${pnl:,.0f}"

                    ax_price.scatter(
                        i,
                        window_prices[i],
                        marker="X",
                        s=150,
                        color=pnl_color,
                        edgecolors="white",
                        linewidths=1.5,
                        zorder=10,
                    )
                    ax_price.annotate(
                        pnl_text,
                        xy=(i, window_prices[i]),
                        xytext=(i + 2, window_prices[i]),
                        fontsize=10,
                        color=pnl_color,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor=THEME["bg_color"],
                            edgecolor=pnl_color,
                            alpha=0.9,
                        ),
                        arrowprops=dict(arrowstyle="-", color=pnl_color, alpha=0.5),
                    )

            # Price axis formatting
            ax_price.set_ylabel(
                price_label, fontsize=10, color=THEME["text_color"], fontweight="bold"
            )
            ax_price.set_xlim(-2, window_size + 5)
            ax_price.set_ylim(
                min(window_lows) - y_range * 0.2, max(window_highs) + y_range * 0.2
            )
            ax_price.set_xticklabels([])

            # Current price indicator
            current_price = window_closes[-1]
            price_color = (
                THEME["green_color"]
                if current_price >= window_opens[-1]
                else THEME["red_color"]
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

            # Title
            if hasattr(window_dates[-1], "strftime"):
                date_title = window_dates[-1].strftime("%b %d, %Y")
            else:
                date_title = f"Bar {end_idx}"
            ax_price.set_title(
                f"{strategy_name}  |  {date_title}",
                fontsize=13,
                color=THEME["text_color"],
                fontweight="bold",
                loc="left",
                pad=12,
            )

            # === Z-SCORE PANEL (optional) ===
            current_z = window_zscore[-1] if len(window_zscore) > 0 else 0
            z_color = (
                THEME["green_color"]
                if current_z < -2
                else (THEME["red_color"] if current_z > 2 else THEME["text_color"])
            )

            if show_zscore and ax_zscore is not None:
                ax_zscore.fill_between(
                    range(len(window_zscore)),
                    0,
                    window_zscore,
                    where=[z >= 0 for z in window_zscore],
                    color=THEME["red_color"],
                    alpha=0.4,
                )
                ax_zscore.fill_between(
                    range(len(window_zscore)),
                    0,
                    window_zscore,
                    where=[z < 0 for z in window_zscore],
                    color=THEME["green_color"],
                    alpha=0.4,
                )
                ax_zscore.plot(
                    range(len(window_zscore)),
                    window_zscore,
                    color=THEME["blue_color"],
                    linewidth=2,
                )

                ax_zscore.axhline(
                    y=2,
                    color=THEME["red_color"],
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                )
                ax_zscore.axhline(
                    y=-2,
                    color=THEME["green_color"],
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                )
                ax_zscore.axhline(
                    y=0, color=THEME["text_dim"], linestyle="-", alpha=0.5, linewidth=1
                )

                ax_zscore.set_ylabel(
                    "Z-Score", fontsize=10, color=THEME["text_color"], fontweight="bold"
                )
                ax_zscore.set_xlim(-2, window_size + 5)
                ax_zscore.set_ylim(-4.5, 4.5)
                ax_zscore.set_xticklabels([])

            # === EQUITY CURVE PANEL ===
            # Compute cumulative equity UP TO the start of window
            cumulative_eq = initial_capital
            for idx in range(0, start_idx):
                if idx in trade_exits:
                    cumulative_eq += trade_exits[idx].get("pnl", 0)

            # Compute equity through the window (shows running P&L during position)
            equity_slice = [cumulative_eq]
            running_eq = cumulative_eq

            # Track current position for unrealized P&L
            current_position = None
            position_entry_price = 0
            position_direction = None

            for idx in range(start_idx, end_idx):
                # Check for position entry
                if idx in trade_entries:
                    trade = trade_entries[idx]
                    current_position = trade
                    position_entry_price = prices[idx]
                    position_direction = trade.get("direction", "LONG")

                # Check for position exit
                if idx in trade_exits:
                    running_eq += trade_exits[idx].get("pnl", 0)
                    current_position = None

                # Calculate unrealized P&L if in position
                unrealized = 0
                if current_position is not None:
                    current_price = prices[idx]
                    if "LONG" in str(position_direction).upper():
                        unrealized = (current_price - position_entry_price) * 1000
                    else:
                        unrealized = (position_entry_price - current_price) * 1000

                equity_slice.append(running_eq + unrealized)

            x_eq = range(len(equity_slice))

            # Draw equity with gradient fill
            ax_equity.fill_between(
                x_eq,
                initial_capital,
                equity_slice,
                where=[e >= initial_capital for e in equity_slice],
                color=THEME["green_color"],
                alpha=0.3,
            )
            ax_equity.fill_between(
                x_eq,
                initial_capital,
                equity_slice,
                where=[e < initial_capital for e in equity_slice],
                color=THEME["red_color"],
                alpha=0.3,
            )
            ax_equity.plot(
                x_eq, equity_slice, color=THEME["purple_color"], linewidth=2.5
            )

            # Baseline
            ax_equity.axhline(
                y=initial_capital,
                color=THEME["text_dim"],
                linestyle="--",
                alpha=0.5,
                linewidth=1,
            )
            ax_equity.text(
                -1,
                initial_capital,
                f"${initial_capital/1000:.0f}K",
                fontsize=8,
                color=THEME["text_dim"],
                va="center",
                ha="right",
            )

            ax_equity.set_ylabel(
                "Equity ($)", fontsize=10, color=THEME["text_color"], fontweight="bold"
            )
            ax_equity.set_xlim(-2, window_size + 5)

            # Date labels on X-axis
            if hasattr(window_dates[0], "strftime"):
                n_labels = 5
                label_indices = np.linspace(
                    0, len(window_dates) - 1, n_labels, dtype=int
                )
                date_labels = [
                    window_dates[idx].strftime("%b %y") for idx in label_indices
                ]
                ax_equity.set_xticks(label_indices)
                ax_equity.set_xticklabels(
                    date_labels, fontsize=8, color=THEME["text_dim"]
                )

            # Current equity indicator
            final_equity = equity_slice[-1]
            eq_color = (
                THEME["green_color"]
                if final_equity >= initial_capital
                else THEME["red_color"]
            )
            ax_equity.text(
                window_size + 1,
                final_equity,
                f"${final_equity:,.0f}",
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
            total_pnl = final_equity - initial_capital
            pnl_color = THEME["green_color"] if total_pnl >= 0 else THEME["red_color"]
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

            stats_items = [
                (f"P&L: {pnl_text}", pnl_color),
                (f"Return: {return_pct:+.1f}%", pnl_color),
                (f"Trades: {trades_so_far}", THEME["text_color"]),
                (
                    f"Win Rate: {win_rate:.0f}%",
                    THEME["green_color"] if win_rate >= 50 else THEME["red_color"],
                ),
            ]

            if show_zscore:
                stats_items.append((f"Z-Score: {current_z:.2f}", z_color))

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

            plt.subplots_adjust(
                left=0.07, right=0.93, top=0.94, bottom=0.04, hspace=0.12
            )

            # Save frame
            buf = io.BytesIO()
            plt.savefig(
                buf,
                format="png",
                dpi=100,
                facecolor=THEME["bg_color"],
                edgecolor="none",
            )
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()
            plt.close(fig)

        if not frames:
            logger.warning("No frames generated for GIF")
            return None

        # Save GIF
        safe_name = strategy_name.replace(" ", "_").replace("/", "_")
        gif_path = output_dir / f"animation_{safe_name}_{report_timestamp}.gif"

        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
        )

        logger.info(f"Generated trade animation GIF: {gif_path}")
        return str(gif_path)

    except Exception as e:
        logger.error(f"Failed to generate trade animation GIF: {e}")
        import traceback

        traceback.print_exc()
        plt.close("all")
        return None


def generate_options_trade_gif(
    data: pd.DataFrame,
    trades: List[Dict],
    output_dir: Path,
    symbol: str,
    strategy_name: str = "Options",
    report_timestamp: str = None,
    initial_capital: float = 100000,
    **kwargs,
) -> Optional[str]:
    """
    Convenience function to generate GIF for options trades.

    Args:
        data: OHLCV DataFrame for underlying equity
        trades: List of trade dicts
        output_dir: Output directory
        symbol: Underlying symbol (included in title)
        strategy_name: Strategy name
        report_timestamp: Timestamp for filename
        initial_capital: Starting capital
        **kwargs: Additional arguments passed to generate_trade_animation_gif

    Returns:
        Path to generated GIF
    """
    full_strategy_name = f"{strategy_name} - {symbol}"

    return generate_trade_animation_gif(
        data=data,
        trades=trades,
        output_dir=output_dir,
        report_timestamp=report_timestamp,
        strategy_name=full_strategy_name,
        initial_capital=initial_capital,
        price_label=f"{symbol} Price ($)",
        **kwargs,
    )
