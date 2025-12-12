"""
Trading strategy implementations for backtesting.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quantcore.backtesting.engine import run_backtest_with_params, calculate_metrics
from quantcore.utils.formatting import (
    print_info,
    print_success,
    print_error,
    print_section,
)


def backtest_spread_strategy(
    data: pd.DataFrame,
    initial_capital: float,
    entry_zscore: float,
    exit_zscore: float,
    position_size: int,
) -> Dict[str, float]:
    """Backtest spread mean reversion strategy."""
    return run_backtest_with_params(
        data, initial_capital, entry_zscore, exit_zscore, position_size, 0.05, None
    )


def backtest_sma_crossover(
    data: pd.DataFrame,
    initial_capital: float,
    fast: int = 20,
    slow: int = 50,
) -> Dict[str, float]:
    """Backtest SMA crossover strategy on WTI."""
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]

    POSITION_SIZE = 1000
    COST = 0.05

    # Calculate SMAs
    prices = data["wti"].copy()
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()

    for i in range(slow, len(data)):
        price = prices.iloc[i]
        fast_val = sma_fast.iloc[i]
        slow_val = sma_slow.iloc[i]
        prev_fast = sma_fast.iloc[i - 1]
        prev_slow = sma_slow.iloc[i - 1]

        # Golden cross (fast crosses above slow)
        if position == 0 and prev_fast <= prev_slow and fast_val > slow_val:
            position = 1
            entry_price = price
            capital -= COST * POSITION_SIZE
        # Death cross (fast crosses below slow)
        elif position == 0 and prev_fast >= prev_slow and fast_val < slow_val:
            position = -1
            entry_price = price
            capital -= COST * POSITION_SIZE
        # Exit long
        elif position == 1 and fast_val < slow_val:
            pnl = (price - entry_price) * POSITION_SIZE - COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0
        # Exit short
        elif position == -1 and fast_val > slow_val:
            pnl = (entry_price - price) * POSITION_SIZE - COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0

        mtm = capital + (
            position * (price - entry_price) * POSITION_SIZE if position != 0 else 0
        )
        equity_curve.append(mtm)

    return calculate_metrics(capital, initial_capital, trades, equity_curve)


def backtest_bollinger_bands(
    data: pd.DataFrame,
    initial_capital: float,
    period: int = 20,
    std_dev: float = 2.0,
) -> Dict[str, float]:
    """Backtest Bollinger Band mean reversion strategy."""
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]

    POSITION_SIZE = 1000
    COST = 0.05

    prices = data["wti"].copy()
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std

    for i in range(period, len(data)):
        price = prices.iloc[i]
        mid = sma.iloc[i]
        low = lower.iloc[i]

        # Price below lower band - buy
        if position == 0 and price < low:
            position = 1
            entry_price = price
            capital -= COST * POSITION_SIZE
        # Price above upper band - sell
        elif position == 0 and price > upper.iloc[i]:
            position = -1
            entry_price = price
            capital -= COST * POSITION_SIZE
        # Exit long at middle band
        elif position == 1 and price >= mid:
            pnl = (price - entry_price) * POSITION_SIZE - COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0
        # Exit short at middle band
        elif position == -1 and price <= mid:
            pnl = (entry_price - price) * POSITION_SIZE - COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0

        mtm = capital + (
            position * (price - entry_price) * POSITION_SIZE if position != 0 else 0
        )
        equity_curve.append(mtm)

    return calculate_metrics(capital, initial_capital, trades, equity_curve)


def backtest_rsi_strategy(
    data: pd.DataFrame,
    initial_capital: float,
    period: int = 14,
    oversold: int = 30,
    overbought: int = 70,
) -> Dict[str, float]:
    """Backtest RSI overbought/oversold strategy."""
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]

    POSITION_SIZE = 1000
    COST = 0.05

    prices = data["wti"].copy()
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    for i in range(period + 1, len(data)):
        price = prices.iloc[i]
        rsi_val = rsi.iloc[i]
        prev_rsi = rsi.iloc[i - 1]

        # RSI crosses above oversold - buy
        if position == 0 and prev_rsi < oversold and rsi_val >= oversold:
            position = 1
            entry_price = price
            capital -= COST * POSITION_SIZE
        # RSI crosses below overbought - sell
        elif position == 0 and prev_rsi > overbought and rsi_val <= overbought:
            position = -1
            entry_price = price
            capital -= COST * POSITION_SIZE
        # Exit long when RSI hits overbought
        elif position == 1 and rsi_val >= overbought:
            pnl = (price - entry_price) * POSITION_SIZE - COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0
        # Exit short when RSI hits oversold
        elif position == -1 and rsi_val <= oversold:
            pnl = (entry_price - price) * POSITION_SIZE - COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0

        mtm = capital + (
            position * (price - entry_price) * POSITION_SIZE if position != 0 else 0
        )
        equity_curve.append(mtm)

    return calculate_metrics(capital, initial_capital, trades, equity_curve)


def backtest_momentum_strategy(
    data: pd.DataFrame,
    initial_capital: float,
    lookback: int = 10,
) -> Dict[str, float]:
    """Backtest momentum (ROC) strategy."""
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]

    POSITION_SIZE = 1000
    COST = 0.05

    prices = data["wti"].copy()
    roc = prices.pct_change(lookback) * 100  # Rate of change

    for i in range(lookback + 1, len(data)):
        price = prices.iloc[i]
        mom = roc.iloc[i]
        prev_mom = roc.iloc[i - 1]

        # Momentum turns positive - buy
        if position == 0 and prev_mom <= 0 and mom > 0:
            position = 1
            entry_price = price
            capital -= COST * POSITION_SIZE
        # Momentum turns negative - sell
        elif position == 0 and prev_mom >= 0 and mom < 0:
            position = -1
            entry_price = price
            capital -= COST * POSITION_SIZE
        # Exit long when momentum turns negative
        elif position == 1 and mom < 0:
            pnl = (price - entry_price) * POSITION_SIZE - COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0
        # Exit short when momentum turns positive
        elif position == -1 and mom > 0:
            pnl = (entry_price - price) * POSITION_SIZE - COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0

        mtm = capital + (
            position * (price - entry_price) * POSITION_SIZE if position != 0 else 0
        )
        equity_curve.append(mtm)

    return calculate_metrics(capital, initial_capital, trades, equity_curve)


def backtest_macd_strategy(
    data: pd.DataFrame,
    initial_capital: float,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Dict[str, float]:
    """Backtest MACD signal line crossover strategy."""
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]

    POSITION_SIZE = 1000
    COST = 0.05

    prices = data["wti"].copy()
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    for i in range(slow + signal, len(data)):
        price = prices.iloc[i]
        macd = macd_line.iloc[i]
        sig = signal_line.iloc[i]
        prev_macd = macd_line.iloc[i - 1]
        prev_sig = signal_line.iloc[i - 1]

        # MACD crosses above signal - buy
        if position == 0 and prev_macd <= prev_sig and macd > sig:
            position = 1
            entry_price = price
            capital -= COST * POSITION_SIZE
        # MACD crosses below signal - sell
        elif position == 0 and prev_macd >= prev_sig and macd < sig:
            position = -1
            entry_price = price
            capital -= COST * POSITION_SIZE
        # Exit long
        elif position == 1 and macd < sig:
            pnl = (price - entry_price) * POSITION_SIZE - COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0
        # Exit short
        elif position == -1 and macd > sig:
            pnl = (entry_price - price) * POSITION_SIZE - COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0

        mtm = capital + (
            position * (price - entry_price) * POSITION_SIZE if position != 0 else 0
        )
        equity_curve.append(mtm)

    return calculate_metrics(capital, initial_capital, trades, equity_curve)


def backtest_hmm_strategy(
    data: pd.DataFrame,
    hmm_model: object,
    initial_capital: float,
) -> Dict[str, float]:
    """
    Backtest HMM regime-enhanced strategy.
    Only trade when regime is favorable.
    """
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]

    POSITION_SIZE = 1000
    SPREAD_COST = 0.05

    # Get base data for HMM predictions
    base_data = data[["wti", "brent"]].copy()
    base_data["close"] = base_data["wti"]
    base_data["volume"] = 1000000

    for i in range(60, len(data)):
        row = data.iloc[i]
        zscore = row["spread_zscore"]
        spread = row["spread"]

        # Get HMM regime for current window
        try:
            window = base_data.iloc[i - 60 : i]
            regime_result = hmm_model.predict(window)
            is_bullish = regime_result.state.name in ["LOW_VOL_BULL", "HIGH_VOL_BULL"]
            is_stable = regime_result.regime_stability > 0.6
        except:
            is_bullish = True
            is_stable = True

        # Only trade in stable regimes
        if position == 0 and is_stable:
            if zscore < -2 and is_bullish:
                position = 1
                entry_price = spread
                capital -= SPREAD_COST * POSITION_SIZE
            elif zscore > 2 and not is_bullish:
                position = -1
                entry_price = spread
                capital -= SPREAD_COST * POSITION_SIZE
        elif position == 1 and zscore > 0:
            pnl = (spread - entry_price) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0
        elif position == -1 and zscore < 0:
            pnl = (entry_price - spread) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0

        mtm = capital + (
            position * (spread - entry_price) * POSITION_SIZE if position != 0 else 0
        )
        equity_curve.append(mtm)

    return calculate_metrics(capital, initial_capital, trades, equity_curve)


def backtest_changepoint_strategy(
    data: pd.DataFrame,
    changepoint_model: object,
    initial_capital: float,
) -> Dict[str, float]:
    """
    Backtest changepoint-enhanced strategy.
    Reduce position when regime change is likely.

    FIX: Now tracks entry_size separately to ensure P&L uses the size
    at which the position was actually entered.
    """
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_size = 0  # Track size at entry to avoid P&L calculation errors
    trades = []
    equity_curve = [capital]

    POSITION_SIZE = 1000
    SPREAD_COST = 0.05

    # Prepare DataFrame with 'close' column for changepoint detection
    cp_data = data[["wti"]].copy()
    cp_data.columns = ["close"]

    for i in range(60, len(data)):
        row = data.iloc[i]
        zscore = row["spread_zscore"]
        spread = row["spread"]

        # Check changepoint probability
        try:
            window_df = cp_data.iloc[i - 60 : i]
            cp_result = changepoint_model.detect(window_df, feature="returns")
            high_change_prob = cp_result.regime_change_probability > 0.3
        except:
            high_change_prob = False

        # Determine effective size for NEW entries only
        effective_size = POSITION_SIZE // 2 if high_change_prob else POSITION_SIZE

        if position == 0:
            if zscore < -2:
                position = 1
                entry_price = spread
                entry_size = effective_size  # Store size at entry
                capital -= SPREAD_COST * entry_size
            elif zscore > 2:
                position = -1
                entry_price = spread
                entry_size = effective_size  # Store size at entry
                capital -= SPREAD_COST * entry_size
        elif position == 1:
            if zscore > 0 or high_change_prob:
                # Use entry_size for P&L, not current effective_size
                pnl = (spread - entry_price) * entry_size - SPREAD_COST * entry_size
                capital += pnl
                trades.append({"pnl": pnl})
                position = 0
                entry_size = 0
        elif position == -1:
            if zscore < 0 or high_change_prob:
                # Use entry_size for P&L, not current effective_size
                pnl = (entry_price - spread) * entry_size - SPREAD_COST * entry_size
                capital += pnl
                trades.append({"pnl": pnl})
                position = 0
                entry_size = 0

        # Mark to market uses entry_size
        mtm = capital + (
            position * (spread - entry_price) * entry_size if position != 0 else 0
        )
        equity_curve.append(mtm)

    return calculate_metrics(capital, initial_capital, trades, equity_curve)


def backtest_tft_strategy(
    data: pd.DataFrame,
    tft_model: object,
    initial_capital: float,
) -> Dict[str, float]:
    """
    Backtest TFT transformer-enhanced strategy.
    Use TFT regime predictions to filter trades.
    """
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]

    POSITION_SIZE = 1000
    SPREAD_COST = 0.05

    # Prepare data for TFT
    base_data = data[["wti", "brent"]].copy()
    base_data["close"] = base_data["wti"]
    base_data["volume"] = 1000000
    base_data["open"] = base_data["close"]
    base_data["high"] = base_data["close"] * 1.01
    base_data["low"] = base_data["close"] * 0.99

    for i in range(100, len(data)):
        row = data.iloc[i]
        zscore = row["spread_zscore"]
        spread = row["spread"]

        # Get TFT prediction
        try:
            window = base_data.iloc[i - 100 : i]
            tft_result = tft_model.predict(window)
            is_trending_up = tft_result.predicted_regime.name == "TRENDING_UP"
            is_trending_down = tft_result.predicted_regime.name == "TRENDING_DOWN"
            confidence = tft_result.confidence
        except:
            is_trending_up = False
            is_trending_down = False
            confidence = 0.5

        # Trade with TFT guidance
        if position == 0 and confidence > 0.5:
            if zscore < -2 and is_trending_up:
                position = 1
                entry_price = spread
                capital -= SPREAD_COST * POSITION_SIZE
            elif zscore > 2 and is_trending_down:
                position = -1
                entry_price = spread
                capital -= SPREAD_COST * POSITION_SIZE
        elif position == 1 and (zscore > 0 or is_trending_down):
            pnl = (spread - entry_price) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0
        elif position == -1 and (zscore < 0 or is_trending_up):
            pnl = (entry_price - spread) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0

        mtm = capital + (
            position * (spread - entry_price) * POSITION_SIZE if position != 0 else 0
        )
        equity_curve.append(mtm)

    return calculate_metrics(capital, initial_capital, trades, equity_curve)


def backtest_ensemble_strategy(
    data: pd.DataFrame,
    models: Dict[str, object],
    initial_capital: float,
) -> Dict[str, float]:
    """
    Backtest ensemble strategy combining all models.
    Uses voting mechanism to make decisions.
    """
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]

    POSITION_SIZE = 1000
    SPREAD_COST = 0.05

    # Prepare base data
    base_data = data[["wti", "brent"]].copy()
    base_data["close"] = base_data["wti"]
    base_data["volume"] = 1000000
    base_data["open"] = base_data["close"]
    base_data["high"] = base_data["close"] * 1.01
    base_data["low"] = base_data["close"] * 0.99

    returns = data["spread"].pct_change().fillna(0)

    for i in range(100, len(data)):
        row = data.iloc[i]
        zscore = row["spread_zscore"]
        spread = row["spread"]

        # Collect votes from all models
        votes_long = 0
        votes_short = 0

        # Base zscore signal
        if zscore < -2:
            votes_long += 1
        elif zscore > 2:
            votes_short += 1

        # HMM vote
        if "hmm" in models:
            try:
                window = base_data.iloc[i - 60 : i]
                result = models["hmm"].predict(window)
                if result.state.name in ["LOW_VOL_BULL", "HIGH_VOL_BULL"]:
                    votes_long += 1
                else:
                    votes_short += 1
            except:
                pass

        # Changepoint vote
        if "changepoint" in models:
            try:
                window_returns = returns.iloc[i - 60 : i]
                result = models["changepoint"].detect(window_returns)
                if result.regime_change_probability < 0.3:
                    if zscore < -2:
                        votes_long += 1
                    elif zscore > 2:
                        votes_short += 1
            except:
                pass

        # TFT vote
        if "tft" in models:
            try:
                window = base_data.iloc[i - 100 : i]
                result = models["tft"].predict(window)
                if result.predicted_regime.name == "TRENDING_UP":
                    votes_long += 1
                elif result.predicted_regime.name == "TRENDING_DOWN":
                    votes_short += 1
            except:
                pass

        # Make decision based on votes
        threshold = len(models) // 2 + 1

        if position == 0:
            if votes_long >= threshold:
                position = 1
                entry_price = spread
                capital -= SPREAD_COST * POSITION_SIZE
            elif votes_short >= threshold:
                position = -1
                entry_price = spread
                capital -= SPREAD_COST * POSITION_SIZE
        elif position == 1:
            if zscore > 0 or votes_short >= threshold:
                pnl = (
                    spread - entry_price
                ) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
                capital += pnl
                trades.append({"pnl": pnl})
                position = 0
        elif position == -1:
            if zscore < 0 or votes_long >= threshold:
                pnl = (
                    entry_price - spread
                ) * POSITION_SIZE - SPREAD_COST * POSITION_SIZE
                capital += pnl
                trades.append({"pnl": pnl})
                position = 0

        mtm = capital + (
            position * (spread - entry_price) * POSITION_SIZE if position != 0 else 0
        )
        equity_curve.append(mtm)

    return calculate_metrics(capital, initial_capital, trades, equity_curve)


def backtest_buy_hold(data: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
    """Backtest buy and hold WTI."""
    if data.empty:
        return {
            "initial_capital": initial_capital,
            "final_capital": initial_capital,
            "total_pnl": 0,
            "total_return_pct": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "total_trades": 0,
            "profitable_trades": 0,
        }

    start_price = data["wti"].iloc[0]
    end_price = data["wti"].iloc[-1]

    returns = data["wti"].pct_change().dropna()
    total_return = (end_price - start_price) / start_price
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (peak - cumulative) / peak
    max_dd = drawdown.max() * 100

    final_capital = initial_capital * (1 + total_return)
    total_pnl = final_capital - initial_capital

    return {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_pnl": total_pnl,
        "total_return_pct": total_return * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": (returns > 0).mean() * 100,
        "total_trades": 1,
        "profitable_trades": 1 if total_return > 0 else 0,
    }


def backtest_rl_spread_strategy(
    data: pd.DataFrame,
    spread_agent: object,
    initial_capital: float,
) -> Dict[str, float]:
    """
    Backtest using the trained Spread RL agent.
    """
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [capital]

    POSITION_SIZE = 1000
    COST = 0.05

    spread_series = (
        data["spread"].values if "spread" in data.columns else np.zeros(len(data))
    )
    zscore_series = (
        data["spread_zscore"].values
        if "spread_zscore" in data.columns
        else np.zeros(len(data))
    )

    for i in range(60, len(data)):
        spread = spread_series[i]
        zscore = zscore_series[i]

        try:
            if hasattr(spread_agent, "select_action"):
                state_arr = np.array([zscore, 0, 0, 0, position, 0, 0, 0, 0, 0, 0, 0])
                from quantcore.rl.base import State

                state = State(continuous=state_arr[: spread_agent.state_dim])
                action = spread_agent.select_action(state, deterministic=True)
                action_idx = int(action.value)
            else:
                action_idx = 2 if zscore < -2 else (4 if zscore > 2 else 0)
        except:
            action_idx = 2 if zscore < -2 else (4 if zscore > 2 else 0)

        if action_idx in [1, 2] and position == 0:
            position = 1
            entry_price = spread
            capital -= COST * POSITION_SIZE
        elif action_idx in [3, 4] and position == 0:
            position = -1
            entry_price = spread
            capital -= COST * POSITION_SIZE
        elif action_idx == 0 and position != 0:
            if position == 1:
                pnl = (spread - entry_price) * POSITION_SIZE - COST * POSITION_SIZE
            else:
                pnl = (entry_price - spread) * POSITION_SIZE - COST * POSITION_SIZE
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0

        mtm = capital + (
            position * (spread - entry_price) * POSITION_SIZE if position != 0 else 0
        )
        equity_curve.append(mtm)

    return calculate_metrics(capital, initial_capital, trades, equity_curve)


def backtest_rl_enhanced_strategy(
    data: pd.DataFrame,
    rl_agents: Dict[str, object],
    initial_capital: float,
) -> Dict[str, float]:
    """
    Backtest spread strategy enhanced with RL execution and sizing.

    FIX: Now tracks entry_position_size separately to ensure P&L uses the size
    at which the position was actually entered (RL sizing varies each bar).
    """
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_position_size = 0  # Track size at entry to avoid P&L calculation errors
    trades = []
    equity_curve = [capital]

    BASE_POSITION_SIZE = 1000
    COST = 0.05

    sizing_agent = rl_agents.get("sizing")

    for i in range(60, len(data)):
        row = data.iloc[i]
        zscore = row.get("spread_zscore", 0)
        spread = row.get("spread", 0)

        base_signal = 0
        if zscore < -2:
            base_signal = 1
        elif zscore > 2:
            base_signal = -1
        elif position == 1 and zscore > 0:
            base_signal = 0
        elif position == -1 and zscore < 0:
            base_signal = 0
        else:
            base_signal = position

        # Only compute sizing for new entries
        if sizing_agent and base_signal != 0 and position == 0:
            try:
                from quantcore.rl.base import State

                state_arr = np.array(
                    [
                        abs(zscore) / 3,
                        base_signal,
                        0.5,
                        0,
                        0,
                        0.5,
                        position / 1,
                        0,
                        0,
                        0.5,
                    ]
                )
                state = State(continuous=state_arr[: sizing_agent.state_dim])
                action = sizing_agent.select_action(state, deterministic=True)
                size_scale = float(action.value) if hasattr(action, "value") else 0.5
                size_scale = max(0.25, min(1.0, size_scale))
            except:
                size_scale = 0.5
        else:
            size_scale = 0.5

        current_size = int(BASE_POSITION_SIZE * size_scale)

        if base_signal == 1 and position == 0:
            position = 1
            entry_price = spread
            entry_position_size = current_size  # Store size at entry
            capital -= COST * entry_position_size
        elif base_signal == -1 and position == 0:
            position = -1
            entry_price = spread
            entry_position_size = current_size  # Store size at entry
            capital -= COST * entry_position_size
        elif base_signal == 0 and position != 0:
            # Use entry_position_size for P&L, not current size
            if position == 1:
                pnl = (
                    spread - entry_price
                ) * entry_position_size - COST * entry_position_size
            else:
                pnl = (
                    entry_price - spread
                ) * entry_position_size - COST * entry_position_size
            capital += pnl
            trades.append({"pnl": pnl})
            position = 0
            entry_position_size = 0

        # Mark to market uses entry_position_size
        mtm = capital + (
            position * (spread - entry_price) * entry_position_size
            if position != 0
            else 0
        )
        equity_curve.append(mtm)

    return calculate_metrics(capital, initial_capital, trades, equity_curve)


def run_strategy_comparison(
    all_data: Dict[str, pd.DataFrame],
    spread_df: pd.DataFrame,
    regime_models: Dict[str, object],
    initial_capital: float = 100000,
    rl_agents: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Run and compare multiple trading strategies.
    """
    print_section("Strategy Comparison Suite")

    results = []

    if spread_df.empty or len(spread_df) < 500:
        print_error("Insufficient data for strategy comparison")
        return pd.DataFrame()

    valid_data = spread_df.dropna(subset=["spread_zscore"]).copy()

    train_data = valid_data[valid_data.index < "2021-01-01"]
    test_data = valid_data[valid_data.index >= "2021-01-01"]

    print_info("⚠️  TEMPORAL SPLIT (No Lookahead Bias):")
    print_info(
        f"  Train:      {len(train_data):,} bars ({train_data.index[0].date()} to {train_data.index[-1].date()})"
    )
    print_info(
        f"  Test:       {len(test_data):,} bars ({test_data.index[0].date()} to {test_data.index[-1].date()}) ← 2021+ ONLY"
    )
    print()

    # Rule-based strategies
    print("─" * 50)
    print("  RULE-BASED STRATEGIES")
    print("─" * 50)

    print_info("Testing Spread Mean Reversion...")
    baseline_result = backtest_spread_strategy(
        test_data, initial_capital, 2.0, 0.0, 1000
    )
    baseline_result["strategy"] = "Spread Mean Reversion"
    baseline_result["type"] = "Rule-Based"
    results.append(baseline_result)
    print_success(
        f"  Return: {baseline_result['total_return_pct']:.1f}%, Sharpe: {baseline_result['sharpe_ratio']:.2f}"
    )

    print_info("Testing SMA Crossover...")
    sma_result = backtest_sma_crossover(test_data, initial_capital)
    sma_result["strategy"] = "SMA Crossover (20/50)"
    sma_result["type"] = "Rule-Based"
    results.append(sma_result)
    print_success(f"  Return: {sma_result['total_return_pct']:.1f}%")

    print_info("Testing Bollinger Bands...")
    bb_result = backtest_bollinger_bands(test_data, initial_capital)
    bb_result["strategy"] = "Bollinger Bands"
    bb_result["type"] = "Rule-Based"
    results.append(bb_result)
    print_success(f"  Return: {bb_result['total_return_pct']:.1f}%")

    print_info("Testing RSI...")
    rsi_result = backtest_rsi_strategy(test_data, initial_capital)
    rsi_result["strategy"] = "RSI (30/70)"
    rsi_result["type"] = "Rule-Based"
    results.append(rsi_result)
    print_success(f"  Return: {rsi_result['total_return_pct']:.1f}%")

    print_info("Testing Momentum...")
    mom_result = backtest_momentum_strategy(test_data, initial_capital)
    mom_result["strategy"] = "Momentum (ROC-10)"
    mom_result["type"] = "Rule-Based"
    results.append(mom_result)
    print_success(f"  Return: {mom_result['total_return_pct']:.1f}%")

    print_info("Testing MACD...")
    macd_result = backtest_macd_strategy(test_data, initial_capital)
    macd_result["strategy"] = "MACD"
    macd_result["type"] = "Rule-Based"
    results.append(macd_result)
    print_success(f"  Return: {macd_result['total_return_pct']:.1f}%")

    # ML-based strategies
    print()
    print("─" * 50)
    print("  ML-BASED STRATEGIES")
    print("─" * 50)

    if "hmm" in regime_models:
        print_info("Testing HMM Regime...")
        hmm_result = backtest_hmm_strategy(
            test_data, regime_models["hmm"], initial_capital
        )
        hmm_result["strategy"] = "HMM Regime"
        hmm_result["type"] = "ML-Based"
        results.append(hmm_result)
        print_success(f"  Return: {hmm_result['total_return_pct']:.1f}%")

    if "changepoint" in regime_models:
        print_info("Testing Changepoint...")
        cp_result = backtest_changepoint_strategy(
            test_data, regime_models["changepoint"], initial_capital
        )
        cp_result["strategy"] = "Changepoint"
        cp_result["type"] = "ML-Based"
        results.append(cp_result)
        print_success(f"  Return: {cp_result['total_return_pct']:.1f}%")

    if "tft" in regime_models:
        print_info("Testing TFT Transformer...")
        tft_result = backtest_tft_strategy(
            test_data, regime_models["tft"], initial_capital
        )
        tft_result["strategy"] = "TFT Transformer"
        tft_result["type"] = "ML-Based"
        results.append(tft_result)
        print_success(f"  Return: {tft_result['total_return_pct']:.1f}%")

    if len(regime_models) >= 2:
        print_info("Testing ML Ensemble...")
        ensemble_result = backtest_ensemble_strategy(
            test_data, regime_models, initial_capital
        )
        ensemble_result["strategy"] = "ML Ensemble"
        ensemble_result["type"] = "ML-Based"
        results.append(ensemble_result)
        print_success(f"  Return: {ensemble_result['total_return_pct']:.1f}%")

    # RL-based strategies
    if rl_agents:
        print()
        print("─" * 50)
        print("  RL-BASED STRATEGIES")
        print("─" * 50)

        if "spread" in rl_agents:
            print_info("Testing RL Spread Agent...")
            rl_spread_result = backtest_rl_spread_strategy(
                test_data, rl_agents["spread"], initial_capital
            )
            rl_spread_result["strategy"] = "RL Spread Agent"
            rl_spread_result["type"] = "RL-Based"
            results.append(rl_spread_result)
            print_success(f"  Return: {rl_spread_result['total_return_pct']:.1f}%")

        if "execution" in rl_agents or "sizing" in rl_agents:
            print_info("Testing RL-Enhanced...")
            rl_enhanced_result = backtest_rl_enhanced_strategy(
                test_data, rl_agents, initial_capital
            )
            rl_enhanced_result["strategy"] = "RL-Enhanced"
            rl_enhanced_result["type"] = "RL-Based"
            results.append(rl_enhanced_result)
            print_success(f"  Return: {rl_enhanced_result['total_return_pct']:.1f}%")

    # Benchmark
    print()
    print("─" * 50)
    print("  BENCHMARK")
    print("─" * 50)

    print_info("Testing Buy & Hold...")
    bh_result = backtest_buy_hold(test_data, initial_capital)
    bh_result["strategy"] = "Buy & Hold"
    bh_result["type"] = "Benchmark"
    results.append(bh_result)
    print_success(f"  Return: {bh_result['total_return_pct']:.1f}%")

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Print summary
    print()
    print("=" * 100)
    print("  STRATEGY COMPARISON RESULTS")
    print("=" * 100)

    for strategy_type in ["Rule-Based", "ML-Based", "RL-Based", "Benchmark"]:
        type_results = df_results[df_results["type"] == strategy_type]
        if type_results.empty:
            continue

        print()
        print(f"  {strategy_type.upper()}")
        print(f"  {'-' * 110}")
        print(
            f"  {'Strategy':<25} {'P&L':>14} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Win%':>8} {'Trades':>8}"
        )

        for _, row in type_results.iterrows():
            pnl = row.get(
                "total_pnl", row.get("total_return_pct", 0) / 100 * initial_capital
            )
            print(
                f"  {row['strategy']:<25} ${pnl:>12,.0f} {row['total_return_pct']:>9.1f}% {row['sharpe_ratio']:>8.2f} {row['max_drawdown']:>7.1f}% {row['win_rate']:>7.1f}% {int(row['total_trades']):>8}"
            )

    # Find best per category
    print()
    rule_based = df_results[df_results["type"] == "Rule-Based"]
    ml_based = df_results[df_results["type"] == "ML-Based"]
    rl_based = df_results[df_results["type"] == "RL-Based"]

    if not rule_based.empty:
        best_rule = rule_based.loc[rule_based["sharpe_ratio"].idxmax()]
        print(
            f"🥇 Best Rule-Based: {best_rule['strategy']} (Sharpe: {best_rule['sharpe_ratio']:.2f})"
        )

    if not ml_based.empty:
        best_ml = ml_based.loc[ml_based["sharpe_ratio"].idxmax()]
        print(
            f"🥇 Best ML-Based:   {best_ml['strategy']} (Sharpe: {best_ml['sharpe_ratio']:.2f})"
        )

    if not rl_based.empty:
        best_rl = rl_based.loc[rl_based["sharpe_ratio"].idxmax()]
        print(
            f"🥇 Best RL-Based:   {best_rl['strategy']} (Sharpe: {best_rl['sharpe_ratio']:.2f})"
        )

    best = df_results.loc[df_results["sharpe_ratio"].idxmax()]
    print(
        f"\n🏆 OVERALL BEST: {best['strategy']} ({best['type']}) with {best['sharpe_ratio']:.2f} Sharpe"
    )

    return df_results
