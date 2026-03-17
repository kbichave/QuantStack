# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Phase 2 backtesting tools for the QuantPod MCP server.

Extracted from ``server.py`` — contains the backtest engine wrappers,
walk-forward validation, multi-timeframe backtesting, sparse-signal
walk-forward, and options convexity backtesting.  Private helpers for
signal generation, rule evaluation, price data fetching, and position
sizing live here as well.
"""

import asyncio
import os
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quant_pod.mcp.server import mcp
from quant_pod.mcp._state import require_ctx, live_db_or_error, _serialize


# =============================================================================
# Private Helpers
# =============================================================================


def _generate_signals_from_rules(
    price_data: pd.DataFrame,
    entry_rules: list[dict[str, Any]],
    exit_rules: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """
    Generate a signals DataFrame from strategy rules + price data.

    Supports:
      - Indicators: SMA, RSI, ATR, BBands, Z-score, breakout, ADX, CCI,
        Stochastic, price_vs_sma200, regime classification
      - Rule hierarchy: prerequisite (AND gate), confirmation (N-of-M),
        plain (OR, backward-compatible)
      - Exit management: time stops, ATR-based stop-loss/take-profit via
        forward simulation that maintains position state

    Returns DataFrame with 'signal' (0/1) and 'signal_direction' (LONG/SHORT/NONE).
    """
    import numpy as np
    import pandas as pd

    df = price_data.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ── SMA (configurable periods) ──────────────────────────────────────
    for key, val in parameters.items():
        if key.startswith("sma_") and key not in ("sma_fast", "sma_slow"):
            period = int(val)
            df[f"sma_{period}"] = close.rolling(period).mean()

    sma_fast_p = parameters.get("sma_fast", parameters.get("sma_fast_period", 10))
    sma_slow_p = parameters.get("sma_slow", parameters.get("sma_slow_period", 50))
    df["sma_fast"] = close.rolling(int(sma_fast_p)).mean()
    df["sma_slow"] = close.rolling(int(sma_slow_p)).mean()

    # SMA 200 (always computed for price_vs_sma200 support)
    df["sma_200"] = close.rolling(200).mean()

    # ── RSI ──────────────────────────────────────────────────────────────
    rsi_period = int(parameters.get("rsi_period", 14))
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ── ATR ──────────────────────────────────────────────────────────────
    atr_period = int(parameters.get("atr_period", 14))
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(span=atr_period, adjust=False).mean()

    # ── ADX (with +DI / -DI) ────────────────────────────────────────────
    adx_period = int(parameters.get("adx_period", 14))
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_for_adx = tr.ewm(span=adx_period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=adx_period, adjust=False).mean() / (atr_for_adx + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=adx_period, adjust=False).mean() / (atr_for_adx + 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    df["adx"] = dx.ewm(span=adx_period, adjust=False).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # ── Stochastic K/D ──────────────────────────────────────────────────
    stoch_period = int(parameters.get("stoch_period", 14))
    stoch_smooth = int(parameters.get("stoch_smooth", 3))
    lowest_low = low.rolling(stoch_period).min()
    highest_high = high.rolling(stoch_period).max()
    df["stoch_k"] = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    df["stoch_d"] = df["stoch_k"].rolling(stoch_smooth).mean()

    # ── CCI ──────────────────────────────────────────────────────────────
    cci_period = int(parameters.get("cci_period", 20))
    typical_price = (high + low + close) / 3
    cci_ma = typical_price.rolling(cci_period).mean()
    cci_md = typical_price.rolling(cci_period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    df["cci"] = (typical_price - cci_ma) / (0.015 * cci_md + 1e-10)

    # ── Price vs SMA200 (signed percentage distance) ─────────────────────
    df["price_vs_sma200"] = ((close - df["sma_200"]) / (df["sma_200"] + 1e-10)) * 100

    # ── Regime (derived from ADX + DI direction) ─────────────────────────
    df["regime"] = "ranging"
    df.loc[(df["adx"] >= 25) & (df["plus_di"] > df["minus_di"]), "regime"] = "trending_up"
    df.loc[(df["adx"] >= 25) & (df["minus_di"] > df["plus_di"]), "regime"] = "trending_down"

    # ── ATR percentile (rolling 252-day rank) ────────────────────────────
    df["atr_percentile"] = df["atr"].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) == 252 else 50,
        raw=False,
    )

    # ── Bollinger Bands ──────────────────────────────────────────────────
    bb_period = int(parameters.get("bb_period", 20))
    bb_std = float(parameters.get("bb_std", 2.0))
    bb_ma = close.rolling(bb_period).mean()
    bb_sd = close.rolling(bb_period).std()
    df["bb_upper"] = bb_ma + bb_std * bb_sd
    df["bb_lower"] = bb_ma - bb_std * bb_sd
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    # ── Breakout levels ──────────────────────────────────────────────────
    lookback = int(parameters.get("breakout_period", 20))
    df["high_n"] = high.rolling(lookback).max()
    df["low_n"] = low.rolling(lookback).min()

    # ── Z-score of close relative to SMA ─────────────────────────────────
    zscore_period = int(parameters.get("zscore_period", 20))
    zscore_ma = close.rolling(zscore_period).mean()
    zscore_sd = close.rolling(zscore_period).std()
    df["zscore"] = (close - zscore_ma) / (zscore_sd + 1e-10)

    # ── Evaluate entry rules with prerequisite/confirmation hierarchy ────
    prerequisite_rules = [r for r in entry_rules if r.get("type") == "prerequisite"]
    confirmation_rules = [r for r in entry_rules if r.get("type") == "confirmation"]
    plain_rules = [r for r in entry_rules if r.get("type") not in ("prerequisite", "confirmation")]

    # Prerequisites: ALL must be True (AND gate)
    prereq_pass = pd.Series(True, index=df.index)
    for rule in prerequisite_rules:
        cond = _evaluate_rule(df, rule, parameters)
        prereq_pass = prereq_pass & cond

    # Confirmations: N-of-M must be True
    min_confirmations = int(parameters.get("min_confirmations_required", 1))
    if confirmation_rules:
        confirmation_count = pd.Series(0, index=df.index, dtype=int)
        for rule in confirmation_rules:
            cond = _evaluate_rule(df, rule, parameters)
            confirmation_count = confirmation_count + cond.astype(int)
        confirm_pass = confirmation_count >= min_confirmations
    else:
        confirm_pass = pd.Series(True, index=df.index)

    # Plain rules: OR logic (backward-compatible with existing strategies)
    entry_long = pd.Series(False, index=df.index)
    entry_short = pd.Series(False, index=df.index)
    for rule in plain_rules:
        cond = _evaluate_rule(df, rule, parameters)
        direction = rule.get("direction", "long").lower()
        if direction == "long":
            entry_long = entry_long | cond
        elif direction == "short":
            entry_short = entry_short | cond

    # Combine: structured entries (prerequisite AND confirmation) OR'd with plain
    if prerequisite_rules or confirmation_rules:
        structured_entry = prereq_pass & confirm_pass
        # Determine direction from parameters or prerequisite rules
        struct_dir = parameters.get("direction", "LONG").upper()
        if struct_dir == "SHORT":
            entry_short = entry_short | structured_entry
        else:
            entry_long = entry_long | structured_entry

    # ── Parse exit rules for simulation ──────────────────────────────────
    time_stop_days: int | None = None
    tp_atr_mult: float | None = None
    sl_atr_mult: float | None = None
    for rule in exit_rules:
        rule_type = rule.get("type", "")
        if rule_type == "time_stop":
            time_stop_days = int(rule.get("days", 5))
        elif rule_type == "take_profit":
            tp_atr_mult = float(rule.get("atr_multiple", 2.5))
        elif rule_type == "stop_loss":
            sl_atr_mult = float(rule.get("atr_multiple", 1.5))

    # Evaluate non-structural exit rules (indicator-based exits)
    exit_signal = pd.Series(False, index=df.index)
    structural_exit_types = {"time_stop", "take_profit", "stop_loss", "event_blackout"}
    for rule in exit_rules:
        if rule.get("type") in structural_exit_types:
            continue
        cond = _evaluate_rule(df, rule, parameters)
        exit_signal = exit_signal | cond

    # ── Build initial signal DataFrame ───────────────────────────────────
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["signal_direction"] = "NONE"

    signals.loc[entry_long, "signal"] = 1
    signals.loc[entry_long, "signal_direction"] = "LONG"
    signals.loc[entry_short, "signal"] = 1
    signals.loc[entry_short, "signal_direction"] = "SHORT"
    signals.loc[exit_signal, "signal"] = 0
    signals.loc[exit_signal, "signal_direction"] = "NONE"

    # ── Forward simulation for position-aware exits ──────────────────────
    # The BacktestEngine exits whenever signal=0, so we must maintain
    # signal=1 while in position and only set signal=0 on the exit bar.
    needs_simulation = (
        time_stop_days is not None or tp_atr_mult is not None or sl_atr_mult is not None
    )
    if needs_simulation:
        sig_vals = signals["signal"].values.copy()
        dir_vals = signals["signal_direction"].values.copy()
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        atrs = df["atr"].values

        in_position = False
        entry_bar = 0
        entry_price = 0.0
        entry_atr = 0.0

        for i in range(len(df)):
            if not in_position:
                if sig_vals[i] == 1:
                    in_position = True
                    entry_bar = i
                    entry_price = closes[i]
                    entry_atr = atrs[i] if not np.isnan(atrs[i]) else 0.0
            else:
                bars_held = i - entry_bar
                should_exit = False

                # Time stop
                if time_stop_days and bars_held >= time_stop_days:
                    should_exit = True

                # Take profit (high breaches TP level for longs)
                if tp_atr_mult and entry_atr > 0 and not should_exit:
                    tp_level = entry_price + tp_atr_mult * entry_atr
                    if highs[i] >= tp_level:
                        should_exit = True

                # Stop loss (low breaches SL level for longs)
                if sl_atr_mult and entry_atr > 0 and not should_exit:
                    sl_level = entry_price - sl_atr_mult * entry_atr
                    if lows[i] <= sl_level:
                        should_exit = True

                if should_exit:
                    sig_vals[i] = 0
                    dir_vals[i] = "NONE"
                    in_position = False
                else:
                    # Maintain position — keep signal=1 so engine stays in trade
                    sig_vals[i] = 1
                    dir_vals[i] = "LONG"
                    # Suppress any new entry signals while in position
                    # (engine ignores them anyway when position != 0)

        signals["signal"] = sig_vals
        signals["signal_direction"] = dir_vals

    return signals


def _evaluate_rule(
    df: pd.DataFrame,
    rule: dict[str, Any],
    parameters: dict[str, Any],
) -> pd.Series:
    """Evaluate a single rule dict against the indicator DataFrame.

    Supports:
      - Any pre-computed column name as indicator
      - Special indicators: sma_crossover, breakout, regime
      - Conditions: above, below, crosses_above, crosses_below, between,
        within_pct (absolute value <=), not_in / in (for string columns)
    """
    import pandas as pd

    indicator = rule.get("indicator", "")
    condition = rule.get("condition", "")
    value = rule.get("value")

    # ── Special: regime (string column) ──────────────────────────────────
    if indicator == "regime":
        if "regime" not in df.columns:
            return pd.Series(False, index=df.index)
        if condition == "not_in" and isinstance(value, list):
            return ~df["regime"].isin(value)
        elif condition == "in" and isinstance(value, list):
            return df["regime"].isin(value)
        elif condition == "equals":
            return df["regime"] == value
        return pd.Series(False, index=df.index)

    # ── Resolve indicator column ─────────────────────────────────────────
    if indicator in df.columns:
        series = df[indicator]
    elif indicator == "close":
        series = df["close"]
    elif indicator == "sma_crossover":
        if condition == "crosses_above":
            prev_fast = df["sma_fast"].shift(1)
            prev_slow = df["sma_slow"].shift(1)
            return (prev_fast <= prev_slow) & (df["sma_fast"] > df["sma_slow"])
        elif condition == "crosses_below":
            prev_fast = df["sma_fast"].shift(1)
            prev_slow = df["sma_slow"].shift(1)
            return (prev_fast >= prev_slow) & (df["sma_fast"] < df["sma_slow"])
        return pd.Series(False, index=df.index)
    elif indicator == "breakout":
        if condition == "above":
            return df["close"] > df["high_n"].shift(1)
        elif condition == "below":
            return df["close"] < df["low_n"].shift(1)
        return pd.Series(False, index=df.index)
    else:
        return pd.Series(False, index=df.index)

    # ── Evaluate condition ───────────────────────────────────────────────
    if value is None:
        return pd.Series(False, index=df.index)

    # within_pct: absolute value of series is within threshold
    if condition == "within_pct":
        return series.abs() <= float(value)

    value = float(value)
    if condition in ("above", "greater_than"):
        return series > value
    elif condition in ("below", "less_than"):
        return series < value
    elif condition == "crosses_above":
        return (series.shift(1) <= value) & (series > value)
    elif condition == "crosses_below":
        return (series.shift(1) >= value) & (series < value)
    elif condition == "between":
        upper = float(rule.get("upper", value))
        lower = float(rule.get("lower", 0))
        return (series >= lower) & (series <= upper)
    else:
        return pd.Series(False, index=df.index)


def _fetch_price_data(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    timeframe: str = "daily",
) -> pd.DataFrame | None:
    """Fetch OHLCV price data using the configured provider registry.

    Resolution order:
      1. Local DuckDB cache (fastest, no network)
      2. Provider registry (DATA_PROVIDER_PRIORITY from .env)

    Args:
        timeframe: Bar size string — "daily", "1h", "4h", "1m", "5m", "15m", "30m".

    Returns None if all sources fail.
    """
    import pandas as pd
    from quantcore.config.settings import get_settings
    from quantcore.config.timeframes import Timeframe
    from quantcore.data.base import AssetClass
    from quantcore.data.registry import DataProviderRegistry

    _TF_MAP = {
        "daily": Timeframe.D1, "1d": Timeframe.D1,
        "1h": Timeframe.H1, "hourly": Timeframe.H1,
        "4h": Timeframe.H4,
        "30m": Timeframe.M30, "30min": Timeframe.M30,
        "15m": Timeframe.M15, "15min": Timeframe.M15,
        "5m": Timeframe.M5,  "5min": Timeframe.M5,
        "1m": Timeframe.M1,  "1min": Timeframe.M1,
    }
    tf = _TF_MAP.get(timeframe.lower(), Timeframe.D1)

    def _apply_date_filter(df: pd.DataFrame) -> pd.DataFrame:
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        return df

    # 1. Local DuckDB cache (read-only — no write lock competition)
    try:
        from quantcore.data.storage import DataStore

        with DataStore(read_only=True) as store:
            df = store.load_ohlcv(symbol, tf)
            if df is not None and not df.empty:
                return _apply_date_filter(df)
    except Exception as exc:
        logger.debug(f"DuckDB cache miss for {symbol} [{timeframe}]: {exc}")

    # 2. Provider registry — respects DATA_PROVIDER_PRIORITY
    try:
        from datetime import datetime, timedelta

        settings = get_settings()
        registry = DataProviderRegistry.from_settings(settings)

        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        start_dt = (
            datetime.strptime(start_date, "%Y-%m-%d")
            if start_date
            else end_dt - timedelta(days=365 * 6)
        )

        df = registry.fetch_ohlcv(symbol, AssetClass.EQUITY, tf, start_dt, end_dt)
        if df is not None and not df.empty:
            return df
    except Exception as exc:
        logger.warning(f"Provider registry fetch failed for {symbol} [{timeframe}]: {exc}")

    return None


def _calc_quantity_from_size(position_size: str, equity: float, current_price: float) -> int:
    """Convert a position_size label ('full', 'half', 'quarter') to shares."""
    fractions = {"full": 0.10, "half": 0.05, "quarter": 0.025}
    frac = fractions.get(position_size, 0.025)
    if current_price <= 0:
        return 0
    return max(1, int((equity * frac) / current_price))


# =============================================================================
# Phase 2: Backtesting Tools
# =============================================================================


@mcp.tool()
async def run_backtest(
    strategy_id: str,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.10,
    commission: float = 1.0,
    slippage_pct: float = 0.001,
) -> dict[str, Any]:
    """
    Backtest a registered strategy against historical price data.

    Fetches the strategy from the registry, generates entry/exit signals from
    its rules, then runs the BacktestEngine.  Results are stored back on the
    strategy record as backtest_summary and status is updated to 'backtested'.

    Args:
        strategy_id: Strategy to backtest.
        symbol: Ticker symbol for price data.
        start_date: Start date (YYYY-MM-DD). None = earliest available.
        end_date: End date (YYYY-MM-DD). None = latest available.
        initial_capital: Starting capital.
        position_size_pct: Fraction of capital per trade.
        commission: Commission per trade in dollars.
        slippage_pct: Slippage as fraction of price.

    Returns:
        BacktestResult dict with metrics.
    """
    import json as _json

    from quant_pod.mcp.server import get_strategy

    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        # 1. Load strategy
        strat_result = await get_strategy.fn(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": strat_result.get("error", "Strategy not found")}
        strat = strat_result["strategy"]

        # 2. Fetch price data
        price_data = await asyncio.get_event_loop().run_in_executor(
            None, _fetch_price_data, symbol, start_date, end_date
        )
        if price_data is None or price_data.empty:
            return {"success": False, "error": f"No price data available for {symbol}"}

        # 3. Generate signals from rules
        entry_rules = strat.get("entry_rules", [])
        exit_rules = strat.get("exit_rules", [])
        parameters = strat.get("parameters", {})

        if not entry_rules:
            return {"success": False, "error": "Strategy has no entry_rules"}

        signals = _generate_signals_from_rules(price_data, entry_rules, exit_rules, parameters)

        # 4. Run backtest
        from quantcore.backtesting.engine import BacktestConfig, BacktestEngine

        config = BacktestConfig(
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            commission_per_trade=commission,
            slippage_pct=slippage_pct,
        )
        engine = BacktestEngine(config=config)
        result = engine.run(signals, price_data)

        # 5. Compute additional metrics
        import numpy as np

        calmar = 0.0
        if result.max_drawdown > 0:
            calmar = (result.total_return / 100.0) / (result.max_drawdown / 100.0)

        avg_pnl = 0.0
        if result.trades:
            avg_pnl = np.mean([t["pnl"] for t in result.trades])

        summary = {
            "symbol": symbol,
            "total_trades": result.total_trades,
            "win_rate": round(result.win_rate, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 4),
            "max_drawdown": round(result.max_drawdown, 2),
            "total_return_pct": round(result.total_return, 2),
            "profit_factor": round(result.profit_factor, 4),
            "calmar_ratio": round(calmar, 4),
            "avg_trade_pnl": round(avg_pnl, 2),
            "start_date": str(price_data.index[0].date())
            if hasattr(price_data.index[0], "date")
            else str(price_data.index[0]),
            "end_date": str(price_data.index[-1].date())
            if hasattr(price_data.index[-1], "date")
            else str(price_data.index[-1]),
            "bars_tested": len(price_data),
            "trades": result.trades,
        }

        # 6. Persist summary on strategy record
        ctx.db.execute(
            "UPDATE strategies SET backtest_summary = ?, status = CASE WHEN status = 'draft' THEN 'backtested' ELSE status END, updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
            [_json.dumps(summary), strategy_id],
        )

        return {"success": True, **summary, "strategy_id": strategy_id}

    except Exception as e:
        logger.error(f"[quantpod_mcp] run_backtest failed: {e}")
        return {"success": False, "error": str(e), "strategy_id": strategy_id, "symbol": symbol}


@mcp.tool()
async def run_backtest_mtf(
    strategy_id: str,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.05,
) -> dict[str, Any]:
    """
    Multi-timeframe backtest for strategies with setup + trigger timeframes.

    Loads the strategy's parameters (setup_tf, trigger_tf, thresholds, stops)
    and runs a cross-timeframe backtest: higher-TF setup condition triggers a
    search window on the lower-TF for a precision entry.

    Requires strategy parameters:
        setup_tf, trigger_tf, setup_rsi_threshold, trigger_rsi_threshold,
        sma_proximity_pct, stop_loss_atr, take_profit_atr,
        time_stop_days (optional), max_trigger_wait_days or max_trigger_wait_hours

    Args:
        strategy_id: Registered strategy with MTF parameters.
        symbol: Ticker symbol.
        start_date: Start date (YYYY-MM-DD). None = earliest available.
        end_date: End date (YYYY-MM-DD). None = latest available.
        initial_capital: Starting capital.
        position_size_pct: Fraction of capital per trade.

    Returns:
        Dict with sharpe, win_rate, total_trades, profit_factor,
        avg_return_pct, max_drawdown, and trade list.
    """
    import json as _json
    import numpy as np

    from quant_pod.mcp.server import get_strategy

    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        # 1. Load strategy
        strat_result = await get_strategy.fn(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": strat_result.get("error", "Strategy not found")}
        strat = strat_result["strategy"]
        params = strat.get("parameters", {})

        setup_tf_str = params.get("setup_tf", "1d")
        trigger_tf_str = params.get("trigger_tf", "1h")
        setup_rsi_threshold = params.get("setup_rsi_threshold", 35)
        trigger_rsi_threshold = params.get("trigger_rsi_threshold", 35)
        sma_proximity_pct = params.get("sma_proximity_pct", 3.0)
        sl_atr_mult = params.get("stop_loss_atr", 1.5)
        tp_atr_mult = params.get("take_profit_atr", 2.5)
        time_stop_days = params.get("time_stop_days")
        max_wait_days = params.get("max_trigger_wait_days", 3)
        max_wait_hours = params.get("max_trigger_wait_hours")
        use_sma_filter = True

        # Convert max_trigger_wait to setup-TF bars
        if max_wait_hours is not None:
            # 4H setup → hours / 4 = bars
            tf_hours = {"4h": 4, "1h": 1, "1d": 24}.get(setup_tf_str, 4)
            max_trigger_wait_bars = max(1, int(max_wait_hours / tf_hours))
        else:
            max_trigger_wait_bars = max_wait_days

        # 2. Load and compute indicators for both timeframes
        from quantcore.data.storage import DataStore
        from quantcore.config.timeframes import Timeframe
        from quantcore.features.technical_indicators import TechnicalIndicators

        tf_map = {"1d": Timeframe.D1, "daily": Timeframe.D1,
                   "4h": Timeframe.H4, "1h": Timeframe.H1}
        setup_tf = tf_map.get(setup_tf_str)
        trigger_tf = tf_map.get(trigger_tf_str)
        if setup_tf is None or trigger_tf is None:
            return {"success": False, "error": f"Unsupported timeframe: {setup_tf_str} or {trigger_tf_str}"}

        def _load_with_indicators(sym, tf_enum):
            store = DataStore(read_only=True)
            df = store.load_ohlcv(sym, tf_enum)
            store.close()
            if df.empty:
                return df
            ti = TechnicalIndicators(tf_enum)
            df = ti.compute(df)
            # Standardize column names
            col_map = {}
            for col in df.columns:
                cl = col.lower()
                if "rsi" in cl and "14" in cl:
                    col_map[col] = "rsi"
                elif cl in ("sma_200", "sma200"):
                    col_map[col] = "sma200"
                elif "atr" in cl and "14" in cl:
                    col_map[col] = "atr"
            df = df.rename(columns=col_map)
            # Fallback computations
            if "rsi" not in df.columns:
                delta = df["close"].diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain / (loss + 1e-10)
                df["rsi"] = 100 - (100 / (1 + rs))
            if "sma200" not in df.columns:
                df["sma200"] = df["close"].rolling(200).mean()
            if "atr" not in df.columns:
                import pandas as pd
                tr = pd.concat([
                    df["high"] - df["low"],
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs(),
                ], axis=1).max(axis=1)
                df["atr"] = tr.rolling(14).mean()
            return df

        setup_df = await asyncio.get_event_loop().run_in_executor(
            None, _load_with_indicators, symbol, setup_tf
        )
        trigger_df = await asyncio.get_event_loop().run_in_executor(
            None, _load_with_indicators, symbol, trigger_tf
        )

        if setup_df.empty or trigger_df.empty:
            return {"success": False, "error": f"No data for {symbol} at {setup_tf_str} or {trigger_tf_str}"}

        # Apply date filters
        import pandas as pd
        if start_date:
            start_ts = pd.Timestamp(start_date)
            setup_df = setup_df[setup_df.index >= start_ts]
            trigger_df = trigger_df[trigger_df.index >= start_ts]
        if end_date:
            end_ts = pd.Timestamp(end_date)
            setup_df = setup_df[setup_df.index <= end_ts]
            trigger_df = trigger_df[trigger_df.index <= end_ts]

        # 3. Run MTF backtest
        setup_df = setup_df.dropna(subset=["rsi", "sma200"]).copy()
        trigger_df = trigger_df.dropna(subset=["rsi", "atr"]).copy()
        trigger_df["rsi_prev"] = trigger_df["rsi"].shift(1)

        trades = []
        in_trade = False
        setup_bar = None
        trade = {}

        for i, (ts, row) in enumerate(setup_df.iterrows()):
            if not in_trade and setup_bar is None:
                sma_dist = abs(row["close"] - row["sma200"]) / row["sma200"] * 100
                if row["rsi"] < setup_rsi_threshold:
                    if not use_sma_filter or sma_dist < sma_proximity_pct:
                        setup_bar = ts

            if setup_bar is not None and not in_trade:
                setup_idx = setup_df.index.get_loc(setup_bar)
                end_idx = min(setup_idx + max_trigger_wait_bars, len(setup_df) - 1)
                window_end_ts = setup_df.index[end_idx]

                if ts > window_end_ts:
                    setup_bar = None
                    continue

                trigger_window = trigger_df[
                    (trigger_df.index >= setup_bar) &
                    (trigger_df.index <= window_end_ts)
                ]

                for t_ts, t_row in trigger_window.iterrows():
                    if (pd.notna(t_row["rsi_prev"]) and
                            t_row["rsi_prev"] < trigger_rsi_threshold and
                            t_row["rsi"] >= trigger_rsi_threshold):
                        entry_price = t_row["close"]
                        entry_atr = t_row["atr"]
                        tp = entry_price + tp_atr_mult * entry_atr
                        sl = entry_price - sl_atr_mult * entry_atr

                        deadline_ts = None
                        if time_stop_days is not None:
                            stop_idx = min(setup_idx + time_stop_days, len(setup_df) - 1)
                            deadline_ts = setup_df.index[stop_idx]

                        in_trade = True
                        trade = {
                            "setup_ts": str(setup_bar),
                            "entry_ts": str(t_ts),
                            "entry_price": entry_price,
                            "tp": tp, "sl": sl,
                            "deadline_ts": str(deadline_ts) if deadline_ts else None,
                            "atr": entry_atr,
                        }
                        setup_bar = None
                        break

                if not in_trade and ts >= window_end_ts:
                    setup_bar = None

            if in_trade:
                entry_ts = pd.Timestamp(trade["entry_ts"])
                scan = trigger_df[trigger_df.index > entry_ts]
                bars_held = 0
                for e_ts, e_row in scan.iterrows():
                    bars_held += 1
                    exit_price = None
                    exit_reason = None

                    if e_row["high"] >= trade["tp"]:
                        exit_price = trade["tp"]
                        exit_reason = "tp"
                    elif e_row["low"] <= trade["sl"]:
                        exit_price = trade["sl"]
                        exit_reason = "sl"
                    elif trade["deadline_ts"] is not None and e_ts >= pd.Timestamp(trade["deadline_ts"]):
                        exit_price = e_row["close"]
                        exit_reason = "time_stop"

                    if exit_price is not None:
                        pnl = (exit_price - trade["entry_price"]) / trade["entry_price"]
                        trades.append({
                            **trade,
                            "exit_ts": str(e_ts),
                            "exit_price": exit_price,
                            "exit_reason": exit_reason,
                            "pnl_pct": round(pnl, 6),
                        })
                        in_trade = False
                        break
                if in_trade:
                    continue

        # 4. Compute metrics
        if not trades:
            summary = {
                "success": True, "sharpe": 0.0, "win_rate": 0.0,
                "total_trades": 0, "profit_factor": 0.0,
                "avg_return_pct": 0.0, "max_drawdown": 0.0,
                "trades": [], "strategy_id": strategy_id, "symbol": symbol,
            }
        else:
            returns = [t["pnl_pct"] * position_size_pct for t in trades]
            equity = initial_capital * (1 + pd.Series(returns)).cumprod()
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r <= 0]

            sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252)
            peak = equity.cummax()
            max_dd = ((equity - peak) / peak).min()

            summary = {
                "success": True,
                "sharpe": round(float(sharpe), 3),
                "win_rate": round(len(wins) / len(returns), 3),
                "total_trades": len(trades),
                "profit_factor": round(sum(wins) / (abs(sum(losses)) + 1e-9), 2),
                "avg_return_pct": round(float(np.mean(returns)) * 100, 3),
                "max_drawdown": round(float(max_dd) * 100, 2),
                "trades": trades,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "setup_tf": setup_tf_str,
                "trigger_tf": trigger_tf_str,
            }

        # 5. Persist summary on strategy record
        bt_summary = {k: v for k, v in summary.items() if k != "trades"}
        ctx.db.execute(
            "UPDATE strategies SET backtest_summary = ?, "
            "status = CASE WHEN status = 'draft' THEN 'backtested' ELSE status END, "
            "updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
            [_json.dumps(bt_summary), strategy_id],
        )

        return summary

    except Exception as e:
        logger.error(f"[quantpod_mcp] run_backtest_mtf failed: {e}")
        return {"success": False, "error": str(e), "strategy_id": strategy_id, "symbol": symbol}


@mcp.tool()
async def run_walkforward(
    strategy_id: str,
    symbol: str,
    n_splits: int = 5,
    test_size: int = 252,
    min_train_size: int = 504,
    gap: int = 0,
    expanding: bool = True,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.10,
) -> dict[str, Any]:
    """
    Walk-forward validation of a registered strategy.

    Splits price data into successive train/test folds.  On each fold, signals
    are generated from the strategy rules and a backtest is run on the test
    period.  Returns per-fold IS/OOS metrics and aggregate statistics.

    Args:
        strategy_id: Strategy to validate.
        symbol: Ticker symbol.
        n_splits: Number of walk-forward folds.
        test_size: Bars per test fold.
        min_train_size: Minimum training bars.
        gap: Embargo bars between train and test.
        expanding: Expanding (True) or rolling (False) window.
        initial_capital: Starting capital per fold.
        position_size_pct: Position size fraction.

    Returns:
        WalkForwardResult dict with per-fold and aggregate metrics.
    """
    import json as _json

    import numpy as np

    from quant_pod.mcp.server import get_strategy

    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        # 1. Load strategy
        strat_result = await get_strategy.fn(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": strat_result.get("error", "Strategy not found")}
        strat = strat_result["strategy"]

        entry_rules = strat.get("entry_rules", [])
        exit_rules = strat.get("exit_rules", [])
        parameters = strat.get("parameters", {})

        if not entry_rules:
            return {"success": False, "error": "Strategy has no entry_rules"}

        # 2. Fetch price data
        price_data = await asyncio.get_event_loop().run_in_executor(
            None, _fetch_price_data, symbol, None, None
        )
        if price_data is None or price_data.empty:
            return {"success": False, "error": f"No price data for {symbol}"}

        n = len(price_data)
        total_needed = min_train_size + n_splits * test_size
        if n < total_needed:
            return {
                "success": False,
                "error": f"Insufficient data: need {total_needed} bars, have {n}",
            }

        # 3. Walk-forward splits
        from quantcore.backtesting.engine import BacktestConfig, BacktestEngine

        config = BacktestConfig(
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
        )

        fold_results = []
        first_test_start = min_train_size + gap

        for i in range(n_splits):
            test_start = first_test_start + i * test_size
            test_end = min(test_start + test_size, n)
            if expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - gap - min_train_size)
            train_end = test_start - gap

            train_data = price_data.iloc[train_start:train_end]
            test_data = price_data.iloc[test_start:test_end]

            # Generate signals for each period
            train_signals = _generate_signals_from_rules(
                train_data, entry_rules, exit_rules, parameters
            )
            test_signals = _generate_signals_from_rules(
                test_data, entry_rules, exit_rules, parameters
            )

            # Run backtests
            engine_is = BacktestEngine(config=config)
            result_is = engine_is.run(train_signals, train_data)

            engine_oos = BacktestEngine(config=config)
            result_oos = engine_oos.run(test_signals, test_data)

            fold_results.append(
                {
                    "fold": i + 1,
                    "train_bars": len(train_data),
                    "test_bars": len(test_data),
                    "is_sharpe": round(result_is.sharpe_ratio, 4),
                    "is_return_pct": round(result_is.total_return, 2),
                    "is_trades": result_is.total_trades,
                    "oos_sharpe": round(result_oos.sharpe_ratio, 4),
                    "oos_return_pct": round(result_oos.total_return, 2),
                    "oos_trades": result_oos.total_trades,
                    "oos_max_dd": round(result_oos.max_drawdown, 2),
                }
            )

        # 4. Aggregate
        is_sharpes = [f["is_sharpe"] for f in fold_results]
        oos_sharpes = [f["oos_sharpe"] for f in fold_results]
        is_mean = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        oos_mean = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        is_std = float(np.std(is_sharpes)) if is_sharpes else 0.0
        oos_std = float(np.std(oos_sharpes)) if oos_sharpes else 0.0
        overfit_ratio = is_mean / oos_mean if oos_mean != 0 else float("inf")
        oos_positive = sum(1 for s in oos_sharpes if s > 0)
        degradation = ((is_mean - oos_mean) / abs(is_mean) * 100) if is_mean != 0 else 0.0

        summary = {
            "symbol": symbol,
            "n_folds": n_splits,
            "is_sharpe_mean": round(is_mean, 4),
            "oos_sharpe_mean": round(oos_mean, 4),
            "is_sharpe_std": round(is_std, 4),
            "oos_sharpe_std": round(oos_std, 4),
            "overfit_ratio": round(overfit_ratio, 4),
            "oos_positive_folds": oos_positive,
            "oos_degradation_pct": round(degradation, 2),
            "fold_results": fold_results,
        }

        # 5. Persist
        ctx.db.execute(
            "UPDATE strategies SET walkforward_summary = ?, updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
            [_json.dumps(summary), strategy_id],
        )

        return {"success": True, "strategy_id": strategy_id, **summary}

    except Exception as e:
        logger.error(f"[quantpod_mcp] run_walkforward failed: {e}")
        return {"success": False, "error": str(e), "strategy_id": strategy_id, "symbol": symbol}


# =============================================================================
# Walk-Forward MTF + Sparse Signal + Options Backtest Tools
# =============================================================================


@mcp.tool()
async def run_walkforward_mtf(
    strategy_id: str,
    symbol: str,
    n_splits: int = 5,
    test_size_days: int = 90,
    min_train_size_days: int = 365,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.05,
) -> dict[str, Any]:
    """
    Walk-forward validation for multi-timeframe strategies (setup_tf + trigger_tf).

    The standard run_walkforward engine generates signals from a single timeframe.
    This tool uses run_backtest_mtf on each IS/OOS fold so the cross-TF entry logic
    (daily setup + hourly trigger) is correctly evaluated.

    Each fold:
    - IS window: min_train_size_days calendar days
    - OOS window: test_size_days calendar days
    - Gap: 0 (expanding window)
    - Both IS and OOS use run_backtest_mtf under the hood

    Args:
        strategy_id: MTF strategy registered via register_strategy with setup_tf/trigger_tf params.
        symbol: Ticker symbol.
        n_splits: Number of folds (default 5).
        test_size_days: OOS window size in calendar days (default 90).
        min_train_size_days: Minimum IS window size in calendar days (default 365).
        initial_capital: Starting capital per fold.
        position_size_pct: Fraction of capital per trade.

    Returns:
        Dict with fold_results, is_sharpe_mean, oos_sharpe_mean, overfit_ratio,
        oos_positive_folds, oos_degradation_pct.
    """
    import json as _json
    import numpy as np
    import pandas as pd

    from quant_pod.mcp.server import get_strategy

    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        # 1. Load strategy and verify MTF params
        strat_result = await get_strategy.fn(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": strat_result.get("error", "Strategy not found")}
        strat = strat_result["strategy"]
        params = strat.get("parameters", {})

        if "setup_tf" not in params or "trigger_tf" not in params:
            return {
                "success": False,
                "error": "Strategy must have setup_tf and trigger_tf in parameters. "
                         "Use run_walkforward for single-timeframe strategies.",
            }

        setup_tf_str = params["setup_tf"]

        # 2. Fetch full setup-TF data to derive fold boundaries
        setup_tf_map = {
            "1d": "daily", "daily": "daily",
            "4h": "4h", "1h": "1h", "hourly": "1h",
        }
        setup_tf_fetch = setup_tf_map.get(setup_tf_str.lower(), "daily")

        def _fetch_setup_data():
            return _fetch_price_data(symbol, None, None, timeframe=setup_tf_fetch)

        setup_df = await asyncio.get_event_loop().run_in_executor(None, _fetch_setup_data)

        if setup_df is None or setup_df.empty:
            return {"success": False, "error": f"No {setup_tf_str} data for {symbol}"}

        # 3. Build fold boundaries in calendar days
        all_dates = setup_df.index
        total_days = (all_dates[-1] - all_dates[0]).days
        min_total_days = min_train_size_days + n_splits * test_size_days

        if total_days < min_total_days:
            return {
                "success": False,
                "error": (
                    f"Insufficient data: {total_days} days available, "
                    f"need {min_total_days} ({min_train_size_days} IS + "
                    f"{n_splits}×{test_size_days} OOS)."
                ),
            }

        start_ts = all_dates[0]
        fold_results = []

        for i in range(n_splits):
            train_start = start_ts
            train_end   = start_ts + pd.Timedelta(days=min_train_size_days + i * test_size_days)
            oos_start   = train_end
            oos_end     = oos_start + pd.Timedelta(days=test_size_days)

            if oos_end > all_dates[-1]:
                break  # No more data for another fold

            # Run IS backtest via run_backtest_mtf
            is_result = await run_backtest_mtf.fn(
                strategy_id=strategy_id,
                symbol=symbol,
                start_date=str(train_start.date()),
                end_date=str(train_end.date()),
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
            )
            # Run OOS backtest via run_backtest_mtf
            oos_result = await run_backtest_mtf.fn(
                strategy_id=strategy_id,
                symbol=symbol,
                start_date=str(oos_start.date()),
                end_date=str(oos_end.date()),
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
            )

            fold_results.append({
                "fold":         i + 1,
                "train_start":  str(train_start.date()),
                "train_end":    str(train_end.date()),
                "oos_start":    str(oos_start.date()),
                "oos_end":      str(oos_end.date()),
                "is_sharpe":    round(is_result.get("sharpe", 0.0), 4),
                "is_trades":    is_result.get("total_trades", 0),
                "oos_sharpe":   round(oos_result.get("sharpe", 0.0), 4),
                "oos_trades":   oos_result.get("total_trades", 0),
                "oos_max_dd":   round(oos_result.get("max_drawdown", 0.0), 2),
            })

        if not fold_results:
            return {"success": False, "error": "No folds generated — check date range and data availability."}

        is_sharpes  = [f["is_sharpe"]  for f in fold_results]
        oos_sharpes = [f["oos_sharpe"] for f in fold_results]
        oos_trades  = [f["oos_trades"] for f in fold_results]

        is_mean  = float(np.mean(is_sharpes))
        oos_mean = float(np.mean(oos_sharpes))
        overfit  = is_mean / oos_mean if oos_mean != 0 else float("inf")
        degradation = ((is_mean - oos_mean) / abs(is_mean) * 100) if is_mean != 0 else 0.0
        oos_positive = sum(1 for s in oos_sharpes if s > 0)
        total_oos_trades = sum(oos_trades)

        summary = {
            "success":            True,
            "strategy_id":        strategy_id,
            "symbol":             symbol,
            "setup_tf":           setup_tf_str,
            "trigger_tf":         params.get("trigger_tf"),
            "n_folds":            len(fold_results),
            "total_oos_trades":   total_oos_trades,
            "is_sharpe_mean":     round(is_mean, 4),
            "oos_sharpe_mean":    round(oos_mean, 4),
            "overfit_ratio":      round(overfit, 4),
            "oos_positive_folds": oos_positive,
            "oos_degradation_pct": round(degradation, 2),
            "fold_results":       fold_results,
            "sparse_warning":     total_oos_trades < n_splits * 2,  # <2 trades/fold avg
        }

        ctx.db.execute(
            "UPDATE strategies SET walkforward_summary = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE strategy_id = ?",
            [_json.dumps(summary), strategy_id],
        )
        return summary

    except Exception as e:
        logger.error(f"[quantpod_mcp] run_walkforward_mtf failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def walk_forward_sparse_signal(
    strategy_id: str,
    symbol: str,
    min_oos_trades: int = 3,
    n_splits: int = 5,
    max_test_size_pct: float = 0.40,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.05,
) -> dict[str, Any]:
    """
    Walk-forward validation that auto-adjusts test_size to guarantee min trades per fold.

    Solves the sparse-signal problem: strategies like RSI<35 mean-reversion fire only
    ~2 times per 150-bar window. Standard walk-forward produces 0 OOS trades per fold,
    making validation impossible.

    This tool runs a full-history backtest first to measure signal frequency, then
    computes the minimum OOS window size that would yield min_oos_trades per fold on
    average, and runs the walk-forward at that adjusted size.

    Args:
        strategy_id: Strategy ID (single-TF or MTF — auto-detects via parameters).
        symbol: Ticker symbol.
        min_oos_trades: Minimum average OOS trades per fold (default 3).
        n_splits: Number of folds (default 5).
        max_test_size_pct: Maximum fraction of total data allowed as OOS window (default 0.40).
        initial_capital: Starting capital per fold.
        position_size_pct: Fraction of capital per trade.

    Returns:
        Dict with adjusted_test_size_days, fold_results, oos_sharpe_mean, sparse_warning
        (True if even the maximum window couldn't guarantee min_oos_trades).
    """
    import json as _json
    import numpy as np

    from quant_pod.mcp.server import get_strategy

    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        # 1. Load strategy — detect if MTF
        strat_result = await get_strategy.fn(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": strat_result.get("error", "Strategy not found")}
        strat = strat_result["strategy"]
        params = strat.get("parameters", {})
        is_mtf = "setup_tf" in params and "trigger_tf" in params

        # 2. Run full-history backtest to measure signal frequency
        if is_mtf:
            full_bt = await run_backtest_mtf.fn(
                strategy_id=strategy_id,
                symbol=symbol,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
            )
        else:
            full_bt = await run_backtest.fn(
                strategy_id=strategy_id,
                symbol=symbol,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
            )

        if not full_bt.get("success", False):
            return {"success": False, "error": f"Full backtest failed: {full_bt.get('error')}"}

        total_trades  = full_bt.get("total_trades", 0) or full_bt.get("total_trades", 0)
        bars_tested   = full_bt.get("bars_tested", 1) or 1
        start_date    = full_bt.get("start_date")
        end_date      = full_bt.get("end_date")

        if total_trades == 0:
            return {
                "success":             False,
                "error":               "Strategy produced 0 trades in full history — cannot estimate signal frequency.",
                "total_trades":        0,
                "sparse_warning":      True,
            }

        # 3. Estimate required OOS window size
        # trades_per_bar = total_trades / bars_tested
        # to get min_oos_trades per fold: oos_bars = min_oos_trades / trades_per_bar
        trades_per_bar = total_trades / bars_tested
        required_oos_bars = int(np.ceil(min_oos_trades / trades_per_bar))
        max_oos_bars = int(bars_tested * max_test_size_pct)
        adjusted_oos_bars = min(required_oos_bars, max_oos_bars)
        sparse_warning = required_oos_bars > max_oos_bars

        min_train_bars = max(int(bars_tested * 0.30), adjusted_oos_bars * 2)
        total_needed = min_train_bars + n_splits * adjusted_oos_bars
        if total_needed > bars_tested:
            # Reduce splits to fit
            n_splits = max(2, (bars_tested - min_train_bars) // adjusted_oos_bars)

        # 4. Run walk-forward with adjusted sizes
        if is_mtf:
            # Convert bars to days (approximate: 1 daily bar = 1 day)
            setup_tf_str = params.get("setup_tf", "1d").lower()
            bars_per_day = {"1d": 1, "daily": 1, "4h": 1.5, "1h": 6.5}.get(setup_tf_str, 1)
            test_size_days = max(30, int(adjusted_oos_bars / bars_per_day))
            min_train_days = max(90, int(min_train_bars / bars_per_day))
            wf_result = await run_walkforward_mtf.fn(
                strategy_id=strategy_id,
                symbol=symbol,
                n_splits=n_splits,
                test_size_days=test_size_days,
                min_train_size_days=min_train_days,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
            )
        else:
            wf_result = await run_walkforward.fn(
                strategy_id=strategy_id,
                symbol=symbol,
                n_splits=n_splits,
                test_size=adjusted_oos_bars,
                min_train_size=min_train_bars,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
            )

        return {
            **wf_result,
            "adjusted_test_size_bars":  adjusted_oos_bars,
            "trades_per_bar":           round(trades_per_bar, 4),
            "required_oos_bars":        required_oos_bars,
            "sparse_warning":           sparse_warning,
            "sparse_note": (
                f"Signal fires ~{trades_per_bar:.3f}×/bar. "
                f"Needed {required_oos_bars} bars/fold for {min_oos_trades} OOS trades; "
                f"{'capped at max' if sparse_warning else 'achievable'}."
            ),
        }

    except Exception as e:
        logger.error(f"[quantpod_mcp] walk_forward_sparse_signal failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def run_backtest_options(
    strategy_id: str,
    symbol: str,
    option_type: str = "call",
    expiry_days: int = 30,
    tp_pct: float = 1.0,
    sl_pct: float = 0.50,
    time_stop_days: int = 15,
    iv_rank_max: float = 1.0,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.05,
    risk_free_rate: float = 0.05,
) -> dict[str, Any]:
    """
    Options convexity backtest: buy ATM calls/puts on the same entry signals as
    the underlying equity strategy, price them with Black-Scholes, and compare
    options P&L to equity P&L.

    Entry signals are sourced from the strategy's entry_rules (same as run_backtest
    or run_backtest_mtf). IV is approximated from 20-day realized vol at entry.

    Exit logic (Variant B):
    - Take profit: option value doubles (tp_pct = 1.0 = 100% gain on premium)
    - Stop loss: option loses sl_pct of its premium value
    - Time stop: time_stop_days from signal date

    iv_rank_max: skip the trade if 20-day vol rank > threshold (0.5 = skip when vol
    is in top 50% of recent history). Set to 1.0 to disable IV filter.

    Args:
        strategy_id: Strategy whose entry signals drive option entries.
        symbol: Ticker symbol (equity underlying).
        option_type: "call" (long setups) or "put" (short setups).
        expiry_days: Days to expiry at option purchase (default 30).
        tp_pct: Take-profit as fraction of entry premium (1.0 = 100% gain).
        sl_pct: Stop-loss as fraction of entry premium (0.50 = 50% loss).
        time_stop_days: Maximum days to hold option from signal date.
        iv_rank_max: Maximum 20-day vol rank allowed at entry (0–1). 1.0 = no filter.
        start_date: Backtest start (YYYY-MM-DD). None = earliest available.
        end_date: Backtest end (YYYY-MM-DD). None = latest available.
        initial_capital: Starting capital.
        position_size_pct: Fraction of capital allocated per option position.
        risk_free_rate: Annual risk-free rate for Black-Scholes (default 0.05).

    Returns:
        Dict with sharpe, win_rate, total_trades, avg_premium_return_pct,
        avg_iv_at_entry, iv_crush_pct, equity_comparison (from underlying strategy),
        and per-trade list.
    """
    import json as _json
    import math
    import numpy as np

    from quant_pod.mcp.server import get_strategy

    ctx, err = live_db_or_error()
    if err:
        return err

    try:
        # ── 1. Load strategy ──────────────────────────────────────────────────
        strat_result = await get_strategy.fn(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": strat_result.get("error", "Strategy not found")}
        strat = strat_result["strategy"]
        params = strat.get("parameters", {})
        is_mtf = "setup_tf" in params and "trigger_tf" in params

        # ── 2. Fetch daily price data (needed for IV computation and BS pricing) ──
        def _fetch_daily():
            return _fetch_price_data(symbol, start_date, end_date, timeframe="daily")

        price_df = await asyncio.get_event_loop().run_in_executor(None, _fetch_daily)
        if price_df is None or price_df.empty:
            return {"success": False, "error": f"No price data for {symbol}"}

        # ── 3. Get entry signals ──────────────────────────────────────────────
        # Run the underlying equity backtest to harvest entry dates and prices
        if is_mtf:
            eq_bt = await run_backtest_mtf.fn(
                strategy_id=strategy_id,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
            )
        else:
            eq_bt = await run_backtest.fn(
                strategy_id=strategy_id,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
            )

        if not eq_bt.get("success", False):
            return {"success": False, "error": f"Equity backtest failed: {eq_bt.get('error')}"}

        raw_trades = eq_bt.get("trades", [])
        if not raw_trades:
            return {
                "success": True,
                "total_trades": 0,
                "sharpe": 0.0,
                "note": "No equity signals in date range — options backtest empty.",
                "equity_sharpe": eq_bt.get("sharpe", eq_bt.get("sharpe_ratio", 0)),
            }

        # ── 4. Black-Scholes pricing helpers ──────────────────────────────────
        def _norm_cdf(x: float) -> float:
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> tuple:
            """Returns (price, delta). T in years."""
            if T <= 1e-6 or sigma <= 1e-6:
                intrinsic = max(S - K, 0.0)
                return intrinsic, (1.0 if S > K else 0.0)
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
            delta = _norm_cdf(d1)
            return max(price, 0.0), delta

        def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> tuple:
            """Returns (price, delta). T in years."""
            if T <= 1e-6 or sigma <= 1e-6:
                intrinsic = max(K - S, 0.0)
                return intrinsic, (-1.0 if S < K else 0.0)
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
            delta = _norm_cdf(d1) - 1
            return max(price, 0.0), delta

        price_fn = _bs_call if option_type.lower() == "call" else _bs_put

        # ── 5. Compute 20-day realised vol and rolling vol-rank ───────────────
        log_rets = np.log(price_df["close"] / price_df["close"].shift(1)).fillna(0)
        rv_20    = log_rets.rolling(20).std() * math.sqrt(252)
        rv_min   = rv_20.rolling(252, min_periods=60).min()
        rv_max   = rv_20.rolling(252, min_periods=60).max()
        vol_rank = (rv_20 - rv_min) / (rv_max - rv_min + 1e-9)

        # ── 6. Options backtest loop ──────────────────────────────────────────
        opt_trades = []

        for eq_trade in raw_trades:
            entry_date_str = eq_trade.get("entry_date") or eq_trade.get("entry_ts")
            entry_price    = eq_trade.get("entry_price")
            if not entry_date_str or not entry_price:
                continue

            entry_ts = pd.Timestamp(entry_date_str).normalize()

            # Get closes from entry_ts onward
            future_bars = price_df.loc[price_df.index >= entry_ts].head(time_stop_days + 5)
            if future_bars.empty:
                continue

            # IV at entry: 20-day realised vol
            try:
                iv_at_entry = float(rv_20.loc[:entry_ts].iloc[-1])
            except (IndexError, KeyError):
                iv_at_entry = 0.20
            if math.isnan(iv_at_entry) or iv_at_entry < 0.05:
                iv_at_entry = 0.15

            # Vol-rank filter
            try:
                rank_at_entry = float(vol_rank.loc[:entry_ts].iloc[-1])
            except (IndexError, KeyError):
                rank_at_entry = 0.5
            if not math.isnan(rank_at_entry) and rank_at_entry > iv_rank_max:
                continue  # IV too high — skip

            # Price ATM option at entry
            K           = round(entry_price)   # nearest dollar ≈ ATM
            T_entry     = expiry_days / 365.0
            entry_prem, entry_delta = price_fn(entry_price, K, T_entry, risk_free_rate, iv_at_entry)
            if entry_prem <= 0.01:
                continue

            contracts = max(1, int((initial_capital * position_size_pct) / (entry_prem * 100)))

            # Simulate day-by-day exit
            exit_prem   = None
            exit_reason = None
            exit_date   = None
            iv_at_exit  = iv_at_entry

            for days_held, (ts, row) in enumerate(future_bars.iterrows()):
                if days_held == 0:
                    continue
                T_rem = max((expiry_days - days_held) / 365.0, 1 / 365.0)
                try:
                    iv_now = float(rv_20.loc[:ts].iloc[-1])
                    if math.isnan(iv_now) or iv_now < 0.05:
                        iv_now = iv_at_entry
                except (IndexError, KeyError):
                    iv_now = iv_at_entry

                current_prem, _ = price_fn(row["close"], K, T_rem, risk_free_rate, iv_now)
                pnl_pct = (current_prem - entry_prem) / entry_prem

                if pnl_pct >= tp_pct:
                    exit_prem   = current_prem
                    exit_reason = "tp"
                    exit_date   = ts
                    iv_at_exit  = iv_now
                    break
                if pnl_pct <= -sl_pct:
                    exit_prem   = entry_prem * (1 - sl_pct)
                    exit_reason = "sl"
                    exit_date   = ts
                    iv_at_exit  = iv_now
                    break
                if days_held >= time_stop_days:
                    exit_prem   = current_prem
                    exit_reason = "time_stop"
                    exit_date   = ts
                    iv_at_exit  = iv_now
                    break

            if exit_prem is None:
                # Ran out of data
                last_ts, last_row = list(future_bars.iterrows())[-1]
                T_rem = max((expiry_days - len(future_bars)) / 365.0, 1 / 365.0)
                exit_prem, _ = price_fn(last_row["close"], K, T_rem, risk_free_rate, iv_at_entry)
                exit_reason  = "time_stop"
                exit_date    = last_ts

            premium_return_pct = (exit_prem - entry_prem) / entry_prem * 100
            capital_return_pct = (exit_prem - entry_prem) * 100 * contracts / initial_capital * 100

            opt_trades.append({
                "entry_date":          str(entry_ts.date()),
                "exit_date":           str(exit_date.date()) if exit_date else None,
                "entry_price":         round(entry_price, 2),
                "strike":              K,
                "entry_premium":       round(entry_prem, 3),
                "exit_premium":        round(exit_prem, 3),
                "entry_delta":         round(entry_delta, 3),
                "iv_at_entry_pct":     round(iv_at_entry * 100, 1),
                "iv_at_exit_pct":      round(iv_at_exit * 100, 1),
                "iv_crush":            iv_at_exit < iv_at_entry,
                "contracts":           contracts,
                "premium_return_pct":  round(premium_return_pct, 1),
                "capital_return_pct":  round(capital_return_pct, 2),
                "exit_reason":         exit_reason,
            })

        if not opt_trades:
            return {
                "success":       True,
                "total_trades":  0,
                "sharpe":        0.0,
                "note":          f"All signals filtered by iv_rank_max={iv_rank_max}.",
                "equity_sharpe": eq_bt.get("sharpe", eq_bt.get("sharpe_ratio", 0)),
            }

        # ── 7. Aggregate metrics ──────────────────────────────────────────────
        cap_returns = [t["capital_return_pct"] / 100 for t in opt_trades]
        wins        = [r for r in cap_returns if r > 0]
        losses      = [r for r in cap_returns if r <= 0]
        iv_crush_n  = sum(1 for t in opt_trades if t["iv_crush"])

        sharpe = (
            (np.mean(cap_returns) / (np.std(cap_returns) + 1e-9))
            * math.sqrt(252 / max(time_stop_days, 1))
        )

        result = {
            "success":                True,
            "strategy_id":            strategy_id,
            "symbol":                 symbol,
            "option_type":            option_type,
            "expiry_days":            expiry_days,
            "sharpe":                 round(sharpe, 3),
            "win_rate":               round(len(wins) / len(cap_returns), 3),
            "total_trades":           len(opt_trades),
            "profit_factor":          round(sum(wins) / (abs(sum(losses)) + 1e-9), 2),
            "avg_premium_return_pct": round(np.mean([t["premium_return_pct"] for t in opt_trades]), 1),
            "avg_capital_return_pct": round(np.mean(cap_returns) * 100, 2),
            "avg_iv_at_entry_pct":    round(np.mean([t["iv_at_entry_pct"] for t in opt_trades]), 1),
            "iv_crush_pct":           round(iv_crush_n / len(opt_trades) * 100, 1),
            "iv_rank_filter_applied": iv_rank_max < 1.0,
            "trades_filtered_by_iv":  len(raw_trades) - len(opt_trades),
            "equity_comparison": {
                "sharpe":       eq_bt.get("sharpe", eq_bt.get("sharpe_ratio", 0)),
                "win_rate":     eq_bt.get("win_rate", 0),
                "total_trades": eq_bt.get("total_trades", 0),
            },
            "trades": opt_trades,
        }

        # Persist summary alongside strategy
        ctx.db.execute(
            "UPDATE strategies SET updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
            [strategy_id],
        )
        logger.info(
            f"[quantpod_mcp] run_backtest_options({symbol}): "
            f"{len(opt_trades)} trades, Sharpe={result['sharpe']}, "
            f"avg_premium_return={result['avg_premium_return_pct']}%"
        )
        return result

    except Exception as e:
        logger.error(f"[quantpod_mcp] run_backtest_options failed: {e}")
        return {"success": False, "error": str(e)}
