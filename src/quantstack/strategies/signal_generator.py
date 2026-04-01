"""Signal generation from strategy rules — core logic used by backtesting, GP, and param optimization.

This module contains the rule-to-signal pipeline that was previously embedded
in the MCP backtesting tool. Extracted here to break the circular dependency
(grammar_gp → mcp.tools.backtesting → mcp.server → mcp.tools.backtesting).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from datetime import datetime, timedelta

from quantstack.config.settings import get_settings
from quantstack.config.timeframes import Timeframe
from quantstack.data.base import AssetClass
from quantstack.data.registry import DataProviderRegistry
from quantstack.data.storage import DataStore
from quantstack.features.enricher import FeatureEnricher


def generate_signals_from_rules(
    price_data: pd.DataFrame,
    entry_rules: list[dict[str, Any]],
    exit_rules: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Generate a signals DataFrame from strategy rules + price data.

    Supports:
      - Indicators: SMA, RSI, ATR, BBands, Z-score, breakout, ADX, CCI,
        Stochastic, price_vs_sma200, regime classification
      - Rule hierarchy: prerequisite (AND gate), confirmation (N-of-M),
        plain (OR, backward-compatible)
      - Exit management: time stops, ATR-based stop-loss/take-profit via
        forward simulation that maintains position state

    Returns DataFrame with 'signal' (0/1) and 'signal_direction' (LONG/SHORT/NONE).
    """
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
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
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
    plus_di = 100 * (
        plus_dm.ewm(span=adx_period, adjust=False).mean() / (atr_for_adx + 1e-10)
    )
    minus_di = 100 * (
        minus_dm.ewm(span=adx_period, adjust=False).mean() / (atr_for_adx + 1e-10)
    )
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

    # ── Price vs SMA200 ─────────────────────────────────────────────────
    df["price_vs_sma200"] = ((close - df["sma_200"]) / (df["sma_200"] + 1e-10)) * 100

    # ── Regime (derived from ADX + DI direction) ─────────────────────────
    df["regime"] = "ranging"
    df.loc[(df["adx"] >= 25) & (df["plus_di"] > df["minus_di"]), "regime"] = (
        "trending_up"
    )
    df.loc[(df["adx"] >= 25) & (df["minus_di"] > df["plus_di"]), "regime"] = (
        "trending_down"
    )

    # ── ATR percentile (rolling 252-day rank) ────────────────────────────
    df["atr_percentile"] = (
        df["atr"]
        .rolling(252)
        .apply(
            lambda x: (
                pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) == 252 else 50
            ),
            raw=False,
        )
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

    # ── Enrich with fundamental/macro/flow features on-demand ────────────
    try:
        enricher = FeatureEnricher()
        tiers = enricher.detect_needed_tiers(entry_rules + exit_rules)
        if tiers.any_active():
            symbol = str(parameters.get("symbol", ""))
            if symbol:
                df = enricher.enrich(df, symbol=symbol, tiers=tiers)
    except Exception as exc:
        logger.debug(f"Feature enrichment skipped: {exc}")

    # ── Evaluate entry rules with prerequisite/confirmation hierarchy ────
    prerequisite_rules = [r for r in entry_rules if r.get("type") == "prerequisite"]
    confirmation_rules = [r for r in entry_rules if r.get("type") == "confirmation"]
    plain_rules = [
        r for r in entry_rules if r.get("type") not in ("prerequisite", "confirmation")
    ]

    prereq_pass = pd.Series(True, index=df.index)
    for rule in prerequisite_rules:
        prereq_pass = prereq_pass & evaluate_rule(df, rule, parameters)

    min_confirmations = int(parameters.get("min_confirmations_required", 1))
    if confirmation_rules:
        confirmation_count = pd.Series(0, index=df.index, dtype=int)
        for rule in confirmation_rules:
            confirmation_count = confirmation_count + evaluate_rule(
                df, rule, parameters
            ).astype(int)
        confirm_pass = confirmation_count >= min_confirmations
    else:
        confirm_pass = pd.Series(True, index=df.index)

    entry_long = pd.Series(False, index=df.index)
    entry_short = pd.Series(False, index=df.index)
    default_direction = parameters.get("direction", "LONG").lower()
    for rule in plain_rules:
        cond = evaluate_rule(df, rule, parameters)
        direction = rule.get("direction", default_direction).lower()
        if direction == "long":
            entry_long = entry_long | cond
        elif direction == "short":
            entry_short = entry_short | cond

    if prerequisite_rules or confirmation_rules:
        structured_entry = prereq_pass & confirm_pass
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

    exit_signal = pd.Series(False, index=df.index)
    structural_exit_types = {"time_stop", "take_profit", "stop_loss", "event_blackout"}
    for rule in exit_rules:
        if rule.get("type") in structural_exit_types:
            continue
        exit_signal = exit_signal | evaluate_rule(df, rule, parameters)

    # ── Enforce direction constraint ─────────────────────────────────────
    # When the strategy declares a single direction (e.g. SHORT-only), suppress
    # the opposite side.  This prevents bidirectional signals from strategies
    # that intend to trade only one direction.
    direction_constraint = parameters.get("direction", "").upper()
    if direction_constraint == "SHORT":
        entry_long = pd.Series(False, index=df.index)
    elif direction_constraint == "LONG":
        entry_short = pd.Series(False, index=df.index)

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
    needs_simulation = (
        time_stop_days is not None or tp_atr_mult is not None or sl_atr_mult is not None
    )
    if needs_simulation:
        _simulate_exits(signals, df, time_stop_days, tp_atr_mult, sl_atr_mult)

    return signals


def _simulate_exits(
    signals: pd.DataFrame,
    df: pd.DataFrame,
    time_stop_days: int | None,
    tp_atr_mult: float | None,
    sl_atr_mult: float | None,
) -> None:
    """Forward-simulate position-aware exits (time stop, TP, SL). Mutates signals in place."""
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
    entry_direction = "LONG"

    for i in range(len(df)):
        if not in_position:
            if sig_vals[i] == 1:
                in_position = True
                entry_bar = i
                entry_price = closes[i]
                entry_atr = atrs[i] if not np.isnan(atrs[i]) else 0.0
                entry_direction = dir_vals[i]
        else:
            bars_held = i - entry_bar
            should_exit = False

            if time_stop_days and bars_held >= time_stop_days:
                should_exit = True
            if tp_atr_mult and entry_atr > 0 and not should_exit:
                if entry_direction == "SHORT":
                    if lows[i] <= entry_price - tp_atr_mult * entry_atr:
                        should_exit = True
                else:
                    if highs[i] >= entry_price + tp_atr_mult * entry_atr:
                        should_exit = True
            if sl_atr_mult and entry_atr > 0 and not should_exit:
                if entry_direction == "SHORT":
                    if highs[i] >= entry_price + sl_atr_mult * entry_atr:
                        should_exit = True
                else:
                    if lows[i] <= entry_price - sl_atr_mult * entry_atr:
                        should_exit = True

            if should_exit:
                sig_vals[i] = 0
                dir_vals[i] = "NONE"
                in_position = False
            else:
                sig_vals[i] = 1
                dir_vals[i] = entry_direction

    signals["signal"] = sig_vals
    signals["signal_direction"] = dir_vals


def evaluate_rule(
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
    from quantstack.strategies.rule_engine import _normalize_indicator

    indicator = _normalize_indicator(rule.get("indicator", ""))
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

    if condition == "within_pct":
        return series.abs() <= float(value)

    # Support column references (e.g. value="sma_50" means compare against df["sma_50"])
    def _resolve(v: Any) -> "pd.Series | float":
        if isinstance(v, str) and not v.replace(".", "", 1).replace("-", "", 1).isdigit():
            col = _normalize_indicator(v)
            return df[col] if col in df.columns else pd.Series(float("nan"), index=df.index)
        return float(v)

    rhs = _resolve(value)
    if condition in ("above", "greater_than"):
        return series > rhs
    elif condition in ("below", "less_than"):
        return series < rhs
    elif condition == "crosses_above":
        rhs_prev = rhs.shift(1) if isinstance(rhs, pd.Series) else rhs
        return (series.shift(1) <= rhs_prev) & (series > rhs)
    elif condition == "crosses_below":
        rhs_prev = rhs.shift(1) if isinstance(rhs, pd.Series) else rhs
        return (series.shift(1) >= rhs_prev) & (series < rhs)
    elif condition == "between":
        upper = float(rule.get("upper", value))
        lower = float(rule.get("lower", 0))
        return (series >= lower) & (series <= upper)
    else:
        return pd.Series(False, index=df.index)


def fetch_price_data(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    timeframe: str = "daily",
) -> pd.DataFrame | None:
    """Fetch OHLCV price data using the configured provider registry.

    Resolution order:
      1. Local PostgreSQL cache (fastest, no network)
      2. Provider registry (DATA_PROVIDER_PRIORITY from .env)

    Returns None if all sources fail.
    """
    _TF_MAP = {
        "daily": Timeframe.D1,
        "1d": Timeframe.D1,
        "1h": Timeframe.H1,
        "hourly": Timeframe.H1,
        "4h": Timeframe.H4,
        "30m": Timeframe.M30,
        "30min": Timeframe.M30,
        "15m": Timeframe.M15,
        "15min": Timeframe.M15,
        "5m": Timeframe.M5,
        "5min": Timeframe.M5,
        "1m": Timeframe.M1,
        "1min": Timeframe.M1,
    }
    tf = _TF_MAP.get(timeframe.lower(), Timeframe.D1)

    def _apply_date_filter(frame: pd.DataFrame) -> pd.DataFrame:
        if start_date:
            frame = frame[frame.index >= start_date]
        if end_date:
            frame = frame[frame.index <= end_date]
        return frame

    # 1. Local PostgreSQL cache
    try:
        with DataStore(read_only=True) as store:
            df = store.load_ohlcv(symbol, tf)
            if df is not None and not df.empty:
                return _apply_date_filter(df)
    except Exception as exc:
        logger.debug(f"PostgreSQL cache miss for {symbol} [{timeframe}]: {exc}")

    # 2. Provider registry
    try:
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
        logger.warning(
            f"Provider registry fetch failed for {symbol} [{timeframe}]: {exc}"
        )

    return None
