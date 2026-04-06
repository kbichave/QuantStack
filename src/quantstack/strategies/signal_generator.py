"""Signal generation from strategy rules — core logic used by backtesting, GP, and param optimization.

This module contains the rule-to-signal pipeline, extracted to break circular
dependencies and shared across backtesting, GP, and parameter optimization.
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
from quantstack.db import db_conn
from quantstack.features.enricher import FeatureEnricher
from quantstack.strategies.rule_engine import (
    _COMPOUND_CONDITIONS,
    _KNOWN_INDICATORS,
    _normalize_indicator,
)


def _rules_reference_signal(rules: list[dict[str, Any]], signal_name: str) -> bool:
    """Return True if any rule's 'indicator' key matches signal_name."""
    return any(r.get("indicator") == signal_name for r in rules)


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

    # ── Enrich macro/institutional signals referenced by rules ────────────
    # These signals live outside the DataFrame pipeline (capitulation_score
    # comes from capitulation tools; credit_regime from the macro
    # collector). Rather than pulling in those full dependency chains, we
    # compute lightweight proxies from the OHLCV data already in df and read
    # the DB-persisted value for credit_regime, then broadcast as scalar
    # columns so the rule engine can compare them normally.
    all_rules = entry_rules + exit_rules
    if _rules_reference_signal(all_rules, "capitulation_score"):
        try:
            # Proxy: fraction of bars in the last 20 days that are down AND
            # volume is above the 20-day average volume.  Normalized to 0-1.
            lookback_cap = 20
            down_bar = close.diff() < 0
            vol_above_avg = df["volume"] > df["volume"].rolling(lookback_cap).mean()
            cap_proxy = (down_bar & vol_above_avg).rolling(lookback_cap).mean()
            # Fill NaN (insufficient history) with 0 so the column is always present.
            df["capitulation_score"] = cap_proxy.fillna(0.0)
        except Exception as exc:
            logger.warning(f"capitulation_score proxy failed, defaulting to 0: {exc}")
            df["capitulation_score"] = 0.0

    if _rules_reference_signal(all_rules, "credit_regime"):
        try:
            credit_regime_val = "unknown"
            with db_conn() as conn:
                row = conn.execute(
                    "SELECT value FROM system_state WHERE key = 'credit_regime'"
                ).fetchone()
                if row:
                    credit_regime_val = str(row[0])
            df["credit_regime"] = credit_regime_val
        except Exception as exc:
            logger.warning(
                f"credit_regime DB read failed, defaulting to 'unknown': {exc}"
            )
            df["credit_regime"] = "unknown"

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

    # Detect if all plain rules are semantic (no indicator key).  Semantic
    # rules represent convergence criteria and should be ANDed, not ORed.
    all_semantic = plain_rules and all("indicator" not in r for r in plain_rules)
    if all_semantic:
        # AND-combine: all conditions must be true simultaneously
        and_mask = pd.Series(True, index=df.index)
        for rule in plain_rules:
            and_mask = and_mask & evaluate_rule(df, rule, parameters)
        direction = parameters.get("direction", "LONG").lower()
        if direction == "short":
            entry_short = and_mask
        else:
            entry_long = and_mask
    else:
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
    _ALL_EXIT_ALIASES = {
        "time_stop", "time_exit", "holding_period_exit",
        "take_profit", "profit_target", "price_target_hit",
        "take_profit_target", "take_profit_atr", "mean_reversion_target",
        "stop_loss", "stop_loss_hit", "stop_loss_atr", "hvn_break",
        "event_blackout", "trailing_stop",
    }
    time_stop_days: int | None = None
    tp_atr_mult: float | None = None
    sl_atr_mult: float | None = None
    for rule in exit_rules:
        # LLMs put exit type in "type", "condition", or "rule" key.
        # Check all three and use whichever matches a structural exit type.
        _candidates = [
            (rule.get("type", "") or "").strip().lower(),
            (rule.get("rule", "") or "").strip().lower(),
            (rule.get("condition", "") or "").strip().lower(),
        ]
        rule_type = ""
        for c in _candidates:
            if c in _ALL_EXIT_ALIASES:
                rule_type = c
                break
        if rule_type in ("time_stop", "time_exit", "holding_period_exit"):
            time_stop_days = int(rule.get("days", rule.get("bars", rule.get("max_bars", 5))))
        elif rule_type in ("take_profit", "profit_target", "price_target_hit",
                           "take_profit_target", "take_profit_atr", "mean_reversion_target"):
            tp_atr_mult = float(rule.get("atr_multiple", rule.get("multiplier", rule.get("target_pct", 2.5))))
        elif rule_type in ("stop_loss", "stop_loss_hit", "stop_loss_atr", "hvn_break"):
            sl_atr_mult = float(rule.get("atr_multiple", rule.get("multiplier", 1.5)))

    exit_signal = pd.Series(False, index=df.index)
    for rule in exit_rules:
        _exit_candidates = [
            (rule.get("type", "") or "").strip().lower(),
            (rule.get("rule", "") or "").strip().lower(),
            (rule.get("condition", "") or "").strip().lower(),
        ]
        if any(c in _ALL_EXIT_ALIASES for c in _exit_candidates):
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


def _evaluate_semantic_condition(
    df: pd.DataFrame,
    rule: dict[str, Any],
    parameters: dict[str, Any],
) -> pd.Series | None:
    """Translate LLM-generated semantic conditions to indicator logic.

    LLMs generate rules like {"condition": "bullish_trend", "sma_period": 50}
    without an "indicator" key.  This mapper converts the most common semantic
    conditions into real indicator evaluations using columns already in df.

    Returns a boolean Series if the condition is recognised, or None to fall
    through to the standard evaluate_rule path.
    """
    # Only activate when no indicator key is present
    if "indicator" in rule:
        return None

    cond = rule.get("condition", "")
    if not cond or not isinstance(cond, str):
        return None

    cond_lower = cond.strip().lower()

    # ── Trend conditions ─────────────────────────────────────────────────
    if cond_lower in ("bullish_trend", "bullish_trend_sma50", "neutral_to_bullish_trend",
                       "positive_trend_regime"):
        sma_period = int(rule.get("sma_period", 50))
        col = f"sma_{sma_period}"
        if col not in df.columns:
            df[col] = df["close"].rolling(sma_period).mean()
        if cond_lower == "neutral_to_bullish_trend":
            # Close above SMA OR ADX < 25 (not strongly trending down)
            above_sma = df["close"] > df[col]
            not_bearish = (df["minus_di"] <= df["plus_di"]) if "plus_di" in df.columns else pd.Series(True, index=df.index)
            return above_sma | not_bearish
        return df["close"] > df[col]

    # ── Volatility conditions ────────────────────────────────────────────
    if cond_lower in ("elevated_volatility", "volatility_context", "volatility_spike"):
        var_threshold = float(rule.get("var_threshold", 2.5))
        if "atr" in df.columns:
            atr_pct = (df["atr"] / (df["close"] + 1e-10)) * 100
            return atr_pct > var_threshold
        return pd.Series(True, index=df.index)  # Can't compute — permissive

    # ── Support/pullback conditions ──────────────────────────────────────
    if cond_lower in ("price_at_hvn_support", "price_pullback_to_hvn_support"):
        hvn_level = rule.get("hvn_level")
        tolerance = float(rule.get("tolerance_pct", 1.5))
        if hvn_level is not None:
            hvn = float(hvn_level)
            lower = hvn * (1 - tolerance / 100)
            upper = hvn * (1 + tolerance / 100)
            return (df["close"] >= lower) & (df["close"] <= upper)
        return pd.Series(True, index=df.index)

    if cond_lower in ("price_at_support",):
        # Generic support — close near 20-day low
        low_20 = df["low"].rolling(20).min()
        return df["close"] <= low_20 * 1.02

    if cond_lower in ("price_pullback_to_sma20",):
        if "sma_20" not in df.columns:
            df["sma_20"] = df["close"].rolling(20).mean()
        # Close within 1.5% of SMA20
        return ((df["close"] - df["sma_20"]).abs() / (df["sma_20"] + 1e-10)) * 100 <= 1.5

    if cond_lower == "price_near_bb_lower":
        if "bb_lower" in df.columns:
            return df["close"] <= df["bb_lower"] * 1.01
        return pd.Series(False, index=df.index)

    # ── RSI conditions ───────────────────────────────────────────────────
    if cond_lower == "rsi_oversold":
        threshold = float(rule.get("threshold", 30))
        if "rsi" in df.columns:
            return df["rsi"] < threshold
        return pd.Series(False, index=df.index)

    # ── Volume conditions ────────────────────────────────────────────────
    if cond_lower in ("volume_exhaustion", "volume_spike"):
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(20).mean()
            if cond_lower == "volume_exhaustion":
                return df["volume"] < vol_ma * 0.7
            return df["volume"] > vol_ma * 1.5
        return pd.Series(True, index=df.index)

    # ── Compound AND/OR expressions (e.g. "close <= 175 AND close >= 171") ─
    import re
    if " AND " in cond or " OR " in cond:
        joiner = " AND " if " AND " in cond else " OR "
        parts = cond.split(joiner)
        sub_results = []
        for part in parts:
            sub_rule = {**rule, "condition": part.strip()}
            sub_result = _evaluate_semantic_condition(df, sub_rule, parameters)
            if sub_result is not None:
                sub_results.append(sub_result)
        if sub_results:
            if joiner == " AND ":
                combined = sub_results[0]
                for sr in sub_results[1:]:
                    combined = combined & sr
            else:
                combined = sub_results[0]
                for sr in sub_results[1:]:
                    combined = combined | sr
            return combined

    # ── Expression-style conditions (e.g. "rsi_14 < 40", "close > 210") ─
    expr_match = re.match(
        r"^(\w+)\s*(>=|<=|>|<|==|!=)\s*([\d.]+)$", cond.strip()
    )
    if expr_match:
        ind_name, op, val_str = expr_match.groups()
        col = _normalize_indicator(ind_name)
        if col in df.columns:
            v = float(val_str)
            ops = {">=": df[col] >= v, "<=": df[col] <= v,
                   ">": df[col] > v, "<": df[col] < v,
                   "==": df[col] == v, "!=": df[col] != v}
            return ops.get(op, pd.Series(False, index=df.index))
        # Column not available — permissive (don't block AND chains)
        logger.debug(f"Semantic expression '{cond}' references unavailable column '{col}' — skipping (permissive)")
        return pd.Series(True, index=df.index)

    # ── Regime filter conditions ─────────────────────────────────────────
    if cond_lower.startswith("regime_trend") or cond_lower.startswith("regime_volatility"):
        if "regime" in df.columns:
            # Extract target value after == or "in"
            regime_match = re.search(r"==\s*(\w+)", cond)
            if regime_match:
                target = regime_match.group(1)
                return df["regime"] == target
            in_match = re.search(r"in\s*\[(.+)\]", cond)
            if in_match:
                targets = [t.strip().strip("'\"") for t in in_match.group(1).split(",")]
                return df["regime"].isin(targets)
        return pd.Series(True, index=df.index)  # Can't filter — permissive

    # Not recognised — if rule has no indicator key, be permissive rather
    # than returning None (which would fall to evaluate_rule and return False,
    # killing AND chains).
    if "indicator" not in rule:
        logger.warning(
            f"Unrecognised semantic condition '{cond}' — skipping (permissive). "
            "Add a mapper in _evaluate_semantic_condition if this should filter."
        )
        return pd.Series(True, index=df.index)

    # Not recognised — fall through to standard path
    return None


def evaluate_rule(
    df: pd.DataFrame,
    rule: dict[str, Any],
    parameters: dict[str, Any],
) -> pd.Series:
    """Evaluate a single rule dict against the indicator DataFrame.

    Supports:
      - Semantic conditions (no indicator key): bullish_trend, elevated_volatility,
        price_at_hvn_support, rsi_oversold, volume_exhaustion, etc.
      - Any pre-computed column name as indicator
      - Special indicators: sma_crossover, breakout, regime
      - Conditions: above, below, crosses_above, crosses_below, between,
        within_pct (absolute value <=), not_in / in (for string columns)
    """
    # ── Semantic condition mapper (rules without indicator key) ───────────
    semantic_result = _evaluate_semantic_condition(df, rule, parameters)
    if semantic_result is not None:
        return semantic_result

    indicator = _normalize_indicator(rule.get("indicator", ""))
    condition = rule.get("condition", "")
    value = rule.get("value")

    # ── Compound condition expansion ────────────────────────────────────
    # LLMs generate conditions like "below_lower_band" which combine an
    # indicator + condition + reference into a single string.  Rewrite them
    # to canonical (indicator, condition, value) before standard evaluation.
    if condition in _COMPOUND_CONDITIONS:
        comp_ind, condition, value = _COMPOUND_CONDITIONS[condition]
        indicator = _normalize_indicator(comp_ind)
        logger.debug(
            f"Expanded compound condition → indicator={indicator}, "
            f"condition={condition}, value={value}"
        )

    # ── Auto-compute dynamic SMA columns ────────────────────────────────
    # If indicator is sma_XX and not yet in df, compute it on the fly.
    if indicator.startswith("sma_") and indicator not in df.columns:
        suffix = indicator[4:]
        if suffix.isdigit():
            period = int(suffix)
            df[indicator] = df["close"].rolling(period).mean()
            logger.debug(f"Auto-computed {indicator} (period={period})")

    # Also auto-compute sma columns referenced as value (e.g. value="SMA_50")
    if isinstance(value, str):
        norm_value = _normalize_indicator(value)
        if norm_value.startswith("sma_") and norm_value not in df.columns:
            suffix = norm_value[4:]
            if suffix.isdigit():
                period = int(suffix)
                df[norm_value] = df["close"].rolling(period).mean()
                logger.debug(f"Auto-computed reference {norm_value} (period={period})")

    # ── Special: regime_disagreement ────────────────────────────────────
    # LLMs generate this to mean "the regime classification is uncertain".
    # In backtesting context we don't have two competing classifiers, so
    # we proxy it: True when ADX is near the trending threshold (20-30),
    # meaning the trend/range classification is ambiguous.
    if indicator == "regime_disagreement":
        if condition == "true":
            if "adx" in df.columns:
                return (df["adx"] >= 20) & (df["adx"] <= 30)
            return pd.Series(True, index=df.index)
        return pd.Series(False, index=df.index)

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
        if indicator not in _KNOWN_INDICATORS:
            logger.warning(
                f"evaluate_rule: indicator '{indicator}' is not in df.columns and not "
                "in _KNOWN_INDICATORS — rule will always return False. "
                "Add enrichment for this signal or check the rule definition."
            )
        return pd.Series(False, index=df.index)

    # ── Evaluate condition ───────────────────────────────────────────────
    # LLMs sometimes put the comparison target in 'reference' instead of 'value'
    if value is None:
        value = rule.get("reference")
    if value is None:
        # Check for threshold in rule-specific fields (e.g. zscore threshold)
        for fallback_key in ("threshold", "level", "zscore"):
            if fallback_key in rule and fallback_key != "indicator":
                value = rule[fallback_key]
                break
    if value is None:
        return pd.Series(False, index=df.index)

    # Support column references (e.g. value="sma_50" means compare against df["sma_50"])
    def _resolve(v: Any) -> "pd.Series | float":
        if isinstance(v, (list, dict)):
            logger.warning(f"evaluate_rule: value is {type(v).__name__}, expected scalar — returning NaN")
            return float("nan")
        if isinstance(v, str) and not v.replace(".", "", 1).replace("-", "", 1).isdigit():
            col = _normalize_indicator(v)
            # Auto-compute dynamic SMA if not yet in df
            if col not in df.columns and col.startswith("sma_") and col[4:].isdigit():
                period = int(col[4:])
                df[col] = df["close"].rolling(period).mean()
            return df[col] if col in df.columns else pd.Series(float("nan"), index=df.index)
        return float(v)

    # List membership — handle before _resolve since value is a list, not a scalar/column
    if condition in ("in", "not_in") and isinstance(value, list):
        result = series.isin(value)
        return ~result if condition == "not_in" else result

    if condition == "within_pct":
        # value is the reference (column name or float), pct_range/tolerance is the threshold
        pct = rule.get("pct_range", rule.get("tolerance", None))
        rhs = _resolve(value)
        if pct is not None and isinstance(rhs, pd.Series):
            # e.g. close within 3% of sma_200: |close - sma_200| / sma_200 <= 0.03
            return ((series - rhs).abs() / (rhs.abs() + 1e-10)) * 100 <= float(pct)
        # Fallback: value is a numeric pct threshold, series is already pct-like
        return series.abs() <= _resolve(value)

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
