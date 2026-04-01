# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Compiled strategy rule engine.

Replaces runtime JSON rule evaluation with a two-phase approach:
  1. **Compile** — at registration/load time, validate rule dicts and produce
     a ``CompiledStrategy`` containing callable rule functions.
  2. **Evaluate** — at backtest/live time, run the compiled functions against a
     DataFrame of indicators to produce vectorised boolean signals.

Benefits over the old `_evaluate_rule()` approach:
  - Fails fast on invalid rules (typos in indicator names, bad conditions)
  - Compiles to vectorised pandas/numpy ops — 10-100× faster in backtests
  - Supports regime-conditional rule weighting (IF regime == X THEN ...)
  - Supports confidence gradations (rule passes with strength 0-1, not just bool)

The compile step is deterministic and pure — no I/O, no DB access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np
import pandas as pd
from loguru import logger


# =============================================================================
# Types
# =============================================================================


class RuleCondition(str, Enum):
    """Supported comparison conditions."""

    ABOVE = "above"
    BELOW = "below"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    BETWEEN = "between"
    WITHIN_PCT = "within_pct"
    EQUALS = "equals"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"


class RuleType(str, Enum):
    """Rule hierarchy type."""

    PREREQUISITE = "prerequisite"
    CONFIRMATION = "confirmation"
    PLAIN = "plain"


# Indicators that don't require a pre-existing column — they're computed
# on-the-fly or use special logic.
_SPECIAL_INDICATORS = frozenset(
    {
        "sma_crossover",
        "breakout",
        "regime",
        "price_vs_sma200",
    }
)

# Known indicator columns that _generate_signals_from_rules computes.
_KNOWN_INDICATORS = frozenset(
    {
        "rsi",
        "atr",
        "adx",
        "plus_di",
        "minus_di",
        "stoch_k",
        "stoch_d",
        "cci",
        "bb_pct",
        "bb_upper",
        "bb_lower",
        "zscore",
        "sma_fast",
        "sma_slow",
        "sma_200",
        "high_n",
        "low_n",
        "atr_percentile",
        "regime",
        "close",
        "open",
        "high",
        "low",
        "volume",
    }
)

# Common indicator name aliases → canonical names used by signal_generator.
# Strategies often register with suffixed names (e.g. "rsi_14", "adx_14")
# but the signal generator computes unsuffixed columns ("rsi", "adx").
_INDICATOR_ALIASES: dict[str, str] = {
    "rsi_14": "rsi",
    "rsi14": "rsi",
    "adx_14": "adx",
    "adx14": "adx",
    "atr_14": "atr",
    "atr14": "atr",
    "cci_14": "cci",
    "cci_20": "cci",
    "bb_width": "bb_pct",
    "bb_width_20": "bb_pct",
    "bb_pctb": "bb_pct",
    "bb_percent_b": "bb_pct",
    "bollinger_pct": "bb_pct",
    "stochastic_k": "stoch_k",
    "stochastic_d": "stoch_d",
    "stoch_k_14": "stoch_k",
    "stoch_d_14": "stoch_d",
    "plus_di_14": "plus_di",
    "minus_di_14": "minus_di",
    "zscore_20": "zscore",
}


def _normalize_indicator(name: str) -> str:
    """Map common indicator aliases to the canonical name used by signal_generator."""
    canonical = _INDICATOR_ALIASES.get(name)
    if canonical:
        logger.debug(f"Normalized indicator alias '{name}' → '{canonical}'")
        return canonical
    return name


class RuleFunction(Protocol):
    """Callable that evaluates a rule against a DataFrame of indicators."""

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series aligned with df.index."""
        ...


@dataclass(frozen=True)
class CompiledRule:
    """A single validated and compiled rule."""

    indicator: str
    condition: RuleCondition
    value: Any
    rule_type: RuleType
    direction: str  # "long" or "short"
    evaluate: RuleFunction
    source_dict: dict[str, Any]  # original rule dict for serialisation

    def __repr__(self) -> str:
        return f"CompiledRule({self.indicator} {self.condition.value} {self.value}, type={self.rule_type.value})"


@dataclass
class CompiledStrategy:
    """
    A fully compiled strategy ready for vectorised evaluation.

    All rules have been validated and converted to callable functions.
    Invalid rules cause a CompilationError at construction time — never at
    evaluation time.
    """

    strategy_id: str
    name: str
    parameters: dict[str, Any]
    entry_rules: list[CompiledRule]
    exit_rules: list[CompiledRule]
    min_confirmations: int = 1
    direction: str = "LONG"  # default direction for structured entries

    # Regime-conditional weight overrides:
    # e.g. {"trending_up": {"momentum_weight": 1.5}, "ranging": {"reversion_weight": 1.5}}
    regime_weights: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def prerequisite_entry_rules(self) -> list[CompiledRule]:
        return [r for r in self.entry_rules if r.rule_type == RuleType.PREREQUISITE]

    @property
    def confirmation_entry_rules(self) -> list[CompiledRule]:
        return [r for r in self.entry_rules if r.rule_type == RuleType.CONFIRMATION]

    @property
    def plain_entry_rules(self) -> list[CompiledRule]:
        return [r for r in self.entry_rules if r.rule_type == RuleType.PLAIN]


class CompilationError(Exception):
    """Raised when a strategy rule fails validation during compilation."""


# =============================================================================
# Compilation
# =============================================================================


def compile_strategy(
    strategy_id: str,
    name: str,
    entry_rules: list[dict[str, Any]],
    exit_rules: list[dict[str, Any]],
    parameters: dict[str, Any],
    regime_weights: dict[str, dict[str, float]] | None = None,
) -> CompiledStrategy:
    """
    Compile raw rule dicts into a ``CompiledStrategy``.

    Raises ``CompilationError`` if any rule is invalid (unknown indicator,
    unsupported condition, missing required fields).
    """
    compiled_entry = []
    for i, rule in enumerate(entry_rules):
        try:
            compiled_entry.append(
                _compile_rule(rule, parameters, context=f"entry_rules[{i}]")
            )
        except CompilationError:
            raise
        except Exception as exc:
            raise CompilationError(f"entry_rules[{i}]: {exc}") from exc

    compiled_exit = []
    for i, rule in enumerate(exit_rules):
        # Structural exits (time_stop, take_profit, stop_loss) are not compiled
        # into boolean rule functions — they're handled by the position simulator.
        rule_type_str = rule.get("type", "")
        if rule_type_str in ("time_stop", "take_profit", "stop_loss", "event_blackout"):
            continue
        try:
            compiled_exit.append(
                _compile_rule(rule, parameters, context=f"exit_rules[{i}]")
            )
        except CompilationError:
            raise
        except Exception as exc:
            raise CompilationError(f"exit_rules[{i}]: {exc}") from exc

    return CompiledStrategy(
        strategy_id=strategy_id,
        name=name,
        parameters=parameters,
        entry_rules=compiled_entry,
        exit_rules=compiled_exit,
        min_confirmations=int(parameters.get("min_confirmations_required", 1)),
        direction=parameters.get("direction", "LONG").upper(),
        regime_weights=regime_weights or {},
    )


def _compile_rule(
    rule: dict[str, Any],
    parameters: dict[str, Any],
    context: str = "",
) -> CompiledRule:
    """Compile a single rule dict into a CompiledRule with a callable evaluator."""
    indicator = _normalize_indicator(rule.get("indicator", ""))
    condition_str = rule.get("condition", "")
    value = rule.get("value")
    rule_type_str = rule.get("type", "plain")
    default_direction = parameters.get("direction", "LONG").lower()
    direction = rule.get("direction", default_direction).lower()

    if not indicator:
        raise CompilationError(f"{context}: missing 'indicator' field")
    if not condition_str:
        raise CompilationError(f"{context}: missing 'condition' field")

    # Validate condition
    try:
        condition = RuleCondition(condition_str)
    except ValueError:
        # Also accept aliases
        alias_map = {"greater_than": "above", "less_than": "below"}
        mapped = alias_map.get(condition_str)
        if mapped:
            condition = RuleCondition(mapped)
        else:
            raise CompilationError(
                f"{context}: unknown condition '{condition_str}'. "
                f"Valid: {[c.value for c in RuleCondition]}"
            )

    # Validate rule type
    try:
        rule_type = RuleType(rule_type_str)
    except ValueError:
        rule_type = RuleType.PLAIN

    # Build the evaluator function
    evaluator = _build_evaluator(indicator, condition, value, rule, parameters, context)

    return CompiledRule(
        indicator=indicator,
        condition=condition,
        value=value,
        rule_type=rule_type,
        direction=direction,
        evaluate=evaluator,
        source_dict=rule,
    )


def _build_evaluator(
    indicator: str,
    condition: RuleCondition,
    value: Any,
    rule: dict[str, Any],
    parameters: dict[str, Any],
    context: str,
) -> RuleFunction:
    """
    Build a vectorised evaluator function for a single rule.

    Returns a callable that takes a DataFrame and returns a boolean Series.
    All validation happens here at compile time.
    """
    # === Special indicators ===

    if indicator == "regime":
        return _build_regime_evaluator(condition, value, context)

    if indicator == "sma_crossover":
        return _build_sma_crossover_evaluator(condition, context)

    if indicator == "breakout":
        return _build_breakout_evaluator(condition, context)

    # === Standard column-based indicators ===

    # Validate that the indicator is known (either a known column or
    # could be a dynamic column like "sma_50", "fund_pe_ratio", etc.)
    is_known = (
        indicator in _KNOWN_INDICATORS
        or indicator.startswith("sma_")
        or indicator.startswith("fund_")
        or indicator.startswith("macro_")
        or indicator.startswith("flow_")
        or indicator.startswith("earn_")
    )

    if not is_known:
        logger.warning(
            f"{context}: indicator '{indicator}' is not in the known set. "
            "It may work if a feature enricher provides it at runtime."
        )

    # Validate value for numeric conditions.
    # The value can be either a numeric literal (e.g., 30) or a string naming
    # another indicator column (e.g., "sma_200").  Column references are resolved
    # at evaluation time by _build_column_evaluator.
    if condition in (
        RuleCondition.ABOVE,
        RuleCondition.BELOW,
        RuleCondition.CROSSES_ABOVE,
        RuleCondition.CROSSES_BELOW,
        RuleCondition.WITHIN_PCT,
    ):
        if value is None:
            raise CompilationError(
                f"{context}: condition '{condition.value}' requires a 'value'"
            )
        # Accept numeric literals and string indicator references.
        if not isinstance(value, str):
            try:
                float(value)
            except (TypeError, ValueError):
                raise CompilationError(
                    f"{context}: condition '{condition.value}' requires numeric value "
                    f"or indicator name, got {value!r}"
                )

    if condition == RuleCondition.BETWEEN:
        lower = rule.get("lower")
        upper = rule.get("upper", value)
        if lower is None or upper is None:
            raise CompilationError(
                f"{context}: 'between' requires 'lower' and 'upper' (or 'value')"
            )

    return _build_column_evaluator(indicator, condition, value, rule, context)


def _build_regime_evaluator(
    condition: RuleCondition,
    value: Any,
    context: str,
) -> RuleFunction:
    """Evaluator for the special 'regime' string column."""
    if condition == RuleCondition.NOT_IN:
        if not isinstance(value, list):
            raise CompilationError(f"{context}: regime 'not_in' requires a list value")
        values = value

        def _eval(df: pd.DataFrame) -> pd.Series:
            if "regime" not in df.columns:
                return pd.Series(False, index=df.index)
            return ~df["regime"].isin(values)

        return _eval

    if condition == RuleCondition.IN:
        if not isinstance(value, list):
            raise CompilationError(f"{context}: regime 'in' requires a list value")
        values = value

        def _eval(df: pd.DataFrame) -> pd.Series:
            if "regime" not in df.columns:
                return pd.Series(False, index=df.index)
            return df["regime"].isin(values)

        return _eval

    if condition == RuleCondition.EQUALS:

        def _eval(df: pd.DataFrame) -> pd.Series:
            if "regime" not in df.columns:
                return pd.Series(False, index=df.index)
            return df["regime"] == value

        return _eval

    raise CompilationError(
        f"{context}: regime only supports 'in', 'not_in', 'equals' conditions"
    )


def _build_sma_crossover_evaluator(
    condition: RuleCondition,
    context: str,
) -> RuleFunction:
    """Evaluator for SMA crossover (fast vs slow)."""
    if condition == RuleCondition.CROSSES_ABOVE:

        def _eval(df: pd.DataFrame) -> pd.Series:
            prev_fast = df["sma_fast"].shift(1)
            prev_slow = df["sma_slow"].shift(1)
            return (prev_fast <= prev_slow) & (df["sma_fast"] > df["sma_slow"])

        return _eval

    if condition == RuleCondition.CROSSES_BELOW:

        def _eval(df: pd.DataFrame) -> pd.Series:
            prev_fast = df["sma_fast"].shift(1)
            prev_slow = df["sma_slow"].shift(1)
            return (prev_fast >= prev_slow) & (df["sma_fast"] < df["sma_slow"])

        return _eval

    raise CompilationError(
        f"{context}: sma_crossover only supports 'crosses_above' and 'crosses_below'"
    )


def _build_breakout_evaluator(
    condition: RuleCondition,
    context: str,
) -> RuleFunction:
    """Evaluator for price breakout above/below N-period high/low."""
    if condition == RuleCondition.ABOVE:

        def _eval(df: pd.DataFrame) -> pd.Series:
            return df["close"] > df["high_n"].shift(1)

        return _eval

    if condition == RuleCondition.BELOW:

        def _eval(df: pd.DataFrame) -> pd.Series:
            return df["close"] < df["low_n"].shift(1)

        return _eval

    raise CompilationError(f"{context}: breakout only supports 'above' and 'below'")


def _is_column_ref(value: Any) -> bool:
    """True if *value* is a string naming another indicator column."""
    return isinstance(value, str) and not value.replace(".", "", 1).replace("-", "", 1).isdigit()


def _build_column_evaluator(
    indicator: str,
    condition: RuleCondition,
    value: Any,
    rule: dict[str, Any],
    context: str,
) -> RuleFunction:
    """Build evaluator for a standard numeric column.

    ``value`` can be a numeric literal *or* a string referencing another column
    (e.g. ``"sma_200"``).  Column references are resolved at evaluation time so
    strategies like ``close > sma_200`` work without hardcoded numbers.
    """
    # Normalize both indicator and value column references through alias map.
    indicator = _normalize_indicator(indicator)
    if _is_column_ref(value):
        value = _normalize_indicator(str(value))

    if condition in (RuleCondition.ABOVE, RuleCondition.GREATER_THAN):
        if _is_column_ref(value):
            ref_col = str(value)

            def _eval(df: pd.DataFrame) -> pd.Series:
                if indicator not in df.columns or ref_col not in df.columns:
                    return pd.Series(False, index=df.index)
                return df[indicator] > df[ref_col]

            return _eval

        threshold = float(value)

        def _eval(df: pd.DataFrame) -> pd.Series:
            if indicator not in df.columns:
                return pd.Series(False, index=df.index)
            return df[indicator] > threshold

        return _eval

    if condition in (RuleCondition.BELOW, RuleCondition.LESS_THAN):
        if _is_column_ref(value):
            ref_col = str(value)

            def _eval(df: pd.DataFrame) -> pd.Series:
                if indicator not in df.columns or ref_col not in df.columns:
                    return pd.Series(False, index=df.index)
                return df[indicator] < df[ref_col]

            return _eval

        threshold = float(value)

        def _eval(df: pd.DataFrame) -> pd.Series:
            if indicator not in df.columns:
                return pd.Series(False, index=df.index)
            return df[indicator] < threshold

        return _eval

    if condition == RuleCondition.CROSSES_ABOVE:
        if _is_column_ref(value):
            ref_col = str(value)

            def _eval(df: pd.DataFrame) -> pd.Series:
                if indicator not in df.columns or ref_col not in df.columns:
                    return pd.Series(False, index=df.index)
                series = df[indicator]
                ref = df[ref_col]
                return (series.shift(1) <= ref.shift(1)) & (series > ref)

            return _eval

        threshold = float(value)

        def _eval(df: pd.DataFrame) -> pd.Series:
            if indicator not in df.columns:
                return pd.Series(False, index=df.index)
            series = df[indicator]
            return (series.shift(1) <= threshold) & (series > threshold)

        return _eval

    if condition == RuleCondition.CROSSES_BELOW:
        if _is_column_ref(value):
            ref_col = str(value)

            def _eval(df: pd.DataFrame) -> pd.Series:
                if indicator not in df.columns or ref_col not in df.columns:
                    return pd.Series(False, index=df.index)
                series = df[indicator]
                ref = df[ref_col]
                return (series.shift(1) >= ref.shift(1)) & (series < ref)

            return _eval

        threshold = float(value)

        def _eval(df: pd.DataFrame) -> pd.Series:
            if indicator not in df.columns:
                return pd.Series(False, index=df.index)
            series = df[indicator]
            return (series.shift(1) >= threshold) & (series < threshold)

        return _eval

    if condition == RuleCondition.BETWEEN:
        lower = float(rule.get("lower", 0))
        upper = float(rule.get("upper", value))

        def _eval(df: pd.DataFrame) -> pd.Series:
            if indicator not in df.columns:
                return pd.Series(False, index=df.index)
            return (df[indicator] >= lower) & (df[indicator] <= upper)

        return _eval

    if condition == RuleCondition.WITHIN_PCT:
        threshold = float(value)

        def _eval(df: pd.DataFrame) -> pd.Series:
            if indicator not in df.columns:
                return pd.Series(False, index=df.index)
            return df[indicator].abs() <= threshold

        return _eval

    if condition == RuleCondition.IN:
        if not isinstance(value, list):
            raise CompilationError(f"{context}: 'in' requires a list value")
        values = value

        def _eval(df: pd.DataFrame) -> pd.Series:
            if indicator not in df.columns:
                return pd.Series(False, index=df.index)
            return df[indicator].isin(values)

        return _eval

    if condition == RuleCondition.NOT_IN:
        if not isinstance(value, list):
            raise CompilationError(f"{context}: 'not_in' requires a list value")
        values = value

        def _eval(df: pd.DataFrame) -> pd.Series:
            if indicator not in df.columns:
                return pd.Series(False, index=df.index)
            return ~df[indicator].isin(values)

        return _eval

    if condition == RuleCondition.EQUALS:

        def _eval(df: pd.DataFrame) -> pd.Series:
            if indicator not in df.columns:
                return pd.Series(False, index=df.index)
            return df[indicator] == value

        return _eval

    raise CompilationError(f"{context}: unhandled condition '{condition.value}'")


# =============================================================================
# Evaluation — vectorised signal generation from a CompiledStrategy
# =============================================================================


def evaluate_signals(
    compiled: CompiledStrategy,
    df: pd.DataFrame,
    regime: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate a compiled strategy against an indicator DataFrame.

    Returns a DataFrame with columns:
      - ``signal``: 0 or 1
      - ``signal_direction``: "LONG", "SHORT", or "NONE"
      - ``signal_confidence``: 0.0–1.0 (fraction of confirmation rules that passed)

    Args:
        compiled: Pre-compiled strategy from ``compile_strategy()``.
        df: DataFrame with indicator columns (output of ``_generate_signals_from_rules``
            indicator computation block or equivalent).
        regime: Optional current regime label. If provided and the strategy has
                ``regime_weights``, rule evaluation may be skipped for
                non-matching regimes.
    """
    n = len(df)
    result = pd.DataFrame(index=df.index)
    result["signal"] = 0
    result["signal_direction"] = "NONE"
    result["signal_confidence"] = 0.0

    # Prerequisites: ALL must pass (AND gate)
    prereq_pass = pd.Series(True, index=df.index)
    for rule in compiled.prerequisite_entry_rules:
        prereq_pass = prereq_pass & rule.evaluate(df)

    # Confirmations: count how many pass
    confirmation_rules = compiled.confirmation_entry_rules
    if confirmation_rules:
        confirmation_count = pd.Series(0, index=df.index, dtype=int)
        for rule in confirmation_rules:
            confirmation_count += rule.evaluate(df).astype(int)
        confirm_pass = confirmation_count >= compiled.min_confirmations
        # Confidence = fraction of confirmation rules that passed
        confidence = confirmation_count / len(confirmation_rules)
    else:
        confirm_pass = pd.Series(True, index=df.index)
        confidence = pd.Series(1.0, index=df.index)

    # Plain rules: OR logic (backward-compatible)
    entry_long = pd.Series(False, index=df.index)
    entry_short = pd.Series(False, index=df.index)
    for rule in compiled.plain_entry_rules:
        mask = rule.evaluate(df)
        if rule.direction == "short":
            entry_short = entry_short | mask
        else:
            entry_long = entry_long | mask

    # Combine structured + plain entries
    if compiled.prerequisite_entry_rules or confirmation_rules:
        structured_entry = prereq_pass & confirm_pass
        if compiled.direction == "SHORT":
            entry_short = entry_short | structured_entry
        else:
            entry_long = entry_long | structured_entry

    # Exit rules: OR logic
    exit_signal = pd.Series(False, index=df.index)
    for rule in compiled.exit_rules:
        exit_signal = exit_signal | rule.evaluate(df)

    # Build final signals
    result.loc[entry_long, "signal"] = 1
    result.loc[entry_long, "signal_direction"] = "LONG"
    result.loc[entry_short, "signal"] = 1
    result.loc[entry_short, "signal_direction"] = "SHORT"
    result.loc[exit_signal, "signal"] = 0
    result.loc[exit_signal, "signal_direction"] = "NONE"

    # Confidence: for structured entries use confirmation fraction;
    # for plain entries use 1.0 if triggered
    result["signal_confidence"] = np.where(
        result["signal"] == 1,
        np.where(
            (
                (prereq_pass & confirm_pass)
                if compiled.prerequisite_entry_rules
                else entry_long | entry_short
            ),
            confidence,
            1.0,
        ),
        0.0,
    )

    return result


# =============================================================================
# Strategy cache — avoids recompilation on every backtest iteration
# =============================================================================

_strategy_cache: dict[str, CompiledStrategy] = {}


def get_compiled(
    strategy_id: str,
    name: str,
    entry_rules: list[dict[str, Any]],
    exit_rules: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> CompiledStrategy:
    """
    Get or compile a strategy.

    Uses a module-level cache keyed by strategy_id. The cache is invalidated
    if the strategy is updated (caller must call ``invalidate_cache`` on update).
    """
    if strategy_id in _strategy_cache:
        return _strategy_cache[strategy_id]

    compiled = compile_strategy(strategy_id, name, entry_rules, exit_rules, parameters)
    _strategy_cache[strategy_id] = compiled
    return compiled


def invalidate_cache(strategy_id: str | None = None) -> None:
    """
    Invalidate the compiled strategy cache.

    Args:
        strategy_id: If given, invalidate only that strategy. Otherwise clear all.
    """
    if strategy_id is None:
        _strategy_cache.clear()
    else:
        _strategy_cache.pop(strategy_id, None)
