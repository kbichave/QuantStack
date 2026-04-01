# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
AlphaDiscoveryEngine — overnight strategy discovery from historical data.

Runs without a Claude Code session. Discovers parameter combinations that
pass statistical filters and registers them as 'draft' strategies for
human review in /workshop sessions.

Process per symbol:
1. Load daily OHLCV from DataStore (read-only, no network calls)
2. Detect current dominant regime (WeeklyRegimeClassifier)
3. Select parameter templates appropriate for that regime
4. Iterate bounded parameter grid (≤ 200 combinations per template)
5. Two-stage filter: IS screen (fast) → OOS walk-forward (thorough)
6. Register passing candidates as status='draft', source='generated'
7. Never auto-promote to forward_testing

Invariants:
- Never writes live/forward_testing strategies — only 'draft'
- DB is opened read-only for price data; strategy writes go through the
  MCP context's write connection (require_live_db)
- Exits cleanly on DB lock conflict via _connect_with_lock_guard in db.py
- All failures are logged and counted — never raises to the caller

Usage:
    python -m quantstack.alpha_discovery.engine --symbols XOM MSFT --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from quantstack.alpha_discovery.filter import CandidateFilter, IS_MIN_TRADES
from quantstack.alpha_discovery.grammar_gp import GrammarGP
from quantstack.alpha_discovery.hypothesis_agent import HypothesisAgent
from quantstack.alpha_discovery.registrar import StrategyRegistrar
from quantstack.alpha_discovery.search_space import ParameterGrid, get_templates_for_regime
from quantstack.alpha_discovery.watchlist import WatchlistLoader
from quantstack.config.timeframes import Timeframe
from quantstack.core.hierarchy.regime_classifier import WeeklyRegimeClassifier
from quantstack.data.storage import DataStore


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class DiscoveryResult:
    """Aggregate result of one AlphaDiscoveryEngine run."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    symbols_processed: int = 0
    candidates_screened: int = 0
    is_passed: int = 0
    oos_passed: int = 0
    registered: int = 0
    errors: int = 0
    dry_run: bool = False
    # GP evolution metrics
    gp_candidates_screened: int = 0
    gp_is_passed: int = 0
    gp_oos_passed: int = 0
    gp_registered: int = 0


# =============================================================================
# AlphaDiscoveryEngine
# =============================================================================


class AlphaDiscoveryEngine:
    """
    Runs the full discovery pipeline for a list of symbols.

    Designed to run overnight (22:00 Mon-Fri) when the machine is idle.
    Total runtime budget: 60 minutes (enforced by scheduler timeout).
    """

    def __init__(self, dry_run: bool = False, enable_gp: bool = True) -> None:
        """
        Args:
            dry_run: Run the pipeline but do not write to the DB.
                     Useful for validating the engine without persisting results.
            enable_gp: Run GP evolution after grid search + HypothesisAgent.
                       Disable with --no-gp for backward-compatible runs.
        """
        self._dry_run = dry_run
        self._enable_gp = enable_gp

    def run(self, symbols: list[str] | None = None) -> DiscoveryResult:
        """
        Synchronous entry point (blocking).

        Args:
            symbols: Symbols to search. Defaults to WatchlistLoader output.
        """
        run_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(timezone.utc)

        logger.info(
            f"[AlphaDiscoveryEngine] run={run_id} starting " f"dry_run={self._dry_run}"
        )

        if symbols is None:
            symbols = WatchlistLoader().load()

        result = DiscoveryResult(
            run_id=run_id,
            started_at=started_at,
            finished_at=started_at,
            dry_run=self._dry_run,
        )

        for symbol in symbols:
            self._process_symbol(symbol, result)
            result.symbols_processed += 1

        result.finished_at = datetime.now(timezone.utc)
        elapsed = round((result.finished_at - result.started_at).total_seconds(), 1)
        logger.info(
            f"[AlphaDiscoveryEngine] run={run_id} done — "
            f"symbols={result.symbols_processed} "
            f"screened={result.candidates_screened} "
            f"is_pass={result.is_passed} "
            f"oos_pass={result.oos_passed} "
            f"registered={result.registered} "
            f"gp_screened={result.gp_candidates_screened} "
            f"gp_registered={result.gp_registered} "
            f"errors={result.errors} "
            f"elapsed={elapsed}s"
        )
        return result

    # -------------------------------------------------------------------------
    # Per-Symbol Pipeline
    # -------------------------------------------------------------------------

    def _process_symbol(self, symbol: str, result: DiscoveryResult) -> None:
        """Full discovery pipeline for one symbol."""
        logger.info(f"[AlphaDiscoveryEngine] {symbol}: starting")
        t0 = time.monotonic()

        try:
            price_data = self._load_price_data(symbol)
            if price_data is None or len(price_data) < 252:
                logger.info(
                    f"[AlphaDiscoveryEngine] {symbol}: skipped — "
                    f"insufficient history ({len(price_data) if price_data is not None else 0} bars)"
                )
                return

            regime = self._detect_regime(price_data)
            trend_regime = regime.get("trend_regime", "unknown")
            logger.info(f"[AlphaDiscoveryEngine] {symbol}: regime={trend_regime}")

            templates = self._get_templates(trend_regime)
            regime_affinity = {trend_regime: 0.8} if trend_regime != "unknown" else {}

            filt = CandidateFilter()
            reg = StrategyRegistrar()

            # Collect seeds for GP: specs with sufficient trades (even if
            # they failed Sharpe/PF — structurally interesting seeds).

            seed_specs: list[dict] = []
            grid_total_combinations = 0

            for template_name, template_rules in templates:
                param_space, entry_rules, exit_rules = template_rules
                grid = ParameterGrid(param_space)
                grid_total_combinations += grid.total_combinations

                logger.info(
                    f"[AlphaDiscoveryEngine] {symbol}: template={template_name} "
                    f"combinations={grid.total_combinations}"
                )

                for params in grid:
                    result.candidates_screened += 1
                    # Substitute concrete parameter values into rule templates
                    concrete_entry = _resolve_rule_params(entry_rules, params)
                    concrete_exit = _resolve_rule_params(exit_rules, params)
                    spec = {
                        "entry_rules": concrete_entry,
                        "exit_rules": concrete_exit,
                        "parameters": params,
                    }

                    filter_result = filt.apply(
                        spec, price_data, n_trials=grid.total_combinations
                    )

                    # Collect GP seed candidates (enough trades to be interesting)
                    if filter_result.is_trades >= IS_MIN_TRADES:
                        seed_specs.append(spec)

                    if not filter_result.passed:
                        if filter_result.stage_rejected == "oos_validation":
                            result.is_passed += 1
                        continue

                    result.is_passed += 1
                    result.oos_passed += 1

                    if self._dry_run:
                        logger.info(
                            f"[AlphaDiscoveryEngine][dry-run] {symbol}/{template_name}: "
                            f"IS={filter_result.is_sharpe:.2f} "
                            f"OOS={filter_result.oos_sharpe_mean:.2f} "
                            f"overfit={filter_result.overfit_ratio:.2f}"
                        )
                        continue

                    strategy_name = (
                        f"disc_{template_name}_{symbol}_"
                        f"is{filter_result.is_sharpe:.1f}_"
                        f"oos{filter_result.oos_sharpe_mean:.1f}"
                    )
                    strategy_id = reg.register(
                        name=strategy_name,
                        template_name=template_name,
                        parameters=params,
                        entry_rules=concrete_entry,
                        exit_rules=concrete_exit,
                        regime_affinity=regime_affinity,
                        is_sharpe=filter_result.is_sharpe,
                        oos_sharpe_mean=filter_result.oos_sharpe_mean,
                        symbol=symbol,
                    )
                    if strategy_id:
                        result.registered += 1

            # --- HypothesisAgent: additive novel candidates after template loop ---
            # Runs only if we have enough history. Falls back to [] on timeout/error.
            if len(price_data) >= 252:
                hypothesis_candidates = _run_hypothesis_agent_sync(
                    symbol=symbol,
                    regime=regime,
                    price_data=price_data,
                )
                for spec in hypothesis_candidates:
                    result.candidates_screened += 1
                    n_hyp = len(hypothesis_candidates)
                    filter_result = filt.apply(spec, price_data, n_trials=max(n_hyp, 1))

                    if not filter_result.passed:
                        if filter_result.stage_rejected == "oos_validation":
                            result.is_passed += 1
                        continue

                    result.is_passed += 1
                    result.oos_passed += 1

                    if self._dry_run:
                        logger.info(
                            f"[AlphaDiscoveryEngine][dry-run][hypothesis] {symbol}: "
                            f"IS={filter_result.is_sharpe:.2f} "
                            f"OOS={filter_result.oos_sharpe_mean:.2f}"
                        )
                        continue

                    strategy_id = reg.register(
                        name=f"hyp_{symbol}_is{filter_result.is_sharpe:.1f}_oos{filter_result.oos_sharpe_mean:.1f}",
                        template_name="hypothesis",
                        parameters=spec.get("parameters", {}),
                        entry_rules=spec["entry_rules"],
                        exit_rules=spec["exit_rules"],
                        regime_affinity=regime_affinity,
                        is_sharpe=filter_result.is_sharpe,
                        oos_sharpe_mean=filter_result.oos_sharpe_mean,
                        symbol=symbol,
                    )
                    if strategy_id:
                        result.registered += 1

            # --- GP evolution: seeded by IS survivors ---
            if self._enable_gp and len(price_data) >= 252:
                try:
                    gp = GrammarGP()
                    gp_survivors, gp_evals = gp.evolve(
                        seed_population=seed_specs,
                        price_data=price_data,
                        n_prior_trials=grid_total_combinations,
                    )
                    n_trials_gp = grid_total_combinations + gp_evals

                    for gp_spec in gp_survivors:
                        result.gp_candidates_screened += 1
                        gp_filter = filt.apply(
                            gp_spec, price_data, n_trials=n_trials_gp
                        )

                        if not gp_filter.passed:
                            if gp_filter.stage_rejected == "oos_validation":
                                result.gp_is_passed += 1
                            continue

                        result.gp_is_passed += 1
                        result.gp_oos_passed += 1

                        if self._dry_run:
                            logger.info(
                                f"[AlphaDiscoveryEngine][dry-run][gp] {symbol}: "
                                f"IS={gp_filter.is_sharpe:.2f} "
                                f"OOS={gp_filter.oos_sharpe_mean:.2f} "
                                f"overfit={gp_filter.overfit_ratio:.2f}"
                            )
                            continue

                        strategy_name = (
                            f"gp_{symbol}_"
                            f"is{gp_filter.is_sharpe:.1f}_"
                            f"oos{gp_filter.oos_sharpe_mean:.1f}"
                        )
                        strategy_id = reg.register(
                            name=strategy_name,
                            template_name="gp_evolved",
                            parameters=gp_spec.get("parameters", {}),
                            entry_rules=gp_spec["entry_rules"],
                            exit_rules=gp_spec["exit_rules"],
                            regime_affinity=regime_affinity,
                            is_sharpe=gp_filter.is_sharpe,
                            oos_sharpe_mean=gp_filter.oos_sharpe_mean,
                            symbol=symbol,
                            source="evolved",
                        )
                        if strategy_id:
                            result.gp_registered += 1

                except Exception as gp_exc:
                    logger.warning(
                        f"[AlphaDiscoveryEngine] {symbol}: GP evolution failed — {gp_exc}"
                    )

        except Exception as exc:
            logger.error(
                f"[AlphaDiscoveryEngine] {symbol}: unhandled error — {exc}",
                exc_info=True,
            )
            result.errors += 1

        logger.debug(
            f"[AlphaDiscoveryEngine] {symbol}: done in " f"{time.monotonic() - t0:.1f}s"
        )

    # -------------------------------------------------------------------------
    # Data helpers
    # -------------------------------------------------------------------------

    def _load_price_data(self, symbol: str) -> Any:
        """Load daily OHLCV from DataStore (read-only, no network calls)."""
        try:
            with DataStore(read_only=True) as store:
                df = store.load_ohlcv(symbol, Timeframe.D1)
                if df is not None and not df.empty:
                    df = df[df.index >= pd.Timestamp("2010-01-01")]
                    return df
            return None
        except Exception as exc:
            logger.warning(
                f"[AlphaDiscoveryEngine] {symbol}: could not load data — {exc}"
            )
            return None

    def _detect_regime(self, price_data: Any) -> dict[str, Any]:
        """Classify regime from loaded OHLCV (no re-fetch)."""
        try:
            classifier = WeeklyRegimeClassifier()
            ctx = classifier.classify(price_data)
            if ctx is None:
                return {"trend_regime": "unknown", "confidence": 0.0}

            # Map RegimeType enum to string (mirrors collectors/regime.py)
            label = str(getattr(ctx, "regime_type", ctx)).lower()
            if "bull" in label:
                trend = "trending_up"
            elif "bear" in label:
                trend = "trending_down"
            elif "sideways" in label or "ranging" in label:
                trend = "ranging"
            else:
                trend = "unknown"

            return {
                "trend_regime": trend,
                "confidence": getattr(ctx, "confidence", 0.5),
            }
        except Exception as exc:
            logger.debug(f"[AlphaDiscoveryEngine] regime detection failed: {exc}")
            return {"trend_regime": "unknown", "confidence": 0.0}

    def _get_templates(
        self, trend_regime: str
    ) -> list[tuple[str, tuple[dict, list, list]]]:
        """
        Return (template_name, (param_space, entry_rules, exit_rules)) tuples.

        Entry/exit rule templates are simple enough for deterministic rule
        evaluation by _generate_signals_from_rules — no LLM needed.
        The entry/exit rules here are TEMPLATES — parameter substitution
        happens in _process_symbol when iterating the ParameterGrid.
        """
        regime_templates = get_templates_for_regime(trend_regime)
        result = []
        for name, param_space in regime_templates:
            entry_rules, exit_rules = _get_rules_for_template(name)
            result.append((name, (param_space, entry_rules, exit_rules)))
        return result


# =============================================================================
# Template rule library
# =============================================================================


def _get_rules_for_template(template_name: str) -> tuple[list[dict], list[dict]]:
    """
    Return (entry_rules, exit_rules) for a named template.

    Rules follow the schema consumed by _generate_signals_from_rules:
    - Indicator-based rules: {"indicator", "condition", "value"} where value is
      a concrete float or a column name string (like "sma_slow").
    - Structural exit rules: {"type": "stop_loss", "atr_multiple": float}

    Parameter-dependent values (like rsi_oversold) are expressed as column
    references or rely on the parameters dict that _generate_signals_from_rules
    uses to build indicators. For threshold comparisons, the parameters dict
    contains the threshold — we use "_param_" prefix convention to signal
    a lookup (handled by _resolve_rule_params below before calling the filter).
    """
    if template_name == "rsi_mean_reversion":
        entry_rules = [
            # RSI crosses below oversold threshold — param resolved at grid iteration time
            {
                "indicator": "rsi",
                "condition": "crosses_below",
                "value": 30,
                "type": "prerequisite",
                "_param_value": "rsi_oversold",
            },
            {
                "indicator": "sma_fast",
                "condition": "above",
                "value": "sma_slow",
                "type": "confirmation",
            },
        ]
        exit_rules = [
            {
                "indicator": "rsi",
                "condition": "crosses_above",
                "value": 70,
                "_param_value": "rsi_overbought",
            },
            {
                "type": "stop_loss",
                "atr_multiple": 2.0,
                "_param_atr_multiple": "stop_loss_atr",
            },
        ]

    elif template_name == "trend_momentum":
        entry_rules = [
            {
                "indicator": "sma_crossover",
                "condition": "crosses_above",
                "value": 0,
                "type": "prerequisite",
            },
            {
                "indicator": "adx",
                "condition": "above",
                "value": 25,
                "_param_value": "adx_threshold",
                "type": "confirmation",
            },
        ]
        exit_rules = [
            {"indicator": "sma_crossover", "condition": "crosses_below", "value": 0},
            {
                "type": "stop_loss",
                "atr_multiple": 2.0,
                "_param_atr_multiple": "stop_loss_atr",
            },
        ]

    elif template_name == "breakout":
        entry_rules = [
            {
                "indicator": "breakout",
                "condition": "above",
                "value": 0,
                "type": "prerequisite",
            },
            {
                "indicator": "adx",
                "condition": "above",
                "value": 20,
                "type": "confirmation",
            },
        ]
        exit_rules = [
            {
                "type": "stop_loss",
                "atr_multiple": 1.5,
                "_param_atr_multiple": "stop_loss_atr",
            },
            {"type": "take_profit", "atr_multiple": 3.0},
        ]

    elif template_name == "mean_reversion_bollinger":
        entry_rules = [
            {
                "indicator": "bb_pct",
                "condition": "below",
                "value": 0.1,
                "type": "prerequisite",
            },
            {
                "indicator": "rsi",
                "condition": "below",
                "value": 35,
                "_param_value": "rsi_oversold",
                "type": "confirmation",
            },
        ]
        exit_rules = [
            {"indicator": "bb_pct", "condition": "above", "value": 0.9},
            {
                "type": "stop_loss",
                "atr_multiple": 2.0,
                "_param_atr_multiple": "stop_loss_atr",
            },
        ]

    else:
        entry_rules = []
        exit_rules = []

    return entry_rules, exit_rules


def _resolve_rule_params(rules: list[dict], params: dict[str, Any]) -> list[dict]:
    """
    Substitute _param_* keys with concrete values from the parameter set.

    Rules that carry `_param_value` or `_param_atr_multiple` get their
    default values replaced with the concrete parameter value for this
    grid iteration.
    """
    resolved = []
    for rule in rules:
        r = dict(rule)
        if "_param_value" in r:
            param_key = r.pop("_param_value")
            if param_key in params:
                r["value"] = params[param_key]
        if "_param_atr_multiple" in r:
            param_key = r.pop("_param_atr_multiple")
            if param_key in params:
                r["atr_multiple"] = params[param_key]
        resolved.append(r)
    return resolved


# =============================================================================
# Async helper for HypothesisAgent (called from synchronous _process_symbol)
# =============================================================================


def _run_hypothesis_agent_sync(
    symbol: str,
    regime: dict,
    price_data: Any,
) -> list[dict]:
    """
    Synchronous wrapper around HypothesisAgent.generate().

    AlphaDiscoveryEngine.run() is a blocking synchronous method called by
    the scheduler. We need to run an async coroutine from a sync context.
    Uses asyncio.new_event_loop() to avoid conflicts if a loop is already
    running in the calling thread (e.g., scheduler's asyncio loop).

    Falls back to [] on any failure — template discovery is unaffected.
    """
    try:
        # Build a brief summary from the most recent bar (no network call needed)
        brief_summary: dict = {}
        try:
            if hasattr(price_data, "iloc") and len(price_data) > 0:
                last = price_data.iloc[-1]

                # Use .get() for dict-like rows; attribute access for named tuples
                def _get(key: str, default: Any) -> Any:
                    try:
                        return last[key]
                    except (KeyError, TypeError):
                        return default

                brief_summary = {
                    "rsi_14": round(float(_get("rsi_14", 50)), 1),
                    "adx_14": round(float(_get("adx_14", 20)), 1),
                    "macd_hist": round(float(_get("macd_hist", 0)), 4),
                    "bb_pct": round(float(_get("bb_pct", 0.5)), 3),
                    "weekly_trend": regime.get("trend_regime", "unknown"),
                }
        except Exception:
            pass  # brief_summary stays empty — HypothesisAgent handles it gracefully

        agent = HypothesisAgent()

        # Run in a fresh event loop — safe from any thread, no loop contamination
        loop = asyncio.new_event_loop()
        try:
            candidates = loop.run_until_complete(
                agent.generate(
                    symbol=symbol, regime=regime, signal_brief_summary=brief_summary
                )
            )
        finally:
            loop.close()

        logger.info(
            f"[AlphaDiscoveryEngine] {symbol}: HypothesisAgent returned "
            f"{len(candidates)} candidates"
        )
        return candidates
    except Exception as exc:
        logger.debug(
            f"[AlphaDiscoveryEngine] {symbol}: _run_hypothesis_agent_sync failed: {exc}"
        )
        return []


# =============================================================================
# CLI entry point
# =============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaDiscoveryEngine — overnight strategy discovery"
    )
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to search")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run pipeline, skip DB writes"
    )
    parser.add_argument(
        "--no-gp",
        action="store_true",
        help="Disable GP evolution (grid search + hypothesis only)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    engine = AlphaDiscoveryEngine(dry_run=args.dry_run, enable_gp=not args.no_gp)
    report = engine.run(symbols=args.symbols)
    print(
        f"Discovery run {report.run_id}: "
        f"screened={report.candidates_screened} "
        f"registered={report.registered} "
        f"gp_registered={report.gp_registered} "
        f"errors={report.errors}"
    )
