# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
PortfolioOptimizerAgent — cross-symbol position sizing via mean-variance optimization.

Why this exists:
    The original _calculate_quantity() sizes each symbol independently using
    full/half/quarter equity buckets.  Four "full" decisions = 80% deployed with
    no cross-correlation check.  This agent replaces that with Markowitz
    mean-variance optimization (Ledoit-Wolf shrinkage, SLSQP) so position sizing
    respects the joint covariance structure of the book.

Design decisions:
  - Wraps MeanVarianceOptimizer from quantcore/portfolio/optimizer.py; no
    reimplementation.
  - Signal vector: position_size_fraction × confidence.  This preserves the
    crew's qualitative conviction while giving the optimizer a numeric forecast.
  - Falls back to signal-proportional equal-weight when returns_df has < 30 bars
    (insufficient sample for stable covariance estimation).
  - Max weight = 20% per name (same ceiling as the legacy "full" bucket).
  - SELL decisions are handled by mapping the decision symbol to zero weight in
    the optimizer's output (the caller reduces/closes the position).

Failure modes:
  - SLSQP does not converge: optimizer's own fallback → equal-weight. Logged.
  - returns_df is None: signal-proportional weights, no covariance used.
  - Single symbol: optimizer is trivially max_weight = min(1.0, signal).
  - All signals zero: equal-weight across active BUY symbols; no trades.
"""

from __future__ import annotations

import os

import pandas as pd
from loguru import logger
from quantstack.core.portfolio.optimizer import (
    MeanVarianceOptimizer,
    OptimizationObjective,
    OptimizationResult,
    PortfolioConstraints,
    covariance_matrix,
)

# Minimum bars required for a well-conditioned covariance estimate.
_MIN_BARS_FOR_COV = int(os.getenv("PORT_OPT_MIN_BARS", "30"))

# Maximum single-name weight (matches legacy "full" bucket cap).
_MAX_WEIGHT = float(os.getenv("PORT_OPT_MAX_WEIGHT", "0.20"))

# Risk-free rate for Sharpe-maximising objective (annualised fraction).
_RISK_FREE_RATE = float(os.getenv("PORT_OPT_RISK_FREE_RATE", "0.05"))

# Fraction of equity allocated to a "full" position in the legacy sizing scheme.
# Used to map position_size labels → numeric signal magnitudes.
_SIZE_FRACTIONS: dict[str, float] = {
    "full": 0.20,
    "half": 0.10,
    "quarter": 0.05,
    "none": 0.00,
}


class PortfolioOptimizerAgent:
    """
    Converts a list of per-symbol trade decisions into portfolio-optimal weights.

    The caller (TradingDayFlow) passes all approved BUY/SELL decisions plus
    optional historical returns data.  The agent returns a {symbol: weight}
    mapping where each weight is a fraction of total portfolio NAV.

    SELL decisions are excluded from the optimizer and returned as weight=0.0
    so the caller can close/reduce those positions.
    """

    def __init__(self) -> None:
        self._optimizer = MeanVarianceOptimizer(risk_free_rate=_RISK_FREE_RATE)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def optimize(
        self,
        decisions: list[dict],
        returns_df: pd.DataFrame | None,
        current_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Compute target portfolio weights for the given trade decisions.

        Args:
            decisions: List of TradeDecision dicts (must have 'symbol', 'action',
                       'position_size', 'confidence').
            returns_df: DataFrame of daily returns (columns=symbols, rows=dates).
                        Use None or < 30 rows to trigger signal-proportional fallback.
            current_weights: Current portfolio weights {symbol: fraction}.  When
                             provided, a turnover cost penalty is applied to the
                             objective so the optimizer resists trivial churn.

        Returns:
            Dict mapping symbol → target weight fraction.
            SELL symbols → 0.0 (caller closes position).
            HOLD symbols → not included (unchanged).
        """
        buy_decisions = [d for d in decisions if d.get("action") == "buy"]
        sell_symbols = {d["symbol"] for d in decisions if d.get("action") == "sell"}

        # SELL decisions: return zero weight so caller reduces/closes
        result_weights: dict[str, float] = dict.fromkeys(sell_symbols, 0.0)

        if not buy_decisions:
            logger.debug("[PortOpt] No BUY decisions — returning sell zeroes only")
            return result_weights

        signals = self._build_signal_vector(buy_decisions)
        buy_symbols = list(signals.keys())

        # Fallback path: insufficient history for covariance estimation
        if returns_df is None or len(returns_df) < _MIN_BARS_FOR_COV:
            fallback_weights = self._signal_proportional_weights(signals)
            logger.info(
                f"[PortOpt] Fallback to signal-proportional weights "
                f"(returns_df rows={len(returns_df) if returns_df is not None else 0}, "
                f"need≥{_MIN_BARS_FOR_COV})"
            )
            result_weights.update(fallback_weights)
            return result_weights

        # Align returns_df to the symbols in play
        available_cols = [s for s in buy_symbols if s in returns_df.columns]
        missing = set(buy_symbols) - set(available_cols)
        if missing:
            logger.debug(
                f"[PortOpt] Symbols missing from returns_df — fallback: {missing}"
            )
            # Use signal-proportional for missing symbols; optimize only available
            for sym in missing:
                result_weights[sym] = self._single_signal_weight(
                    signals[sym], len(buy_symbols)
                )

        if not available_cols:
            result_weights.update(self._signal_proportional_weights(signals))
            return result_weights

        aligned_returns = returns_df[available_cols].dropna(how="all").fillna(0.0)
        if len(aligned_returns) < _MIN_BARS_FOR_COV:
            result_weights.update(
                self._signal_proportional_weights(
                    {s: signals[s] for s in available_cols}
                )
            )
            return result_weights

        try:
            cov = covariance_matrix(aligned_returns, annualise=True, shrinkage=True)
        except Exception as cov_err:
            logger.warning(
                f"[PortOpt] Covariance build failed ({cov_err}) — using fallback"
            )
            result_weights.update(self._signal_proportional_weights(signals))
            return result_weights

        constraints = PortfolioConstraints(
            max_weight=_MAX_WEIGHT,
            min_weight=0.0,
            max_leverage=1.0,
        )

        opt_signals = {s: signals[s] for s in available_cols}
        opt_current = (
            {s: current_weights.get(s, 0.0) for s in available_cols}
            if current_weights
            else None
        )

        try:
            opt_result: OptimizationResult = self._optimizer.optimize(
                signals=opt_signals,
                cov_matrix=cov,
                constraints=constraints,
                objective=OptimizationObjective.MAX_SHARPE,
                current_weights=opt_current,
            )

            if not opt_result.converged:
                logger.warning(
                    f"[PortOpt] SLSQP did not converge ({opt_result.solver_message}) "
                    "— optimizer fell back to equal-weight internally"
                )

            logger.info(
                f"[PortOpt] Optimised {len(available_cols)} symbols | "
                f"E[ret]={opt_result.expected_return:.1%} "
                f"E[vol]={opt_result.expected_volatility:.1%} "
                f"Sharpe={opt_result.expected_sharpe:.2f} "
                f"converged={opt_result.converged}"
            )

            result_weights.update(opt_result.target_weights)

        except Exception as opt_err:
            logger.error(
                f"[PortOpt] Optimizer raised {opt_err} — using signal-proportional fallback"
            )
            result_weights.update(self._signal_proportional_weights(opt_signals))

        return result_weights

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _build_signal_vector(self, buy_decisions: list[dict]) -> dict[str, float]:
        """
        Convert crew decisions into numeric return forecasts for the optimizer.

        Formula: position_size_fraction × confidence
        e.g. "full" (0.20) × 0.80 confidence = 0.16 (16% expected annual return proxy)

        This is not a real return forecast — it preserves the crew's relative
        conviction across symbols, which is sufficient for MAX_SHARPE ordering.
        """
        signals: dict[str, float] = {}
        for d in buy_decisions:
            symbol = d.get("symbol")
            if not symbol:
                continue
            size_label = d.get("position_size", "quarter")
            size_frac = _SIZE_FRACTIONS.get(size_label, 0.05)
            confidence = float(d.get("confidence", 0.5))
            signals[symbol] = size_frac * confidence
        return signals

    def _signal_proportional_weights(
        self, signals: dict[str, float]
    ) -> dict[str, float]:
        """
        Distribute weight proportionally to signal magnitudes, capped at _MAX_WEIGHT.

        Used when covariance estimation is unavailable (< 30 bars).
        """
        total = sum(signals.values())
        if total <= 0.0:
            # All signals zero → equal-weight
            n = len(signals)
            return {s: min(_MAX_WEIGHT, 1.0 / n) for s in signals} if n > 0 else {}

        weights = {}
        for sym, sig in signals.items():
            raw = sig / total
            weights[sym] = min(raw, _MAX_WEIGHT)

        # Re-normalise after capping to ensure weights sum ≤ 1.0
        cap_total = sum(weights.values())
        if cap_total > 1.0:
            weights = {s: w / cap_total for s, w in weights.items()}

        return weights

    def _single_signal_weight(self, signal: float, n_symbols: int) -> float:
        """Compute a single symbol's share when only its signal is known."""
        # Equal share capped at max_weight as a conservative estimate
        return min(_MAX_WEIGHT, 1.0 / max(n_symbols, 1))
