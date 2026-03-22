# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
ParameterGrid — bounded cross-product for strategy parameter search.

Hard cap: MAX_COMBINATIONS_PER_TEMPLATE = 200
This is enforced before any backtest runs. The cap is not negotiable:
an unbounded grid causes overnight jobs to run into market open.

Each template defines a parameter space. The grid emits concrete parameter
dicts by iterating the cross-product and sampling if over the cap.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Iterator

MAX_COMBINATIONS_PER_TEMPLATE = 200


class ParameterGrid:
    """
    Bounded cross-product iterator over a parameter space.

    Args:
        param_space: Dict of param_name → list of values to try.
            Example: {"rsi_period": [10, 14, 20], "rsi_oversold": [25, 30, 35]}
        seed: Random seed for reproducible sampling when over the cap.
    """

    def __init__(self, param_space: dict[str, list[Any]], seed: int = 42) -> None:
        self._space = param_space
        self._seed = seed

    @property
    def total_combinations(self) -> int:
        """Total cross-product size before sampling."""
        total = 1
        for values in self._space.values():
            total *= len(values)
        return total

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield at most MAX_COMBINATIONS_PER_TEMPLATE parameter dicts."""
        keys = list(self._space.keys())
        all_combos = list(itertools.product(*[self._space[k] for k in keys]))

        if len(all_combos) <= MAX_COMBINATIONS_PER_TEMPLATE:
            for combo in all_combos:
                yield dict(zip(keys, combo))
            return

        # Sample without replacement — use fixed seed for reproducibility
        rng = random.Random(self._seed)
        sampled = rng.sample(all_combos, MAX_COMBINATIONS_PER_TEMPLATE)
        for combo in sampled:
            yield dict(zip(keys, combo))


# =============================================================================
# Built-in template library
# =============================================================================

# These are reasonable starting grids. AlphaDiscoveryEngine picks templates
# based on the current dominant regime — a ranging regime gets the MR template,
# a trending regime gets the momentum templates.

RSI_MEAN_REVERSION_SPACE = {
    "rsi_period": [10, 14, 20],
    "rsi_oversold": [25, 30, 35],
    "rsi_overbought": [65, 70, 75],
    "sma_fast_period": [10, 20],
    "sma_slow_period": [50, 100, 200],
    "stop_loss_atr": [1.5, 2.0, 2.5],
}

TREND_MOMENTUM_SPACE = {
    "sma_fast_period": [10, 20, 50],
    "sma_slow_period": [50, 100, 200],
    "adx_threshold": [20, 25, 30],
    "rsi_period": [10, 14],
    "stop_loss_atr": [1.5, 2.0, 3.0],
}

BREAKOUT_SPACE = {
    "breakout_period": [10, 20, 30, 50],
    "atr_period": [10, 14, 20],
    "volume_confirmation": [True, False],
    "stop_loss_atr": [1.0, 1.5, 2.0],
}

MEAN_REVERSION_BOLLINGER_SPACE = {
    "bb_period": [15, 20, 25],
    "bb_std": [1.5, 2.0, 2.5],
    "rsi_period": [10, 14],
    "rsi_oversold": [30, 35],
    "stop_loss_atr": [1.5, 2.0],
}

TEMPLATE_REGISTRY: dict[str, dict[str, list[Any]]] = {
    "rsi_mean_reversion": RSI_MEAN_REVERSION_SPACE,
    "trend_momentum": TREND_MOMENTUM_SPACE,
    "breakout": BREAKOUT_SPACE,
    "mean_reversion_bollinger": MEAN_REVERSION_BOLLINGER_SPACE,
}


def get_templates_for_regime(
    trend_regime: str,
) -> list[tuple[str, dict[str, list[Any]]]]:
    """Return template names and spaces appropriate for the current regime."""
    if trend_regime in ("trending_up", "trending_down"):
        return [
            ("trend_momentum", TREND_MOMENTUM_SPACE),
            ("breakout", BREAKOUT_SPACE),
        ]
    elif trend_regime == "ranging":
        return [
            ("rsi_mean_reversion", RSI_MEAN_REVERSION_SPACE),
            ("mean_reversion_bollinger", MEAN_REVERSION_BOLLINGER_SPACE),
        ]
    else:
        # Unknown regime: try all templates at reduced depth
        return list(TEMPLATE_REGISTRY.items())
