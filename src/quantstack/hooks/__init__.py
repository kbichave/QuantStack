# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hooks — fire-and-forget callbacks for the execution pipeline."""

from quantstack.hooks.trade_hooks import (  # noqa: F401
    find_similar_situations,
    get_reflexion_episodes,
    on_daily_close,
    on_trade_close,
)
