# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Intraday trading loop — continuous M1/M5 streaming, signal evaluation,
execution, and position management with flatten-at-close.

Usage:
    from quantstack.intraday import LiveIntradayLoop

    loop = LiveIntradayLoop(symbols=["SPY", "QQQ"], timeframe="M1")
    report = asyncio.run(loop.run())
"""

from quantstack.intraday.loop import IntradayReport, LiveIntradayLoop
from quantstack.intraday.position_manager import IntradayPositionManager
from quantstack.intraday.signal_evaluator import IntradaySignalEvaluator

__all__ = [
    "LiveIntradayLoop",
    "IntradayReport",
    "IntradayPositionManager",
    "IntradaySignalEvaluator",
]
