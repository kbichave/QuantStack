# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Broker factory — returns the correct broker based on USE_REAL_TRADING flag.

USE_REAL_TRADING=false (default):
    Returns PaperBroker — local simulation with realistic slippage.
    No eTrade credentials required. Safe for development and backtesting.

USE_REAL_TRADING=true:
    Returns EtradeBroker — live execution via eTrade API.
    Requires ETRADE_CONSUMER_KEY, ETRADE_CONSUMER_SECRET, and prior OAuth.
    ETRADE_SANDBOX=true (default): uses apisb.etrade.com (test environment)
    ETRADE_SANDBOX=false: uses api.etrade.com (real money / paper account)

The broker interface is identical in both cases: execute(OrderRequest) -> Fill.
The flow, risk gate, and audit trail never know which broker is active.

Usage:
    broker = get_broker()
    fill = broker.execute(order_request)

    # Inspect which mode is active
    mode = get_broker_mode()  # "paper" | "etrade_sandbox" | "etrade_live"
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Union

from loguru import logger

from quant_pod.execution.paper_broker import PaperBroker, get_paper_broker

if TYPE_CHECKING:
    from quant_pod.execution.etrade_broker import EtradeBroker


def _use_real_trading() -> bool:
    """Read USE_REAL_TRADING env var. Defaults to False (paper broker)."""
    val = os.getenv("USE_REAL_TRADING", "false").strip().lower()
    return val in ("true", "1", "yes")


def get_broker_mode() -> str:
    """
    Return a human-readable label for the active broker mode.

    Used in /health and /dashboard responses so operators can confirm
    which mode the system is in without needing to inspect env vars.
    """
    if not _use_real_trading():
        return "paper"
    sandbox = os.getenv("ETRADE_SANDBOX", "true").strip().lower() in ("true", "1", "yes")
    return "etrade_sandbox" if sandbox else "etrade_live"


def get_broker() -> Union[PaperBroker, "EtradeBroker"]:
    """
    Return the active broker singleton.

    Lazily imports EtradeBroker only when USE_REAL_TRADING=true, so
    systems without eTrade credentials installed never pay the import cost.
    """
    if not _use_real_trading():
        return get_paper_broker()

    # Validate credentials before attempting init
    if not os.getenv("ETRADE_CONSUMER_KEY") or not os.getenv("ETRADE_CONSUMER_SECRET"):
        logger.error(
            "[BROKER] USE_REAL_TRADING=true but ETRADE_CONSUMER_KEY / "
            "ETRADE_CONSUMER_SECRET not set. Falling back to PaperBroker."
        )
        return get_paper_broker()

    try:
        from quant_pod.execution.etrade_broker import get_etrade_broker
        broker = get_etrade_broker()
        mode = get_broker_mode()
        logger.info(f"[BROKER] Active broker: EtradeBroker (mode={mode})")
        return broker
    except Exception as e:
        logger.error(
            f"[BROKER] EtradeBroker init failed ({e}). "
            "Falling back to PaperBroker — set USE_REAL_TRADING=false to suppress this warning."
        )
        return get_paper_broker()
