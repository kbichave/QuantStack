# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Broker factory — returns the correct broker based on USE_REAL_TRADING flag
and available credentials.

USE_REAL_TRADING=false (default):
    PaperBroker — local simulation with realistic slippage. No credentials needed.

USE_REAL_TRADING=true, ALPACA_API_KEY set:
    AlpacaBroker — Alpaca paper (ALPACA_PAPER=true, default) or live trading.
    No OAuth needed — API key + secret are enough.

USE_REAL_TRADING=true, no Alpaca keys, ETRADE_CONSUMER_KEY set:
    EtradeBroker — eTrade paper sandbox or live account. Requires prior OAuth.

Fallback: PaperBroker if USE_REAL_TRADING=true but no broker credentials are set.

The broker interface is identical in all cases: execute(OrderRequest) -> Fill.
The flow, risk gate, and audit trail never know which broker is active.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

from quantstack.execution.paper_broker import BrokerProtocol, PaperBroker, get_paper_broker

# TYPE_CHECKING guard: gives static analysers the full type without paying the
# import cost at runtime. The deferred runtime imports in get_broker() are the
# *one* legitimate use of deferred imports here — each broker wraps an optional
# external SDK that may not be installed.
if TYPE_CHECKING:
    from quantstack.execution.alpaca_broker import AlpacaBroker  # noqa: F401
    from quantstack.execution.etrade_broker import EtradeBroker  # noqa: F401


def _use_real_trading() -> bool:
    val = os.getenv("USE_REAL_TRADING", "false").strip().lower()
    return val in ("true", "1", "yes")


def _alpaca_keys_set() -> bool:
    return bool(os.getenv("ALPACA_API_KEY")) and bool(os.getenv("ALPACA_SECRET_KEY"))


def _etrade_keys_set() -> bool:
    return bool(os.getenv("ETRADE_CONSUMER_KEY")) and bool(os.getenv("ETRADE_CONSUMER_SECRET"))


def get_broker_mode() -> str:
    """Human-readable label for the active broker mode."""
    if not _use_real_trading():
        return "paper"

    if _alpaca_keys_set():
        paper = os.getenv("ALPACA_PAPER", "true").strip().lower() in ("true", "1", "yes")
        return "alpaca_paper" if paper else "alpaca_live"

    if _etrade_keys_set():
        sandbox = os.getenv("ETRADE_SANDBOX", "true").strip().lower() in ("true", "1", "yes")
        return "etrade_sandbox" if sandbox else "etrade_live"

    return "paper"  # fallback


def get_broker() -> BrokerProtocol:
    """
    Return the active broker singleton.

    Priority when USE_REAL_TRADING=true:
      1. AlpacaBroker  (ALPACA_API_KEY + ALPACA_SECRET_KEY set)
      2. EtradeBroker  (ETRADE_CONSUMER_KEY + ETRADE_CONSUMER_SECRET set)
      3. PaperBroker   (fallback — logs a warning)
    """
    if not _use_real_trading():
        return get_paper_broker()

    if _alpaca_keys_set():
        try:
            from quantstack.execution.alpaca_broker import get_alpaca_broker
            broker = get_alpaca_broker()
            logger.info(f"[BROKER] Active broker: AlpacaBroker (mode={get_broker_mode()})")
            return broker
        except Exception as e:
            logger.error(
                f"[BROKER] AlpacaBroker init failed ({e}). "
                "Falling back to PaperBroker."
            )
            return get_paper_broker()

    if _etrade_keys_set():
        try:
            from quantstack.execution.etrade_broker import get_etrade_broker
            broker = get_etrade_broker()
            logger.info(f"[BROKER] Active broker: EtradeBroker (mode={get_broker_mode()})")
            return broker
        except Exception as e:
            logger.error(
                f"[BROKER] EtradeBroker init failed ({e}). "
                "Falling back to PaperBroker."
            )
            return get_paper_broker()

    logger.warning(
        "[BROKER] USE_REAL_TRADING=true but no broker credentials found. "
        "Set ALPACA_API_KEY+ALPACA_SECRET_KEY (preferred) or "
        "ETRADE_CONSUMER_KEY+ETRADE_CONSUMER_SECRET. Falling back to PaperBroker."
    )
    return get_paper_broker()
