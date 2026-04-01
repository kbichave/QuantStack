# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Backward-compat re-export — canonical location is quantstack.execution.adapters.etrade."""

from quantstack.execution.adapters.etrade import (
    Account,
    AccountBalance,
    AuthStatus,
    ETradeAuthManager,
    ETradeClient,
    Order,
    OrderAction,
    OrderDuration,
    OrderLeg,
    OrderPreview,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    SecurityType,
)

__all__ = [
    "ETradeAuthManager",
    "ETradeClient",
    "Account",
    "AccountBalance",
    "AuthStatus",
    "Order",
    "OrderAction",
    "OrderDuration",
    "OrderLeg",
    "OrderPreview",
    "OrderRequest",
    "OrderStatus",
    "OrderType",
    "Position",
    "Quote",
    "SecurityType",
]
