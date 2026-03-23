"""
BrokerInterface ABC — provider-agnostic trading operations contract.

All broker clients (Alpaca, IBKR) implement this interface.  The strategy
layer and execution engine depend only on ``BrokerInterface``, never on
provider-specific clients.

Design
------
All methods are **synchronous**.  Broker calls are blocking I/O (REST or
gateway socket).  The async execution layer wraps these calls in
``asyncio.get_event_loop().run_in_executor(None, broker.place_order, ...)``
when needed to avoid blocking the event loop.  Keeping the interface sync
is simpler and avoids the awkward mix of sync and async in providers like
ib_insync that expose a synchronous API.

Error handling
--------------
All methods raise ``BrokerError`` (or subclasses) on failure.  Callers
should catch ``BrokerError`` and never let raw provider exceptions propagate
into strategy logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from quantstack.core.execution.unified_models import (
    UnifiedAccount,
    UnifiedBalance,
    UnifiedOrder,
    UnifiedOrderPreview,
    UnifiedOrderResult,
    UnifiedPosition,
    UnifiedQuote,
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


from quantstack.shared.exceptions import BrokerError, BrokerConnectionError  # noqa: F401, E402


class BrokerOrderError(BrokerError):
    """Raised when an order is rejected or cannot be placed."""

    def __init__(self, message: str, order: UnifiedOrder | None = None) -> None:
        super().__init__(message)
        self.order = order


class BrokerAuthError(BrokerError):
    """Raised when API credentials are invalid or expired."""


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BrokerInterface(ABC):
    """Provider-agnostic trading and account-data interface.

    Implementations must translate between ``UnifiedOrder`` / ``UnifiedPosition``
    / etc. and the provider's native models.

    All methods are synchronous (blocking).  The async execution layer wraps
    calls in a thread-pool executor.
    """

    # ── Account & balance ─────────────────────────────────────────────────────

    @abstractmethod
    def get_accounts(self) -> list[UnifiedAccount]:
        """Return all accounts accessible with the current credentials."""
        ...

    @abstractmethod
    def get_balance(self, account_id: str) -> UnifiedBalance:
        """Return cash, buying power, and portfolio value for ``account_id``."""
        ...

    # ── Positions ─────────────────────────────────────────────────────────────

    @abstractmethod
    def get_positions(self, account_id: str) -> list[UnifiedPosition]:
        """Return all open positions for ``account_id``."""
        ...

    # ── Market data ───────────────────────────────────────────────────────────

    @abstractmethod
    def get_quote(self, symbols: list[str]) -> list[UnifiedQuote]:
        """Return real-time (or delayed) best-bid/offer quotes.

        Args:
            symbols: List of ticker symbols (e.g. ["SPY", "AAPL"]).

        Returns:
            One ``UnifiedQuote`` per symbol in the same order as ``symbols``.
        """
        ...

    # ── Orders ────────────────────────────────────────────────────────────────

    @abstractmethod
    def preview_order(
        self, account_id: str, order: UnifiedOrder
    ) -> UnifiedOrderPreview:
        """Estimate cost and commission for ``order`` without submitting it."""
        ...

    @abstractmethod
    def place_order(self, account_id: str, order: UnifiedOrder) -> UnifiedOrderResult:
        """Submit ``order`` to the broker.

        Raises:
            BrokerOrderError: If the order is rejected.
            BrokerConnectionError: If the broker cannot be reached.
        """
        ...

    @abstractmethod
    def cancel_order(self, account_id: str, order_id: str) -> bool:
        """Cancel an open order.

        Returns:
            True if the cancellation was accepted, False if order was not found
            or already terminal.
        """
        ...

    @abstractmethod
    def get_orders(
        self,
        account_id: str,
        status: str | None = None,
    ) -> list[UnifiedOrderResult]:
        """Return orders for ``account_id``.

        Args:
            account_id: Account identifier.
            status:     Filter — "open", "filled", "cancelled", or None for all.

        Returns:
            List of orders, most-recent first.
        """
        ...

    # ── Optional: auth / connectivity check ──────────────────────────────────

    def check_auth(self) -> bool:
        """Return True if credentials are valid and the broker is reachable.

        Default implementation attempts ``get_accounts()`` and catches
        ``BrokerError``.  Providers may override with a lighter-weight check.
        """
        try:
            self.get_accounts()
            return True
        except BrokerError:
            return False
