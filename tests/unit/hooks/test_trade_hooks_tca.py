"""Tests for TCA EWMA wiring in trade hooks (QS-E6).

Verifies:
  - _update_tca_ewma calls update_ewma_after_fill with correct args
  - Exceptions are caught and logged, never raised
  - _on_trade_fill always fires TCA update
  - Existing OutcomeTracker logic is unaffected
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestUpdateTcaEwma:
    """Unit tests for the _update_tca_ewma helper."""

    @patch("quantstack.hooks.trade_hooks.db_conn", create=True)
    @patch("quantstack.hooks.trade_hooks.update_ewma_after_fill", create=True)
    def test_calls_update_ewma_with_correct_args(self, mock_update, mock_db):
        from quantstack.hooks.trade_hooks import _update_tca_ewma

        mock_conn = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        ts = datetime(2026, 4, 7, 10, 30, tzinfo=timezone.utc)
        _update_tca_ewma(
            order_id="ord-123",
            symbol="AAPL",
            fill_timestamp=ts,
            arrival_price=150.0,
            fill_price=150.05,
            fill_quantity=100,
            daily_volume=5_000_000,
        )

        mock_update.assert_called_once_with(
            conn=mock_conn,
            order_id="ord-123",
            symbol="AAPL",
            fill_timestamp=ts,
            arrival_price=150.0,
            fill_price=150.05,
            fill_quantity=100,
            adv=5_000_000.0,
        )

    def test_exception_caught_and_logged(self):
        """_update_tca_ewma never raises — catches and logs."""
        from quantstack.hooks.trade_hooks import _update_tca_ewma

        with patch(
            "quantstack.execution.tca_ewma.update_ewma_after_fill",
            side_effect=RuntimeError("DB down"),
        ):
            # Should not raise
            _update_tca_ewma(
                order_id="ord-fail",
                symbol="FAIL",
                fill_timestamp=datetime.now(timezone.utc),
                arrival_price=100.0,
                fill_price=100.05,
                fill_quantity=50,
                daily_volume=1_000_000,
            )


class TestOnTradeFillTcaIntegration:
    """Tests for TCA EWMA integration in _on_trade_fill."""

    @patch("quantstack.hooks.trade_hooks._update_tca_ewma")
    @patch("quantstack.hooks.trade_hooks.OutcomeTracker")
    def test_always_calls_tca(self, mock_tracker_cls, mock_tca):
        from quantstack.hooks.trade_hooks import _on_trade_fill

        ts = datetime(2026, 4, 7, 10, 30, tzinfo=timezone.utc)
        _on_trade_fill(
            strategy_id="strat-1",
            symbol="AAPL",
            action="buy",
            fill_price=150.05,
            fill_quantity=100,
            fill_timestamp=ts,
            arrival_price=150.0,
            daily_volume=5_000_000,
            order_id="ord-123",
        )

        mock_tca.assert_called_once_with(
            order_id="ord-123",
            symbol="AAPL",
            fill_timestamp=ts,
            arrival_price=150.0,
            fill_price=150.05,
            fill_quantity=100,
            daily_volume=5_000_000,
        )

    @patch("quantstack.hooks.trade_hooks._update_tca_ewma")
    @patch("quantstack.hooks.trade_hooks.OutcomeTracker")
    def test_outcome_tracker_still_called(self, mock_tracker_cls, mock_tca):
        """Existing OutcomeTracker logic fires alongside TCA."""
        from quantstack.hooks.trade_hooks import _on_trade_fill

        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        ts = datetime(2026, 4, 7, 10, 30, tzinfo=timezone.utc)
        _on_trade_fill(
            strategy_id="strat-1",
            symbol="AAPL",
            action="buy",
            fill_price=150.05,
            fill_quantity=100,
            fill_timestamp=ts,
            arrival_price=150.0,
            daily_volume=5_000_000,
            order_id="ord-123",
        )

        mock_tracker.record_entry.assert_called_once()
        mock_tca.assert_called_once()
